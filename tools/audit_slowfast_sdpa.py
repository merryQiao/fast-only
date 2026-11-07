"""Isolated FA4 native/cropped/masked-prefix SDPA replay; no model updates.

Run only after the readability GPU job has ended. Capture one predeclared real
Fast forward, then replay identical post-RoPE Q/K/V under explicit geometries.
Full-query FP32/math attention is deliberately excluded to avoid quadratic RAM.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F


EXTERNAL_TOKENS=8255
_ACTIVE=False


def tensor_difference(reference,actual):
    if reference.shape!=actual.shape:raise ValueError('Comparison tensor layouts differ')
    a=reference.detach().float();b=actual.detach().float();delta=b-a
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite numerical audit tensor')
    return dict(exact=bool(torch.equal(reference,actual)),max_abs_difference=float(delta.abs().max()),
        rms_difference=float(delta.square().mean().sqrt()),
        relative_l2=float(delta.norm()/a.norm().clamp_min(1e-12)))


@contextmanager
def capture_attention(attention):
    """Scoped read-only hooks plus SDPA interception, filtered to one module call."""
    global _ACTIVE
    if _ACTIVE or torch.is_grad_enabled() or attention.training:
        raise RuntimeError('Capture requires isolated eval/no_grad inference')
    _ACTIVE=True;records={};handles=[];active=False;original=F.scaled_dot_product_attention
    def put(name,value):
        if name in records:raise ValueError('Expected exactly one target-attention prefill')
        records[name]=value.detach().clone(memory_format=torch.preserve_format)
    def begin(module,args,kwargs):
        nonlocal active
        if 'position_embeddings' in records:raise ValueError('Expected one target-attention forward')
        positions=kwargs['position_embeddings'] if 'position_embeddings' in kwargs else args[1]
        records['position_embeddings']=tuple(t.detach().clone() for t in positions)
        mask=kwargs.get('attention_mask',args[2] if len(args)>2 else None)
        records['native_mask']=None if mask is None else mask.detach().clone()
        active=True
    def end(module,args,output):
        nonlocal active
        active=False
    def gate(module,args,output):
        b,t,_=output.shape
        put('gate',output.view(b,t,-1,2*attention.head_dim).chunk(2,dim=-1)[1].reshape(b,t,-1))
    def wrapped(query,key,value,*args,**kwargs):
        output=original(query,key,value,*args,**kwargs)
        if active:
            if args:raise ValueError('Unexpected native positional SDPA options')
            put('query',query);put('key',key);put('value',value);put('native_sdpa_output',output)
            records['sdpa_options']={name:kwargs.get(name,default) for name,default in
                [('dropout_p',0.),('is_causal',False),('scale',None),('enable_gqa',False)]}
            actual_mask=kwargs.get('attn_mask')
            records['sdpa_mask']=None if actual_mask is None else actual_mask.detach().clone()
        return output
    try:
        handles.append(attention.register_forward_pre_hook(begin,with_kwargs=True))
        handles.append(attention.register_forward_hook(end,always_call=True))
        handles.append(attention.q_norm.register_forward_hook(lambda m,a,o:put('normalized_query',o)))
        handles.append(attention.k_norm.register_forward_hook(lambda m,a,o:put('normalized_key',o)))
        handles.append(attention.q_proj.register_forward_hook(gate))
        handles.append(attention.o_proj.register_forward_pre_hook(lambda m,a:put('post_gate',a[0])))
        F.scaled_dot_product_attention=wrapped
        yield records
    finally:
        F.scaled_dot_product_attention=original
        for handle in handles:handle.remove()
        _ACTIVE=False


@torch.no_grad()
def replay_native(attention,captured):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    q,k,v=(captured[name] for name in ('query','key','value'))
    options=captured['sdpa_options']
    result,_=sdpa_attention_forward(attention,q,k,v,captured['sdpa_mask'],
        dropout=options['dropout_p'],scaling=options['scale'],is_causal=options['is_causal'])
    post=result.reshape(*captured['gate'].shape).contiguous()*captured['gate'].sigmoid()
    expected=captured['native_sdpa_output'].transpose(1,2).contiguous()
    rq,rk=apply_rotary_pos_emb(captured['normalized_query'].transpose(1,2),
        captured['normalized_key'].transpose(1,2),*captured['position_embeddings'])
    return dict(post_gate_exact=bool(torch.equal(post,captured['post_gate'])),
        post_gate_difference=tensor_difference(captured['post_gate'],post),
        native_attention_difference=tensor_difference(expected,result),
        recomputed_rope_query_exact=bool(torch.equal(q,rq)),recomputed_rope_key_exact=bool(torch.equal(k,rk)),
        recomputed_rope_query_difference=tensor_difference(q,rq),recomputed_rope_key_difference=tensor_difference(k,rk))


def attention_geometry(query,key,value,*,queries=8,external_tokens=EXTERNAL_TOKENS,scale=None):
    """Three mathematically equivalent crops using the correct bottom-right mask."""
    if (query.ndim!=4 or key.ndim!=4 or value.shape!=key.shape or key.shape[0]!=query.shape[0]
            or query.shape[-2]!=key.shape[-2] or query.shape[-1]!=key.shape[-1]
            or not 0<queries<=query.shape[-2] or query.shape[1]%key.shape[1] or external_tokens<0):
        raise ValueError('Unsupported fixed Q/K/V crop layout')
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    b,h,t,d=query.shape;groups=h//key.shape[1];q=query[:,:,-queries:]
    mask=(torch.arange(t,device=query.device)[None,:]<=torch.arange(t-queries,t,device=query.device)[:,None])[None,None]
    mask=mask.expand(b,1,-1,-1)
    common=dict(query=q,attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale)
    cases={'crop_gqa':dict(common,key=key,value=value,enable_gqa=True),
           'crop_repeat':dict(common,key=repeat_kv(key,groups),value=repeat_kv(value,groups),enable_gqa=False)}
    source_shape=(b,key.shape[1],external_tokens,d)
    extended_key=repeat_kv(torch.cat((key.new_zeros(source_shape),key),dim=2),groups)
    extended_value=repeat_kv(torch.cat((value.new_zeros(source_shape),value),dim=2),groups)
    extended_mask=torch.cat((torch.zeros(b,1,queries,external_tokens,device=query.device,dtype=torch.bool),mask),dim=-1)
    cases['crop_repeat_masked_external']=dict(common,key=extended_key,value=extended_value,
        attn_mask=extended_mask,enable_gqa=False)
    return cases


def _tensor_identity(tensor):
    value=tensor.detach().cpu().contiguous()
    digest=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),sha256=digest)


def _profile_call(function):
    # CPU dispatcher names expose the actual SDPA backend without requiring CUPTI.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output=function()
    return output,sorted({event.key for event in profile.key_averages() if 'scaled_dot_product' in event.key})


@torch.no_grad()
def run(root,output):
    from fast_only.common import fingerprint,write_json
    from slowfast_v11.validation import canonical_state,state_digest
    from tools.train_slowfast_expanded import ExpandedExperiment
    from tools.reload_slowfast_expanded import checked_checkpoint
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    from transformers.integrations import sdpa_attention
    from torch.nn.attention import sdpa_kernel,SDPBackend

    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Use a new numerical audit JSON path')
    started=time.monotonic();original=json.loads((root/'result.json').read_text())
    experiment=ExpandedExperiment(copy.deepcopy(original['config']))
    checkpoint_path=root/'joint_final.pt';checkpoint_sha=original['arms']['joint']['checkpoint_sha256']
    saved=checked_checkpoint(checkpoint_path,experiment,original,'joint',800,checkpoint_sha)
    row=experiment.cohorts['new_val'][0]  # Same first predeclared Step1 source, no error-based selection.
    references=[r for r in saved['reference_predictions'] if r['token']==row['token'] and r['mode']=='none'
                and r['split']=='new_val' and r['stage']=='joint_final']
    if len(references)!=1:raise ValueError('Expected exactly one fixed-token checkpoint none reference')
    reference_prediction=torch.tensor(references[0]['prediction'],dtype=torch.float32)
    if reference_prediction.shape!=(8,3):raise ValueError('Invalid checkpoint trajectory reference')
    del saved
    for name in ('fast','slow'):getattr(experiment,name).eval().requires_grad_(False)
    before={name:state_digest(canonical_state(getattr(experiment,name))) for name in ('fast','slow')}
    versions={name:tuple(p._version for p in getattr(experiment,name).parameters()) for name in ('fast','slow')}
    batch=experiment.batch(row['token'],'fast');attention=experiment.fast.backbone.language_model.layers[3].self_attn
    with torch.autocast('cuda',enabled=False):
        with capture_attention(attention) as captured:prediction=experiment.fast(**batch)
        native_replay,operators=_profile_call(lambda:replay_native(attention,captured))
    if captured['sdpa_mask'] is not None or captured['native_mask'] is not None:
        raise ValueError('Expected the actual unpadded native mask=None prefill')
    options=captured['sdpa_options']
    if (not options['enable_gqa'] or not options['is_causal'] or options['dropout_p']!=0.
            or options['scale']!=.0625 or captured['query'].shape[1]!=8 or captured['key'].shape[1]!=2
            or captured['query'].shape[-1]!=256):
        raise ValueError('Actual native attention regime differs from the fixed audit contract')
    q,k,v=(captured[name] for name in ('query','key','value'))
    cases=[];outputs={}
    for dtype_name,dtype,regimes in [('bf16',torch.bfloat16,('default','math')),
                                    ('fp32',torch.float32,('math',))]:
        geometry=attention_geometry(q.to(dtype),k.to(dtype),v.to(dtype),scale=options['scale'])
        for regime in regimes:
            for name,arguments in geometry.items():
                def evaluate(arguments=arguments):
                    if regime=='math':
                        with sdpa_kernel(backends=[SDPBackend.MATH]):return F.scaled_dot_product_attention(**arguments)
                    return F.scaled_dot_product_attention(**arguments)
                with torch.autocast('cuda',enabled=False):value,kernels=_profile_call(evaluate)
                label=f'{dtype_name}_{regime}_{name}';outputs[label]=value.detach()
                cases.append(dict(label=label,dtype=dtype_name,backend_policy=regime,geometry=name,
                    query_shape=list(arguments['query'].shape),key_shape=list(arguments['key'].shape),
                    mask_shape=list(arguments['attn_mask'].shape),enable_gqa=arguments['enable_gqa'],operators=kernels))
        del geometry
    fp32_reference=outputs['fp32_math_crop_gqa'];native_crop=captured['native_sdpa_output'][:,:,-8:]
    for case in cases:
        value=outputs[case['label']]
        case['vs_native_auto_q8']=tensor_difference(native_crop,value)
        case['vs_fp32_math_crop_gqa']=tensor_difference(fp32_reference,value)
        # Match the BF16 native gate exactly; this separates attention drift from gating.
        gate=captured['gate'][:,-8:]
        post=value.to(gate.dtype).transpose(1,2).contiguous().reshape(*gate.shape)*gate.sigmoid()
        case['post_gate_vs_native']=tensor_difference(captured['post_gate'][:,-8:],post)
    after={name:state_digest(canonical_state(getattr(experiment,name))) for name in ('fast','slow')}
    unchanged=before==after and all(tuple(p._version for p in getattr(experiment,name).parameters())==versions[name]
                                  for name in ('fast','slow'))
    if not unchanged:raise ValueError('Numerical audit mutated source parameters')
    evidence=dict(schema='slowfast_sdpa_numerical_audit_v1',token=row['token'],cohort=row['cohort'],route=row['route'],
        source_checkpoint=checkpoint_path.name,source_checkpoint_sha256=checkpoint_sha,
        source_arm='joint',source_step=800,assets=experiment.assets,data_identities=experiment.data_identities,
        source_parameters_unchanged=unchanged,parameter_digests=after,original_none_prediction=prediction[0].float().cpu().tolist(),
        original_none_vs_checkpoint_reference=tensor_difference(reference_prediction,prediction[0].float().cpu()),
        native_sdpa_options=options,native_helper_replay=native_replay,native_replay_operators=operators,
        captured={name:_tensor_identity(captured[name]) for name in
            ('query','key','value','normalized_query','normalized_key','gate','post_gate')},
        rotary_embeddings=[_tensor_identity(t) for t in captured['position_embeddings']],
        external_tokens=EXTERNAL_TOKENS,external_content='finite zeros; every external column masked',
        comparisons=cases,native_auto_vs_fp32_math_crop=tensor_difference(fp32_reference,native_crop),
        full_query_math_run=False,full_query_fp32_run=False,
        runtime=dict(torch=version('torch'),transformers=version('transformers'),gpu=torch.cuda.get_device_name(0),
            cuda=torch.version.cuda,reproducibility=experiment.c['reproducibility']),
        source_code_sha256={'tools/audit_slowfast_sdpa.py':fingerprint(__file__),
            'slowfast_v11/attention.py':fingerprint('slowfast_v11/attention.py'),
            'transformers/modeling_qwen3_5.py':fingerprint(native.__file__),
            'transformers/sdpa_attention.py':fingerprint(sdpa_attention.__file__)},
        elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        new_model_updates=0,new_download_bytes=0,full_eval_run=False)
    write_json(output,evidence)
    print(json.dumps(dict(native_post_gate_exact=native_replay['post_gate_exact'],cases=len(cases),
                         source_parameters_unchanged=unchanged)),flush=True)
    if not native_replay['post_gate_exact']:raise ValueError('Native helper replay did not reproduce actual post-gate tensor exactly')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/slowfast_v11_expanded/run_01')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.run,args.output)


if __name__=='__main__':main()
