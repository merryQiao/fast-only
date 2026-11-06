"""One-sample/one-update real-weight V1.1 interface audit; never full training."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
import yaml
from fast_navsim.collate import NavsimCollator
from fast_navsim.data import NavsimDataset
from fast_navsim.model import pose_loss
from fast_only.common import to_device, seed_all
from slowfast_v11.attention import check_compatibility
from slowfast_v11.model import FastWithSlowKV, SlowPrefill
from slowfast_v11.protocol import LayerKV, SourceMetadata, admit_snapshot, cache_metadata


def validate_config(c):
    if c['schema_version']!='slowfast_v11_p0':raise ValueError('Wrong V1.1 audit schema')
    if c['audit']['samples']!=1 or c['audit']['optimizer_updates']!=1:
        raise ValueError('This entry point is limited to one sample and one update')
    if c['fast']['device']==c['slow']['device'] and not c['audit']['allow_single_gpu']:
        raise ValueError('Single GPU requires explicit allow_single_gpu audit override')
    required={'layers_one_based':[4,8,12,16,20,24],'planning_queries':8,
        'fast_position_offset':4096,'native_state':'reset_each_forward',
        'source_key':'after_knorm_before_rope','loss':'existing_navsim_mean_l1_pose',
        'gradient_checkpointing':False,
        'metadata_fields':['age_s','dx_m','dy_m','sin_dyaw','cos_dyaw','enabled'],
        'metadata_units':'raw_SI','no_cache_policy':'learned_missing_token_when_absent_or_disabled',
        'fusion':'planning_rows_only_sdpa_before_native_output_projection'}
    if any(c['interface'][k]!=v for k,v in required.items()):
        raise ValueError('Unsupported interface config drift')
    data_required={'frequency_hz':2,'image_policy':'original_pad','camera':'CAM_F0','pad_multiple':32,
        'history_times_s':[-1.5,-1.,-.5,0.], 'future_times_s':[.5*i for i in range(1,9)]}
    if any(c['data'][k]!=v for k,v in data_required.items()):
        raise ValueError('Requires native-resolution NAVSIM 2Hz input')
    if c['model']['dtype']!='bfloat16':raise ValueError('Real audit requires BF16 base weights')


def _gradients(parameters):
    grads=[p.grad for p in parameters if p.grad is not None]
    return {'count':len(grads),'finite':bool(grads) and all(torch.isfinite(g).all().item() for g in grads),
        'nonzero':any(g.count_nonzero().item()>0 for g in grads)}


def prepare_memory_measurement(devices):
    """Initialize/validate real devices before asking the CUDA allocator for stats."""
    for device in devices:
        torch.cuda.get_device_properties(device)
        torch.cuda.reset_peak_memory_stats(device)


def run_audit(c):
    validate_config(c);seed_all(c['seed']);torch.set_num_threads(4)
    devices=sorted(set([c['fast']['device'],c['slow']['device']]))
    if any(not d.startswith('cuda:') for d in devices):raise ValueError('Real audit requires CUDA devices')
    prepare_memory_measurement(devices)
    output=Path(c['audit']['output'])
    if output.exists():raise FileExistsError('Preserve previous evidence; choose a new audit output')
    started=time.monotonic()
    def log(message):print(f'[{time.monotonic()-started:.1f}s] {message}',flush=True)
    kwargs={k:v for k,v in c['model'].items() if k!='dtype'}
    kwargs['position_offset']=c['interface']['fast_position_offset']
    kwargs['dtype']=torch.bfloat16
    log('Loading complete Fast and Slow native backbones')
    fast=FastWithSlowKV.from_pretrained(c['fast']['model_path'],**kwargs).to(c['fast']['device']).eval()
    slow=SlowPrefill.from_pretrained(c['slow']['model_path'],**kwargs).to(c['slow']['device']).eval()
    compatibility=check_compatibility(fast.backbone.language_model,slow.backbone.language_model)
    dataset=NavsimDataset(c['data']['manifest'],c['data']['stats'])
    if not len(dataset):raise ValueError('Empty mini manifest')
    if any(row['split']!='train' for row in dataset.rows):raise ValueError('Audit update uses TRAIN only')
    sample=dataset[0]
    if len(sample['images'])!=4:raise ValueError('Exactly four history images')
    batches={}
    for name in ['fast','slow']:
        collator=NavsimCollator(c[name]['model_path'],policy=c['data']['image_policy'],
            pad_multiple=c['data']['pad_multiple'],history_times_s=c['data']['history_times_s'])
        batches[name]=to_device(collator([sample]),c[name]['device'])
    f,s=batches['fast'],batches['slow']
    sync=lambda:[torch.cuda.synchronize(d) for d in devices]
    def invoke_fast(**kw):return fast(f['native'],f['ego'],f['command'],**kw)
    def invoke_slow(feedback):return slow(s['native'],s['ego'],s['command'],feedback=feedback)
    log('Real native-resolution candidate plan and Slow snapshot')
    with torch.no_grad():
        candidate=invoke_fast().detach().to(c['slow']['device'])
        slow_result=invoke_slow(candidate)
        published=slow_result.snapshot.inference_copy(c['fast']['device'])
    del slow_result
    # Synthetic mapped times for admission plumbing only: no measured async-age claim.
    source=SourceMetadata(0.,.3,'audit_episode','audit_nav','local_2B',(0.,0.,0.))
    assert admit_snapshot(source,image_time=.5,selection_time=.6,
        episode_id='audit_episode',navigation_id='audit_nav',model_id='local_2B')
    published=replace(published,metadata=source)
    meta=cache_metadata(source,image_time=.5,current_pose=(0.,0.,0.),device=c['fast']['device'])[None]
    enabled=torch.ones(1,device=c['fast']['device'],dtype=torch.bool)
    disabled=~enabled
    changed=replace(published,layers=tuple(LayerKV(z.key+1,z.value+1) for z in published.layers))
    log('Absent/masked/content and context-isolation checks')
    with torch.no_grad():
        plain=invoke_fast()
        masked=invoke_fast(snapshot=published,metadata=meta,enabled=disabled)
        masked_other=invoke_fast(snapshot=changed,metadata=meta,enabled=disabled)
        captures=[]
        handle=fast.backbone.language_model.register_forward_hook(
            lambda module,args,out:captures.append(out.last_hidden_state.detach().cpu()))
        try:
            active=invoke_fast(snapshot=published,metadata=meta,enabled=enabled)
            other=invoke_fast(snapshot=changed,metadata=meta,enabled=enabled)
        finally:handle.remove()
        sync();begin=time.monotonic()
        invoke_fast(snapshot=published,metadata=meta,enabled=enabled)
        sync();active_ms=(time.monotonic()-begin)*1000
    checks={'masked_equals_absent':torch.equal(plain,masked),
        'masked_content_independent':torch.equal(masked,masked_other),
        'context_content_independent':torch.equal(captures[0][:,:-8],captures[1][:,:-8]),
        'active_content_changes_plan':not torch.equal(active,other),
        'active_max_plan_change_mixed_units':(active-other).abs().max().item()}
    del captures,published,changed,plain,masked,masked_other,active,other
    for d in devices:torch.cuda.empty_cache()
    log('Joint Fast-only-loss backward through live Slow KV (no detach)')
    # Eval disables stochastic dropout; gradients remain enabled throughout both LMs.
    # This is a graph/optimizer audit, not an optimization-quality experiment.
    live=invoke_slow(candidate)
    for layer in live.snapshot.layers:
        layer.key.retain_grad();layer.value.retain_grad()
    moved=live.snapshot.to(c['fast']['device'])
    prediction=invoke_fast(snapshot=moved,metadata=meta,enabled=enabled)
    loss=pose_loss(prediction,f['trajectory_gt'])
    if not torch.isfinite(loss):raise AssertionError('Nonfinite planning loss')
    loss.backward();sync()
    grads={
        'slow_lora_from_fast_loss':_gradients(p for n,p in slow.named_parameters() if 'lora_' in n),
        'slow_feedback_from_fast_loss':_gradients(slow.feedback_encoder.parameters()),
        'fast_metadata':_gradients(fast.metadata_encoder.parameters()),
        'slow_kv_layers':[{'key':_gradients([z.key]),'value':_gradients([z.value])} for z in live.snapshot.layers],
        'fast_visual_frozen':all(not p.requires_grad and p.grad is None for p in fast.backbone.visual.parameters()),
        'slow_visual_frozen':all(not p.requires_grad and p.grad is None for p in slow.backbone.visual.parameters())}
    params=[p for model in (fast,slow) for p in model.parameters() if p.requires_grad]
    all_finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    before=fast.head[-1].weight.detach().clone()
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step();sync()
    checks['optimizer_changed_fast_head']=not torch.equal(before,fast.head[-1].weight)
    checks['all_trainable_gradients_finite']=all_finite
    checks['fast_loss_reaches_slow']=grads['slow_lora_from_fast_loss']['nonzero'] and all(
        z[field]['nonzero'] and z[field]['finite'] for z in grads['slow_kv_layers'] for field in ['key','value'])
    checks['new_input_gradients_valid']=all(grads[name]['finite'] and grads[name]['nonzero']
        for name in ['slow_feedback_from_fast_loss','fast_metadata'])
    checks['visual_frozen_and_without_gradient']=grads['fast_visual_frozen'] and grads['slow_visual_frozen']
    memory={d:{'gpu':torch.cuda.get_device_name(d),
        'peak_allocated_gib':torch.cuda.max_memory_allocated(d)/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved(d)/2**30} for d in devices}
    evidence={'passed':all(v for v in checks.values() if isinstance(v,bool)),
        'scope':'P0/P1 interface audit; untrained planning additions; not KV planning benefit',
        'multi_gpu_validated':len(devices)>1,'device_assignment':{n:c[n]['device'] for n in ['fast','slow']},
        'clock_test':'synthetic mapped times only; actual async publication not implemented',
        'compatibility':compatibility,'checks':checks,'gradients':grads,
        'image_audit':{n:batches[n]['image_audit'] for n in ['fast','slow']},
        'layout':{'fast':fast.last_layout,'slow':slow.last_layout},
        'samples':1,'optimizer_updates':1,'fast_loss_only':loss.item(),'grad_norm':float(norm),
        'active_fast_model_only_ms_single_measurement':active_ms,'memory':memory,
        'elapsed_s':time.monotonic()-started,'torch':torch.__version__,
        'seed':c['seed'],'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'feature_future_head':'not implemented',
        'native_training_checkpoint_resume':'not implemented for joint prototype'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(evidence,indent=2)+'\n')
    log(f'Audit written; passed={evidence["passed"]}')
    if not evidence['passed']:raise AssertionError('Interface audit failed; inspect saved evidence')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/slowfast_v11_p0.yaml')
    for flag in ['fast-model','slow-model','fast-device','slow-device','manifest','stats','output']:
        parser.add_argument('--'+flag)
    parser.add_argument('--allow-single-gpu',action='store_true')
    args=parser.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    for name in ['fast','slow']:
        for arg,key in [('model','model_path'),('device','device')]:
            value=getattr(args,f'{name}_{arg}')
            if value is not None:c[name][key]=value
    for arg in ['manifest','stats']:
        if getattr(args,arg) is not None:c['data'][arg]=getattr(args,arg)
    if args.output is not None:c['audit']['output']=args.output
    if args.allow_single_gpu:c['audit']['allow_single_gpu']=True
    run_audit(c)


if __name__=='__main__':main()
