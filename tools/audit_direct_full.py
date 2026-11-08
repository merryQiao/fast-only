"""Real H800 gate for full language masters, acc4 memory and base-weight updates."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from fast_only.common import apply_reproducibility,seed_all,fingerprint,write_json
from fast_navsim.model import pose_loss
from slowfast_v11.full_training import configure_full_fast,make_full_optimizer,full_state,restore_full_state,full_precision
from slowfast_v11.validation import state_digest
from tools.direct_full_common import read_config,load_fast,train_rows,vision_packet,LegacySources,equal_source_layout,direct_forward
from tools.train_slowfast_joint import capture_rng,restore_rng,compare_trees


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def block_probes(model):
    out=[]
    for i,block in enumerate(model.backbone.language_model.layers):
        named=[(n,p) for n,p in block.named_parameters() if 'lora_' not in n and p.grad is not None]
        out.append(dict(block=i+1,finite=all(bool(torch.isfinite(p.grad).all()) for _,p in named),
            nonzero=any(bool(p.grad.count_nonzero()) for _,p in named),tensors_with_grad=len(named)))
    return out


def run(config_path,output):
    c=read_config(config_path);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh P0 output')
    output.mkdir(parents=True);started=time.monotonic()
    source=json.loads((Path(c['source_run'])/'result.json').read_text());apply_reproducibility(source['config']);seed_all(42);torch.set_num_threads(4)
    model,collator,stats,_=load_fast(c);model.eval().requires_grad_(False)
    rows=train_rows(c);selected=rows[:2];reference=min(r['token'] for r in rows)
    packets={r['token']:vision_packet(model,collator,r,stats) for r in selected};store=LegacySources(c)
    sources={t:store.packet(t) for t in set(packets)|{reference}}
    for t in packets:equal_source_layout(sources[t],sources[reference])
    scope=configure_full_fast(model);initial=full_state(model);digest=state_digest(initial)
    torch.save(dict(schema='direct_full_shared_init_v1',state=initial,digest=digest,config=c,source_fast=c['fast_checkpoint_sha256']),output/'shared_init.pt')
    results={};precision=[];roe=[]
    handles=[]
    handles.append(model.head.register_forward_hook(lambda mod,args,out:precision.append(dict(kind='head',input=str(args[0].dtype),output=str(out.dtype),amp=torch.is_autocast_enabled('cuda')))))
    handles.append(model.ego_mlp.register_forward_hook(lambda mod,args,out:precision.append(dict(kind='ego_mlp',input=str(args[0].dtype),output=str(out.dtype),amp=torch.is_autocast_enabled('cuda')))))
    for name,module in model.backbone.language_model.named_modules():
        if '.lora_A.' in name or '.lora_B.' in name:
            if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(lambda mod,args,out:precision.append(dict(kind='lora',input=str(args[0].dtype),output=str(out.dtype)))))
    handles.append(model.backbone.language_model.rotary_emb.register_forward_hook(lambda mod,args,out:roe.append(dict(positions=args[1].detach().clone(),cos=out[0].detach().clone(),sin=out[1].detach().clone()))))
    with full_precision(model):
        model.eval()
        with torch.no_grad():
            initialization_checks=[]
            for row in selected:
                token=row['token'];p=direct_forward(model,packets[token],'full_fast_only').float().cpu()
                initialization_checks.append(dict(token=token,max_abs_difference=float((p-packets[token]['original_prediction']).abs().max())))
        for arm in c['arms']:
            restore_full_state(model,initial);model.zero_grad(set_to_none=True);model.train();model.backbone.visual.eval()
            opt=make_full_optimizer(model,language_lr=c['language_lr'],new_lr=c['new_lr'],weight_decay=c['weight_decay'])
            if state_digest(full_state(model))!=digest:raise ValueError('P0 arm did not reset to shared full init')
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();entries=[];start=time.monotonic()
            sampled={i:next((n,p) for n,p in block.named_parameters() if 'lora_' not in n and p.ndim==2)
                     for i,block in enumerate(model.backbone.language_model.layers)}
            probes_before={i:p.detach().clone() for i,(_,p) in sampled.items()}
            before_second=None
            for step in [1,2]:
                if step==2:before_second=dict(model=full_state(model),optimizer=cpu_tree(opt.state_dict()),rng=capture_rng())
                opt.zero_grad(set_to_none=True)
                for micro in range(4):
                    row=selected[micro%2];token=row['token'];s=None if arm=='full_fast_only' else sources[token if arm=='full_correct_kv' else reference]
                    prediction=direct_forward(model,packets[token],arm,s)
                    loss=pose_loss(prediction,torch.tensor([row['trajectory_gt']],device='cuda:0'))/4
                    loss.backward()
                    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite full-language P0 loss')
                probes=block_probes(model);norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                if not all(r['finite'] and r['nonzero'] for r in probes):raise ValueError('Full base blocks did not receive gradients')
                special={n:dict(finite=bool(torch.isfinite(p.grad).all()),nonzero=bool(p.grad.count_nonzero())) for n,p in model.backbone.language_model.named_parameters()
                         if p.grad is not None and (n=='embed_tokens.weight' or n=='norm.weight')}
                opt.step();entries.append(dict(step=step,loss=float(loss.detach())*4,grad_norm=float(norm),blocks=probes,embedding_and_final_norm=special))
            after_second=dict(model=full_state(model),optimizer=cpu_tree(opt.state_dict()),rng=capture_rng())
            restore_full_state(model,before_second['model']);opt.load_state_dict(before_second['optimizer']);restore_rng(before_second['rng'])
            opt.zero_grad(set_to_none=True)
            for micro in range(4):
                row=selected[micro%2];token=row['token'];s=None if arm=='full_fast_only' else sources[token if arm=='full_correct_kv' else reference]
                replay_loss=pose_loss(direct_forward(model,packets[token],arm,s),torch.tensor([row['trajectory_gt']],device='cuda:0'))/4
                replay_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
            replay={k:compare_trees(after_second[k],v) for k,v in dict(model=full_state(model),optimizer=cpu_tree(opt.state_dict()),rng=capture_rng()).items()}
            if not all(x['passed'] for x in replay.values()):raise ValueError('Full-state accumulated-update replay differs')
            deltas=[]
            for i,block in enumerate(model.backbone.language_model.layers):
                p=next(p for n,p in block.named_parameters() if 'lora_' not in n and p.ndim==2)
                delta=float((p.detach()-probes_before[i]).abs().max());deltas.append(delta)
            if any(v==0 for v in deltas):raise ValueError('Dense base weights did not actually update')
            peak=torch.cuda.max_memory_allocated()/2**30
            if peak>c['max_allocated_gib']:raise MemoryError('Actual full-language acc4 AdamW exceeds declared memory cap')
            if any(p.requires_grad or p.grad is not None for p in model.backbone.visual.parameters()):raise ValueError('Vision/merger received gradients')
            results[arm]=dict(updates=entries,base_max_abs_changes=deltas,peak_allocated_gib=peak,elapsed_s=time.monotonic()-start,
                              optimizer_state_parameters=len(opt.state),trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              sampled_base_parameter_names={str(i+1):n for i,(n,_) in sampled.items()},actual_accumulated_update_replay=replay)
            print(f'P0 {arm}:peak={peak:.2f}GiB,elapsed={time.monotonic()-start:.1f}s',flush=True)
            del opt,probes_before,prediction,loss,before_second,after_second,replay_loss;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        restore_full_state(model,initial)
    for handle in handles:handle.remove()
    if state_digest(full_state(model))!=digest:raise ValueError('Disposable P0 weights leaked into initial checkpoint')
    # Native and bridge calls with identical current positions must produce identical BF16 RoPE.
    matched=0
    for i,a in enumerate(roe):
        if a['cos'].dtype!=torch.bfloat16 or a['sin'].dtype!=torch.bfloat16:raise ValueError('RoPE must match native BF16 language input precision')
        for b in roe[:i]:
            if torch.equal(a['positions'],b['positions']):
                if not torch.equal(a['cos'],b['cos']) or not torch.equal(a['sin'],b['sin']):raise ValueError('Native/bridge RoPE precision mismatch')
                matched+=1;break
    if not matched or any(x['input']!='torch.float32' or x['output']!='torch.float32' or x.get('amp',False) for x in precision):raise ValueError('Head/LoRA FP32 compute contract failed')
    write_json(output/'result.json',dict(passed=True,scope=scope,initial_digest=digest,shared_init_sha256=fingerprint(output/'shared_init.pt'),
        initialization_checks=initialization_checks,arms=results,precision_probes=len(precision),matching_rope_checks=matched,
        dtype_observations=sorted({json.dumps(r,sort_keys=True) for r in precision}),
        image_audits={t:p['image_audit'] for t,p in packets.items()},raw_KV_manifest_sha256=fingerprint(store.root/'manifest.json'),
        raw_KV_shards_verified=sorted(store.verified),
        source_checkpoint=c['source_checkpoint_sha256'],fast_checkpoint=c['fast_checkpoint_sha256'],normalizer_sha256=source['data_identities']['stats'],
        p0_updates_discarded=True,full_init_restored=True,gradient_checkpointing=False,elapsed_s=time.monotonic()-started,
        source_code_sha256={p:fingerprint(p) for p in ['tools/audit_direct_full.py','tools/direct_full_common.py','slowfast_v11/full_training.py']}))
    print('Full-language direct-KV P0 complete; all engineering updates discarded',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_direct_full.yaml');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)


if __name__=='__main__':main()
