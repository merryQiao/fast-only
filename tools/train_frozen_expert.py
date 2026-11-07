"""Bounded frozen-Slow ActionExpert vs complete Fast+LoRA control."""
import argparse
from contextlib import contextmanager,nullcontext
import copy
import gc
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import yaml
from fast_only.common import fingerprint,write_json,seed_all,to_device
from fast_navsim.model import pose_loss,trajectory_metrics
from slowfast_v11.protocol import LayerKV,KVSnapshot
from slowfast_v11.validation import canonical_state,state_digest,set_stage,make_optimizer,restore_weights
from slowfast_v11.expanded_protocol import epoch_plan,WarmupCosineScheduler
from tools.train_slowfast_expanded import ExpandedExperiment
from tools.reload_slowfast_expanded import checked_checkpoint
from tools.diagnose_slowfast_readability import FeatureCache,state_cpu
from tools.train_slowfast_joint import capture_rng


@contextmanager
def cached_vision(backbone,features,grid):
    original=backbone.get_image_features
    def get(pixel_values,image_grid_thw,**kwargs):
        if not torch.equal(image_grid_thw,grid):raise ValueError('Cached vision grid mismatch')
        return SimpleNamespace(pooler_output=features)
    backbone.get_image_features=get
    try:yield
    finally:backbone.get_image_features=original


def snapshot_for(cache,target,source,kv_heads=2,masked=False):
    i,j=cache.index[target],cache.index[source];length,width=cache.keys.shape[2:]
    layers=tuple(LayerKV(cache.keys[j,l].view(1,length,kv_heads,width//kv_heads).transpose(1,2),
                        cache.values[j,l].view(1,length,kv_heads,width//kv_heads).transpose(1,2)) for l in range(6))
    valid=cache.valid[i:i+1];valid=torch.zeros_like(valid) if masked else valid
    return KVSnapshot(layers,cache.positions[:,i:i+1],valid)


def load_cache(root,expected_source,expected_data):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['source_checkpoint']!=expected_source or manifest['data_identities']!=expected_data:
        raise ValueError('Frozen cache source/data identity mismatch')
    tokens=[t for s in manifest['shards'] for t in s['tokens']]
    shape=manifest['raw_KV_shape']
    if shape[0]!=len(tokens) or len(set(tokens))!=len(tokens) or shape[1:]!=[6,8255,512]:
        raise ValueError('Frozen cache geometry or tokens differ')
    cache=object.__new__(FeatureCache);cache.tokens=tokens;cache.index={t:i for i,t in enumerate(tokens)}
    cache.keys=torch.empty(shape,device='cuda:0',dtype=torch.bfloat16);cache.values=torch.empty_like(cache.keys)
    cache.positions=torch.empty(3,len(tokens),shape[2],device='cuda:0',dtype=torch.long)
    cache.groups=torch.empty(len(tokens),shape[2],device='cuda:0',dtype=torch.long)
    cache.valid=torch.empty(len(tokens),shape[2],device='cuda:0',dtype=torch.bool)
    cache.final=torch.zeros(len(tokens),8,1,device='cuda:0')  # unused in consumer controls
    for number,shard in enumerate(manifest['shards']):
        path=root/shard['path']
        if fingerprint(path)!=shard['sha256']:raise ValueError('Frozen cache shard SHA mismatch')
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if saved['tokens']!=shard['tokens']:raise ValueError('Frozen cache token order differs')
        indices=torch.tensor([cache.index[t] for t in saved['tokens']],device='cuda:0')
        for name in ['keys','values','groups','valid']:getattr(cache,name)[indices]=saved[name].to('cuda:0')
        cache.positions[:,indices]=saved['positions'].to('cuda:0')
        if number%8==0:print(f'loaded frozen shard{number+1}/{len(manifest["shards"])}',flush=True)
    return cache


@torch.no_grad()
def prepare_fast_inputs(e,cache):
    vision={};inputs={};checks=[];original=e.fast.backbone.get_image_features
    test_tokens={e.train[0]['token'],e.cohorts['new_val'][0]['token']}
    for number,row in enumerate(e.rows,1):
        token=row['token'];batch=e.batch(token,'fast');native=batch['native']
        features=original(native['pixel_values'],native['image_grid_thw'],return_dict=True).pooler_output
        vision[token]=tuple(t.detach().clone() for t in features)
        small=dict(native);small['pixel_values']=torch.empty(0,device='cuda:0')
        inputs[token]=dict(batch,native=small)
        if token in test_tokens:
            snap=snapshot_for(cache,token,token);metadata=e.metadata(e.fresh[token])
            before=e.fast(**batch,snapshot=snap,metadata=metadata)
            with cached_vision(e.fast.backbone,vision[token],small['image_grid_thw']):
                after=e.fast(**inputs[token],snapshot=snap,metadata=metadata)
            difference=float((before-after).abs().max());checks.append(dict(token=token,max_abs_difference=difference))
            if difference:raise ValueError('Cached frozen vision changed planning prediction')
        if number%32==0:print(f'cached original frozen vision{number}/{len(e.rows)}',flush=True)
    e.batch_cache.clear();gc.collect();torch.cuda.empty_cache()
    return inputs,vision,checks


def control_metrics(records):
    return trajectory_metrics(torch.tensor([r['prediction'] for r in records]),torch.tensor([r['target'] for r in records]))


def run(config,arm,output):
    c=yaml.safe_load(Path(config).read_text());output=Path(output)
    if output.exists():raise FileExistsError('Use a new output directory')
    if (c['schema']!='frozen_slow_expert_control_v1' or c['steps']!=4096 or c['micro_batch']!=1 or c['seed']!=42
            or c['train_size']!=256 or c['expert']!=dict(width=512,ffn_width=1536,layers=24,query_heads=8,kv_heads=2,head_dim=256,position_offset=4096)):
        raise ValueError('Bounded consumer-control protocol differs')
    output.mkdir(parents=True);started=time.monotonic();root=Path(c['source_run'])
    source=json.loads((root/'result.json').read_text());e=ExpandedExperiment(copy.deepcopy(source['config']));e.output=output
    saved=checked_checkpoint(root/'joint_final.pt',e,source,'joint',800,source['arms']['joint']['checkpoint_sha256']);del saved
    order=epoch_plan(e.train,c['steps'],c['seed']);reference=min(r['token'] for r in e.train)
    text_config=copy.deepcopy(e.slow.backbone.config.text_config)
    cache=load_cache(Path(c['feature_run'])/'frozen_cache',source['arms']['joint']['checkpoint_sha256'],e.data_identities)
    if set(cache.tokens)!=set(e.by_token):raise ValueError('Frozen cache source coverage differs')
    del e.slow;e.fast.zero_grad(set_to_none=True);e.fast.eval();gc.collect();torch.cuda.empty_cache()
    seed_all(c['seed']);inputs=vision=None;vision_checks=[]
    if arm=='expert':
        from slowfast_v11.action_expert import ActionExpert
        del e.fast;gc.collect();torch.cuda.empty_cache()
        model=ActionExpert(text_config=text_config,**c['expert']).cuda()
        block=[p for n,p in model.named_parameters() if n.startswith('blocks.')]
        new=[p for n,p in model.named_parameters() if not n.startswith('blocks.')]
        optimizer=torch.optim.AdamW([dict(params=block,lr=c['block_lr']),dict(params=new,lr=c['conditioning_head_lr'])],weight_decay=c['weight_decay'])
        initializer='new_random_145M_full_parameter_expert'
    elif arm=='fast':
        model=e.fast
        inputs,vision,vision_checks=prepare_fast_inputs(e,cache)
        set_stage(model,'fast_consumer');optimizer=make_optimizer(model,dict(lora_lr=c['block_lr'],new_lr=c['conditioning_head_lr'],weight_decay=c['weight_decay']))
        initializer='existing_joint_final_Fast;prior1312exposures'
    else:raise ValueError('Unknown consumer arm')
    schedule=WarmupCosineScheduler(optimizer,c['steps'],c['warmup_steps'],c['min_lr_ratio'])
    source_versions=tuple(t._version for t in [cache.keys,cache.values,cache.positions,cache.groups,cache.valid])
    protocol=dict(config=c,arm=arm,source_checkpoint=source['arms']['joint']['checkpoint_sha256'],
        source_data=e.data_identities,assets=e.assets,order=order,initializer=initializer,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),prior_Fast_exposures=1312 if arm=='fast' else 0,
        frozen_source_versions=source_versions,vision_checks=vision_checks,new_download_bytes=0,
        feature_manifest_sha256=fingerprint(Path(c['feature_run'])/'frozen_cache/manifest.json'),
        caveat='same_new_samples_not_same_params_FLOPs_initialization_or_visual_path',
        source_code_sha256={str(p):fingerprint(p) for p in [Path(__file__),Path('slowfast_v11/action_expert.py')]})
    write_json(output/'protocol.json',protocol)
    def checkpoint(step,reference_prediction=None):
        return dict(schema='frozen_slow_expert_control_v1',arm=arm,config=c,
            model=state_cpu(model) if arm=='expert' else canonical_state(model),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
            rng=capture_rng(),order=order,next_index=step,protocol=protocol,text_config=text_config.to_dict(),reference_prediction=reference_prediction)
    def forward(token,mode='correct'):
        source_token=token if mode in ['correct','masked'] else reference if mode=='reference' else e.mismatches[token]
        row=e.by_token[token];metadata=e.metadata(e.fresh[token])
        if arm=='expert':
            batch,_=cache.batch([token],[source_token])
            if mode=='masked':batch['valid']=torch.zeros_like(batch['valid'])
            ego=torch.tensor([[(x-m)/s for x,m,s in zip(row['ego'],e.stats['ego_mean'],e.stats['ego_std'])]],device='cuda:0')
            command=torch.tensor([row['command']],device='cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):return model(**batch,ego=ego,command=command,metadata=metadata)
        with cached_vision(model.backbone,vision[token],inputs[token]['native']['image_grid_thw']):
            return model(**inputs[token],snapshot=snapshot_for(cache,token,source_token,masked=mode=='masked'),metadata=metadata)
    @torch.no_grad()
    def evaluate(label,rows,modes):
        model.eval();records=[]
        for row in rows:
            for mode in modes:
                prediction=forward(row['token'],mode)[0].float().cpu().tolist()
                records.append(dict(stage=label,token=row['token'],cohort=row['cohort'],route=row['route'],mode=mode,
                                    speed_mps=math.hypot(*row['ego'][:2]),prediction=prediction,target=row['trajectory_gt']))
        return records
    curves={};model.train();losses=[]
    for step,token in enumerate(order,1):
        if step==4096:
            torch.save(checkpoint(4095),output/'before_final.pt')
            packet=cache.packet([token]);row=e.by_token[token]
            packet.update(token=token,target=row['trajectory_gt'],metadata=e.metadata(e.fresh[token]).cpu(),
                ego=torch.tensor([[(x-m)/s for x,m,s in zip(row['ego'],e.stats['ego_mean'],e.stats['ego_std'])]]),
                command=torch.tensor([row['command']]))
            if arm=='fast':
                packet['native']=to_device(inputs[token]['native'],'cpu')
                packet['vision']=tuple(t.cpu() for t in vision[token])
                packet['ego']=inputs[token]['ego'].cpu().clone()
                packet['command']=inputs[token]['command'].cpu().clone()
            torch.save(packet,output/'replay_input.pt')
        optimizer.zero_grad(set_to_none=True);prediction=forward(token)
        target=torch.tensor([e.by_token[token]['trajectory_gt']],device='cuda:0')
        loss=pose_loss(prediction,target);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip'])
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite consumer training')
        grads=None
        if arm=='expert' and step in [1,1024,2048,3072,4096]:
            grads=[dict(block=i+1,finite=all(bool(torch.isfinite(p.grad).all()) for p in b.parameters() if p.grad is not None),
                        nonzero=any(bool(torch.count_nonzero(p.grad)) for p in b.parameters() if p.grad is not None)) for i,b in enumerate(model.blocks)]
            if not all(g['finite'] and g['nonzero'] for g in grads):raise ValueError('Inactive/nonfinite Expert block gradients')
        rates=[g['lr'] for g in optimizer.param_groups];optimizer.step();schedule.step()
        entry=dict(step=step,token=token,loss=float(loss),grad_norm=float(norm),lr=rates,windows_seen=step,epochs=step/256,expert_grads=grads)
        losses.append(entry)
        with (output/'train.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        if step==1 or step%64==0:print(f'{arm} {step}/4096 loss={float(loss):.6f} grad={float(norm):.4f} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if step in c['curve_steps']:
            diag=[e.by_token[t] for t in e.diagnostic_tokens];curves[step]=control_metrics(evaluate('train_curve',diag,['correct']));model.train()
            write_json(output/'curves.json',curves)
            print(f'{arm} fixed TRAIN32 step{step} ADE={curves[step]["ADE"]:.6f}',flush=True)
        if torch.cuda.max_memory_allocated()/2**30>c['max_allocated_gib']:raise MemoryError('Consumer working allocation cap exceeded')
    final=evaluate('final',e.cohorts['new_val']+e.cohorts['legacy_val'],['correct','masked','reference','shuffled'])
    summary={name:{mode:control_metrics([r for r in final if r['cohort']==name and r['mode']==mode])
                   for mode in ['correct','masked','reference','shuffled']} for name in ['new_val','legacy_val']}
    versions=tuple(t._version for t in [cache.keys,cache.values,cache.positions,cache.groups,cache.valid])
    if versions!=source_versions:raise ValueError('Frozen source cache mutated')
    with torch.no_grad():reference_prediction=forward(order[-1])[0].float().cpu()
    torch.save(checkpoint(4096,reference_prediction),output/'final.pt')
    write_json(output/'predictions.json',final)
    write_json(output/'result.json',dict(completed=True,arm=arm,summary=summary,curves=curves,protocol=protocol,
        checkpoint_sha256=fingerprint(output/'final.pt'),frozen_source_unchanged=True,
        before_final_sha256=fingerprint(output/'before_final.pt'),replay_input_sha256=fingerprint(output/'replay_input.pt'),
        elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        new_consumer_updates=4096,new_Slow_updates=0,new_download_bytes=0,val_used_for_selection=False,test_used=False))
    print(f'{arm} frozen-Slow control complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/slowfast_action_expert.yaml')
    parser.add_argument('--arm',required=True,choices=['expert','fast']);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.config,args.arm,args.output)


if __name__=='__main__':main()
