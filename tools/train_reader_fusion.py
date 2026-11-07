"""Bounded, frozen Fast+Reader residual experiment with fixed-source control."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import yaml

from fast_only.common import fingerprint,seed_all,write_json
from fast_navsim.model import pose_loss,trajectory_metrics
from slowfast_v11.expanded_protocol import epoch_plan,WarmupCosineScheduler
from slowfast_v11.readability import KVReader
from slowfast_v11.reader_fusion import ResidualFusion,fused_head,reader_features
from slowfast_v11.validation import canonical_state,restore_weights,state_digest
from tools.diagnose_slowfast_readability import state_cpu
from tools.reload_slowfast_expanded import checked_checkpoint
from tools.train_frozen_expert import load_cache
from tools.train_slowfast_expanded import ExpandedExperiment
from tools.train_slowfast_joint import capture_rng

MODES=('correct','off','reference','shuffled')


def make_head(state,device):
    width=state['0.weight'].numel()
    head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),nn.Linear(256,3)).to(device)
    head.load_state_dict(state);return head.eval().requires_grad_(False)


def cached_forward(fusion,head,bank,tokens,mode):
    if mode not in MODES:raise ValueError('Unknown fusion source mode')
    indices=torch.tensor([bank['index'][t] for t in tokens],device=bank['fast'].device)
    current=bank['fast'][indices]
    # No source access or all-masked Reader invocation in this branch.
    if mode=='off':return head(current)
    return head(fusion(current,bank['reader'][mode][indices]))


@torch.no_grad()
def evaluate(fusion,head,bank,rows,modes=MODES):
    fusion.eval();records=[]
    for row in rows:
        for mode in modes:
            p=cached_forward(fusion,head,bank,[row['token']],mode)[0].float().cpu()
            if mode=='off' and not torch.equal(p,bank['baseline'][bank['index'][row['token']]].cpu()):
                raise ValueError('Disabled residual changed native Fast prediction')
            records.append(dict(token=row['token'],cohort=row['cohort'],route=row['route'],command=row['command'],mode=mode,
                speed_mps=math.hypot(*row['ego'][:2]),source_token=None if mode=='off' else bank['sources'][mode][row['token']],
                prediction=p.tolist(),target=row['trajectory_gt']))
    return records


def metrics(records):
    return trajectory_metrics(torch.tensor([r['prediction'] for r in records]),torch.tensor([r['target'] for r in records]))


def train_arm(c,arm,bank,head,rows,diagnostics,output,protocol):
    output.mkdir();train=[r for r in rows if r['cohort']=='train'];by_token={r['token']:r for r in train}
    if len(train)!=c['train_size'] or any(r['split']!='train' for r in train):raise ValueError('TRAIN-only optimizer rows required')
    device=bank['fast'].device;seed_all(c['seed'])
    fusion=ResidualFusion(fast_width=bank['fast'].shape[-1],**c['fusion']).to(device)
    initial_digest=state_digest(state_cpu(fusion));head_digest=state_digest(state_cpu(head))
    optimizer=torch.optim.AdamW(fusion.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scheduler=WarmupCosineScheduler(optimizer,c['steps'],c['warmup_steps'],c['min_lr_ratio'])
    order=epoch_plan(train,c['steps']*c['batch_size'],c['seed'])
    source_versions=tuple(t._version for t in [bank['fast'],*bank['reader'].values()])
    curves={};started=time.monotonic()
    def checkpoint(step,reference_prediction=None):
        return dict(schema=c['schema'],arm=arm,config=c,protocol=protocol,fusion=state_cpu(fusion),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),order=order,next_index=step,
            initial_digest=initial_digest,reference_prediction=reference_prediction)
    for step in range(1,c['steps']+1):
        tokens=order[(step-1)*c['batch_size']:step*c['batch_size']]
        if step==c['steps']:torch.save(checkpoint(step-1),output/'before_final.pt')
        optimizer.zero_grad(set_to_none=True);fusion.train()
        prediction=cached_forward(fusion,head,bank,tokens,arm)
        target=torch.tensor([by_token[t]['trajectory_gt'] for t in tokens],device=device)
        loss=pose_loss(prediction,target);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(fusion.parameters(),c['grad_clip'],error_if_nonfinite=True)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite fusion loss')
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        grads={name:dict(finite=bool(torch.isfinite(p.grad).all()),nonzero=bool(p.grad.count_nonzero()))
               for name,p in fusion.named_parameters() if p.grad is not None} if step in [1,2,128,512] else None
        entry=dict(step=step,tokens=tokens,loss=float(loss),grad_norm=float(norm),lr=lr,
                   windows_seen=step*c['batch_size'],epochs=step*c['batch_size']/len(train),gradients=grads)
        with (output/'train.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        if step==1 or step%64==0:print(f'{arm} fusion {step}/{c["steps"]} loss={float(loss):.6f} grad={float(norm):.4f}',flush=True)
        if step in c['curve_steps']:
            curves[step]=metrics(evaluate(fusion,head,bank,diagnostics,('correct' if arm=='correct' else 'reference',)))
            write_json(output/'curves.json',curves)
        if torch.cuda.is_available() and torch.cuda.max_memory_allocated()/2**30>c['max_allocated_gib']:
            raise MemoryError('Fusion experiment exceeded allocation cap')
    final=evaluate(fusion,head,bank,[r for r in rows if r['cohort']!='train'])
    with torch.no_grad():last=cached_forward(fusion,head,bank,tokens,arm).cpu()
    torch.save(checkpoint(c['steps'],last),output/'final.pt')
    if source_versions!=tuple(t._version for t in [bank['fast'],*bank['reader'].values()]):raise ValueError('Frozen feature bank changed')
    if state_digest(state_cpu(head))!=head_digest or any(p.grad is not None for p in head.parameters()):raise ValueError('Original head changed/received parameter gradients')
    summary={cohort:{mode:metrics([r for r in final if r['cohort']==cohort and r['mode']==mode]) for mode in MODES}
             for cohort in ['new_val','legacy_val']}
    write_json(output/'predictions.json',final)
    result=dict(completed=True,arm=arm,summary=summary,curves=curves,initial_digest=initial_digest,order=order,
        trainable_parameters=sum(p.numel() for p in fusion.parameters()),frozen_head_unchanged=True,frozen_features_unchanged=True,
        checkpoint_sha256=fingerprint(output/'final.pt'),before_final_sha256=fingerprint(output/'before_final.pt'),
        elapsed_s=time.monotonic()-started,new_exposures=len(order),val_used_for_selection=False)
    write_json(output/'result.json',result);return fusion,result


@torch.no_grad()
def extract(c,e,output):
    parent=json.loads((Path(c['source_run'])/'result.json').read_text())
    checked=checked_checkpoint(Path(c['source_run'])/'joint_final.pt',e,parent,'joint',800,parent['arms']['joint']['checkpoint_sha256'])
    del checked
    fast_result=json.loads((Path(c['fast_run'])/'result.json').read_text())
    if (fast_result['checkpoint_sha256']!=c['fast_checkpoint_sha256'] or fingerprint(Path(c['fast_run'])/'final.pt')!=c['fast_checkpoint_sha256']
            or fast_result['protocol']['source_checkpoint']!=parent['arms']['joint']['checkpoint_sha256']
            or fast_result['protocol']['source_data']!=e.data_identities or fast_result['protocol']['assets']!=e.assets):
        raise ValueError('Current Fast/source/data/assets mismatch')
    saved=torch.load(Path(c['fast_run'])/'final.pt',map_location='cpu',weights_only=False)
    if saved['arm']!='fast' or saved['next_index']!=4096:raise ValueError('Expected completed latest Fast consumer')
    if json.loads(json.dumps(saved['protocol']))!=fast_result['protocol']:raise ValueError('Fast embedded/result protocol differs')
    pinned_manifest=saved['protocol']['feature_manifest_sha256']
    if fingerprint(Path(c['feature_run'])/'frozen_cache/manifest.json')!=pinned_manifest:
        raise ValueError('Frozen KV manifest differs from authenticated Fast source cache')
    restore_weights(e.fast,saved['model'])
    if state_digest(canonical_state(e.fast))!=state_digest(saved['model']):raise ValueError('Restored current Fast state differs')
    del saved
    e.fast.eval().requires_grad_(False);e.slow.bridge.close();del e.slow
    gc.collect();torch.cuda.empty_cache()
    fast_digest=state_digest(canonical_state(e.fast));fast_versions=tuple(p._version for p in e.fast.parameters())
    reader_result=json.loads((Path(c['feature_run'])/'result.json').read_text())
    if reader_result['probes']['kv']['checkpoint_sha256']!=c['reader_checkpoint_sha256'] or fingerprint(Path(c['feature_run'])/'kv_final.pt')!=c['reader_checkpoint_sha256']:
        raise ValueError('Readable Reader checkpoint identity mismatch')
    saved=torch.load(Path(c['feature_run'])/'kv_final.pt',map_location='cpu',weights_only=False)
    if saved['source_checkpoint']!=parent['arms']['joint']['checkpoint_sha256']:raise ValueError('Reader provider mismatch')
    reader=KVReader(**saved['architecture']).cuda();reader.load_state_dict(saved['state']);reader.eval().requires_grad_(False)
    if state_digest(state_cpu(reader))!=state_digest(saved['state']):raise ValueError('Restored Reader state differs')
    del saved
    reader_digest=state_digest(state_cpu(reader));reader_versions=tuple(p._version for p in reader.parameters())
    cache=load_cache(Path(c['feature_run'])/'frozen_cache',parent['arms']['joint']['checkpoint_sha256'],e.data_identities)
    reference=min(r['token'] for r in e.train);tokens=[r['token'] for r in e.rows]
    sources={mode:{t:t if mode=='correct' else reference if mode=='reference' else e.mismatches[t] for t in tokens}
             for mode in ['correct','reference','shuffled']}
    old=json.loads((Path(c['feature_run'])/'predictions.json').read_text())
    expected={(r['token'],r['mode']):torch.tensor(r['prediction']) for r in old if r['probe']=='kv'}
    # Recreate source Reader's actual historical B8 + final4 inference.
    historical_exact=0
    for mode in sources:
        for start in range(0,len(tokens),8):
            selected=tokens[start:start+8];batch,_=cache.batch(selected,[sources[mode][t] for t in selected])
            with torch.autocast('cuda',dtype=torch.bfloat16):features,prediction=reader_features(reader,batch)
            for t,p in zip(selected,prediction.float().cpu()):
                if not torch.equal(p,expected[t,mode]):raise ValueError('Historical Reader prediction differs')
                historical_exact+=1
    bank=dict(tokens=tokens,index={t:i for i,t in enumerate(tokens)},sources=sources,reader={})
    reader_b1_records=[];geometry_max=0.
    for mode in sources:
        all_features=[]
        for t in tokens:
            batch,_=cache.batch([t],[sources[mode][t]])
            with torch.autocast('cuda',dtype=torch.bfloat16):features,prediction=reader_features(reader,batch)
            all_features.append(features.float())
            prediction=prediction[0].float().cpu();geometry_max=max(geometry_max,float((prediction-expected[t,mode]).abs().max()))
            row=e.by_token[t]
            reader_b1_records.append(dict(token=t,cohort=row['cohort'],mode=mode,prediction=prediction.tolist(),target=row['trajectory_gt']))
        bank['reader'][mode]=torch.cat(all_features)
        print(f'frozen Reader B1 features: {mode}',flush=True)
    reference_features=bank['reader']['reference']
    if not torch.equal(reference_features,reference_features[:1].expand_as(reference_features)):raise ValueError('Fixed reference carries target-specific features')
    live_tokens=[e.train[0]['token'],e.cohorts['new_val'][0]['token']]
    live_packets={t:cache.batch([t])[0] for t in live_tokens}
    # Clone the two small source rows; do not retain the 33GB cache by views.
    live_packets={t:{k:v.detach().clone() for k,v in b.items()} for t,b in live_packets.items()}
    del batch,cache;gc.collect();torch.cuda.empty_cache()
    features=[];predictions=[]
    for number,row in enumerate(e.rows,1):
        captured=[];handle=e.fast.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        try:p=e.fast(**e.batch(row['token'],'fast'))
        finally:handle.remove()
        if len(captured)!=1 or not torch.equal(e.fast.head(captured[0]),p):raise ValueError('Native Fast head feature extraction differs')
        features.append(captured[0]);predictions.append(p)
        if number%32==0:print(f'original-resolution native Fast features {number}/{len(e.rows)}',flush=True)
    bank['fast']=torch.cat(features).float();bank['baseline']=torch.cat(predictions).float()
    bank['head']=state_cpu(e.fast.head)
    cpu=dict(bank,fast=bank['fast'].cpu(),baseline=bank['baseline'].cpu(),reader={k:v.cpu() for k,v in bank['reader'].items()})
    torch.save(cpu,output/'frozen_features.pt')
    write_json(output/'reader_b1_predictions.json',reader_b1_records)
    export=dict(historical_reader_exact_count=historical_exact,native_Fast_head_exact_count=len(tokens),
        reader_b1_vs_historical_max_abs_prediction_difference=geometry_max,reference_features_identical=True,
        fast_baseline_mode='latest_Fast_native_no_external_KV_no_cache_token',fast_head_width=bank['fast'].shape[-1],
        feature_bank_sha256=fingerprint(output/'frozen_features.pt'),pinned_feature_manifest_sha256=pinned_manifest,
        fast_digest=fast_digest,reader_digest=reader_digest,
        fast_versions=fast_versions,reader_versions=reader_versions,live_tokens=live_tokens)
    return bank,reader,live_packets,export


def run(config,output):
    c=yaml.safe_load(Path(config).read_text());output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh output directory')
    if (c['schema']!='frozen_reader_residual_v1' or c['arms']!=['correct','reference'] or c['steps']!=512
            or c['batch_size']!=8 or c['train_size']!=256 or c['seed']!=42 or c['evaluation_geometry']!='batch1'
            or c['reader_feature_geometry']!='batch1' or c['fast_mode']!='native_no_external_KV'):
        raise ValueError('Fixed residual fusion experiment protocol differs')
    output.mkdir(parents=True);started=time.monotonic()
    parent=json.loads((Path(c['source_run'])/'result.json').read_text())
    e=ExpandedExperiment(copy.deepcopy(parent['config']));e.output=output
    bank,reader,live_packets,export=extract(c,e,output)
    write_json(output/'rows.json',e.rows)
    protocol=dict(schema=c['schema'],config=c,source_data=e.data_identities,assets=e.assets,export=export,
        source_checkpoint=parent['arms']['joint']['checkpoint_sha256'],fast_checkpoint=c['fast_checkpoint_sha256'],
        reader_checkpoint=c['reader_checkpoint_sha256'],feature_manifest_sha256=export['pinned_feature_manifest_sha256'],
        config_sha256=fingerprint(config),frozen_features_sha256=export['feature_bank_sha256'],rows_sha256=fingerprint(output/'rows.json'),
        source_code_sha256={p:fingerprint(p) for p in ['slowfast_v11/reader_fusion.py','tools/train_reader_fusion.py']},
        train_protocol='correct vs reference train; identical512x8 order/init; VALonlyend',new_download_bytes=0)
    protocol=json.loads(json.dumps(protocol))
    write_json(output/'protocol.json',protocol)
    head=make_head(bank['head'],'cuda:0');results={};models={}
    for arm in c['arms']:
        models[arm],results[arm]=train_arm(c,arm,bank,head,e.rows,[e.by_token[t] for t in e.diagnostic_tokens],output/arm,protocol)
    if results['correct']['initial_digest']!=results['reference']['initial_digest'] or results['correct']['order']!=results['reference']['order']:
        raise ValueError('Two fusion arms do not share initialization/order')
    # Online path uses original Fast and actual raw-KV Reader B1, not cached F.
    live=[]
    with torch.no_grad():
        for arm,fusion in models.items():
            for token,batch in live_packets.items():
                fast_batch=e.batch(token,'fast');i=bank['index'][token]
                def read():
                    with torch.autocast('cuda',dtype=torch.bfloat16):r,_=reader_features(reader,batch)
                    return r.float()
                with fused_head(e.fast.head,fusion,read,enabled=False):off=e.fast(**fast_batch)
                with fused_head(e.fast.head,fusion,read,enabled=True):on=e.fast(**fast_batch)
                cached=cached_forward(fusion,head,bank,[token],'correct')
                diff=float((on-cached).abs().max());off_exact=torch.equal(off[0],bank['baseline'][i])
                if not off_exact or diff:raise ValueError('Live original-resolution fusion disagrees with cached experiment')
                live.append(dict(arm=arm,token=token,cohort=e.by_token[token]['cohort'],off_exact=off_exact,on_max_abs_difference=diff))
    frozen=(state_digest(canonical_state(e.fast))==export['fast_digest'] and state_digest(state_cpu(reader))==export['reader_digest']
        and list(p._version for p in e.fast.parameters())==list(export['fast_versions'])
        and list(p._version for p in reader.parameters())==list(export['reader_versions'])
        and all(p.grad is None and not p.requires_grad for p in e.fast.parameters())
        and all(p.grad is None and not p.requires_grad for p in reader.parameters()))
    if not frozen:raise ValueError('Frozen Fast/Reader changed')
    write_json(output/'result.json',dict(completed=True,protocol=protocol,arms=results,live=live,frozen_Fast_Reader_unchanged=True,
        elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        new_Fast_updates=0,new_Reader_updates=0,new_Slow_updates=0,new_download_bytes=0,test_used=False,val_used_for_selection=False))
    print('Frozen Reader residual fusion experiment completed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_reader_fusion.yaml');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)


if __name__=='__main__':main()
