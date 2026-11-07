"""Frozen Slow export readability plus empty fusion-path equivalence."""
import argparse
import copy
import gc
import json
import math
from pathlib import Path
import time

import torch
from fast_only.common import fingerprint,write_json,seed_all
from fast_navsim.model import pose_loss,trajectory_metrics
from slowfast_v11.content_diagnostics import source_groups,attention_probe
from slowfast_v11.path_diagnostics import force_no_cache,trace_fast,compare_traces
from slowfast_v11.readability import KVReader,FinalFeatureReader
from slowfast_v11.expanded_protocol import epoch_plan,WarmupCosineScheduler
from slowfast_v11.validation import canonical_state,state_digest
from tools.train_slowfast_expanded import ExpandedExperiment
from tools.reload_slowfast_expanded import checked_checkpoint
from tools.train_slowfast_joint import capture_rng

MODES=('correct','shuffled','reference')


def state_cpu(module):return {n:t.detach().cpu().clone() for n,t in module.state_dict().items()}


class FeatureCache:
    """Preallocated full-token cache. Labels are never part of a reader input."""
    def __init__(self,tokens,snapshot,features,groups):
        self.tokens=list(tokens);self.index={t:i for i,t in enumerate(tokens)};n=len(tokens)
        _,heads,length,width=snapshot.layers[0].key.shape
        device=snapshot.valid_mask.device
        shape=(n,6,length,heads*width)
        self.keys=torch.empty(shape,dtype=snapshot.layers[0].key.dtype,device=device)
        self.values=torch.empty_like(self.keys)
        self.positions=torch.empty(3,n,length,dtype=torch.long,device=device)
        self.groups=torch.empty(n,length,dtype=torch.long,device=device)
        self.valid=torch.empty(n,length,dtype=torch.bool,device=device)
        self.final=torch.empty(n,*features.shape[1:],dtype=features.dtype,device=device)
        self.group_names=list(groups);self.layout_positions=snapshot.position_ids.clone()
        self.layout_valid=snapshot.valid_mask.clone();self.layout_groups=torch.stack(list(groups.values())).long().argmax(0)
        self.initialized=set()
    @torch.no_grad()
    def put(self,token,snapshot,features,groups):
        if token in self.initialized:raise ValueError('Duplicate frozen export')
        if (not torch.equal(self.layout_positions,snapshot.position_ids) or not torch.equal(self.layout_valid,snapshot.valid_mask)
                or self.group_names!=list(groups) or not torch.equal(self.layout_groups,torch.stack(list(groups.values())).long().argmax(0))):
            raise ValueError('Raw cache layout changed; no padding/pooling fallback')
        i=self.index[token]
        for l,z in enumerate(snapshot.layers):
            self.keys[i,l].copy_(z.key[0].transpose(0,1).flatten(1))
            self.values[i,l].copy_(z.value[0].transpose(0,1).flatten(1))
        self.positions[:,i].copy_(snapshot.position_ids[:,0]);self.valid[i].copy_(snapshot.valid_mask[0])
        self.groups[i].copy_(self.layout_groups);self.final[i].copy_(features[0]);self.initialized.add(token)
    def batch(self,tokens,sources=None):
        indices=torch.tensor([self.index[t] for t in tokens],device=self.keys.device)
        selected=indices if sources is None else torch.tensor([self.index[t] for t in sources],device=self.keys.device)
        return dict(keys=self.keys[selected],values=self.values[selected],positions=self.positions[:,indices],
                    group_ids=self.groups[indices],valid=self.valid[indices]),self.final[selected]
    def packet(self,tokens):
        indices=torch.tensor([self.index[t] for t in tokens],device=self.keys.device)
        return dict(tokens=tokens,keys=self.keys[indices].cpu(),values=self.values[indices].cpu(),
                    positions=self.positions[:,indices].cpu(),groups=self.groups[indices].cpu(),
                    valid=self.valid[indices].cpu(),final=self.final[indices].cpu())


@torch.no_grad()
def training_rms(cache,rows):
    if not rows or any(r['split']!='train' or r['cohort']!='train' for r in rows):
        raise ValueError('RMS requires TRAIN-only rows')
    sums_k=torch.zeros(6,device=cache.keys.device,dtype=torch.float64);sums_v=sums_k.clone();elements=0
    for row in rows:
        i=cache.index[row['token']]
        sums_k+=cache.keys[i].float().square().sum((1,2)).double()
        sums_v+=cache.values[i].float().square().sum((1,2)).double();elements+=cache.keys[i,0].numel()
    return (sums_k/elements).sqrt().float().cpu(),(sums_v/elements).sqrt().float().cpu()


@torch.no_grad()
def path_audit(e,row,snapshot,groups,references,full_context):
    batch=e.batch(row['token'],'fast');metadata=e.metadata(e.fresh[row['token']]);predictions={};traces={}
    for mode in ['none','empty','masked','correct']:
        if mode=='none':
            with trace_fast(e.fast,full_context=full_context) as trace:prediction=e.fast(**batch)
        else:
            from contextlib import nullcontext
            with (force_no_cache(e.fast) if mode=='empty' else nullcontext()):
                with attention_probe(groups,mask_external=mode!='correct'):
                    with trace_fast(e.fast,full_context=full_context) as trace:
                        prediction=e.fast(**batch,snapshot=snapshot,metadata=metadata)
        predictions[mode]=prediction[0].float().cpu().tolist();traces[mode]=trace
        if mode!='empty':
            expected=torch.tensor(references[row['token'],mode])
            if not torch.equal(prediction[0].float().cpu(),expected):raise ValueError('Historical path prediction changed')
    differences={f'{a}_to_{b}':compare_traces(traces[a],traces[b]) for a,b in
                 [('none','empty'),('empty','masked'),('masked','correct'),('none','correct')]}
    delta=float((torch.tensor(predictions['none'])-torch.tensor(predictions['empty'])).abs().max())
    return dict(token=row['token'],cohort=row['cohort'],route=row['route'],target=row['trajectory_gt'],
                speed_mps=math.hypot(*row['ego'][:2]),
                empty_path_comparison=dict(max_abs_prediction_difference=delta,exact=delta==0.,within_1e_4=delta<=1e-4),
                predictions=predictions,differences=differences,
                gate_summary={m:{n:dict(mean=float(t.mean()),minimum=float(t.min()),maximum=float(t.max()))
                                 for n,t in trace.items() if n.startswith('gate_')} for m,trace in traces.items()})


@torch.no_grad()
def extract(e,original,root,output):
    checkpoint=checked_checkpoint(root/'joint_final.pt',e,original,'joint',800,original['arms']['joint']['checkpoint_sha256'])
    old=[json.loads(s) for s in Path('evidence/slowfast_kv_content/records.jsonl').read_text().splitlines()]
    references={(r['token'],m):r['predictions'][m] for r in old for m in ['none','masked','correct']}
    del checkpoint
    e.fast.eval().requires_grad_(False);e.slow.eval().requires_grad_(False)
    before={n:state_digest(canonical_state(getattr(e,n))) for n in ['fast','slow']}
    versions={n:tuple(p._version for p in getattr(e,n).parameters()) for n in ['fast','slow']}
    rows=e.cohorts['new_val']+e.cohorts['legacy_val']+e.train;tokens=[r['token'] for r in rows]
    cache=None;path_records=[];baseline={}
    for index,row in enumerate(rows,1):
        token=row['token'];batch=e.batch(token,'slow');captured=[]
        handle=e.slow.head.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone()))
        try:source=e.slow(**batch,feedback=e.feedback[token].to('cuda:0'))
        finally:handle.remove()
        if len(captured)!=1 or not torch.equal(e.slow.head(captured[0]),source.trajectory):
            raise ValueError('Final feature positive-control head is not exact')
        groups=source_groups(batch['native'],e.slow.backbone.config.image_token_id,e.slow.backbone.visual.spatial_merge_size)
        if cache is None:cache=FeatureCache(tokens,source.snapshot,captured[0],groups)
        cache.put(token,source.snapshot,captured[0],groups)
        if row['split']!='train':
            record=path_audit(e,row,source.snapshot,groups,references,full_context=not path_records)
            path_records.append(record)
            with (output/'path_records.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        baseline[token]=dict(slow=source.trajectory[0].float().cpu().tolist(),feedback=e.feedback[token][0].float().cpu().tolist())
        del source,captured,batch
        if torch.cuda.memory_allocated()>64*2**30:raise MemoryError('Frozen export exceeds64GiB cap')
        if index==1 or index%16==0 or index==len(rows):e.log(f'frozen export {index}/{len(rows)} allocated={torch.cuda.memory_allocated()/2**30:.2f}GiB')
    if len(cache.initialized)!=len(rows):raise ValueError('Incomplete full-token export')
    for name in ['fast','slow']:
        model=getattr(e,name)
        if state_digest(canonical_state(model))!=before[name] or tuple(p._version for p in model.parameters())!=versions[name]:
            raise ValueError('Extraction mutated model')
        model.bridge.close()
    feature_versions=tuple(t._version for t in [cache.keys,cache.values,cache.final,cache.positions,cache.groups,cache.valid])
    key_scales,value_scales=training_rms(cache,e.train)
    export=dict(source_parameters_unchanged=True,original_head_exact_count=len(rows),raw_KV_shape=list(cache.keys.shape),
        raw_KV_dtype=str(cache.keys.dtype),raw_KV_bytes=cache.keys.numel()*cache.keys.element_size()*2,
        key_scales=key_scales.tolist(),value_scales=value_scales.tolist(),
        scales_source='TRAIN256_only',group_names=cache.group_names,source_versions=feature_versions,
        path_exact_references=len(path_records)*3,empty_equivalence_prediction_tolerance=1e-4,
        before_digests=before)
    del model,e.fast,e.slow,e.collators;e.batch_cache.clear();gc.collect();torch.cuda.empty_cache()
    export['post_model_unload_allocated_gib']=torch.cuda.memory_allocated()/2**30
    return cache,path_records,baseline,export


def predict(reader,cache,tokens,sources=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if isinstance(reader,KVReader):
            kv,_=cache.batch(tokens,sources)
            return reader(**kv)
        indices=torch.tensor([cache.index[t] for t in (tokens if sources is None else sources)],device=cache.final.device)
        return reader(cache.final[indices])


@torch.no_grad()
def evaluate(reader,cache,rows,donors,reference,modes=MODES):
    reader.eval();records=[]
    for mode in modes:
        for start in range(0,len(rows),8):
            selected=rows[start:start+8];tokens=[r['token'] for r in selected]
            sources=tokens if mode=='correct' else [donors[t] for t in tokens] if mode=='shuffled' else [reference]*len(tokens)
            predictions=predict(reader,cache,tokens,sources).float().cpu()
            for row,source,p in zip(selected,sources,predictions):
                records.append(dict(token=row['token'],cohort=row['cohort'],route=row['route'],mode=mode,source=source,
                                    speed_mps=math.hypot(*row['ego'][:2]),
                                    prediction=p.tolist(),target=row['trajectory_gt']))
    return records


def metrics(records):
    return trajectory_metrics(torch.tensor([r['prediction'] for r in records]),torch.tensor([r['target'] for r in records]))


def train_reader(name,reader,cache,e,output,order):
    reader.to('cuda:0').train();opt=torch.optim.AdamW(reader.parameters(),lr=5e-4,weight_decay=.01)
    schedule=WarmupCosineScheduler(opt,1024,32,.1);curves={};losses=[];start=time.monotonic()
    train_by_token={r['token']:r for r in e.train};diag=[e.by_token[t] for t in e.diagnostic_tokens]
    for step in range(1,1025):
        tokens=order[(step-1)*8:step*8]
        target=torch.tensor([train_by_token[t]['trajectory_gt'] for t in tokens],device='cuda:0')
        opt.zero_grad(set_to_none=True);prediction=predict(reader,cache,tokens);loss=pose_loss(prediction,target)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite probe training')
        lr=opt.param_groups[0]['lr'];opt.step();schedule.step()
        entry=dict(probe=name,step=step,loss=float(loss),grad_norm=float(norm),lr=lr,windows_seen=step*8,equivalent_epochs=step*8/256)
        losses.append(entry)
        with (output/'train.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        if step in [256,512,1024]:
            records=evaluate(reader,cache,diag,e.mismatches,min(train_by_token),modes=('correct',));curves[step]=metrics(records);reader.train()
        if step==1 or step%64==0:print(f'{name} {step}/1024 loss={float(loss):.6f} gradient={float(norm):.4f} elapsed={time.monotonic()-start:.1f}s',flush=True)
        if torch.cuda.memory_allocated()>64*2**30:raise MemoryError('Readout training exceeds64GiB cap')
    checkpoint=dict(schema='slowfast_readability_probe_v1',probe=name,state=state_cpu(reader),
        optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=capture_rng(),order=order,
        source_checkpoint=e.source_checkpoint_sha,architecture=dict(kv_width=512,hidden=128,layers=6,queries=8,groups=9) if name=='kv' else dict(width=cache.final.shape[-1],hidden=256))
    torch.save(checkpoint,output/f'{name}_final.pt')
    return reader,dict(curves=curves,elapsed_s=time.monotonic()-start,parameters=sum(p.numel() for p in reader.parameters()),
                       checkpoint_sha256=fingerprint(output/f'{name}_final.pt'),train_losses=losses)


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True);started=time.monotonic();original=json.loads((root/'result.json').read_text())
    e=ExpandedExperiment(copy.deepcopy(original['config']));e.output=output
    e.source_checkpoint_sha=original['arms']['joint']['checkpoint_sha256']
    order=epoch_plan(e.train,1024*8,42);reference=min(r['token'] for r in e.train)
    rows=e.cohorts['new_val']+e.cohorts['legacy_val'];reload_tokens=[min(r['token'] for r in rows if r['route']==route)
                                                                for route in sorted({r['route'] for r in e.cohorts['new_val']})]
    protocol=dict(schema='slowfast_readability_v1',source_checkpoint_sha256=e.source_checkpoint_sha,
        data_identities=e.data_identities,assets=e.assets,probe_steps=1024,batch=8,seed=42,train_order=order,
        optimizer=dict(lr=5e-4,weight_decay=.01,clip=1.,warmup_steps=32,min_lr_ratio=.1),reproducibility=e.c['reproducibility'],
        source_tokens=[r['token'] for r in e.rows],reference_token=reference,reload_tokens=reload_tokens,
        metadata_path_comparison=['none','empty_fusion_no_cache_token','masked_valid_metadata','correct'],
        empty_equivalence_prediction_tolerance=1e-4,group_prior='equal_logit_prior_not_posterior_mass',
        stage3='pending_separate_protocol_review_after_readability',new_download_bytes=0,
        historical_reference_sha256=fingerprint('evidence/slowfast_kv_content/records.jsonl'),
        frozen_local_cache='all324_full_KV_shards_for_dedicated_expert_followup;not_uploaded',
        source_code_sha256={name:fingerprint(name) for name in ['tools/diagnose_slowfast_readability.py',
                            'slowfast_v11/path_diagnostics.py','slowfast_v11/readability.py']})
    write_json(output/'protocol.json',protocol)
    cache,path_records,baseline,export=extract(e,original,root,output)
    write_json(output/'export.json',export);write_json(output/'baselines.json',baseline)
    shards=[]
    for start in range(0,len(cache.tokens),8):
        path=output/'frozen_cache'/f'{start:04d}.pt';path.parent.mkdir(exist_ok=True)
        tokens=cache.tokens[start:start+8];torch.save(cache.packet(tokens),path)
        shards.append(dict(path=path.name,tokens=tokens,sha256=fingerprint(path),bytes=path.stat().st_size))
    write_json(output/'frozen_cache/manifest.json',dict(shards=shards,source_checkpoint=e.source_checkpoint_sha,
        data_identities=e.data_identities,group_names=cache.group_names,raw_KV_shape=export['raw_KV_shape']))
    print('Frozen full-token cache saved locally; no pooling or downloads',flush=True)
    seed_all(42);kv=KVReader().cuda();kv.set_scales(export['key_scales'],export['value_scales'])
    kv,kv_info=train_reader('kv',kv,cache,e,output,order)
    seed_all(42);final=FinalFeatureReader(cache.final.shape[-1]).cuda()
    final,final_info=train_reader('final',final,cache,e,output,order)
    records=[];summaries={};reload_batches={};packet_tokens=set()
    for name,reader in [('kv',kv),('final',final)]:
        part=evaluate(reader,cache,e.train+rows,e.mismatches,reference)
        records.extend([dict(r,probe=name) for r in part]);summaries[name]={}
        for cohort in ['train','new_val','legacy_val']:
            summaries[name][cohort]={mode:metrics([r for r in part if r['cohort']==cohort and r['mode']==mode]) for mode in MODES}
        reload_batches[name]=[]
        for mode in MODES:
            selected=[r for r in part if r['mode']==mode]
            for start in range(0,len(selected),8):
                batch=selected[start:start+8]
                if any(r['token'] in reload_tokens for r in batch):
                    reload_batches[name].append(batch)
                    packet_tokens.update(r['token'] for r in batch)
                    packet_tokens.update(r['source'] for r in batch)
    versions=tuple(t._version for t in [cache.keys,cache.values,cache.final,cache.positions,cache.groups,cache.valid])
    if versions!=export['source_versions']:raise ValueError('Probe training mutated frozen source cache')
    torch.save(cache.packet(sorted(packet_tokens)),output/'reload_inputs.pt')
    for name,info in [('kv',kv_info),('final',final_info)]:info.pop('train_losses')
    result=dict(schema='slowfast_readability_result_v1',completed=True,summary=summaries,
        probes=dict(kv=kv_info,final=final_info),reload_batches=reload_batches,
        frozen_cache_unchanged=True,export=export,elapsed_s=time.monotonic()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        reload_inputs_sha256=fingerprint(output/'reload_inputs.pt'),new_base_training_updates=0,
        new_probe_updates=2048,new_download_bytes=0,val_used_for_selection=False,test_used=False)
    write_json(output/'predictions.json',records);write_json(output/'result.json',result)
    print('Frozen Slow readability experiment complete',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/slowfast_v11_expanded/run_01')
    parser.add_argument('--output',default='outputs/slowfast_readability/run_01')
    args=parser.parse_args();run(args.run,args.output)


if __name__=='__main__':main()
