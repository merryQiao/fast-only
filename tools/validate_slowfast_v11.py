"""Bounded offline KV adaptation on existing mini data; no full training/download."""
import argparse
from collections import OrderedDict,Counter
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import yaml
from fast_navsim.collate import NavsimCollator
from fast_navsim.data import NavsimDataset
from fast_navsim.evaluate import constant_velocity_metrics
from fast_navsim.model import trajectory_metrics
from fast_only.common import to_device,seed_all,write_json,fingerprint
from slowfast_v11.attention import check_compatibility
from slowfast_v11.diagnostic_data import observation_at,build_pairs,load_frames_by_log,deterministic_mismatches
from slowfast_v11.model import FastWithSlowKV,SlowPrefill
from slowfast_v11.protocol import relative_source_pose
from slowfast_v11.validation import (validate_config,set_stage,make_optimizer,perform_step,
    canonical_state,state_digest,cycle,sampling_plan,paired_summary)
from tools.slowfast_v11_audit import prepare_memory_measurement


class Experiment:
    def __init__(self,c,*,config_validator=validate_config,plan_factory=None,
                 mismatch_factory=deterministic_mismatches):
        config_validator(c);self.c=c;self.started=time.monotonic()
        seed=c['train_seed'] if 'train_seed' in c else c['seed']
        seed_all(seed);torch.set_num_threads(4)
        self.rows=[]
        for split in ['train','val']:
            dataset=NavsimDataset(c['data'][split+'_manifest'],c['data']['stats'])
            if not 0<len(dataset)<=c['data']['max_'+split+'_rows'] or any(r['split']!=split for r in dataset.rows):
                raise ValueError('Requires bounded, correctly labelled TRAIN/VAL cohort')
            self.rows.extend(dataset.rows)
        self.by_token={r['token']:r for r in self.rows};self.train=[r for r in self.rows if r['split']=='train']
        self.stats=json.loads(Path(c['data']['stats']).read_text())
        self.frames=load_frames_by_log(self.rows)
        self.pairs,self.coverage=build_pairs(self.rows,self.frames,c['data']['sensor_root'],c['data']['ages_s'])
        self.mismatches,self.mismatch_report=mismatch_factory(self.rows)
        self.fresh={p.target_token:p for p in self.pairs if p.age_s==0}
        if set(self.fresh)!=set(self.by_token):raise ValueError('Fresh raw history must exist for every manifest row')
        stale=[asdict(p) for p in self.pairs if p.split=='train' and p.age_s>0]
        self.plan=(plan_factory(self.train,stale,c) if plan_factory is not None else
                   sampling_plan(self.train,stale,c['stages']['fast_consumer'],seed,c['consumer_modes']))
        self.sources={p.source_token:(p.source_route,p.source_index) for p in self.pairs}
        if len(self.sources)>c['data']['max_source_observations']:raise ValueError('Source-observation cap exceeded')
        self.data_identities={key:fingerprint(c['data'][key]) for key in ['train_manifest','val_manifest','stats']}
        self.data_identities['metadata']={r['route']:r['source']['metadata_sha256'] for r in self.rows}
        images={self.frames[route][i]['cams']['CAM_F0']['data_path']
            for route,index in self.sources.values() for i in range(index-3,index+1)}
        self.data_identities['images']={name:fingerprint(Path(c['data']['sensor_root'])/name) for name in sorted(images)}
        self.records=[];self.feedback={};self.cache={};self.batch_cache=OrderedDict();self.audits={};self.training=[]
        self.devices=sorted(set(c[n]['device'] for n in ['fast','slow']))
        prepare_memory_measurement(self.devices)
        kwargs=dict(c['model'],dtype=torch.bfloat16)
        self.log('Loading complete pretrained Fast/Slow models')
        self.fast=FastWithSlowKV.from_pretrained(c['fast']['model_path'],**kwargs).to(c['fast']['device'])
        self.slow=SlowPrefill.from_pretrained(c['slow']['model_path'],**kwargs).to(c['slow']['device'])
        self.compatibility=check_compatibility(self.fast.backbone.language_model,self.slow.backbone.language_model)
        self.collators={n:NavsimCollator(c[n]['model_path'],**c['processor']) for n in ['fast','slow']}
        self.assets=self.verify_assets()

    def log(self,message):print(f'[{time.monotonic()-self.started:.1f}s] {message}',flush=True)

    def verify_assets(self):
        expected=json.loads(Path('evidence/slowfast_v11/model_assets.json').read_text())['models']
        identities={}
        for name,size in [('fast','Qwen3.5-0.8B'),('slow','Qwen3.5-2B')]:
            identities[name]={}
            for filename,info in expected[size]['files'].items():
                digest=fingerprint(Path(self.c[name]['model_path'])/filename)
                if digest!=info['sha256']:raise ValueError('Model/processor assets differ from verified P0 identity')
                identities[name][filename]=digest
        return identities

    def batch(self,token,which):
        key=(token,which)
        if key not in self.batch_cache:
            route,index=self.sources[token]
            obs=observation_at(self.frames[route],index,self.c['data']['sensor_root'],self.stats)
            # The shared collator requires a label slot. An empty, unused sentinel
            # lets it process unlabeled source observations without reading future GT.
            packed=self.collators[which]([dict(obs,trajectory_gt=torch.empty(0))])
            self.audits[which]=packed['image_audit']
            self.batch_cache[key]={k:packed[k] for k in ['native','ego','command']}
            while len(self.batch_cache)>2:self.batch_cache.popitem(last=False)
        self.batch_cache.move_to_end(key)
        return to_device(self.batch_cache[key],self.c[which]['device'])

    def target(self,token,which):
        return torch.tensor([self.by_token[token]['trajectory_gt']],device=self.c[which]['device'],dtype=torch.float32)

    def metadata(self,pair):
        dx,dy,yaw=relative_source_pose(pair.source_pose,pair.current_pose)
        return torch.tensor([[pair.actual_age_s,dx,dy,math.sin(yaw),math.cos(yaw),1.]],
            device=self.c['fast']['device'],dtype=torch.float32)

    def run_stage(self,name):
        which='slow' if name.startswith('slow') else 'fast';model=getattr(self,which)
        set_stage(model,name);opt=make_optimizer(model,self.c['optimizer'])
        count=self.c['stages'][name]
        items=self.plan if name=='fast_consumer' else [dict(target_token=t,mode='none' if which=='fast' else 'feedback')
            for t in cycle([r['token'] for r in self.train],count,self.c['seed'])]
        losses=[]
        for step,item in enumerate(items,1):
            token=item['target_token'];batch=self.batch(token,which);kwargs={}
            if which=='slow':kwargs['feedback']=self.feedback[token].to(self.c['slow']['device'])
            elif name=='fast_consumer' and item['mode']!='none':
                pair=next(p for p in self.pairs if p.target_token==token and p.age_s==item['age_s'])
                kwargs=dict(snapshot=self.cache[pair.source_token].to(self.c['fast']['device']),metadata=self.metadata(pair))
            result=perform_step(model,opt,batch,self.target(token,which),self.c['optimizer']['grad_clip'],**kwargs)
            row=dict(stage=name,step=step,token=token,mode=item['mode'],age_s=item.get('age_s',0),
                windows_seen=step,equivalent_train_epochs=step/len(self.train),**result)
            self.training.append(row);losses.append(result['loss'])
            with (Path(self.c['output'])/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step==1 or step%10==0 or step==count:
                self.log(f'{name} {step}/{count}; last10 loss={np.mean(losses[-10:]):.6f}')
        model.zero_grad(set_to_none=True);model.eval()
        return opt

    @torch.no_grad()
    def make_feedback(self):
        self.fast.eval()
        self.feedback_producer=canonical_state(self.fast)
        self.feedback_digest=state_digest(self.feedback_producer)
        self.log(f'Generating completed numeric feedback for {len(self.sources)} source observations')
        for token in sorted(self.sources):
            b=self.batch(token,'fast')
            self.feedback[token]=self.fast(b['native'],b['ego'],b['command']).detach().cpu()

    @torch.no_grad()
    def export_cache(self):
        self.slow.requires_grad_(False);self.slow.eval();self.slow.zero_grad(set_to_none=True)
        self.log(f'Exporting frozen Slow snapshots for {len(self.sources)} observed histories')
        for token in sorted(self.sources):
            b=self.batch(token,'slow')
            output=self.slow(b['native'],b['ego'],b['command'],feedback=self.feedback[token].to(self.c['slow']['device']))
            self.cache[token]=output.snapshot.inference_copy('cpu')
        self.slow_digest=state_digest(canonical_state(self.slow))

    @torch.no_grad()
    def evaluate_slow(self,label):
        self.slow.eval();out={}
        for split in ['train','val']:
            preds=[];targets=[]
            for row in self.rows:
                if row['split']!=split:continue
                b=self.batch(row['token'],'slow')
                pred=self.slow(b['native'],b['ego'],b['command'],feedback=self.feedback[row['token']].to(self.c['slow']['device'])).trajectory
                preds.append(pred.cpu());targets.append(self.target(row['token'],'slow').cpu())
            out[split]=trajectory_metrics(torch.cat(preds),torch.cat(targets))
        self.log(f'Slow {label} TRAIN ADE={out["train"]["ADE"]:.4f}, VAL ADE={out["val"]["ADE"]:.4f} (diagnostic only)')
        return out

    @torch.no_grad()
    def evaluate(self,stage):
        self.fast.eval();self.log(f'Paired evaluation: {stage}')
        for row in self.rows:
            token=row['token'];b=self.batch(token,'fast');fresh=self.fresh[token]
            variants=[('none',fresh,None),('fresh',fresh,token)]
            if token in self.mismatches:variants.append(('shuffled',fresh,self.mismatches[token]))
            variants.extend(('stale',p,p.source_token) for p in self.pairs if p.target_token==token and p.age_s>0)
            for mode,pair,source in variants:
                kwargs={} if source is None else dict(snapshot=self.cache[source].to(self.c['fast']['device']),metadata=self.metadata(pair))
                pred=self.fast(b['native'],b['ego'],b['command'],**kwargs).float().cpu()
                self.records.append(dict(stage=stage,split=row['split'],token=token,source_token=source,
                    mode=mode,age_s=pair.age_s,actual_age_s=pair.actual_age_s,
                    speed_mps=math.hypot(*row['ego'][:2]),command=row['command'],
                    prediction=pred[0].tolist(),target=row['trajectory_gt']))
        write_json(Path(self.c['output'])/'predictions.json',self.records)
        return paired_summary(self.records)

    def run(self):
        output=Path(self.c['output'])
        write_json(output/'coverage.json',dict(coverage=self.coverage,mismatches=self.mismatch_report,
            pairs=[asdict(p) for p in self.pairs]))
        opt=self.run_stage('fast_warmup');del opt;gc.collect()
        self.make_feedback()
        slow_before=self.evaluate_slow('before')
        for stage in ['slow_inputs','slow_lora']:
            opt=self.run_stage(stage);del opt;gc.collect()
        slow_after=self.evaluate_slow('after')
        self.export_cache()
        self.evaluate('before_consumer')
        opt=self.run_stage('fast_consumer')
        summary=self.evaluate('after_consumer')
        frozen_slow_unchanged=state_digest(canonical_state(self.slow))==self.slow_digest
        if not frozen_slow_unchanged:raise AssertionError('Frozen Slow changed during cache consumption')
        checkpoint=dict(schema='slowfast_validation_v1',config=self.c,
            fast=canonical_state(self.fast),slow=canonical_state(self.slow),
            feedback_producer=self.feedback_producer,feedback=self.feedback,
            feedback_producer_digest=self.feedback_digest,assets=self.assets,data_identities=self.data_identities,
            optimizer=opt.state_dict(),scheduler=None,stage='fast_consumer',step=self.c['stages']['fast_consumer'],
            sampler=dict(next_index=len(self.plan),plan=self.plan),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
            reference_predictions=[r for r in self.records if r['stage']=='after_consumer' and r['split']=='val'])
        torch.save(checkpoint,output/'final.pt')
        result=dict(scope='bounded offline diagnostic; not generalization, actual async delivery or closed loop',
            config=self.c,counts={s:sum(r['split']==s for r in self.rows) for s in ['train','val']},
            sources=len(self.sources),updates=self.c['stages'],summary=summary,
            slow=dict(before=slow_before,after=slow_after),coverage=self.coverage,mismatches=self.mismatch_report,
            constant_velocity={},
            compatibility=self.compatibility,data_identities=self.data_identities,feedback_producer_digest=self.feedback_digest,
            final_fast_digest=state_digest(checkpoint['fast']),final_slow_digest=self.slow_digest,
            frozen_slow_unchanged=frozen_slow_unchanged,
            consumer_exposures=dict(Counter(item['mode'] for item in self.plan)),
            image_audit=self.audits,checkpoint_sha256=fingerprint(output/'final.pt'),
            memory={d:dict(gpu=torch.cuda.get_device_name(d),peak_allocated_gib=torch.cuda.max_memory_allocated(d)/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(d)/2**30) for d in self.devices},
            elapsed_s=time.monotonic()-self.started,seed=self.c['seed'],test_used=False,val_used_for_selection=False,
            joint_resume_verified=False,multi_gpu_validated=len(self.devices)>1)
        # CV helper needs the normal dataset sequence protocol, not a synthetic model input.
        for split in ['train','val']:
            dataset=NavsimDataset(self.c['data'][split+'_manifest'],self.c['data']['stats'])
            result['constant_velocity'][split]=constant_velocity_metrics(dataset)
        write_json(output/'result.json',result)
        self.log('All fixed-budget stages finished; final checkpoint and paired results saved')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/slowfast_v11_validation.yaml')
    for name in ['train-manifest','val-manifest','stats','sensor-root','fast-model','slow-model','output']:
        parser.add_argument('--'+name)
    args=parser.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    for name in ['train_manifest','val_manifest','stats','sensor_root']:
        value=getattr(args,name)
        if value is not None:c['data'][name]=value
    for name in ['fast','slow']:
        if getattr(args,name+'_model') is not None:c[name]['model_path']=getattr(args,name+'_model')
    if args.output is not None:c['output']=args.output
    output=Path(c['output'])
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new output directory; no overwrite/resume claim')
    output.mkdir(parents=True,exist_ok=True)
    Experiment(c).run()


if __name__=='__main__':main()
