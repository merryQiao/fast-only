"""Expanded, bounded original-resolution NAVSIM diagnostic; no full-data claim."""
import argparse
from collections import Counter, OrderedDict
import gc
import json
import math
from pathlib import Path
import time

import torch
import yaml

from fast_only.common import apply_reproducibility, fingerprint, seed_all, write_json
from fast_navsim.evaluate import constant_velocity_metrics
from fast_navsim.model import trajectory_metrics
from slowfast_v11.expanded_protocol import (validate_expanded_config, expanded_plan, epoch_plan,
    scalable_mismatches, select_train_diagnostics, WarmupCosineScheduler, speed_bin)
from slowfast_v11.joint_protocol import fresh_probe_steps
from slowfast_v11.joint_training import joint_step
from slowfast_v11.validation import (canonical_state, restore_weights, state_digest,
    make_optimizer, set_stage, perform_step, paired_summary)
from tools.train_slowfast_joint import capture_rng
from tools.validate_slowfast_v11 import Experiment


class SnapshotLRU:
    def __init__(self, capacity=2):
        if not isinstance(capacity,int) or capacity<1: raise ValueError('Positive cache capacity required')
        self.capacity=capacity; self.items=OrderedDict()
    def get(self,key,factory):
        if key not in self.items:
            # Evict before constructing the next large device snapshot.
            if len(self.items)>=self.capacity: self.items.popitem(last=False)
            self.items[key]=factory()
        self.items.move_to_end(key)
        return self.items[key]
    def clear(self): self.items.clear()
    def __len__(self): return len(self.items)


class RowsView:
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)


class ExpandedExperiment(Experiment):
    def __init__(self,c):
        apply_reproducibility(c)
        super().__init__(c,config_validator=validate_expanded_config,plan_factory=expanded_plan,
                         mismatch_factory=scalable_mismatches)
        self.output=Path(c['output'])
        self.snapshots=SnapshotLRU(c['eval_cache_capacity'])
        self.diagnostic_tokens=select_train_diagnostics(self.train,c['train_diagnostic_count'])
        self.probe_steps=fresh_probe_steps(self.plan,c['gradient_probe_count'])
        self.replay_step=self.probe_steps[-1]
        self.shared_initialization=None
        self.cohorts={name:[r for r in self.rows if r['cohort']==name]
                      for name in ['train','new_val','legacy_val']}
        if {k:len(v) for k,v in self.cohorts.items()}!={'train':256,'new_val':64,'legacy_val':4}:
            raise ValueError('Expanded experiment requires exact TRAIN256/newVAL64/legacyVAL4')
        if any(r['split']!=('train' if r['cohort']=='train' else 'val') for r in self.rows):
            raise ValueError('Cohort and optimizer split disagree')
        logs={name:{r['route'] for r in rows} for name,rows in self.cohorts.items()}
        if any(logs[a]&logs[b] for a,b in [('train','new_val'),('train','legacy_val'),('new_val','legacy_val')]):
            raise ValueError('Cohort log leakage')
        if {k:len(v) for k,v in logs.items()}!={'train':4,'new_val':2,'legacy_val':1}:
            raise ValueError('Expanded experiment requires4/2/1 disjoint logs')

    def append_training(self,row):
        self.training.append(row)
        with (self.output/'train.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')

    def scheduler(self,optimizer,count):
        c=self.c['optimizer']
        return WarmupCosineScheduler(optimizer,count,c['warmup_steps'],c['min_lr_ratio'])

    @torch.no_grad()
    def snapshot(self,token):
        def build():
            batch=self.batch(token,'slow')
            source=self.slow(**batch,feedback=self.feedback[token].to(self.c['slow']['device']))
            return source.snapshot.inference_copy(self.c['fast']['device'])
        return self.snapshots.get(token,build)

    def trainable(self):
        self.snapshots.clear()
        set_stage(self.fast,'fast_consumer');set_stage(self.slow,'slow_lora')

    @torch.no_grad()
    def evaluate_rows(self,stage,rows):
        self.snapshots.clear(); self.fast.eval(); self.slow.eval()
        start=len(self.records)
        for row in rows:
            token=row['token'];batch=self.batch(token,'fast');pair=self.fresh[token]
            variants=[('none',None),('fresh',token)]
            if token in self.mismatches: variants.append(('shuffled',self.mismatches[token]))
            for mode,source in variants:
                kwargs={} if source is None else dict(snapshot=self.snapshot(source),metadata=self.metadata(pair))
                prediction=self.fast(**batch,**kwargs).float().cpu()
                self.records.append(dict(stage=stage,split=row['cohort'],manifest_split=row['split'],
                    token=token,source_token=source,mode=mode,age_s=0.,actual_age_s=0.,
                    route=row['route'],speed_mps=math.hypot(*row['ego'][:2]),speed_bin=speed_bin(row),
                    command=row['command'],prediction=prediction[0].tolist(),target=row['trajectory_gt']))
        self.snapshots.clear()
        write_json(self.output/'predictions.json',self.records)
        return paired_summary(self.records[start:])

    @torch.no_grad()
    def slow_metrics(self,rows):
        self.slow.eval(); groups={}
        for row in rows:
            token=row['token'];batch=self.batch(token,'slow')
            prediction=self.slow(**batch,feedback=self.feedback[token].to(self.c['slow']['device'])).trajectory.cpu()
            group=groups.setdefault(row['cohort'],[[],[]])
            group[0].append(prediction);group[1].append(self.target(token,'slow').cpu())
        return {key:trajectory_metrics(torch.cat(predictions),torch.cat(targets))
                for key,(predictions,targets) in groups.items()}

    def warmup(self,which):
        model=getattr(self,which); stage='fast_warmup' if which=='fast' else 'slow_adapt'
        set_stage(model,'fast_warmup' if which=='fast' else 'slow_lora')
        count=self.c['stages'][stage]; optimizer=make_optimizer(model,self.c['optimizer'])
        schedule=self.scheduler(optimizer,count)
        order=epoch_plan(self.train,count,self.c['train_seed'])
        for step,token in enumerate(order,1):
            rates=[group['lr'] for group in optimizer.param_groups]
            kwargs={} if which=='fast' else dict(feedback=self.feedback[token].to(self.c['slow']['device']))
            metrics=perform_step(model,optimizer,self.batch(token,which),self.target(token,which),
                self.c['optimizer']['grad_clip'],**kwargs)
            schedule.step()
            self.append_training(dict(stage=stage,step=step,token=token,learning_rates=rates,
                windows_seen=step,equivalent_train_epochs=step/len(self.train),**metrics))
            if step==1 or step%20==0 or step==count:
                self.log(f'{stage} {step}/{count} loss={metrics["loss"]:.5f} lr={rates}')
        model.zero_grad(set_to_none=True);model.eval()
        return dict(final_state=state_digest(canonical_state(model)),schedule=schedule.state_dict())

    def paired_update(self,arm,index):
        item=self.plan[index];token=item['target_token']
        metrics=joint_step(self.fast,self.slow,self.opt_fast,self.opt_slow,
            self.batch(token,'fast'),self.batch(token,'slow'),self.target(token,'fast'),self.target(token,'slow'),
            self.feedback[token].to(self.c['slow']['device']),arm=arm,mode=item['mode'],
            slow_weight=self.c['loss']['slow_weight'],clip=self.c['optimizer']['grad_clip'],
            probe=index+1 in self.probe_steps)
        self.schedule_fast.step();self.schedule_slow.step()
        return metrics

    @torch.no_grad()
    def replay_prediction(self,token):
        self.fast.eval();self.slow.eval();self.snapshots.clear()
        batch=self.batch(token,'fast')
        prediction=self.fast(**batch,snapshot=self.snapshot(token),metadata=self.metadata(self.fresh[token]))
        result=prediction.detach().cpu()
        self.trainable()
        return result

    def save_checkpoint(self,arm,step,name,references=None,replay_prediction=None):
        checkpoint=dict(schema='slowfast_expanded_v1',config=self.c,arm=arm,step=step,
            fast=canonical_state(self.fast),slow=canonical_state(self.slow),
            optimizer_fast=self.opt_fast.state_dict(),optimizer_slow=self.opt_slow.state_dict(),
            scheduler_fast=self.schedule_fast.state_dict(),scheduler_slow=self.schedule_slow.state_dict(),
            sampler=dict(next_index=step,plan=self.plan),rng=capture_rng(),
            feedback=self.feedback,feedback_producer=self.feedback_producer,feedback_producer_digest=self.feedback_digest,
            assets=self.assets,data_identities=self.data_identities,shared_initialization=self.shared_initialization,
            probe_steps=self.probe_steps,diagnostic_tokens=self.diagnostic_tokens,
            reference_predictions=references or [],replay_prediction=replay_prediction)
        path=self.output/name;torch.save(checkpoint,path)
        return dict(checkpoint_sha256=fingerprint(path),fast_digest=state_digest(checkpoint['fast']),
                    slow_digest=state_digest(checkpoint['slow']))

    def run(self):
        diagnostic_rows=[self.by_token[t] for t in self.diagnostic_tokens]
        write_json(self.output/'protocol.json',dict(config=self.c,plan=self.plan,diagnostic_tokens=self.diagnostic_tokens,
            probe_steps=self.probe_steps,replay_step=self.replay_step,coverage=self.coverage,mismatches=self.mismatch_report,
            data_identities=self.data_identities,assets=self.assets,selection_before_training=True))
        warmups={'fast':self.warmup('fast')}
        self.make_feedback()
        slow_before=self.slow_metrics(diagnostic_rows)
        warmups['slow']=self.warmup('slow')
        slow_after=self.slow_metrics(diagnostic_rows)
        common_fast=canonical_state(self.fast);common_slow=canonical_state(self.slow)
        self.shared_initialization=dict(fast_digest=state_digest(common_fast),slow_digest=state_digest(common_slow),
                                        feedback_digest=self.feedback_digest)
        torch.save(dict(fast=common_fast,slow=common_slow,feedback=self.feedback,
                        feedback_producer=self.feedback_producer,identity=self.shared_initialization),self.output/'common_init.pt')
        arms={};replay_checkpoints={}
        for arm in self.c['arms']:
            self.log(f'Starting expanded arm {arm}: fresh common initialization,800 paired updates')
            restore_weights(self.fast,common_fast);restore_weights(self.slow,common_slow)
            seed_all(self.c['train_seed']);self.trainable()
            self.opt_fast=make_optimizer(self.fast,self.c['optimizer']);self.opt_slow=make_optimizer(self.slow,self.c['optimizer'])
            self.schedule_fast=self.scheduler(self.opt_fast,len(self.plan));self.schedule_slow=self.scheduler(self.opt_slow,len(self.plan))
            for step,item in enumerate(self.plan,1):
                rates={name:[g['lr'] for g in getattr(self,'opt_'+name).param_groups] for name in ['fast','slow']}
                if arm=='joint' and step==self.replay_step:
                    replay_checkpoints['before']=self.save_checkpoint(arm,step-1,'replay_before.pt')
                metrics=self.paired_update(arm,step-1)
                self.append_training(dict(stage=arm,step=step,token=item['target_token'],sampled_mode=item['mode'],
                    actual_mode='none' if arm=='no_kv' else item['mode'],speed_bin=speed_bin(self.by_token[item['target_token']]),
                    learning_rates=rates,windows_seen=step,equivalent_train_epochs=step/len(self.train),
                    equivalent_total_model_epochs=(512+step)/len(self.train),**metrics))
                if step==1 or step%20==0:
                    self.log(f'{arm} {step}/{len(self.plan)} LF={metrics["loss_fast"]:.5f} LS={metrics["loss_slow"]:.5f} lr={rates["fast"]}')
                if arm=='joint' and step==self.replay_step:
                    prediction=self.replay_prediction(item['target_token'])
                    replay_checkpoints['after']=self.save_checkpoint(arm,step,'replay_after.pt',replay_prediction=prediction)
                if step in self.c['train_curve_steps']:
                    self.log(f'{arm} fixed TRAIN32 diagnostic at step{step}; VAL is not evaluated')
                    self.evaluate_rows(f'{arm}_step{step}',diagnostic_rows)
                    self.trainable()
            val_rows=self.cohorts['new_val']+self.cohorts['legacy_val']
            slow_final=self.slow_metrics(diagnostic_rows+val_rows)
            self.evaluate_rows(f'{arm}_final',val_rows)
            references=[r for r in self.records if r['stage']==f'{arm}_final']
            identities=self.save_checkpoint(arm,len(self.plan),f'{arm}_final.pt',references=references)
            arms[arm]=dict(**identities,slow_metrics=slow_final,
                actual_modes=dict(Counter('none' if arm=='no_kv' else p['mode'] for p in self.plan)))
            write_json(self.output/'partial_result.json',dict(arms=arms))
            del self.opt_fast,self.opt_slow,self.schedule_fast,self.schedule_slow
            self.fast.zero_grad(set_to_none=True);self.slow.zero_grad(set_to_none=True);gc.collect()
        result=dict(schema='slowfast_expanded_result_v1',scope='expanded mini diagnostic; not official navtrain/closed-loop',
            config=self.c,arms=arms,summary=paired_summary(self.records),shared_initialization=self.shared_initialization,
            slow_shared_adapt=dict(before=slow_before,after=slow_after),warmups=warmups,
            counts={k:len(v) for k,v in self.cohorts.items()},diagnostic_tokens=self.diagnostic_tokens,
            budget=dict(new_iterations=2624,total_optimizer_steps=4224,fast_optimizer_steps=2112,slow_optimizer_steps=2112,
                        final_updates_per_model_per_arm=1312,train_size=256),
            probe_steps=self.probe_steps,replay_step=self.replay_step,replay_checkpoints=replay_checkpoints,
            data_identities=self.data_identities,assets=self.assets,feedback_producer_digest=self.feedback_digest,
            coverage=self.coverage,image_audit=self.audits,mismatches=self.mismatch_report,
            constant_velocity={name:constant_velocity_metrics(RowsView(rows))
                               for name,rows in dict(self.cohorts,train_diagnostic=diagnostic_rows).items()},
            memory={d:dict(gpu=torch.cuda.get_device_name(d),peak_allocated_gib=torch.cuda.max_memory_allocated(d)/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(d)/2**30) for d in self.devices},
            elapsed_s=time.monotonic()-self.started,test_used=False,val_used_for_selection=False,
            resume_verified=False,multi_gpu_validated=False)
        write_json(self.output/'result.json',result)
        self.log('Expanded fixed-budget training complete; independent reload/replay required')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/slowfast_v11_expanded.yaml')
    for name in ['train-manifest','val-manifest','stats','sensor-root','fast-model','slow-model','output']:
        parser.add_argument('--'+name)
    args=parser.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    for name in ['train_manifest','val_manifest','stats','sensor_root']:
        if getattr(args,name) is not None:c['data'][name]=getattr(args,name)
    for name in ['fast','slow']:
        if getattr(args,name+'_model') is not None:c[name]['model_path']=getattr(args,name+'_model')
    if args.output is not None:c['output']=args.output
    validate_expanded_config(c)
    output=Path(c['output'])
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use new output directory; no overwrite')
    output.mkdir(parents=True,exist_ok=True)
    ExpandedExperiment(c).run()


if __name__=='__main__':main()
