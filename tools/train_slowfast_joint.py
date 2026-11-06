"""Fixed-budget, three-arm live-KV diagnostic on the existing NAVSIM mini cohort."""
import argparse
from collections import Counter
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

from fast_only.common import apply_reproducibility, fingerprint, seed_all, write_json
from fast_navsim.model import trajectory_metrics
from fast_navsim.data import NavsimDataset
from fast_navsim.evaluate import constant_velocity_metrics
from slowfast_v11.joint_protocol import (validate_joint_config, balanced_plan,
                                         balanced_source_order, fresh_probe_steps)
from slowfast_v11.joint_training import joint_step
from slowfast_v11.validation import (canonical_state, restore_weights, state_digest,
                                     make_optimizer, set_stage, perform_step, paired_summary)
from tools.validate_slowfast_v11 import Experiment


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def compare_trees(expected, actual):
    """Exact recursive comparison, including optimizer moments/dtypes and keys."""
    errors = []; maximum = 0.; tensor_count = 0
    def visit(a, b, path):
        nonlocal maximum, tensor_count
        if isinstance(a, torch.Tensor):
            tensor_count += 1
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                errors.append(path + ': tensor metadata'); return
            a, b = a.detach().cpu(), b.detach().cpu()
            if not torch.equal(a, b):
                difference = float((a.double()-b.double()).abs().max()) if a.numel() else 0.
                maximum = max(maximum, difference)
                errors.append(path + ': tensor values')
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                errors.append(path + ': keys'); return
            for key in a: visit(a[key], b[key], f'{path}/{key}')
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                errors.append(path + ': sequence'); return
            for i, (x, y) in enumerate(zip(a,b)): visit(x,y,f'{path}/{i}')
        elif isinstance(a, np.ndarray):
            if not isinstance(b,np.ndarray) or a.dtype != b.dtype or not np.array_equal(a,b):
                errors.append(path + ': array')
        elif type(a) is not type(b) or a != b:
            errors.append(path + ': scalar')
    visit(expected, actual, '')
    return dict(passed=not errors, max_abs_difference=maximum, tensor_count=tensor_count,
                differing_fields=len(errors), first_differences=errors[:12])


class JointExperiment(Experiment):
    def __init__(self, c):
        apply_reproducibility(c)
        super().__init__(c, config_validator=validate_joint_config, plan_factory=balanced_plan)
        self.probe_steps = fresh_probe_steps(self.plan, c['gradient_probe_count'])
        self.replay_step = self.probe_steps[-1]
        self.output = Path(c['output'])
        self.parent_sha = None

    def load_parent(self):
        path = Path(self.c['initial_checkpoint'])
        parent_result = json.loads((path.parent/'result.json').read_text())
        self.parent_sha = fingerprint(path)
        if self.parent_sha != parent_result['checkpoint_sha256']:
            raise ValueError('Parent checkpoint SHA differs from its result')
        parent = torch.load(path, map_location='cpu', weights_only=False)
        if parent.get('schema') != 'slowfast_validation_v1' or parent.get('step') != 120:
            raise ValueError('Requires the verified prior diagnostic final checkpoint')
        if parent['assets'] != self.assets or parent['data_identities'] != self.data_identities:
            raise ValueError('Parent assets or data identities differ')
        digest = state_digest(parent['feedback_producer'])
        if digest != parent['feedback_producer_digest'] or digest != parent_result['feedback_producer_digest']:
            raise ValueError('Parent completed-feedback teacher identity differs')
        if state_digest(parent['slow']) != parent_result['final_slow_digest']:
            raise ValueError('Parent Slow identity differs')
        self.feedback_producer = parent['feedback_producer']
        self.feedback_digest = digest
        self.feedback = parent['feedback']
        if set(self.feedback) != set(self.sources):
            raise ValueError('Saved feedback must cover exactly the source observations')
        for value in self.feedback.values():
            if value.shape != (1,8,3) or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid completed numerical feedback')
        restore_weights(self.fast, self.feedback_producer)
        restore_weights(self.slow, parent['slow'])
        self.log('Verified parent SHA, assets, cohort and fixed feedback; restored Fast80 / Slow60')

    def append_training(self, row):
        self.training.append(row)
        with (self.output/'train.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    @torch.no_grad()
    def slow_metrics(self, splits):
        self.slow.eval(); result = {}
        for split in splits:
            predictions=[]; targets=[]
            for row in self.rows:
                if row['split'] != split: continue
                token=row['token']; b=self.batch(token,'slow')
                predictions.append(self.slow(**b, feedback=self.feedback[token].to(self.c['slow']['device'])).trajectory.cpu())
                targets.append(self.target(token,'slow').cpu())
            result[split]=trajectory_metrics(torch.cat(predictions),torch.cat(targets))
        return result

    @torch.no_grad()
    def evaluate_selected(self, stage, splits, *, include_stale=False):
        rows=[r for r in self.rows if r['split'] in splits]
        required={r['token'] for r in rows}
        required.update(self.mismatches[t] for t in list(required) if t in self.mismatches)
        if include_stale:
            required.update(p.source_token for p in self.pairs if p.split in splits)
        all_sources=self.sources
        self.sources={t:all_sources[t] for t in sorted(required)}
        self.cache.clear()
        try: self.export_cache()
        finally: self.sources=all_sources
        self.fast.eval(); start=len(self.records)
        for row in rows:
            token=row['token']; b=self.batch(token,'fast'); fresh=self.fresh[token]
            variants=[('none',fresh,None),('fresh',fresh,token)]
            if token in self.mismatches:
                variants.append(('shuffled',fresh,self.mismatches[token]))
            if include_stale:
                variants.extend(('stale',p,p.source_token) for p in self.pairs
                                if p.target_token==token and p.age_s>0)
            for mode,pair,source in variants:
                kwargs={} if source is None else dict(snapshot=self.cache[source].to(self.c['fast']['device']),
                                                      metadata=self.metadata(pair))
                pred=self.fast(**b,**kwargs).float().cpu()
                self.records.append(dict(stage=stage,split=row['split'],token=token,source_token=source,
                    mode=mode,age_s=pair.age_s,actual_age_s=pair.actual_age_s,
                    speed_mps=math.hypot(*row['ego'][:2]),command=row['command'],
                    prediction=pred[0].tolist(),target=row['trajectory_gt']))
        self.cache.clear(); gc.collect()
        write_json(self.output/'predictions.json', self.records)
        return paired_summary(self.records[start:])

    def trainable(self):
        set_stage(self.fast,'fast_consumer'); set_stage(self.slow,'slow_lora')

    def paired_update(self, arm, index):
        item=self.plan[index]; token=item['target_token']
        return joint_step(self.fast,self.slow,self.opt_fast,self.opt_slow,
            self.batch(token,'fast'),self.batch(token,'slow'),self.target(token,'fast'),self.target(token,'slow'),
            self.feedback[token].to(self.c['slow']['device']),arm=arm,mode=item['mode'],
            slow_weight=self.c['loss']['slow_weight'],clip=self.c['optimizer']['grad_clip'],
            probe=index+1 in self.probe_steps)

    @torch.no_grad()
    def replay_prediction(self, token):
        self.fast.eval(); self.slow.eval()
        b=self.batch(token,'slow')
        snapshot=self.slow(**b,feedback=self.feedback[token].to(self.c['slow']['device'])).snapshot
        b=self.batch(token,'fast')
        prediction=self.fast(**b,snapshot=snapshot.to(self.c['fast']['device']),metadata=self.metadata(self.fresh[token]))
        self.trainable()
        return prediction.detach().cpu()

    def save_checkpoint(self, arm, step, name, references=None, replay_prediction=None):
        # torch.save is immediate: optimizer.state_dict tensors must never remain
        # aliased to a live optimizer across subsequent updates.
        checkpoint=dict(schema='slowfast_joint_v1',config=self.c,arm=arm,step=step,
            fast=canonical_state(self.fast),slow=canonical_state(self.slow),
            optimizer_fast=self.opt_fast.state_dict(),optimizer_slow=self.opt_slow.state_dict(),scheduler=None,
            sampler=dict(next_index=step,plan=self.plan),rng=capture_rng(),
            feedback=self.feedback,feedback_producer=self.feedback_producer,feedback_producer_digest=self.feedback_digest,
            assets=self.assets,data_identities=self.data_identities,parent_checkpoint_sha256=self.parent_sha,
            probe_steps=self.probe_steps,reference_predictions=references or [],replay_prediction=replay_prediction)
        path=self.output/name
        torch.save(checkpoint,path)
        return dict(checkpoint_sha256=fingerprint(path),fast_digest=state_digest(checkpoint['fast']),
                    slow_digest=state_digest(checkpoint['slow']))

    def run(self):
        self.load_parent()
        write_json(self.output/'protocol.json',dict(config=self.c,plan=self.plan,probe_steps=self.probe_steps,
            replay_step=self.replay_step,coverage=self.coverage,mismatches=self.mismatch_report,
            pairs=[asdict(p) for p in self.pairs]))
        slow_before=self.slow_metrics(['train'])
        set_stage(self.slow,'slow_lora'); opt=make_optimizer(self.slow,self.c['optimizer'])
        order=balanced_source_order(self.train,self.c['stages']['slow_adapt'],self.c['train_seed'])
        for step,token in enumerate(order,1):
            metrics=perform_step(self.slow,opt,self.batch(token,'slow'),self.target(token,'slow'),
                self.c['optimizer']['grad_clip'],feedback=self.feedback[token].to(self.c['slow']['device']))
            self.append_training(dict(stage='slow_adapt',step=step,token=token,**metrics))
            if step==1 or step%10==0:
                self.log(f'Slow adapt {step}/{len(order)} loss={metrics["loss"]:.5f}')
        del opt
        self.slow.zero_grad(set_to_none=True)
        slow_after=self.slow_metrics(['train'])
        common_fast=canonical_state(self.fast); common_slow=canonical_state(self.slow)
        torch.save(dict(fast=common_fast,slow=common_slow,feedback_digest=self.feedback_digest),self.output/'common_init.pt')
        arms={}; final_slow_controls={}; replay_checkpoints={}
        for arm in self.c['arms']:
            self.log(f'Starting paired arm {arm}: identical Fast80 and strengthened Slow initialization')
            restore_weights(self.fast,common_fast); restore_weights(self.slow,common_slow)
            seed_all(self.c['train_seed']); self.trainable()
            self.opt_fast=make_optimizer(self.fast,self.c['optimizer'])
            self.opt_slow=make_optimizer(self.slow,self.c['optimizer'])
            for step,item in enumerate(self.plan,1):
                if arm=='joint' and step==self.replay_step:
                    replay_checkpoints['before']=self.save_checkpoint(arm,step-1,'replay_before.pt')
                metrics=self.paired_update(arm,step-1)
                self.append_training(dict(stage=arm,step=step,token=item['target_token'],stratum=item['stratum'],
                    sampled_mode=item['mode'],actual_mode='none' if arm=='no_kv' else item['mode'],
                    windows_seen=step,equivalent_train_epochs=step/len(self.train),**metrics))
                if step==1 or step%10==0:
                    self.log(f'{arm} {step}/{len(self.plan)} LF={metrics["loss_fast"]:.5f} LS={metrics["loss_slow"]:.5f}')
                if arm=='joint' and step==self.replay_step:
                    prediction=self.replay_prediction(item['target_token'])
                    replay_checkpoints['after']=self.save_checkpoint(arm,step,'replay_after.pt',replay_prediction=prediction)
                if step in self.c['train_curve_steps']:
                    self.evaluate_selected(f'{arm}_step{step}',['train'])
                    self.trainable()
            slow_final=self.slow_metrics(['train','val'])
            self.evaluate_selected(f'{arm}_final',['train','val'],include_stale=True)
            refs=[r for r in self.records if r['stage']==f'{arm}_final' and r['split']=='val']
            identities=self.save_checkpoint(arm,len(self.plan),f'{arm}_final.pt',references=refs)
            final_slow_controls[arm]=identities['slow_digest']
            arms[arm]=dict(**identities,slow_metrics=slow_final,
                actual_modes=dict(Counter('none' if arm=='no_kv' else p['mode'] for p in self.plan)))
            write_json(self.output/'partial_result.json',dict(arms=arms))
            del self.opt_fast,self.opt_slow
            self.fast.zero_grad(set_to_none=True); self.slow.zero_grad(set_to_none=True)
            gc.collect()
        control_equality=final_slow_controls['no_kv']==final_slow_controls['detached']
        result=dict(schema='slowfast_joint_result_v1',scope='bounded mini diagnostic; not generalization/async/closed-loop',
            config=self.c,arms=arms,summary=paired_summary(self.records),slow_shared_adapt=dict(before=slow_before,after=slow_after),
            common_init=dict(fast_digest=state_digest(common_fast),slow_digest=state_digest(common_slow)),
            no_kv_detached_slow_exactly_equal=control_equality,
            joint_slow_differs=final_slow_controls['joint']!=final_slow_controls['detached'],
            budget=dict(new_iterations=960,slow_optimizer_steps=960,fast_optimizer_steps=720,total_optimizer_steps=1680,
                inherited_fast_steps=80,inherited_slow_module_steps=60,inherited_slow_lora_steps=40),
            probe_steps=self.probe_steps,replay_step=self.replay_step,replay_checkpoints=replay_checkpoints,
            data_identities=self.data_identities,assets=self.assets,parent_checkpoint_sha256=self.parent_sha,
            feedback_producer_digest=self.feedback_digest,coverage=self.coverage,image_audit=self.audits,
            constant_velocity={split:constant_velocity_metrics(NavsimDataset(self.c['data'][split+'_manifest'],self.c['data']['stats']))
                               for split in ['train','val']},
            memory={d:dict(gpu=torch.cuda.get_device_name(d),peak_allocated_gib=torch.cuda.max_memory_allocated(d)/2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(d)/2**30) for d in self.devices},
            elapsed_s=time.monotonic()-self.started,test_used=False,val_used_for_selection=False,
            resume_verified=False,multi_gpu_validated=False)
        write_json(self.output/'result.json',result)
        if not control_equality:
            raise AssertionError('Slow controls differ: investigate before attributing joint differences to LF')
        self.log('Fixed budgets complete; independent-process reload/replay still required')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/slowfast_v11_joint.yaml')
    for name in ['train-manifest','val-manifest','stats','sensor-root','fast-model','slow-model','initial-checkpoint','output']:
        parser.add_argument('--'+name)
    args=parser.parse_args(); c=yaml.safe_load(Path(args.config).read_text())
    for name in ['train_manifest','val_manifest','stats','sensor_root']:
        if getattr(args,name) is not None: c['data'][name]=getattr(args,name)
    for name in ['fast','slow']:
        if getattr(args,name+'_model') is not None: c[name]['model_path']=getattr(args,name+'_model')
    for name in ['initial_checkpoint','output']:
        if getattr(args,name) is not None: c[name]=getattr(args,name)
    validate_joint_config(c)
    output=Path(c['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory; preserve previous evidence')
    output.mkdir(parents=True,exist_ok=True)
    JointExperiment(c).run()


if __name__=='__main__': main()
