"""Fixed matched-Fast-update-budget no-KV control for the bounded diagnostic.

Restore the saved Fast warmup teacher, reset AdamW, and reuse exactly the main
consumer's 120 TRAIN targets with KV absent on every update. Both Fast paths
therefore have 200 updates, while the main path also has 60 Slow updates.
"""
import argparse
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import random
import time


_CONTROL_STEPS = 120
_CONSUMER_MODES = ('none',)*3 + ('fresh',)*4 + ('stale',)*3


def control_targets(plan, rows):
    """Validate the complete saved consumer schedule and preserve its target order."""
    if len(plan) != _CONTROL_STEPS:
        raise ValueError('Matched control requires exactly 120 saved consumer steps')
    by_token = {}
    route_splits = {}
    for row in rows:
        token, split, route = row['token'], row['split'], row['route']
        if (not isinstance(token,str) or not token or token in by_token
                or split not in ('train','val')):
            raise ValueError('Invalid or duplicate TRAIN/VAL manifest identity')
        if route in route_splits and route_splits[route] != split:
            raise ValueError('Route leakage across TRAIN/VAL splits')
        route_splits[route] = split
        by_token[token] = row
    targets = []
    for i,item in enumerate(plan):
        token, mode = item.get('target_token'), item.get('mode')
        if token not in by_token or by_token[token]['split'] != 'train' or item.get('split') != 'train':
            raise ValueError('Every control update must target a known TRAIN row')
        if mode != _CONSUMER_MODES[i % 10]:
            raise ValueError('Saved consumer mode sequence differs from fixed 3/4/3 blocks')
        age, source = item.get('age_s'), item.get('source_token')
        if (isinstance(age,bool) or not isinstance(age,Real) or not math.isfinite(age)
                or not isinstance(source,str) or not source):
            raise ValueError('Saved consumer source/age is malformed')
        if mode == 'stale':
            if age not in (.5,1.,1.5) or source == token:
                raise ValueError('Stale consumer steps require a genuine older source')
        elif age != 0 or source != token:
            raise ValueError('Absent/fresh consumer steps require the current source at age zero')
        targets.append(token)
    return tuple(targets)


def target_order_digest(tokens):
    """SHA256 of the exact ordered token sequence, including repeated exposures."""
    payload = json.dumps(list(tokens),separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def run_control(checkpoint_path, output_directory):
    checkpoint_path, output = Path(checkpoint_path), Path(output_directory)
    if output.exists():
        raise FileExistsError('Matched control requires a new output directory')

    import numpy as np
    import torch
    from fast_only.common import fingerprint, write_json
    from slowfast_v11.validation import (canonical_state,restore_weights,state_digest,
        set_stage,make_optimizer,perform_step,paired_summary)
    from tools.validate_slowfast_v11 import Experiment

    started = time.monotonic()
    checkpoint_sha = fingerprint(checkpoint_path)
    result_path = checkpoint_path.parent/'result.json'
    original_result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if original_result is not None and original_result.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Main checkpoint SHA256 differs from original result.json')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if (checkpoint.get('schema') != 'slowfast_validation_v1' or checkpoint.get('stage') != 'fast_consumer'
            or checkpoint.get('step') != _CONTROL_STEPS
            or checkpoint['config']['stages']['fast_warmup'] != 80
            or checkpoint['config']['stages']['fast_consumer'] != _CONTROL_STEPS):
        raise ValueError('Requires a completed fixed-budget main validation checkpoint')
    teacher_digest = state_digest(checkpoint['feedback_producer'])
    if teacher_digest != checkpoint['feedback_producer_digest']:
        raise ValueError('Saved Fast warmup teacher identity differs')
    if original_result is not None and original_result.get('feedback_producer_digest') != teacher_digest:
        raise ValueError('Warmup teacher identity differs from original main result')

    experiment = Experiment(copy.deepcopy(checkpoint['config']))
    if experiment.assets != checkpoint['assets']:
        raise ValueError('Base model/processor identities differ from main checkpoint')
    if experiment.data_identities != checkpoint.get('data_identities'):
        raise ValueError('Data identities differ from main checkpoint')
    if original_result is not None and original_result.get('data_identities') != experiment.data_identities:
        raise ValueError('Data identities differ from original main result')
    plan = checkpoint['sampler']['plan']
    targets = control_targets(plan,experiment.rows)
    if checkpoint['sampler'].get('next_index') != _CONTROL_STEPS or plan != experiment.plan:
        raise ValueError('Saved consumer schedule differs from rebuilt deterministic TRAIN schedule')
    order_digest = target_order_digest(targets)
    rebuilt_order_digest = target_order_digest(item['target_token'] for item in experiment.plan)
    if order_digest != rebuilt_order_digest:
        raise ValueError('Control/main TRAIN target order differs')

    # The only loaded adaptation state is the completed Fast warmup teacher.
    # The consumer's final Fast state and optimizer are never restored here.
    restore_weights(experiment.fast,checkpoint['feedback_producer'])
    initial_digest = state_digest(canonical_state(experiment.fast))
    if initial_digest != teacher_digest:
        raise ValueError('Restored control initialization differs from shared Fast warmup state')
    experiment.slow.requires_grad_(False)
    experiment.slow.eval()
    set_stage(experiment.fast,'fast_consumer')
    optimizer = make_optimizer(experiment.fast,experiment.c['optimizer'])
    if optimizer.state:
        raise ValueError('Control optimizer must start with empty reset AdamW state')
    output.mkdir(parents=True,exist_ok=False)
    training = []
    with (output/'control_train.jsonl').open('x') as stream:
        for step,token in enumerate(targets,1):
            batch = experiment.batch(token,'fast')
            # No snapshot or cache metadata is supplied on any control update.
            metrics = perform_step(experiment.fast,optimizer,batch,experiment.target(token,'fast'),
                                   experiment.c['optimizer']['grad_clip'])
            record = dict(stage='matched_no_kv_control',step=step,token=token,mode='none',
                source_consumer_mode=plan[step-1]['mode'],windows_seen=step,
                equivalent_train_epochs=step/len(experiment.train),**metrics)
            training.append(record)
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush()
            if step == 1 or step % 10 == 0:
                mean = sum(item['loss'] for item in training[-10:])/len(training[-10:])
                experiment.log(f'matched_no_kv_control {step}/120; last10 loss={mean:.6f}')
    experiment.fast.zero_grad(set_to_none=True)
    experiment.fast.eval()
    predictions = []
    with torch.no_grad():
        for row in experiment.rows:
            token = row['token']
            batch = experiment.batch(token,'fast')
            prediction = experiment.fast(batch['native'],batch['ego'],batch['command']).float().cpu()
            predictions.append(dict(stage='matched_no_kv_control',split=row['split'],token=token,
                source_token=None,mode='none',age_s=0.,prediction=prediction[0].tolist(),
                target=row['trajectory_gt'],speed_mps=math.hypot(*row['ego'][:2]),command=row['command']))
    write_json(output/'control_predictions.json',predictions)
    summary = paired_summary(predictions)
    final_state = canonical_state(experiment.fast)
    final_digest = state_digest(final_state)
    control_checkpoint = dict(schema='slowfast_matched_control_v1',config=experiment.c,
        fast=final_state,optimizer=optimizer.state_dict(),scheduler=None,
        stage='matched_no_kv_control',step=_CONTROL_STEPS,
        source_checkpoint_sha256=checkpoint_sha,feedback_producer_digest=teacher_digest,
        initial_fast_digest=initial_digest,final_fast_digest=final_digest,
        assets=experiment.assets,data_identities=experiment.data_identities,
        sampler=dict(next_index=_CONTROL_STEPS,targets=list(targets),target_order_digest=order_digest),
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()),
        reference_predictions=[record for record in predictions if record['split']=='val'])
    torch.save(control_checkpoint,output/'control_final.pt')
    result = dict(schema='slowfast_matched_control_result_v1',
        scope='matched Fast update count and TRAIN target order; main path additionally trains Slow for 60 updates',
        initialization='exact saved completed Fast warmup state; reset AdamW',
        source_checkpoint=checkpoint_path.name,source_checkpoint_sha256=checkpoint_sha,
        source_checkpoint_sha256_verified_against_result=original_result is not None,
        feedback_producer_digest=teacher_digest,initial_fast_digest=initial_digest,final_fast_digest=final_digest,
        control_checkpoint_sha256=fingerprint(output/'control_final.pt'),
        assets=experiment.assets,data_identities=experiment.data_identities,
        asset_identities_verified=True,data_identities_verified=True,
        target_order_digest=order_digest,source_consumer_target_order_digest=rebuilt_order_digest,
        target_order_verified=True,target_order_count=len(targets),target_tokens=list(targets),
        counts={split:sum(row['split']==split for row in experiment.rows) for split in ('train','val')},
        updates=dict(shared_fast_warmup=80,control_additional_fast=120,control_total_fast=200,
                     main_total_fast=200,main_additional_slow=60,main_run=260,
                     additional_control_run=120,combined_operational=380,control_slow=0),
        microbatch_size=1,optimizer=experiment.c['optimizer'],optimizer_reset=True,
        optimizer_restored_from_main=False,seed=experiment.c['seed'],
        training_modes={'none':120},summary=summary,predictions=predictions,
        image_audit=experiment.audits,devices={name:experiment.c[name]['device'] for name in ('fast','slow')},
        elapsed_s=time.monotonic()-started,test_used=False,val_used_for_selection=False,
        no_feedback_regenerated=True,no_slow_forward=True,continuation_verified=False)
    write_json(output/'control_result.json',result)
    experiment.log('Matched-budget control finished; all TRAIN/VAL no-KV predictions saved')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True,help='New local output directory; existing directories are refused')
    args = parser.parse_args()
    result = run_control(args.checkpoint,args.output)
    print(json.dumps({'target_order_verified':result['target_order_verified'],'updates':result['updates']},sort_keys=True))


if __name__ == '__main__':
    main()
