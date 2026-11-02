"""Single-GPU LoRA training, guarded smoke mode and resumable checkpoints."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import (NativeCollator, experiment_fingerprints, make_model, predict,
                     read_config, seed_all, to_device, write_json)
from .model import planning_loss


def build_scheduler(optimizer, steps, warmup_fraction):
    warmup = max(1, math.ceil(steps * warmup_fraction))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step-warmup) / max(1, steps-warmup))
        return .5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer, scheduler, config, step, best_metric, data_fingerprints,
                    sampler_state=None, windows_seen=None):
    import json
    import importlib.metadata
    stats_path = config.get('data', {}).get('stats')
    state = {'format_version': 1, 'model': model.trainable_state(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'config': config, 'step': step, 'best_metric': best_metric,
             'sampler_state': sampler_state, 'windows_seen': windows_seen,
             'data_fingerprints': data_fingerprints,
             'normalization_stats': json.loads(Path(stats_path).read_text()) if stats_path else None,
             'versions': {'dataset': 'rethinklab/Bench2Drive-V0.0.4',
                          'torch': torch.__version__,
                          'transformers': importlib.metadata.version('transformers'),
                          'peft': importlib.metadata.version('peft')},
             'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                     'torch': torch.get_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary); temporary.replace(path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_trainable_state(state['model'])
    if optimizer is not None: optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None: scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if torch.cuda.is_available() and state['rng']['cuda']:
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def training_stop_step(total, restored_step, stop_after):
    stop = min(total, total if stop_after is None else stop_after)
    if stop <= restored_step:
        raise ValueError(f'Stop step {stop} must exceed restored step {restored_step}')
    return stop


def training_accounting(step, microbatch, accumulation, dataset_size):
    windows = step * microbatch * accumulation
    return {'windows_seen': windows, 'epoch_equivalent': windows / dataset_size}


def _validate_v2_rows(dataset, config, role):
    """Reject a stale/legacy manifest instead of silently changing semantics."""
    from .navigation import load_source_audit
    evidence = load_source_audit(config['data']['source_audit'])
    for row in dataset.rows:
        nav = row.get('navigation_metadata', {})
        if row.get('nav_source') != 'official_annotation_near_route_target':
            raise ValueError(f'{role} manifest has unverified/zero navigation')
        if nav.get('origin') != config['data']['navigation_origin'] or nav.get('gnss_mount_xy_m') != config['data']['gnss_mount_xy_m']:
            raise ValueError(f'{role} navigation origin/extrinsic differs from config')
        if nav.get('privileged_ego_pose_used') is not False:
            raise ValueError(f'{role} navigation uses privileged ego pose')
        if row.get('timing_source') != config['data']['timing_policy']:
            raise ValueError(f'{role} clock differs from configured nominal policy')
        if row.get('timing_metadata', {}).get('source_audit_sha256') != evidence['audit_sha256']:
            raise ValueError(f'{role} manifest source evidence is stale')
        if row.get('label_pose_source') != 'official_ego_vehicle_pose_supervision_only':
            raise ValueError(f'{role} labels do not use verified official ego pose')


def run_training(config, *, resume=None, stop_after=None, p0_only=False):
    from .benchmark import measure
    from .data import FastDataset
    from .evaluate import evaluate_dataset, save_predictions
    from .configuration import validate_training_config
    from types import SimpleNamespace
    validate_training_config(config)
    args = SimpleNamespace(resume=resume, stop_after=stop_after, p0_only=p0_only)
    v2 = config.get('schema_version') == 2
    cap = config.get('execution', {}).get('stop_after')
    if cap is not None:
        if args.stop_after is not None and args.stop_after > cap:
            raise ValueError('Requested stop exceeds the explicit preflight execution cap')
        args.stop_after = min(cap, args.stop_after) if args.stop_after is not None else cap
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if config != saved['config']:
            raise ValueError('Resume requires the same config, including original total-step budget')
    seed = config['train_seed'] if v2 else config['seed']
    seed_all(seed)
    torch.set_num_threads(config.get('cpu_threads', 4))
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output/'last.pt').exists():
        raise FileExistsError(f'{output}/last.pt already exists; use --resume or a new output directory')
    dataset = FastDataset(config['data']['train_manifest'], config['data']['stats'])
    if len(dataset) == 0:
        raise ValueError('Empty training manifest')
    splits = {row['split'] for row in dataset.rows}
    if splits != {'train'}:
        raise ValueError(f'Training manifest must contain only train rows, got {splits}')
    if v2:
        _validate_v2_rows(dataset, config, 'train')
    if config.get('smoke', False):
        if len(dataset) > config.get('smoke_max_windows', 16):
            raise ValueError('Smoke cohort exceeds configured hard sample limit')
        selection_dataset = dataset; selection_cohort = 'train_smoke'
    else:
        if not config['data'].get('val_manifest'):
            raise ValueError('Formal training requires a fixed independent val manifest')
        selection_dataset = FastDataset(config['data']['val_manifest'], config['data']['stats'])
        if {row['split'] for row in selection_dataset.rows} != {'val'}:
            raise ValueError('Validation manifest must contain only val rows')
        if v2:
            _validate_v2_rows(selection_dataset, config, 'val')
        from .data import geographic_group, TIMING_FALLBACK
        for row in dataset.rows + selection_dataset.rows:
            if row.get('timing_source') == TIMING_FALLBACK:
                raise ValueError('Frame-index timing assumptions are permitted only in smoke mode')
        train_groups = {row.get('geographic_group', geographic_group(row['route'])) for row in dataset.rows}
        val_groups = {row.get('geographic_group', geographic_group(row['route'])) for row in selection_dataset.rows}
        if train_groups & val_groups: raise ValueError('Train/VAL geographic route group overlap')
        selection_cohort = 'val'
    fingerprints = experiment_fingerprints(config)
    if args.resume and saved['data_fingerprints'] != fingerprints:
        raise ValueError('Dataset/statistics/model configuration changed since checkpoint')
    write_json(output/'config.json', config)
    write_json(output/'data_fingerprints.json', fingerprints)
    print('Loading native Qwen3.5 vision/language weights and injecting LoRA...', flush=True)
    model = make_model(config)
    write_json(output/'model_inventory.json', {
        'lora_targets': model.lora_targets,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_names': [name for name, p in model.named_parameters() if p.requires_grad],
        'trainable_dtypes': sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
        'base_weight_path': config['model']['path'],
    })
    collator = NativeCollator(config['model']['path'], **config['processor'])
    train_collator = NativeCollator(config['model']['path'], **config['processor'],
                                     brightness=config['train'].get('brightness', 0.))
    lora = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    extra = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    tc = config['train']
    optimizer = torch.optim.AdamW([{'params': lora, 'lr': tc['lora_lr']},
                                   {'params': extra, 'lr': tc['new_lr']}], weight_decay=tc['weight_decay'])
    scheduler = build_scheduler(optimizer, tc['steps'], tc['warmup_fraction'])
    start_step = 0; best = float('inf')
    if args.resume:
        state = restore_checkpoint(args.resume, model, optimizer, scheduler)
        start_step, best = state['step'], state['best_metric']
        if v2:
            expected = {'sampler_version': 'counter_v1', 'seed': seed,
                        'next_batch': start_step*tc['gradient_accumulation'],
                        'microbatch': tc['microbatch'], 'dataset_size': len(dataset)}
            if state.get('sampler_state') != expected:
                raise ValueError('Consumed sampler cursor does not match checkpoint/config')
    max_step = training_stop_step(tc['steps'], start_step, args.stop_after)
    if not args.resume:
        initial, pred, gt = evaluate_dataset(model, dataset, collator, config['device'])
        write_json(output/'initial_train_metrics.json', initial)
        save_predictions(output/'initial', dataset, pred, gt, cohort='train_smoke' if config['smoke'] else 'train')
        probe_samples = [dataset[i % len(dataset)] for i in range(tc['microbatch'])]
        model.train(); batch = to_device(train_collator(probe_samples, seed=seed), config['device'])
        prediction = predict(model, batch)
        loss = planning_loss(prediction, batch['trajectory_gt']); loss.backward()
        report = model.gradient_report()
        if not report['vision_frozen'] or report['vision_has_gradient']:
            raise RuntimeError('Vision freeze check failed')
        for key in ['ego', 'command', 'nav', 'queries', 'head', 'full_attention_lora', 'gdn_layer_ffn_lora']:
            if not report[key]['finite'] or not report[key]['nonzero']:
                raise RuntimeError(f'Missing/nonfinite gradient in {key}: {report[key]}')
        write_json(output/'p0_forward_backward.json', {'prediction_shape': list(prediction.shape),
                    'loss_l1_xy': loss.item(), 'gradients': report, 'layout': model.last_layout,
                    'image_audit': batch.get('image_audit'),
                    'processor_policy': config['processor'],
                    'navigation_policy': config['data'].get('navigation_mode'),
                    'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None})
        optimizer.zero_grad(set_to_none=True)
        # Release probe graph before measuring memory/latency.
        del loss, prediction, batch
        benchmark_config = config.get('benchmark', {})
        timing = None
        if benchmark_config.get('enabled', True):
            options = {key: value for key, value in benchmark_config.items()
                       if key in {'warmup', 'repeats', 'modes', 'profile_components'}}
            timing = measure(model, dataset, collator, config['device'], **options)
            write_json(output/'p0_benchmark.json', timing)
        print({'p0_forward_backward': 'passed', 'initial_train': initial,
               'end_to_end_ms': timing['end_to_end_ms'] if timing else None}, flush=True)
        if args.p0_only: return {'model': model, 'p0_only': True}
    log_path = output/'train.jsonl'
    mode = 'a' if args.resume else 'w'
    start_time = time.perf_counter()
    if torch.device(config['device']).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(config['device'])
    with ExitStack() as stack:
        log = stack.enter_context(log_path.open(mode))
        stream = None
        if v2:
            from .loading import batch_stream
            stream = stack.enter_context(batch_stream(dataset, train_collator, tc['microbatch'],
                         seed, start_step*tc['gradient_accumulation'],
                         (max_step-start_step)*tc['gradient_accumulation'],
                         **{key: value for key, value in config['loader'].items()
                            if key in {'workers', 'prefetch', 'pin_memory'}}))
        for step in range(start_step + 1, max_step + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.
            for _ in range(tc['gradient_accumulation']):
                if stream is not None:
                    prepared = next(stream)
                    batch = to_device(prepared.batch, config['device'])
                else:
                    indices = [random.randrange(len(dataset)) for _ in range(tc['microbatch'])]
                    batch = to_device(train_collator([dataset[i] for i in indices]), config['device'])
                prediction = predict(model, batch)
                loss = planning_loss(prediction, batch['trajectory_gt'])
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at step {step}')
                (loss / tc['gradient_accumulation']).backward()
                accumulated_loss += loss.detach().item() / tc['gradient_accumulation']
            norm = torch.nn.utils.clip_grad_norm_(lora + extra, tc['grad_clip'], error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            row = {'step': step, 'train_loss_l1_xy': accumulated_loss, 'grad_norm': float(norm),
                   'lora_lr': optimizer.param_groups[0]['lr'], 'new_lr': optimizer.param_groups[1]['lr'],
                   'elapsed_s': time.perf_counter()-start_time,
                   'effective_batch': tc['microbatch']*tc['gradient_accumulation'],
                   **training_accounting(step, tc['microbatch'], tc['gradient_accumulation'], len(dataset))}
            if step % tc['eval_every'] == 0 or step == max_step:
                metrics, pred, gt = evaluate_dataset(model, selection_dataset, collator, config['device'])
                row[f'{selection_cohort}_metrics'] = metrics
                if metrics['ADE'] < best:
                    best = metrics['ADE']
                    save_checkpoint(output/f'best_{selection_cohort}.pt', model, optimizer, scheduler,
                                    config, step, best, fingerprints,
                                    sampler_state=stream.state_dict() if stream else None,
                                    windows_seen=row['windows_seen'])
            if step % tc['checkpoint_every'] == 0 or step == max_step:
                save_checkpoint(output/'last.pt', model, optimizer, scheduler, config, step, best, fingerprints,
                                sampler_state=stream.state_dict() if stream else None,
                                windows_seen=row['windows_seen'])
                if v2:
                    save_checkpoint(output/f'step_{step:06d}.pt', model, optimizer, scheduler, config,
                                    step, best, fingerprints, sampler_state=stream.state_dict() if stream else None,
                                    windows_seen=row['windows_seen'])
            log.write(__import__('json').dumps(row, allow_nan=False) + '\n'); log.flush()
            if step == 1 or step % 10 == 0 or step == max_step: print(row, flush=True)
    final, pred, gt = evaluate_dataset(model, dataset, collator, config['device'])
    final.update(cohort='train_smoke' if config['smoke'] else 'train', optimizer_step=max_step)
    final.update(training_accounting(max_step, tc['microbatch'], tc['gradient_accumulation'], len(dataset)))
    if torch.cuda.is_available():
        final['peak_training_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        final['peak_training_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
    write_json(output/'final_train_metrics.json', final)
    save_predictions(output/'final', dataset, pred, gt, cohort=final['cohort'])
    print({'final_train': final, 'last_checkpoint': str(output/'last.pt')}, flush=True)
    return {'model': model, 'final_train': final, 'output': str(output), 'optimizer': optimizer,
            'scheduler': scheduler, 'dataset': dataset, 'collator': collator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int, help='Stop early without changing planned LR schedule')
    parser.add_argument('--p0-only', action='store_true')
    args = parser.parse_args()
    run_training(read_config(args.config), resume=args.resume,
                 stop_after=args.stop_after, p0_only=args.p0_only)


if __name__ == '__main__':
    main()
