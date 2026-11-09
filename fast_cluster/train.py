"""Native full-Fast NAVSIM epochs with honest valid-sample DDP accounting."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from .config import read_config, semantic_config, validate_config, split_history, fast_frame_count
from .distributed import (EpochPlan, initialize, load_checkpoint, save_best_weights,
                          save_checkpoint, valid_scaled_loss)


def per_sample_pose_loss(prediction, target):
    """The existing official NAVSIM loss, reduced over poses and x/y/heading only."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (8, 3):
        raise ValueError('Expected matching [B,8,3] NAVSIM poses')
    return (prediction.float() - target.float()).abs().mean((1, 2))


def _target(batch):
    return batch['target'] if 'target' in batch else batch['trajectory_gt']


def _seed_local(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _apply_reproducibility(config):
    rc = config['reproducibility']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = rc['cublas_workspace_config']
    torch.use_deterministic_algorithms(rc['deterministic_algorithms'], warn_only=rc['warn_only'])
    torch.backends.cudnn.deterministic = rc['cudnn_deterministic']
    torch.backends.cudnn.benchmark = rc['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = rc['allow_tf32']
    torch.backends.cudnn.allow_tf32 = rc['allow_tf32']


def make_scheduler(optimizer, total_updates, warmup_fraction, minimum_lr_ratio):
    warmup = int(total_updates * warmup_fraction)
    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        fraction = min(1., max(0., (step - warmup) / max(1, total_updates - warmup)))
        return minimum_lr_ratio + (1 - minimum_lr_ratio) * .5 * (1 + math.cos(math.pi * fraction))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def perform_update(wrapped, backend, optimizer, scheduler, samples, group, *, world_size, gradient_clip, device,
                   audit_dense=False):
    """One complete optimizer boundary, including a possibly short final group."""
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    for micro, _indices in enumerate(group.microbatches):
        sync = micro == len(group.microbatches) - 1
        context = wrapped.no_sync() if not sync and hasattr(wrapped, 'no_sync') else nullcontext()
        with context:
            batch = backend.prepare(next(samples))
            prediction = wrapped(batch)
            per_sample = per_sample_pose_loss(prediction, _target(batch))
            valid = batch['valid'].bool()
            finite = torch.isfinite(per_sample).all().to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('Nonfinite NAVSIM loss on at least one rank')
            loss = valid_scaled_loss(per_sample, valid, world_size=world_size,
                                     global_valid_count=group.valid_count)
            loss.backward()
            loss_sum += (per_sample.detach().double() * valid).sum()
        del batch, prediction, per_sample, loss
    gradient_report = None
    if audit_dense:
        from tools.train_direct_full import gradient_audit
        gradient_report = gradient_audit(backend.fast)
    parameters = [p for group_ in optimizer.param_groups for p in group_['params']]
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    backend.check_frozen()
    optimizer.step()
    scheduler.step()
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
    result = dict(loss_l1_pose=loss_sum.item() / group.valid_count, grad_norm=float(norm),
                valid_windows=group.valid_count, accumulation_microbatches=len(group.microbatches),
                learning_rates={group_.get('name', str(i)): group_['lr']
                                for i, group_ in enumerate(optimizer.param_groups)})
    if gradient_report is not None:
        result['dense_gradient_audit'] = gradient_report
    if audit_dense and hasattr(backend,'source_gradient_report'):
        result['source_gradient_audit'] = backend.source_gradient_report()
    return result


@torch.no_grad()
def evaluate(backend, dataset, *, rank, world_size, micro_batch, device, reduce=True):
    """Unequal rank slices use the unwrapped module, so forward has no DDP collective."""
    training = backend.training
    backend.eval()
    # ADE, FDE, pose L1, count, eight L2 horizons, wrapped heading mean/final.
    sums = torch.zeros(14, dtype=torch.float64, device=device)
    indices = list(range(rank, len(dataset), world_size))
    for start in range(0, len(indices), micro_batch):
        batch = backend.prepare([dataset[i] for i in indices[start:start + micro_batch]])
        prediction = backend(batch)
        target = _target(batch)
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            sums[:] = float('nan')
            break
        delta = prediction.float() - target.float()
        xy = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        heading = torch.atan2(delta[..., 2].sin(), delta[..., 2].cos()).abs()
        sums += torch.cat((torch.stack((xy.mean(-1).double().sum(), xy[:, -1].double().sum(),
                                        per_sample_pose_loss(prediction, target).double().sum(),
                                        sums.new_tensor(len(prediction)))),
                           xy.double().sum(0),
                           torch.stack((heading.mean(-1).double().sum(), heading[:, -1].double().sum()))))
    if reduce and dist.is_initialized():
        dist.all_reduce(sums)
    if not torch.isfinite(sums).all() or sums[3] <= 0:
        raise ValueError('Evaluation requires nonempty finite independent validation samples')
    result = dict(ADE=(sums[0] / sums[3]).item(), FDE=(sums[1] / sums[3]).item(),
                  loss_l1_pose=(sums[2] / sums[3]).item(), samples=int(sums[3].item()),
                  heading_mae_rad=(sums[12] / sums[3]).item(),
                  heading_final_mae_rad=(sums[13] / sums[3]).item())
    result.update({f'L2@{(i + 1) * .5:g}s': (sums[4 + i] / sums[3]).item() for i in range(8)})
    backend.train(training)
    return result


def _identity_collate(samples):
    return samples


def _loader(dataset, plan, data_config):
    batches = [batch for group in plan for batch in group.microbatches]
    # Dedicated generator prevents DataLoader worker seeding from changing model RNG.
    generator = torch.Generator().manual_seed(plan.seed + plan.epoch)
    kwargs = dict(dataset=dataset, batch_sampler=batches, collate_fn=_identity_collate,
                  num_workers=data_config['workers'], generator=generator)
    if data_config['workers']:
        kwargs['prefetch_factor'] = data_config['prefetch_factor']
        kwargs['multiprocessing_context'] = 'spawn'
    return DataLoader(**kwargs)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def _prepare_output(output, resume):
    error = [None]
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        try:
            output = Path(output)
            if output.exists() and any(output.iterdir()) and not resume:
                raise FileExistsError('Output contains prior evidence; use a new output or explicit resume')
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def run_training(config, *, arm, seed, output, resume=None, stop_after_updates=None, preflight_updates=None):
    validate_config(config)
    if arm not in config['train']['arms'] or seed not in config['train']['seeds']:
        raise ValueError('Arm/seed is outside the configured training protocol')
    if stop_after_updates is not None:
        if (config['execution']['mode'] != 'smoke' or isinstance(stop_after_updates, bool)
                or not isinstance(stop_after_updates, int) or stop_after_updates <= 0):
            raise ValueError('An explicit positive update stop is allowed only in Smoke mode')
    if preflight_updates is not None:
        if (config['execution']['mode'] != 'formal' or stop_after_updates is not None
                or isinstance(preflight_updates, bool) or not isinstance(preflight_updates, int)
                or preflight_updates <= 0):
            raise ValueError('Positive --preflight-updates requires the declared formal 16-GPU topology')
    _apply_reproducibility(config)
    context = initialize(config)
    device, rank, world = context.device, context.rank, context.world_size
    _prepare_output(output, resume)
    output = Path(output)
    from .data import PreparedData
    from .source_contract import validate_source_data
    data = PreparedData(config['data']['bundle'], config['data']['sensor_root'],
                        allow_smoke=config['execution']['mode'] == 'smoke')
    validate_source_data(config, data)
    if split_history(config):
        frames = 4 if arm == 'slow_fast' else fast_frame_count(config,arm)
        train,val = data.dataset('train',image_frames=frames),data.dataset('val',image_frames=frames)
    else:
        train,val = data.dataset('train'),data.dataset('val')
    if not len(train) or not len(val):
        raise ValueError('Nonempty independent TRAIN and VAL are required')
    from .models import build_backend
    from slowfast_v11.full_training import make_full_optimizer
    backend = build_backend(config, data.stats, arm, seed, device)
    frozen_start = backend.audit_frozen()
    identity = dict(semantic_config=semantic_config(config), model=backend.model_identity,
                    data=data.identity, arm=arm, seed=seed, world_size=world)
    tc = config['train']
    micro_count = math.ceil(len(train) / (world * tc['micro_batch']))
    updates_per_epoch = math.ceil(micro_count / tc['gradient_accumulation'])
    total_updates = updates_per_epoch * tc['epochs']
    if stop_after_updates is not None and stop_after_updates > total_updates:
        raise ValueError('Smoke stop exceeds declared full epoch schedule')
    if preflight_updates is not None and preflight_updates >= total_updates:
        raise ValueError('Formal preflight must stop before the complete eight-epoch schedule')
    optimizer = make_full_optimizer(backend.fast, language_lr=tc['language_lr'],
                                    new_lr=tc['new_lr'], weight_decay=tc['weight_decay'])
    if hasattr(backend,'add_source_optimizer_group'):
        backend.add_source_optimizer_group(optimizer,tc['new_lr'])
    scheduler = make_scheduler(optimizer, total_updates, tc['warmup_fraction'], tc['minimum_lr_ratio'])
    wrapped = DistributedDataParallel(backend, device_ids=[context.local_rank],
                output_device=context.local_rank, find_unused_parameters=True,
                broadcast_buffers=False) if world > 1 else backend
    # Provider loading and DDP construction cannot perturb the arm-paired RNG schedule.
    _seed_local(seed + rank, device)
    plan = EpochPlan(len(train), tc['micro_batch'], world, rank, tc['gradient_accumulation'], seed, 0)
    progress = dict(update=0, windows_seen=0, best_val_ADE=None, sampler=plan.state_dict())
    if resume:
        progress = load_checkpoint(resume, backend, optimizer, scheduler,
                                   identity=identity, device=device)
        plan = EpochPlan.from_state(progress['sampler'], rank=rank)
        expected = EpochPlan(len(train), tc['micro_batch'], world, rank,
                             tc['gradient_accumulation'], seed, plan.epoch, plan.cursor).state_dict()
        if plan.state_dict() != expected:
            raise ValueError('Checkpoint sampler dimensions differ from active training')
        expected_updates = plan.epoch * updates_per_epoch + math.ceil(plan.cursor / tc['gradient_accumulation'])
        expected_windows = plan.epoch * len(train) + min(len(train), plan.cursor * world * tc['micro_batch'])
        if progress['update'] != expected_updates or progress['windows_seen'] != expected_windows:
            raise ValueError('Checkpoint optimizer/window counters differ from consumed sampler cursor')
    cap = (preflight_updates if preflight_updates is not None else
           stop_after_updates if stop_after_updates is not None else total_updates)
    if cap <= progress['update']:
        raise ValueError('Requested budget must exceed restored update cursor')
    if rank == 0:
        _write_json(output / 'config.json', config)
        _write_json(output / 'identity.json', identity)
        _write_json(output / 'model_inventory.json', backend.diagnostics())
        _write_json(output / 'frozen_audit_start.json', frozen_start)
        _write_json(output / 'budget.json', dict(total_updates=total_updates, updates_per_epoch=updates_per_epoch,
                    train_windows=len(train), val_windows=len(val), epochs=tc['epochs'],
                    effective_global_batch=world * tc['micro_batch'] * tc['gradient_accumulation'],
                    loss='official_unweighted_pose_l1', checkpoint_full_retention=2))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    invocation_start_windows = progress['windows_seen']
    log = (output / 'train.jsonl').open('a' if resume else 'w') if rank == 0 else None
    last_metrics = None
    try:
        for epoch in range(plan.epoch, tc['epochs']):
            if epoch != plan.epoch:
                plan = EpochPlan(len(train), tc['micro_batch'], world, rank,
                                 tc['gradient_accumulation'], seed, epoch)
            loader = _loader(train, plan, config['data'])
            samples = iter(loader)
            for group in plan:
                row = perform_update(wrapped, backend, optimizer, scheduler, samples, group,
                                     world_size=world, gradient_clip=tc['gradient_clip'], device=device,
                                     audit_dense=progress['update'] == 0)
                plan.commit(group)
                progress['update'] += 1
                progress['windows_seen'] += group.valid_count
                progress['sampler'] = plan.state_dict()
                elapsed = time.perf_counter() - started
                row.update(update=progress['update'], epoch=epoch + 1, windows_seen=progress['windows_seen'],
                           equivalent_epochs=progress['windows_seen'] / len(train),
                           effective_global_batch=world * tc['micro_batch'] * tc['gradient_accumulation'],
                           elapsed_s=elapsed, windows_per_second=(progress['windows_seen'] - invocation_start_windows) / elapsed,
                           peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
                epoch_end = plan.cursor == plan.micro_count
                stopped = progress['update'] == cap
                if epoch_end and preflight_updates is None:
                    backend.set_eval_source('correct')
                    last_metrics = evaluate(backend, val, rank=rank, world_size=world,
                                            micro_batch=tc['micro_batch'], device=device)
                    row['val'] = last_metrics
                    row['val_ADE'], row['val_FDE'] = last_metrics['ADE'], last_metrics['FDE']
                    if progress['best_val_ADE'] is None or last_metrics['ADE'] < progress['best_val_ADE']:
                        progress['best_val_ADE'] = last_metrics['ADE']
                        save_best_weights(output / 'best_weights.pt', backend,
                                          identity=identity, metrics=last_metrics)
                if epoch_end or stopped or progress['update'] % tc['checkpoint_every_updates'] == 0:
                    save_checkpoint(output / f'update_{progress["update"]:09d}.pt', backend, optimizer, scheduler,
                                    identity=identity, progress=progress, device=device,
                                    keep=tc['keep_full_checkpoints'])
                if rank == 0:
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(dict(phase='train', **row), allow_nan=False), flush=True)
                if stopped:
                    break
            del samples, loader
            if progress['update'] == cap:
                break
        completed = progress['update'] == total_updates
        endpoint = {}
        preflight_partial_val = None
        if preflight_updates is not None and not completed:
            subset_size = min(len(val), world * tc['micro_batch'] * 2)
            subset = Subset(val, range(subset_size))
            backend.set_eval_source('correct')
            preflight_partial_val = evaluate(backend, subset, rank=rank, world_size=world,
                                             micro_batch=tc['micro_batch'], device=device)
            preflight_partial_val.update(scope='preflight_partial_val',
                                         full_val_windows=len(val), used_for_model_selection=False)
        if completed:
            modes = config['evaluation']['final_joint_modes'] if arm == 'slow_fast' else ['correct']
            if 'fixed' in modes:
                fixed_index = min(range(len(train)), key=lambda i: train.row(i)['token'])
                backend.initialize_fixed_source(train[fixed_index])
            for mode in modes:
                backend.set_eval_source(mode)
                endpoint[mode] = evaluate(backend, val, rank=rank, world_size=world,
                                         micro_batch=tc['micro_batch'], device=device)
            backend.set_eval_source('correct')
        frozen_end = backend.audit_frozen()
        if frozen_end != frozen_start:
            raise ValueError('Frozen vision/source parameters or buffers changed during training')
        result = dict(completed=completed, execution_mode=config['execution']['mode'],
                      stage='preflight' if preflight_updates is not None else config['execution']['mode'],
                      arm=arm, seed=seed,
                      update=progress['update'], total_updates=total_updates,
                      windows_seen=progress['windows_seen'], equivalent_epochs=progress['windows_seen'] / len(train),
                      best_val_ADE=progress['best_val_ADE'], endpoint=endpoint,
                      validation_scope='native single-GPU smoke' if world == 1 else 'declared formal topology')
        if preflight_partial_val is not None:
            result['preflight_partial_val'] = preflight_partial_val
        if rank == 0:
            _write_json(output / 'frozen_audit_end.json', frozen_end)
            _write_json(output / 'result.json', result)
        return result
    finally:
        if log is not None:
            log.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=['fast_only', 'fast_single', 'fast_history', 'slow_fast'], required=True)
    parser.add_argument('--seed', type=int, choices=[42, 43], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--preflight-updates', type=int,
                        help='Bounded formal-topology preflight; keeps the complete epoch/LR schedule for resume')
    args = parser.parse_args()
    run_training(read_config(args.config), arm=args.arm, seed=args.seed, output=args.output,
                 resume=args.resume, stop_after_updates=args.stop_after_updates,
                 preflight_updates=args.preflight_updates)


if __name__ == '__main__':
    main()
