"""Bounded two-rank CPU Gloo algorithm regression; never a native-model claim."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fast_cluster.distributed import (EpochPlan, load_checkpoint, save_checkpoint,
                                      valid_scaled_loss)


class TinyBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast = nn.Linear(1, 1, bias=False)
        self.fast.register_buffer('protocol_buffer', torch.tensor([3.]), persistent=False)
        with torch.no_grad():
            self.fast.weight.fill_(.25)

    def forward(self, x):
        return self.fast(x)

    def fast_state(self):
        return {key: value.detach().clone() for key, value in self.fast.state_dict().items()}

    def restore_fast_state(self, state):
        self.fast.load_state_dict(state, strict=True)


def _random_draw():
    return [random.random(), float(np.random.rand()), float(torch.rand(()))]


def _update(wrapped, optimizer, scheduler, group, world):
    optimizer.zero_grad(set_to_none=True)
    for micro, slots in enumerate(group.microbatches):
        context = wrapped.no_sync() if micro < len(group.microbatches) - 1 else nullcontext()
        with context:
            _random_draw()  # Checkpoint must recover each rank's independent trajectory.
            x = torch.tensor([[index + 1.] for index, _ in slots]) / 13
            target = x * .8
            losses = (wrapped(x) - target).square().flatten()
            loss = valid_scaled_loss(losses, torch.tensor([valid for _, valid in slots]),
                                     world_size=world, global_valid_count=group.valid_count)
            loss.backward()
    gradient = wrapped.module.fast.weight.grad.detach().clone()
    optimizer.step()
    scheduler.step()
    return gradient


def _worker(rank, world, output, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(seconds=90))
    try:
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        torch.random.default_generator.manual_seed(42 + rank)
        backend = TinyBackend()
        wrapped = DistributedDataParallel(backend, find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(backend.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        plan = EpochPlan(13, 1, world, rank, 3, 42, 0)
        groups = list(plan)
        gradient = _update(wrapped, optimizer, scheduler, groups[0], world)
        # Compare DDP's first gradient against the exact global valid sample mean.
        reference = TinyBackend()
        valid_indices = [index for r in range(world)
                         for slots in list(EpochPlan(13, 1, world, r, 3, 42, 0))[0].microbatches
                         for index, valid in slots if valid]
        x = torch.tensor(valid_indices).float().reshape(-1, 1).add(1).div(13)
        (reference(x) - .8 * x).square().mean().backward()
        mean_equal = torch.allclose(gradient, reference.fast.weight.grad, atol=1e-7, rtol=1e-6)
        plan.commit(groups[0])
        identity = dict(world_size=world, test_model='tiny_linear', data='13_unique_rows')
        progress = dict(update=1, sampler=plan.state_dict())
        checkpoint = Path(output) / 'update_000000001.pt'
        save_checkpoint(checkpoint, backend, optimizer, scheduler, identity=identity,
                        progress=progress, device='cpu')
        expected_random = _random_draw()
        _update(wrapped, optimizer, scheduler, groups[1], world)
        expected_weight = backend.fast.weight.detach().clone()
        expected_after = _random_draw()
        # Recreate every mutable object before reading the real on-disk checkpoint.
        restored = TinyBackend()
        restored_wrapped = DistributedDataParallel(restored, find_unused_parameters=True, broadcast_buffers=False)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: .9 ** step)
        restored_progress = load_checkpoint(checkpoint, restored, restored_optimizer, restored_scheduler,
                                            identity=identity, device='cpu')
        actual_random = _random_draw()
        restored_plan = EpochPlan.from_state(restored_progress['sampler'], rank=rank)
        next_group = next(iter(restored_plan))
        _update(restored_wrapped, restored_optimizer, restored_scheduler, next_group, world)
        actual_after = _random_draw()
        resume_equal = torch.equal(expected_weight, restored.fast.weight)
        restored_plan.commit(next_group)
        # Exercise a final one-microbatch group with only one valid global slot.
        tail = next(iter(restored_plan))
        assert len(tail.microbatches) == 1 and tail.valid_count == 1
        tail_reference = TinyBackend()
        with torch.no_grad():
            tail_reference.fast.weight.copy_(restored.fast.weight)
        valid_tail = [index for r in range(world)
                      for slots in list(EpochPlan(13, 1, world, r, 3, 42, 0))[-1].microbatches
                      for index, valid in slots if valid]
        tail_x = torch.tensor(valid_tail).float().reshape(-1, 1).add(1).div(13)
        (tail_reference(tail_x) - .8 * tail_x).square().mean().backward()
        tail_gradient = _update(restored_wrapped, restored_optimizer, restored_scheduler, tail, world)
        mean_equal = mean_equal and torch.allclose(tail_gradient, tail_reference.fast.weight.grad,
                                                   atol=1e-7, rtol=1e-6)
        restored_plan.commit(tail)
        all_weights = [torch.empty_like(restored.fast.weight) for _ in range(world)]
        dist.all_gather(all_weights, restored.fast.weight.detach())
        record = dict(mean_gradient_equal=bool(mean_equal), resume_equal=bool(resume_equal),
                      rank_rng_equal=expected_random == actual_random and expected_after == actual_after,
                      parameters_equal=all(torch.equal(all_weights[0], weight) for weight in all_weights))
        records = [None] * world
        dist.all_gather_object(records, record)
        # All checkpoint calls are collective; retention is handled by the sole writer.
        for update in (2, 3):
            save_checkpoint(Path(output) / f'update_{update:09d}.pt', restored,
                            restored_optimizer, restored_scheduler, identity=identity,
                            progress=dict(update=update, sampler=restored_plan.state_dict()), device='cpu')
        if rank == 0:
            files = sorted(Path(output).glob('update_*.pt'))
            assert len(files) == 2
            writers = sorted(set(torch.load(path, map_location='cpu', weights_only=False)['writer_rank']
                                 for path in files))
            payload = dict(validation_scope='two-rank CPU algorithm regression; no native model or H200 coverage',
                           world_size=world, backend='gloo', valid_samples=13, checkpoint_writers=writers,
                           retained_checkpoints=len(files),
                           **{key: all(row[key] for row in records) for key in record})
            (Path(output) / 'receipt.json').write_text(json.dumps(payload, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def run_smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rendezvous = output / 'gloo_rendezvous'
    if rendezvous.exists():
        rendezvous.unlink()
    mp.spawn(_worker, args=(2, str(output), str(rendezvous)), nprocs=2, join=True)
    receipt = json.loads((output / 'receipt.json').read_text())
    if not all(receipt[key] for key in ('mean_gradient_equal', 'resume_equal', 'rank_rng_equal', 'parameters_equal')):
        raise AssertionError(f'Distributed algorithm smoke failed: {receipt}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


if __name__ == '__main__':
    main()
