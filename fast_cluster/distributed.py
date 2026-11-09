"""Equal-rank epochs, globally weighted updates, and atomic collective checkpoints."""
from dataclasses import dataclass
from datetime import timedelta
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class AccumulationGroup:
    start_micro: int
    microbatches: tuple
    valid_count: int


class EpochPlan:
    """Seeded permutation with invalid padding; iteration never advances consumption."""
    def __init__(self, dataset_size, micro_batch, world_size, rank, accumulation, seed, epoch, cursor=0):
        for value in (dataset_size, micro_batch, world_size, accumulation):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError('Epoch dimensions must be positive integers')
        if not 0 <= rank < world_size or epoch < 0:
            raise ValueError('Invalid rank or epoch')
        self.dataset_size, self.micro_batch = dataset_size, micro_batch
        self.world_size, self.rank, self.accumulation = world_size, rank, accumulation
        self.seed, self.epoch, self.cursor = seed, epoch, cursor
        self.global_micro_batch = micro_batch * world_size
        self.micro_count = math.ceil(dataset_size / self.global_micro_batch)
        if (not 0 <= cursor <= self.micro_count or
                (cursor != self.micro_count and cursor % accumulation)):
            raise ValueError('Consumed cursor must be an optimizer boundary')
        generator = torch.Generator().manual_seed(seed + epoch)
        self.order = torch.randperm(dataset_size, generator=generator).tolist()

    def __iter__(self):
        for start in range(self.cursor, self.micro_count, self.accumulation):
            end = min(start + self.accumulation, self.micro_count)
            batches = []
            for micro in range(start, end):
                base = micro * self.global_micro_batch + self.rank * self.micro_batch
                batches.append(tuple((self.order[slot] if slot < self.dataset_size else self.order[0],
                                      slot < self.dataset_size)
                                     for slot in range(base, base + self.micro_batch)))
            count = min(end * self.global_micro_batch, self.dataset_size) - start * self.global_micro_batch
            yield AccumulationGroup(start, tuple(batches), count)

    def commit(self, group):
        if group.start_micro != self.cursor or group != next(iter(self), None):
            raise ValueError('Only the next complete optimizer group may be consumed')
        self.cursor += len(group.microbatches)

    def state_dict(self):
        return dict(version='epoch_permutation_invalid_padding_v1', dataset_size=self.dataset_size,
                    micro_batch=self.micro_batch, world_size=self.world_size,
                    accumulation=self.accumulation, seed=self.seed, epoch=self.epoch, cursor=self.cursor)

    @classmethod
    def from_state(cls, state, *, rank):
        state = dict(state)
        if state.pop('version', None) != 'epoch_permutation_invalid_padding_v1':
            raise ValueError('Unknown sampler version')
        return cls(rank=rank, **state)


def valid_scaled_loss(per_sample_loss, valid, *, world_size, global_valid_count):
    if per_sample_loss.ndim != 1 or valid.shape != per_sample_loss.shape or global_valid_count <= 0:
        raise ValueError('Expected per-sample losses, validity mask, and positive global valid count')
    return (per_sample_loss * valid.to(per_sample_loss)).sum() * (world_size / global_valid_count)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    device: torch.device


def initialize(config):
    from .config import validate_topology
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    validate_topology(config, world_size=world, local_world_size=local_world)
    if not 0 <= rank < world or not 0 <= local_rank < local_world:
        raise ValueError('Invalid rank environment')
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError('Native cluster training requires the declared local CUDA device')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group(backend=config['execution']['backend'],
                                timeout=timedelta(seconds=config['execution']['timeout_s']))
    return DistributedContext(rank, world, local_rank, local_world, device)


def capture_rng(device):
    device = torch.device(device)
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    device = torch.device(device)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise ValueError('Checkpoint has no local CUDA RNG state')
        torch.cuda.set_rng_state(state['cuda'], device)


def verify_identity(saved, current):
    if saved.get('world_size') != current.get('world_size'):
        raise ValueError('Resume world_size differs; strict same-world resume is required')
    if saved != current:
        different = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError(f'Resume semantic/content identity differs: {different}')


def _rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, backend, optimizer, scheduler, *, identity, progress, device, keep=2):
    """All ranks contribute local RNG; only rank zero constructs/writes full state."""
    rank, world = _rank_world()
    local_rng = capture_rng(device)
    all_rng = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_rng, all_rng, dst=0)
    else:
        all_rng = [local_rng]
    error = [None]
    if rank == 0:
        try:
            payload = dict(schema='navsim_cluster_checkpoint_v1', writer_rank=0, identity=identity,
                           world_size=world, progress=progress, model=backend.fast_state(),
                           fast_buffers={name: value.detach().cpu().clone()
                                         for name, value in backend.fast.named_buffers()},
                           optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           rng_by_rank=all_rng)
            _atomic_torch_save(payload, path)
            latest = Path(path).parent / 'latest.pt'
            descriptor, temporary = tempfile.mkstemp(prefix='.latest.', dir=latest.parent)
            os.close(descriptor)
            os.unlink(temporary)
            try:
                os.symlink(Path(path).name, temporary)
                os.replace(temporary, latest)
            finally:
                if os.path.lexists(temporary):
                    os.unlink(temporary)
            checkpoints = sorted(Path(path).parent.glob('update_*.pt'))
            for stale in checkpoints[:-keep]:
                stale.unlink()
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'Rank zero checkpoint failed: {error[0]}')


def load_checkpoint(path, backend, optimizer, scheduler, *, identity, device):
    rank, world = _rank_world()
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('schema') != 'navsim_cluster_checkpoint_v1':
        raise ValueError('Unknown full checkpoint schema')
    if state.get('world_size') != world or len(state.get('rng_by_rank', [])) != world:
        raise ValueError('Resume world_size/RNG rank count differs')
    verify_identity(state['identity'], identity)
    expected_buffers = dict(backend.fast.named_buffers())
    buffers = state['fast_buffers']
    if set(buffers) != set(expected_buffers) or any(
            buffers[key].shape != tensor.shape or buffers[key].dtype != tensor.dtype
            for key, tensor in expected_buffers.items()):
        raise ValueError('Fast checkpoint buffer keys/shape/dtype differ')
    backend.restore_fast_state(state['model'])
    with torch.no_grad():
        for name, tensor in expected_buffers.items():
            tensor.copy_(buffers[name])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng_by_rank'][rank], device)
    return state['progress']


def save_best_weights(path, backend, *, identity, metrics):
    rank, world = _rank_world()
    error = [None]
    if rank == 0:
        try:
            _atomic_torch_save(dict(schema='navsim_cluster_best_weights_v1', writer_rank=0,
                                    identity=identity, metrics=metrics, model=backend.fast_state(),
                                    fast_buffers={name: tensor.detach().cpu().clone()
                                                  for name, tensor in backend.fast.named_buffers()}), path)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'Rank zero best weights save failed: {error[0]}')
