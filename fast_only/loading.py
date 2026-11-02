"""Bounded CPU dataset/processor prefetch with a resumable counter sampler.

Threads avoid CUDA fork/spawn and remote-environment interpreter startup. They
never transfer tensors to CUDA. A batch's sample choices and augmentation seed
are pure functions of its counter, independent of thread scheduling/global RNG.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import random

import torch


@dataclass(frozen=True)
class PreparedBatch:
    indices: tuple[int, ...]
    batch_index: int
    batch: dict


def _counter_seed(seed, batch_index):
    payload = f'fast-only-counter-v1:{seed}:{batch_index}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def _cpu_tensors(value, pin_memory):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Dataset prefetch must produce CPU tensors only')
        return value.pin_memory() if pin_memory else value
    if isinstance(value, dict):
        return {key: _cpu_tensors(item, pin_memory) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensors(item, pin_memory) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tensors(item, pin_memory) for item in value)
    return value


class BatchStream:
    """Single-pass ordered stream; checkpoint the consumed cursor, never futures.

    ``num_batches`` counts batches starting at ``start_batch``. ``prefetch`` is
    the maximum number of future batches held/submitted beyond the batch just
    returned to the caller. Use a context manager to promptly close early exits.
    The collator must accept ``seed=`` and avoid mutable global RNG/state.
    """
    def __init__(self, dataset, collator, microbatch, seed, start_batch, num_batches,
                 *, workers=2, prefetch=2, pin_memory=False):
        if len(dataset) == 0:
            raise ValueError('Cannot sample an empty dataset')
        for key, value, minimum in [('microbatch', microbatch, 1), ('start_batch', start_batch, 0),
                                    ('num_batches', num_batches, 0), ('workers', workers, 0),
                                    ('prefetch', prefetch, 1)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        self.dataset = dataset
        self.collator = collator
        self.microbatch = microbatch
        self.seed = seed
        self.next_batch = start_batch
        self.stop_batch = start_batch + num_batches
        self.workers = workers
        self.prefetch = prefetch
        self.pin_memory = pin_memory
        self.closed = False
        self._submitted = start_batch
        self._pending = deque()
        self._pool = None

    def _prepare(self, batch_index):
        seed = _counter_seed(self.seed, batch_index)
        rng = random.Random(seed)
        indices = tuple(rng.randrange(len(self.dataset)) for _ in range(self.microbatch))
        samples = [self.dataset[index] for index in indices]
        batch = self.collator(samples, seed=seed)
        return PreparedBatch(indices, batch_index, _cpu_tensors(batch, self.pin_memory))

    def _fill(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers,
                                            thread_name_prefix='fast-only-loader')
        while len(self._pending) < self.prefetch and self._submitted < self.stop_batch:
            self._pending.append(self._pool.submit(self._prepare, self._submitted))
            self._submitted += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed or self.next_batch >= self.stop_batch:
            self.close()
            raise StopIteration
        try:
            if self.workers:
                self._fill()
                prepared = self._pending.popleft().result()
            else:
                prepared = self._prepare(self.next_batch)
            self.next_batch += 1
            if self.workers:
                self._fill()
            return prepared
        except BaseException:
            self.close()
            raise

    def state_dict(self):
        return {'sampler_version': 'counter_v1', 'seed': self.seed,
                'next_batch': self.next_batch, 'microbatch': self.microbatch,
                'dataset_size': len(self.dataset)}

    def close(self):
        self.closed = True
        for future in self._pending:
            future.cancel()
        self._pending.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def batch_stream(dataset, collator, microbatch, seed, start_batch, num_batches,
                 *, workers=2, prefetch=2, pin_memory=False):
    return BatchStream(dataset, collator, microbatch, seed, start_batch, num_batches,
                       workers=workers, prefetch=prefetch, pin_memory=pin_memory)
