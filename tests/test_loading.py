import random
import threading
import time

import pytest
import torch

from fast_only import loading


class Dataset:
    def __len__(self):
        return 19

    def __getitem__(self, index):
        return index


def collate(samples, *, seed):
    return {'values': torch.tensor(samples), 'augmentation': torch.tensor(random.Random(seed).random())}


def test_counter_sampler_resume_matches_uninterrupted_across_worker_counts():
    whole = list(loading.batch_stream(Dataset(), collate, 3, 42, 0, 12, workers=0, prefetch=1))
    stream = loading.batch_stream(Dataset(), collate, 3, 42, 0, 12, workers=2, prefetch=3)
    with stream:
        first = [next(stream) for _ in range(5)]
        state = stream.state_dict()
        time.sleep(.02)  # completed prefetch cannot advance the consumed cursor
        assert stream.state_dict() == state
    assert state['next_batch'] == 5 and state['seed'] == 42
    rest = list(loading.batch_stream(Dataset(), collate, 3, 42, state['next_batch'], 7,
                                     workers=3, prefetch=2))
    for expected, actual in zip(whole, first + rest, strict=True):
        assert expected.batch_index == actual.batch_index
        assert expected.indices == actual.indices
        assert torch.equal(expected.batch['values'], actual.batch['values'])
        assert torch.equal(expected.batch['augmentation'], actual.batch['augmentation'])
        assert actual.batch['values'].device.type == 'cpu'


def test_prefetch_loads_real_cpu_batches_and_is_bounded():
    calls = []
    class CountingDataset(Dataset):
        def __getitem__(self, index):
            calls.append((index, threading.current_thread().name))
            return index
    stream = loading.batch_stream(CountingDataset(), collate, 2, 1, 0, 100,
                                  workers=2, prefetch=3)
    with stream:
        first = next(stream)
        time.sleep(.05)
        assert 4 <= len(calls) <= 8  # current batch plus at most 3 future batches
        assert all(name.startswith('fast-only-loader') for _, name in calls)
        assert len(first.indices) == 2
        assert iter(stream) is stream
    assert stream.closed
    with pytest.raises(StopIteration):
        next(stream)


def test_worker_exception_is_propagated_and_closes_pool():
    def broken(samples, *, seed):
        raise RuntimeError('bad preprocessing')
    stream = loading.batch_stream(Dataset(), broken, 1, 1, 0, 5, workers=2, prefetch=2)
    with pytest.raises(RuntimeError, match='bad preprocessing'):
        next(stream)
    assert stream.closed


@pytest.mark.parametrize('kwargs', [dict(workers=-1), dict(prefetch=0), dict(microbatch=0),
                                    dict(start_batch=-1), dict(num_batches=-1)])
def test_invalid_loader_settings_fail(kwargs):
    args = dict(microbatch=1, seed=42, start_batch=0, num_batches=1, workers=0, prefetch=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        loading.batch_stream(Dataset(), collate, **args)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match='empty'):
        loading.batch_stream([], collate, 1, 42, 0, 1)
