"""Distributed protocol checks; these do not claim native-model or H200 coverage."""
import random

import numpy as np
import pytest
import torch


def test_epoch_plan_covers_once_padding_is_invalid_and_cursor_is_consumed_only():
    from fast_cluster.distributed import EpochPlan

    plans = [EpochPlan(19, micro_batch=2, world_size=3, rank=r,
                       accumulation=2, seed=42, epoch=0) for r in range(3)]
    groups = [list(p) for p in plans]
    assert [len(g.microbatches) for g in groups[0]] == [2, 2]
    assert [g.valid_count for g in groups[0]] == [12, 7]
    valid = [i for rank_groups in groups for g in rank_groups
             for batch in g.microbatches for i, keep in batch if keep]
    assert sorted(valid) == list(range(19))
    assert len(valid) == len(set(valid))
    assert plans[0].state_dict()['cursor'] == 0  # iteration/prefetch never consumes
    plans[0].commit(groups[0][0])
    assert plans[0].state_dict()['cursor'] == 2
    resumed = EpochPlan.from_state(plans[0].state_dict(), rank=0)
    assert list(resumed) == groups[0][1:]
    assert list(EpochPlan(19, 2, 3, 0, 2, 42, 1)) != groups[0]


def test_partial_accumulation_group_uses_global_valid_mean():
    from fast_cluster.distributed import EpochPlan, valid_scaled_loss

    groups = list(EpochPlan(13, 2, 3, 0, 3, 42, 0))
    assert len(groups) == 1 and len(groups[0].microbatches) == 3
    weight = torch.tensor(2., requires_grad=True)
    total = 0.
    for rank in range(3):
        group = list(EpochPlan(13, 2, 3, rank, 3, 42, 0))[0]
        for batch in group.microbatches:
            values = torch.tensor([i + 1. for i, _ in batch])
            loss = valid_scaled_loss(weight * values,
                                     torch.tensor([keep for _, keep in batch]),
                                     world_size=3, global_valid_count=group.valid_count)
            total = total + loss / 3  # DDP averages gradients across ranks
    total.backward()
    assert weight.grad.item() == pytest.approx(7.)


def test_rng_roundtrip_is_local_device_only(monkeypatch):
    from fast_cluster.distributed import capture_rng, restore_rng

    def forbidden(*args, **kwargs):
        raise AssertionError('all-device CUDA RNG API must never run')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    state = capture_rng(torch.device('cpu'))
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng(state, torch.device('cpu'))
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_real_two_rank_gloo_checkpoint_resume_and_single_writer(tmp_path):
    from tools.smoke_distributed import run_smoke

    receipt = run_smoke(tmp_path)
    assert receipt['world_size'] == 2
    assert receipt['backend'] == 'gloo'
    assert receipt['valid_samples'] == 13
    assert receipt['mean_gradient_equal']
    assert receipt['parameters_equal']
    assert receipt['resume_equal']
    assert receipt['rank_rng_equal']
    assert receipt['checkpoint_writers'] == [0]
    assert (tmp_path / 'latest.pt').resolve().name == 'update_000000003.pt'
