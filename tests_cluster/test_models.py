import copy

import pytest
import torch

from fast_cluster.models import expand_snapshot, normalize_ego, validate_source_bundle
from slowfast_v11.protocol import KVSnapshot, LayerKV
from slowfast_v11.validation import state_digest


def test_distinct_training_and_source_normalizers():
    raw = torch.tensor([[4., 6., 8., 10.]])
    full = {'ego_mean': [0.] * 4, 'ego_std': [2.] * 4}
    source = {'ego_mean': [2.] * 4, 'ego_std': [1.] * 4}
    assert normalize_ego(raw, full).tolist() == [[2., 3., 4., 5.]]
    assert normalize_ego(raw, source).tolist() == [[2., 4., 6., 8.]]
    with pytest.raises(ValueError):
        normalize_ego(raw, dict(full, ego_std=[0.] * 4))


def test_fixed_snapshot_expands_all_native_axes():
    key = torch.randn(1, 2, 3, 4)
    snap = KVSnapshot((LayerKV(key, key.clone()),) * 6,
                      torch.arange(9).view(3, 1, 3), torch.ones(1, 3, dtype=torch.bool))
    expanded = expand_snapshot(snap, 2)
    assert expanded.layers[0].key.shape == (2, 2, 3, 4)
    assert torch.equal(expanded.position_ids[:, 0], expanded.position_ids[:, 1])
    assert expanded.valid_mask.shape == (2, 3)


def test_source_bundle_authenticates_both_task_states():
    state = {'x': torch.ones(2)}
    bundle = dict(schema='navsim_frozen_source_v1', source_checkpoint_sha256='a'*64,
                  slow=state, feedback_teacher=state, slow_digest=state_digest(state),
                  feedback_teacher_digest=state_digest(state),
                  stats={'ego_mean': [0.]*4, 'ego_std': [1.]*4},
                  model_kwargs={}, assets={'fast': {}, 'slow': {}})
    validate_source_bundle(bundle, 'a'*64)
    damaged = copy.deepcopy(bundle)
    damaged['slow']['x'][0] = 9.
    with pytest.raises(ValueError, match='digest'):
        validate_source_bundle(damaged, 'a'*64)
    with pytest.raises(ValueError, match='checkpoint'):
        validate_source_bundle(bundle, 'b'*64)
