import random
import numpy as np
import pytest
import torch

from tools.train_slowfast_joint import capture_rng, restore_rng, compare_trees
from tools.validate_slowfast_v11 import Experiment


def test_rng_roundtrip_reproduces_next_draws():
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(5))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(5))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])


def test_replay_comparison_includes_optimizer_moments_and_dtypes():
    reference = {'state': {0: {'exp_avg': torch.tensor([1., 2.])}}, 'step': 4}
    assert compare_trees(reference, reference)['passed']
    changed = {'state': {0: {'exp_avg': torch.tensor([1., 3.])}}, 'step': 4}
    report = compare_trees(reference, changed)
    assert not report['passed'] and report['max_abs_difference'] == 1.
    assert not compare_trees(torch.ones(1), torch.ones(1, dtype=torch.float64))['passed']
    assert not compare_trees(reference, {})['passed']


def test_new_harness_validator_runs_before_any_dataset_access():
    class Stopped(Exception): pass
    seen = []
    def validator(config):
        seen.append(config)
        raise Stopped
    with pytest.raises(Stopped):
        Experiment({'train_seed': 17}, config_validator=validator)
    assert seen == [{'train_seed': 17}]
