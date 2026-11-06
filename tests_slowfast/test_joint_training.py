"""CPU native-model checks of consumer-only gradient routes and control fairness."""
import json

import pytest
import torch

from slowfast_v11.model import FastWithSlowKV, SlowPrefill
from slowfast_v11.validation import canonical_state, restore_weights


def build_pair(factory):
    # Construct fresh hooks and identical frozen weights; never deepcopy a bridge.
    torch.manual_seed(123)
    kw = dict(lora_rank=2, lora_alpha=4, lora_dropout=0., position_offset=128)
    return FastWithSlowKV(factory(), **kw).train(), SlowPrefill(factory(64), **kw).train()


def batch(native):
    return dict(native=native, ego=torch.ones(1, 4), command=torch.tensor([1]))


@pytest.mark.parametrize('arm,mode,live', [
    ('joint', 'fresh', True), ('detached', 'fresh', False),
    ('no_kv', 'fresh', False), ('joint', 'none', False),
    ('detached', 'none', False),
])
def test_fast_only_probe_separates_live_route_from_slow_own_loss(
        language_factory, native_input, arm, mode, live):
    from slowfast_v11.joint_training import joint_step

    fast, slow = build_pair(language_factory)
    before = canonical_state(slow)
    feedback = torch.randn(1, 8, 3, requires_grad=True)
    optimizers = [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
                  for m in (fast, slow)]
    result = joint_step(fast, slow, *optimizers, batch(native_input), batch(native_input),
                        torch.ones(1, 8, 3), torch.ones(1, 8, 3), feedback,
                        arm=arm, mode=mode, probe=True)
    # Returning tensors would keep graphs / GPU storage alive in runner logs.
    json.dumps(result, allow_nan=False)
    probe = result['probe']
    assert probe['expected_live'] is live
    assert len(probe['kv']) == 6
    for layer in probe['kv']:
        for kind in ['key', 'value']:
            stats = layer[kind]
            assert stats['with_gradient'] == int(live)
            assert stats['nonzero'] is live
            assert stats['finite']
    assert probe['lora_b']['with_gradient'] == (12 if live else 0)
    assert probe['lora_b']['nonzero'] is live
    assert probe['vision_frozen'] and not probe['vision_has_gradient']
    assert feedback.grad is None
    assert any(not torch.equal(before[n], p) for n, p in canonical_state(slow).items())
    assert result['loss_total'] == pytest.approx(result['loss_fast'] + result['loss_slow'])


def test_detached_and_no_kv_slow_updates_equal_with_separate_clipping(
        language_factory, native_input):
    from slowfast_v11.joint_training import joint_step

    fast, slow = build_pair(language_factory)
    initial_fast, initial_slow = canonical_state(fast), canonical_state(slow)
    feedback = torch.ones(1, 8, 3)
    states, reports = [], []
    for arm in ['no_kv', 'detached']:
        fast, slow = build_pair(language_factory)
        restore_weights(fast, initial_fast)
        restore_weights(slow, initial_slow)
        opts = [torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=.1)
                for m in (fast, slow)]
        # The step must reset both optimizers rather than accumulating this junk.
        if arm == 'detached':
            for model in (fast, slow):
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = torch.full_like(p, 10.)
        reports.append(joint_step(fast, slow, *opts, batch(native_input), batch(native_input),
                                  torch.full((1, 8, 3), 100.), torch.ones(1, 8, 3), feedback,
                                  arm=arm, mode='fresh', clip=.01))
        states.append(canonical_state(slow))
    assert reports[0]['loss_fast'] != reports[1]['loss_fast']
    assert reports[0]['fast_grad_norm'] != reports[1]['fast_grad_norm']
    assert reports[0]['slow_grad_norm'] > .01
    assert reports[0]['slow_grad_norm'] == reports[1]['slow_grad_norm']
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
    assert any(not torch.equal(initial_slow[name], states[0][name]) for name in initial_slow)


@pytest.mark.parametrize('arm,mode,weight,clip', [
    ('invalid', 'fresh', 1., 1.), ('joint', 'stale', 1., 1.),
    ('joint', 'fresh', -1., 1.), ('joint', 'fresh', 1., 0.),
])
def test_invalid_step_policy_rejected_before_forward(arm, mode, weight, clip):
    from slowfast_v11.joint_training import joint_step

    with pytest.raises(ValueError):
        joint_step(None, None, None, None, None, None, None, None, None,
                   arm=arm, mode=mode, slow_weight=weight, clip=clip)
