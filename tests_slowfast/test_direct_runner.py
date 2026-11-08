"""CPU contracts for deterministic full-language training and accumulated replay."""
import copy
import random

import pytest
import torch
from torch import nn

from slowfast_v11.full_training import configure_full_fast, make_full_optimizer, full_state
from slowfast_v11.expanded_protocol import WarmupCosineScheduler


def api():
    from tools import train_direct_full
    return train_direct_full


class TinyFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.language_model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(.2), nn.Linear(4, 4))
        self.backbone.visual = nn.Linear(3, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.backbone.language_model(x))


def ready():
    torch.random.default_generator.manual_seed(42)
    model = TinyFull()
    configure_full_fast(model)
    optimizer = make_full_optimizer(model)
    scheduler = WarmupCosineScheduler(optimizer, 4, 1, .1)
    return model, optimizer, scheduler


def rows(count=512):
    return [dict(token=f'{i:016x}', split='train') for i in range(count)]


def test_order_is_eight_permutations_of_authenticated_train_only_and_ambient_rng_independent():
    f = api()
    cohort = rows()
    random.seed(7)
    before = random.getstate()
    first = f.build_draw_order(cohort)
    assert random.getstate() == before
    random.seed(900)
    assert first == f.build_draw_order(cohort)
    assert len(first) == 4096
    reference = random.Random(42)
    for epoch in range(8):
        expected = [r['token'] for r in cohort]
        reference.shuffle(expected)
        assert first[epoch*512:(epoch+1)*512] == expected
    assert first[:512] != first[512:1024]


@pytest.mark.parametrize('damage', ['val', 'duplicate', 'empty'])
def test_sampler_rejects_nontrain_or_ambiguous_rows(damage):
    f = api()
    cohort = rows()
    if damage == 'val':
        cohort[0]['split'] = 'val'
    elif damage == 'duplicate':
        cohort[1]['token'] = cohort[0]['token']
    else:
        cohort = []
    with pytest.raises(ValueError, match='TRAIN|train|unique|empty'):
        f.build_draw_order(cohort)


def test_sampler_boundary_refuses_partial_accumulation_and_detects_order_changes():
    f = api()
    order = f.build_draw_order(rows())
    saved = f.sampler_state(order, draw_index=4092, accumulation=4)
    assert saved['draw_index'] == 4092 and saved['completed_updates'] == 1023
    assert saved['accumulation_index'] == 0
    assert f.validate_sampler(saved, order, 4) == 4092
    for cursor in (-4, 1, 4097):
        with pytest.raises(ValueError, match='cursor|boundary|draw'):
            f.sampler_state(order, draw_index=cursor, accumulation=4)
    altered = order.copy()
    altered[0], altered[1] = altered[1], altered[0]
    with pytest.raises(ValueError, match='sampler|order'):
        f.validate_sampler(saved, altered, 4)
    damaged = dict(saved, completed_updates=1)
    with pytest.raises(ValueError, match='sampler|cursor'):
        f.validate_sampler(damaged, order, 4)


def test_fixed_train64_diagnostics_cover_all_twelve_recording_groups_and_speed_bins():
    f = api()
    from slowfast_v11.adapter_data import recording_group
    from slowfast_v11.expanded_protocol import select_train_diagnostics, speed_bin
    cohort = [dict(r, route=f'recording-{i//43}_00000_00500',
                   ego=[[0., 0., 0., 0.], [5., 0., 0., 0.], [15., 0., 0., 0.]][i%3])
              for i, r in enumerate(rows())]
    selected = f.select_diagnostic_rows(cohort, 64)
    assert len(selected) == 64
    assert len({recording_group(r['route']) for r in selected}) == 12
    assert {speed_bin(r) for r in selected} == {'low_speed', 'moving', 'high_speed'}
    assert [r['token'] for r in selected] == select_train_diagnostics(cohort, 64)
    assert selected == f.select_diagnostic_rows(list(reversed(cohort)), 64)
    with pytest.raises(ValueError, match='TRAIN|train'):
        f.select_diagnostic_rows(cohort+[dict(cohort[0], split='val')], 64)


def test_accumulated_step_averages_all_micro_losses_before_single_update():
    f = api()
    model, optimizer, scheduler = ready()
    model.eval()
    expected = copy.deepcopy(model)
    other = make_full_optimizer(expected)
    other_schedule = WarmupCosineScheduler(other, 4, 1, .1)
    xs = [torch.tensor([[float(i), 1., -1.]]) for i in range(4)]
    loss_fn = lambda token: (model(xs[token])-1).square().mean()
    losses = [(expected(x)-1).square().mean() for x in xs]
    mean = torch.stack(losses).mean()
    mean.backward()
    norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 1., error_if_nonfinite=True)
    other.step()
    other_schedule.step()
    result = f.accumulated_step(model, optimizer, scheduler, list(range(4)), loss_fn, clip=1.)
    assert result['loss'] == pytest.approx(float(mean.detach()), abs=1e-7)
    assert result['grad_norm'] == pytest.approx(float(norm), abs=1e-7)
    assert scheduler.completed_steps == 1
    assert all(torch.equal(v, full_state(expected)[k]) for k, v in full_state(model).items())
    assert all(p.grad is None for p in model.parameters())


def test_complete_checkpoint_replays_real_four_micro_update_with_rng_and_optimizer(monkeypatch):
    f = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model, optimizer, scheduler = ready()
    order = f.build_draw_order(rows(8), updates=4, accumulation=4)
    identity = dict(cache_sha256='cache', init_sha256='init', initial_digest='base')
    config = dict(gradient_accumulation=4)
    xs = {r['token']: torch.tensor([[int(r['token'], 16), 1., -.5]]) for r in rows(8)}
    loss_fn = lambda token: (model(xs[token])-1).square().mean()
    f.accumulated_step(model, optimizer, scheduler, order[:4], loss_fn)
    before = f.checkpoint_payload(model, optimizer, scheduler, order, 4, config, identity, 'full_fast_only')
    f.accumulated_step(model, optimizer, scheduler, order[4:8], loss_fn)
    after = f.checkpoint_payload(model, optimizer, scheduler, order, 8, config, identity, 'full_fast_only')
    assert any('language_model.0.weight' in name for name in before['model'])
    assert all('visual' not in name for name in before['model'])
    assert not torch.equal(before['model']['head.weight'], after['model']['head.weight'])
    restored = f.restore_checkpoint(model, optimizer, scheduler, before, order, config, identity, 'full_fast_only')
    assert restored == 4
    f.accumulated_step(model, optimizer, scheduler, order[4:8], loss_fn)
    replay = f.checkpoint_payload(model, optimizer, scheduler, order, 8, config, identity, 'full_fast_only')
    for key in ['model', 'optimizer', 'scheduler', 'rng', 'sampler']:
        assert f.compare_trees(after[key], replay[key])['passed'], key


def test_checkpoint_tampering_and_identity_drift_fail_before_parameter_copy(monkeypatch):
    f = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model, optimizer, scheduler = ready()
    order = f.build_draw_order(rows(8), updates=4, accumulation=4)
    identity = dict(cache_sha256='cache', init_sha256='init', initial_digest='base')
    config = dict(gradient_accumulation=4)
    saved = f.checkpoint_payload(model, optimizer, scheduler, order, 0, config, identity, 'full_fast_only')
    original = full_state(model)
    for damage in ['digest', 'source', 'cursor', 'missing_base']:
        changed = copy.deepcopy(saved)
        if damage == 'digest':
            changed['model']['head.weight'].add_(1)
        elif damage == 'source':
            changed['identity']['cache_sha256'] = 'different'
        elif damage == 'cursor':
            changed['sampler']['draw_index'] = 1
        else:
            changed['model'].pop('backbone.language_model.0.weight')
        with pytest.raises(ValueError):
            f.restore_checkpoint(model, optimizer, scheduler, changed, order, config, identity, 'full_fast_only')
        assert all(torch.equal(v, full_state(model)[k]) for k, v in original.items())


def test_nonfinite_micro_loss_does_not_step_optimizer():
    f = api()
    model, optimizer, scheduler = ready()
    initial = full_state(model)
    with pytest.raises(ValueError, match='finite'):
        f.accumulated_step(model, optimizer, scheduler, ['bad'], lambda token: model.head.weight.sum()*float('nan'))
    assert scheduler.completed_steps == 0 and len(optimizer.state) == 0
    assert all(torch.equal(v, full_state(model)[k]) for k, v in initial.items())


def test_endpoint_records_keep_complete_eight_pose_predictions_and_targets():
    f = api()
    row = dict(token='target', route='route', command=1, ego=[2., 3., 0., 0.],
               trajectory_gt=[[float(i), 0., .1] for i in range(8)])
    prediction = torch.ones(1, 8, 3)
    record = f.prediction_record(row, prediction, 'full_correct_kv', 'val', source_token='target')
    assert record['prediction'] == [[1., 1., 1.]]*8
    assert record['target'] == row['trajectory_gt']
    assert record['source_token'] == 'target' and record['split'] == 'val'
    assert record['arm'] == 'full_correct_kv'
    with pytest.raises(ValueError, match='pose|shape|finite'):
        f.prediction_record(row, prediction[:, :6], 'full_correct_kv', 'val', source_token='target')


def test_gradient_audit_demands_dense_gradient_in_every_native_block_and_frozen_visual():
    f = api()
    model = TinyFull()
    language = model.backbone.language_model
    language.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(24)])
    configure_full_fast(model)
    for p in language.parameters():
        p.grad = torch.ones_like(p)
    report = f.gradient_audit(model)
    assert len(report['blocks']) == 24
    assert all(r['finite'] and r['nonzero'] for r in report['blocks'])
    assert report['visual_frozen'] is True
    language.layers[-1].weight.grad = None
    language.layers[-1].bias.grad = None
    with pytest.raises(ValueError, match='dense|block|gradient'):
        f.gradient_audit(model)
    model.backbone.visual.weight.grad = torch.ones_like(model.backbone.visual.weight)
    with pytest.raises(ValueError, match='vision|visual'):
        f.gradient_audit(model)


def test_diagnostics_restore_rng_and_training_mode_and_never_draw_from_val(monkeypatch):
    f = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    model = TinyFull().train()
    cohort = [dict(token='train', split='train', route='r', command=0, ego=[0.]*4,
                   trajectory_gt=[[0., 0., 0.]]*8)]
    calls = []

    class Cache:
        def packet(self, token, device):
            calls.append(token)
            return dict(token=token)

    def forward(model, packet, arm, source):
        torch.rand(1)
        return torch.ones(1, 8, 3)

    monkeypatch.setattr(f, 'direct_forward', forward)
    before = torch.get_rng_state().clone()
    result, records = f.evaluate_rows(model, Cache(), cohort, 'full_fast_only', 'train', device='cpu')
    assert result['samples'] == 1 and len(records) == 1 and calls == ['train']
    assert torch.equal(before, torch.get_rng_state())
    assert model.training and not model.backbone.visual.training
    with pytest.raises(ValueError, match='split|TRAIN|train'):
        f.evaluate_rows(model, Cache(), [dict(cohort[0], split='val')], 'full_fast_only', 'train', device='cpu')


def test_training_loop_saves_true_before_and_after_last_update_without_val_draws(tmp_path, monkeypatch):
    f = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.random.default_generator.manual_seed(17)

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(24)])
            self.dropout = nn.Dropout(.1)

        def forward(self, x):
            for layer in self.layers:
                x = x + layer(x).tanh()*.1
            return self.dropout(x)

    model = TinyFull()
    model.backbone.language_model = Language()
    model.head = nn.Linear(3, 3)
    scope = configure_full_fast(model)
    cohort = [dict(token=r['token'], split='train', route='r', command=0, ego=[0.]*4,
                   trajectory_gt=[[0., 0., 0.]]*8) for r in rows(8)]

    class Cache:
        train = cohort
        val = [dict(cohort[0], token=f'{i:016x}', split='val') for i in range(8, 10)]
        calls = []

        def packet(self, token, device):
            self.calls.append(token)
            return dict(token=token)

    def forward(model, packet, arm, source):
        x = torch.tensor([[1., float(int(packet['token'], 16)), -1.]])
        return model(x)[:, None].expand(-1, 8, -1)

    monkeypatch.setattr(f, 'direct_forward', forward)
    config = dict(gradient_accumulation=4, language_lr=1e-5, new_lr=5e-5,
                  weight_decay=.01, optimizer_updates=4, warmup_updates=1,
                  minimum_lr_ratio=.1, gradient_clip=1., train_curve_updates=[1, 2, 3, 4],
                  train_diagnostic_windows=2, max_allocated_gib=76, train_windows=8)
    order = f.build_draw_order(cohort, updates=4)
    cache = Cache()
    result = f.train_loop(model, cache, config, {'cache_sha256': 'cache'}, order,
                          'full_fast_only', tmp_path, scope, device='cpu')
    before = torch.load(tmp_path/'before_final.pt', weights_only=False)
    final = torch.load(tmp_path/'final.pt', weights_only=False)
    assert before['sampler']['draw_index'] == 12
    assert final['sampler']['draw_index'] == 16
    assert len(final['reference_predictions']) == 4
    assert result['optimizer_updates'] == 4 and result['windows_seen'] == 16
    assert result['val_used_for_selection'] is False
    assert set(cache.calls) <= {r['token'] for r in cohort}
    train = [__import__('json').loads(s) for s in (tmp_path/'train.jsonl').read_text().splitlines()]
    assert len(train) == 4
    assert [r['draw_index'] for r in train] == [4, 8, 12, 16]
    assert all(len(r['micro_losses']) == 4 for r in train)
    assert all(len(r['gradient_probes']['blocks']) == 24 for r in train)
    assert all(r['base_weight_deltas'] for r in train)
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        fresh.head.weight.add_(10)
    verification = f.verify_or_replay(fresh, cache, config, {'cache_sha256': 'cache'}, order,
                                      'full_fast_only', tmp_path, device='cpu')
    assert verification['passed'] and verification['samples'] == 2
    assert verification['reference_prediction_check']['passed']
    replay = f.verify_or_replay(fresh, cache, config, {'cache_sha256': 'cache'}, order,
                               'full_fast_only', tmp_path, replay=True, device='cpu')
    assert replay['passed'] and replay['micro_batches'] == 4
    assert all(check['passed'] for check in replay['checks'].values())
