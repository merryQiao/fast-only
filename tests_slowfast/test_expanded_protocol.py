import copy
from collections import Counter
import json
from pathlib import Path

import pytest
import torch
import yaml


def config():
    return yaml.safe_load(Path('configs/slowfast_v11_expanded.yaml').read_text())


def rows(count=256, split='train', cohort='train', logs=4):
    return [dict(token=f'{cohort}-{i:03}', split=split, cohort=cohort,
                 route=f'{cohort}-log-{i % logs}', timestamp_us=i*2_000_000,
                 ego=[float([0, 5, 15][i % 3]), 0., 0., 0.]) for i in range(count)]


def test_fixed_expanded_config_and_reject_policy_drift():
    from slowfast_v11.expanded_protocol import validate_expanded_config

    c = config()
    validate_expanded_config(c)
    changes = [('stages', 'fast_consumer', 768), ('optimizer', 'schedule', 'constant'),
               ('model', 'lora_rank', 8), ('data', 'max_train_rows', 12)]
    for section, key, value in changes:
        bad = copy.deepcopy(c); bad[section][key] = value
        with pytest.raises(ValueError): validate_expanded_config(bad)
    for key, value in [('train_seed', True), ('split_seed', 43),
                       ('initial_checkpoint', 'old-mini.pt'), ('unknown', None)]:
        bad = copy.deepcopy(c); bad[key] = value
        with pytest.raises(ValueError): validate_expanded_config(bad)


def test_epoch_orders_uniform_train_only_and_input_order_independent():
    from slowfast_v11.expanded_protocol import epoch_plan

    r = rows()
    order = epoch_plan(r, 800, 42)
    assert order == epoch_plan(list(reversed(r)), 800, 42)
    expected = {x['token'] for x in r}
    for start in [0, 256, 512]: assert set(order[start:start+256]) == expected
    assert len(set(order[768:])) == 32
    assert max(Counter(order).values()) == 4
    assert order != epoch_plan(r, 800, 43)
    with pytest.raises(ValueError): epoch_plan(r + rows(1, 'val', 'new_val'), 800, 42)
    with pytest.raises(ValueError): epoch_plan([r[0], r[0]], 800, 42)
    with pytest.raises(ValueError): epoch_plan([], 800, 42)


def test_expanded_modes_are_independent_and_exact_with_fixed_probes():
    from slowfast_v11.expanded_protocol import expanded_plan, epoch_plan
    from slowfast_v11.joint_protocol import fresh_probe_steps

    plan = expanded_plan(rows(), [], config())
    assert [x['target_token'] for x in plan] == epoch_plan(rows(), 800, 42)
    assert Counter(x['mode'] for x in plan) == {'none': 240, 'fresh': 560}
    assert all(x['split'] == 'train' and x['source_token'] == x['target_token']
               and x['age_s'] == 0 for x in plan)
    assert len(set(fresh_probe_steps(plan))) == 4
    assert all(plan[i-1]['mode'] == 'fresh' for i in fresh_probe_steps(plan))
    assert plan == expanded_plan(list(reversed(rows())), [object()], config())


def test_polynomial_donors_are_bijective_and_split_cohort_local():
    from slowfast_v11.expanded_protocol import scalable_mismatches

    r = rows() + rows(64, 'val', 'new_val', 2) + rows(4, 'val', 'legacy_val', 1)
    mapping, report = scalable_mismatches(r)
    by_token = {x['token']: x for x in r}
    assert len(mapping) == len(r)
    assert len(set(mapping.values())) == len(r)
    assert mapping == scalable_mismatches(list(reversed(r)))[0]
    for target, donor in mapping.items():
        a, b = by_token[target], by_token[donor]
        assert target != donor and (a['split'], a['cohort']) == (b['split'], b['cohort'])
        if a['cohort'] != 'legacy_val': assert a['route'] != b['route']
        else: assert abs(a['timestamp_us']-b['timestamp_us']) >= 2_000_000
    assert report['val/legacy_val']['available']
    singleton = rows(1, 'val', 'legacy_val', 1)
    mapping, report = scalable_mismatches(singleton)
    assert not mapping and not report['val/legacy_val']['available']
    assert report['val/legacy_val']['count'] == 1


def test_donor_fallback_and_route_leakage_rejection():
    from slowfast_v11.expanded_protocol import scalable_mismatches

    r = rows(4, logs=1)
    mapping, report = scalable_mismatches(r)
    assert len(mapping) == 4 and report['train/train']['cross_log_count'] == 0
    bad = rows(1, 'val', 'new_val'); bad[0]['route'] = r[0]['route']
    with pytest.raises(ValueError): scalable_mismatches(r + bad)


def test_malformed_rows_rejected_explicitly():
    from slowfast_v11.expanded_protocol import epoch_plan, scalable_mismatches

    with pytest.raises(ValueError): epoch_plan([None], 4, 42)
    with pytest.raises(ValueError): scalable_mismatches([None])


def test_fixed_train_diagnostics_cover_logs_and_speeds_without_labels():
    from slowfast_v11.expanded_protocol import select_train_diagnostics

    r = rows()
    selected = select_train_diagnostics(r)
    assert len(selected) == len(set(selected)) == 32
    assert selected == select_train_diagnostics(list(reversed(r)))
    lookup = {x['token']: x for x in r}
    assert {lookup[t]['route'] for t in selected} == {x['route'] for x in r}
    bins = Counter(lookup[t]['ego'][0] for t in selected)
    assert max(bins.values()) - min(bins.values()) <= 1
    for item in r: item['trajectory_gt'] = object()
    assert select_train_diagnostics(r) == selected
    with pytest.raises(ValueError): select_train_diagnostics(r + rows(1, 'val', 'new_val'))
    with pytest.raises(ValueError): select_train_diagnostics(r[:10])


def test_cosine_endpoints_and_real_optimizer_resume_without_lr_aliasing():
    from slowfast_v11.expanded_protocol import WarmupCosineScheduler, lr_factor

    assert lr_factor(1, 512) == pytest.approx(1/32)
    assert lr_factor(32, 512) == 1.
    assert lr_factor(512, 512) == pytest.approx(.1)
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.AdamW([dict(params=[p], lr=1e-4)])
    schedule = WarmupCosineScheduler(opt, 512)
    for _ in range(47):
        opt.zero_grad(); p.square().sum().backward(); opt.step(); schedule.step()
    saved_p, saved_opt, saved_schedule = p.detach().clone(), copy.deepcopy(opt.state_dict()), schedule.state_dict()
    json.dumps(saved_schedule, allow_nan=False)
    opt.zero_grad(); p.square().sum().backward(); opt.step(); schedule.step()
    expected = p.detach().clone(), copy.deepcopy(opt.state_dict()), schedule.state_dict()
    q = torch.nn.Parameter(saved_p.clone())
    resumed = torch.optim.AdamW([dict(params=[q], lr=1e-4)])
    resumed_schedule = WarmupCosineScheduler(resumed, 512)
    resumed.load_state_dict(saved_opt); resumed_schedule.load_state_dict(saved_schedule)
    assert resumed.param_groups[0]['lr'] == saved_opt['param_groups'][0]['lr']
    resumed.zero_grad(); q.square().sum().backward(); resumed.step(); resumed_schedule.step()
    assert torch.equal(q, expected[0])
    assert resumed.param_groups[0]['lr'] == expected[1]['param_groups'][0]['lr']
    assert resumed_schedule.state_dict() == expected[2]
    assert saved_schedule['completed_steps'] == 47
    with pytest.raises(ValueError): WarmupCosineScheduler(resumed, 32)
