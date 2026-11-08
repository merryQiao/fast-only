import copy
import importlib
from types import SimpleNamespace

import pytest
import torch


def api():
    return importlib.import_module('tools.trace_direct_kv')


def test_intervention_selection_is_grouped_balanced_and_label_independent():
    rows = [dict(token=f'{g}-{i:02}', route=f'{g}_00001_00100', command=i % 2,
                 ego=[0. if i < 8 else 12., 0., 0., 0.], split='val')
            for g in ['one', 'two'] for i in range(16)]
    chosen = api().select_interventions(rows, per_group=4)
    assert len(chosen) == 8
    for group in ['one', 'two']:
        part = [r for r in chosen if r['token'].startswith(group)]
        assert {(r['command'], r['ego'][0]) for r in part} == {
            (0, 0.), (1, 0.), (0, 12.), (1, 12.)}
    changed = copy.deepcopy(rows)
    for r in changed:
        r['trajectory_gt'] = [[-999., 999., 5.]] * 8
        r['model_error'] = 100.
    assert [r['token'] for r in chosen] == [
        r['token'] for r in api().select_interventions(list(reversed(changed)), per_group=4)]


def test_selector_refuses_silent_shortening_or_train_rows():
    row = dict(token='x', route='a', command=0, ego=[0., 0., 0., 0.], split='val')
    with pytest.raises(ValueError):
        api().select_interventions([row], per_group=8)
    with pytest.raises(ValueError):
        api().select_interventions([dict(row, split='train')], per_group=1)
    with pytest.raises(ValueError):
        api().select_interventions([row, row], per_group=1)


def test_comparison_reports_reference_scale_and_common_mode_cancellation():
    reference = torch.tensor([[[1., -1.], [2., -2.]]])
    shifted = reference + 2.
    result = api().tensor_comparison(reference, shifted)
    assert result['rms_difference'] == pytest.approx(2.)
    assert result['reference_rms'] == pytest.approx(2.5 ** .5)
    assert result['relative_l2'] == pytest.approx((4. / 2.5) ** .5)
    assert result['equal_fraction'] == 0.
    normalized_a = torch.nn.functional.layer_norm(reference, (2,))
    normalized_b = torch.nn.functional.layer_norm(shifted, (2,))
    assert api().tensor_comparison(normalized_a, normalized_b)['rms_difference'] == 0.


def test_group_summary_does_not_weight_long_logs_as_independent_replicates():
    values = [dict(route='a', value=1.), dict(route='a', value=1.),
              dict(route='b', value=5.)]
    result = api().grouped_distribution(values, lambda r:r['value'])
    assert result['mean'] == pytest.approx(7 / 3)
    assert result['equal_recording_mean'] == 3.
    assert result['recordings'] == 2
    assert result['median'] == 1.


def test_context_guard_checks_all_context_rows_and_cleans_up_after_failure():
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    model = SimpleNamespace(backbone=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))
    original = torch.zeros(1, 12, 3)
    with api().context_trace(model) as (reference, checks):
        value = original
        for layer in layers:
            value = layer(value)
    assert checks == {1:True, 2:True}
    # Changes to query rows are allowed, including all final eight positions.
    allowed = original.clone()
    allowed[:, -8:] = 2.
    with api().context_trace(model, reference) as (_, checks):
        for layer in layers:
            allowed = layer(allowed)
    assert all(checks.values())
    # The last Context row must not be overlooked.
    forbidden = original.clone()
    forbidden[:, -9] = 1.
    with pytest.raises(ValueError, match='Context'):
        with api().context_trace(model, reference):
            layers[0](forbidden)
    assert all(not layer._forward_hooks for layer in layers)
