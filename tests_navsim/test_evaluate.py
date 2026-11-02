"""CPU-only tests of NAVSIM evaluation and descriptive channel interventions."""
from copy import deepcopy
from importlib import import_module
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch


def api():
    assert (Path(__file__).parents[1] / 'fast_navsim/evaluate.py').is_file(), 'NAVSIM evaluator is missing'
    return import_module('fast_navsim.evaluate')


class Dataset:
    def __init__(self, commands=(0, 1, 2, 3), logs=None, split='val'):
        self.rows = []
        for index, command in enumerate(commands):
            self.rows.append({'route': (logs or ['log_a'] * len(commands))[index],
                'scene': f'scene_{index // 2}', 'frame': index, 'token': f'token_{index}',
                'command': command, 'split': split, 'ego': [2., -1., 90., -90.],
                'trajectory_gt': [[i, -.5 * i, 0.] for i in range(1, 9)]})
        self.reads = []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        self.reads.append(index)
        row = self.rows[index]
        return {'images': [Image.new('RGB', (5, 3), (index * 20 + t, 0, 0)) for t in range(4)],
            # Deliberately normalized differently from raw ego in rows.
            'ego': torch.tensor([100., 200., 300., 400.]),
            'command': torch.tensor(row['command'], dtype=torch.long),
            'trajectory_gt': torch.tensor(row['trajectory_gt']), 'metadata': deepcopy(row)}


class Collator:
    def __init__(self):
        self.calls = []

    def __call__(self, samples, seed=None):
        self.calls.append({'pixels': [[im.getpixel((0, 0))[0] for im in s['images']] for s in samples],
            'ego': [s['ego'].clone() for s in samples],
            'commands': [int(s['command']) for s in samples], 'seed': seed})
        return {'native': {'frames': torch.tensor(self.calls[-1]['pixels'], dtype=torch.float32)},
            'nav': None, **{key: torch.stack([s[key] for s in samples])
                           for key in ['ego', 'command', 'trajectory_gt']}}


class Model(torch.nn.Module):
    def forward(self, native, ego, command, nav=None):
        assert nav is None and not self.training and not torch.is_grad_enabled()
        pred = torch.zeros(len(ego), 8, 3)
        pred[..., 0] = (native['frames'] * torch.arange(1, 5)).sum(-1)[:, None]
        pred[..., 1] = ego[:, :1]
        pred[..., 2] = torch.where(command == 0, math.pi - .01, -math.pi + .01)[:, None]
        return pred


def test_evaluate_dataset_returns_official_pose_metrics_and_restores_training_state():
    model = Model().train()
    ds = Dataset(commands=(0, 1))
    metrics, prediction, target = api().evaluate_dataset(model, ds, Collator(), 'cpu')
    assert model.training
    assert prediction.shape == target.shape == (2, 8, 3)
    assert metrics['samples'] == 2 and 'L2@4s' in metrics and 'heading_mae_rad' in metrics
    from fast_navsim.model import trajectory_metrics
    assert metrics == trajectory_metrics(prediction, target)
    assert prediction.device.type == 'cpu' and not prediction.requires_grad


def test_evaluation_restores_eval_mode_and_mode_after_failure():
    model = Model().eval()
    api().evaluate_dataset(model, Dataset(commands=(0,)), Collator(), 'cpu')
    assert not model.training
    model.train()
    def broken(*args, **kwargs):
        raise RuntimeError('broken input')
    with pytest.raises(RuntimeError, match='broken input'):
        api().evaluate_dataset(model, Dataset(), broken, 'cpu')
    assert model.training
    with pytest.raises(ValueError, match='empty'):
        api().evaluate_dataset(model, Dataset(commands=()), Collator(), 'cpu')


def test_constant_velocity_uses_raw_vx_vy_only_at_eight_half_second_horizons():
    ds = Dataset(commands=(0, 0))
    metrics = api().constant_velocity_metrics(ds)
    assert metrics['ADE'] == metrics['FDE'] == metrics['heading_mae_rad'] == 0
    assert metrics['samples'] == 2
    assert 'reference' in metrics['method'] and 'constant' in metrics['heading_assumption']
    assert 'PDM' in metrics['limitation']
    assert ds.reads == [], 'CV should use raw row metadata and never decode images'
    ds.rows[0]['ego'][2:] = [-10000., 10000.]
    assert api().constant_velocity_metrics(ds)['ADE'] == 0


def test_cv_reports_xy_and_heading_separately_and_rejects_nonfinite_velocity():
    ds = Dataset(commands=(0,))
    ds.rows[0]['trajectory_gt'] = [[i, -.5 * i, math.pi] for i in range(1, 9)]
    result = api().constant_velocity_metrics(ds)
    assert result['ADE'] == result['FDE'] == 0
    assert result['heading_mae_rad'] == pytest.approx(math.pi)
    ds.rows[0]['ego'][0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        api().constant_velocity_metrics(ds)


def test_visualization_indices_cover_scene_command_buckets_without_duplicate_padding():
    ds = Dataset(commands=(0, 0, 1, 1, 2), logs=['a', 'a', 'a', 'a', 'b'])
    assert api().select_visualization_indices(ds, max_samples=3) == [0, 2, 4]
    indices = api().select_visualization_indices(ds, max_samples=10)
    assert len(indices) == len(set(indices)) == 5
    assert api().select_visualization_indices(ds, max_samples=0) == []
    assert ds.reads == []


def test_prediction_artifacts_keep_eight_poses_identifiers_and_units(tmp_path):
    ds = Dataset(commands=(0, 1))
    target = torch.tensor([r['trajectory_gt'] for r in ds.rows])
    prediction = target.clone(); prediction[..., 2] = 1
    assert api().save_predictions(tmp_path, ds, prediction, target, cohort='val', max_plots=0) == 2
    payload = json.loads((tmp_path / 'predictions.json').read_text())
    assert payload[0]['scene'] == 'scene_0' and payload[0]['token'] == 'token_0'
    assert payload[1]['command'] == 1 and payload[0]['cohort'] == 'val'
    assert payload[0]['ADE'] == 0 and payload[0]['heading_mae_rad'] == pytest.approx(1)
    assert len(payload[0]['prediction']) == 8 and len(payload[0]['prediction'][0]) == 3
    arrays = np.load(tmp_path / 'predictions.npz', allow_pickle=False)
    assert arrays['prediction'].shape == arrays['trajectory_gt'].shape == (2, 8, 3)
    assert arrays['future_times_s'].tolist() == [i * .5 for i in range(1, 9)]
    csv_header = (tmp_path / 'metrics.csv').read_text().splitlines()[0]
    assert 'heading_mae_rad' in csv_header and 'L2@4s' in csv_header and 'token' in csv_header
    selection = json.loads((tmp_path / 'visualization_selection.json').read_text())
    assert selection['indices'] == [] and selection['available_cases'] == 2
    assert ds.reads == [], 'No plots means no image reads'


def test_plot_labels_four_correct_history_times_and_heading_in_separate_axes(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    captured = []
    close = plt.close
    def capture(fig=None):
        if hasattr(fig, 'axes'):
            captured.append(fig)
        close(fig)
    monkeypatch.setattr(plt, 'close', capture)
    ds = Dataset(commands=(0,))
    target = torch.tensor([ds.rows[0]['trajectory_gt']])
    api().save_predictions(tmp_path, ds, target, target, cohort='val', max_plots=10)
    assert len(list(tmp_path.glob('case_*.png'))) == 1
    assert len(captured) == 1
    axes = captured[0].axes
    assert [ax.get_title() for ax in axes[:4]] == ['t-1.5s', 't-1.0s', 't-0.5s', 't+0.0s']
    assert '(m)' in axes[4].get_xlabel() and '(m)' in axes[4].get_ylabel()
    assert '(rad)' in axes[5].get_ylabel() and '(s)' in axes[5].get_xlabel()


@pytest.mark.parametrize('split', ['train', 'test', 'unknown'])
def test_channel_interventions_are_validation_only_before_model_or_image_access(split):
    ds = Dataset(split=split)
    collator = Collator()
    with pytest.raises(ValueError, match='val'):
        api().channel_sensitivity(Model(), ds, collator, 'cpu')
    assert not ds.reads and not collator.calls


def test_validation_gate_checks_unsampled_rows_too():
    ds = Dataset()
    ds.rows[-1]['split'] = 'test'
    with pytest.raises(ValueError, match='val'):
        api().channel_sensitivity(Model(), ds, Collator(), 'cpu', max_samples=2)
    assert not ds.reads


def test_channel_interventions_swap_all_four_images_and_change_only_command():
    ds = Dataset(commands=(0, 0, 0, 0), logs=['a', 'a', 'a', 'b'])
    before = deepcopy(ds.rows)
    model = Model().train(); collator = Collator()
    report = api().channel_sensitivity(model, ds, collator, 'cpu', max_samples=4)
    assert model.training and ds.rows == before
    image = report['image_swap']; command = report['command_intervention']
    assert image['status'] == command['status'] == 'measured'
    assert image['samples'] == command['samples'] == 4
    assert image['logs'] == command['logs'] == 2
    assert image['same_log_pairs'] + image['cross_log_pairs'] == 4
    assert image['cross_log_pairs'] > 0
    assert 'four_images' in image['method'] and 'cycle' in image['method']
    assert command['method'] == 'command_plus_one_modulo_four'
    assert image['mean_point_change_m'] > 0 and image['mean_heading_change_rad'] == 0
    assert command['mean_point_change_m'] == 0
    assert command['mean_heading_change_rad'] == pytest.approx(.02, abs=1e-6)
    assert 'semantic' in report['limitation'] and 'PDM' in report['limitation']
    for pair in image['pairs']:
        assert pair['index'] != pair['donor_index']
    assert sorted(p['donor_index'] for p in image['pairs']) == list(range(4))
    # Calls are original / image intervention / command intervention per case.
    assert len(collator.calls) == 12
    for original, swapped, intervened in zip(collator.calls[::3], collator.calls[1::3], collator.calls[2::3]):
        assert original['commands'] == swapped['commands'] == [0]
        assert intervened['commands'] == [1]
        assert torch.equal(original['ego'][0], swapped['ego'][0])
        assert torch.equal(original['ego'][0], intervened['ego'][0])
        assert original['pixels'] == intervened['pixels'] and original['pixels'] != swapped['pixels']
        assert len(swapped['pixels'][0]) == 4
        assert original['seed'] == swapped['seed'] == intervened['seed']


def test_capped_sensitivity_covers_distinct_logs_before_additional_windows():
    ds = Dataset(commands=(0, 0, 0, 0), logs=['a', 'a', 'a', 'b'])
    report = api().channel_sensitivity(Model(), ds, Collator(), 'cpu', max_samples=2)
    assert report['image_swap']['samples'] == 2 and report['image_swap']['logs'] == 2
    assert report['image_swap']['cross_log_pairs'] == 2


def test_single_validation_case_has_insufficient_image_swap_but_command_is_measured():
    report = api().channel_sensitivity(Model(), Dataset(commands=(0,)), Collator(), 'cpu')
    assert report['image_swap']['status'] == 'insufficient'
    assert report['image_swap']['mean_point_change_m'] is None
    assert report['command_intervention']['status'] == 'measured'
    assert report['command_intervention']['samples'] == 1


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_sensitivity_rejects_invalid_sample_caps(limit):
    with pytest.raises(ValueError, match='max_samples'):
        api().channel_sensitivity(Model(), Dataset(), Collator(), 'cpu', max_samples=limit)


def test_interventions_restore_model_mode_after_failure():
    model = Model().train()
    def broken(*args, **kwargs):
        raise RuntimeError('broken input')
    with pytest.raises(RuntimeError, match='broken input'):
        api().channel_sensitivity(model, Dataset(), broken, 'cpu')
    assert model.training
