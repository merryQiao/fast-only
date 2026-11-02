import pytest
import torch

from fast_only import diagnostics


def sample(index, route='route_a', command=4, scene='Straight', split='val'):
    speed = float(index + 2)
    times = torch.arange(1, 7, dtype=torch.float32) * .5
    target = torch.stack((speed * times, torch.zeros(6)), dim=-1)
    return {'images': [torch.tensor(float(index + j)) for j in range(4)],
            'ego': torch.tensor([-100., 8., 9.]), 'nav': torch.tensor([index + 1., 7.]),
            'command': torch.tensor(command), 'trajectory_gt': target,
            'metadata': {'route': route, 'frame': index, 'scene': scene, 'split': split,
                         'ego': [speed, 0., 0.], 'command': command}}


def test_constant_velocity_uses_raw_speed_and_exact_six_horizons():
    dataset = [sample(0), sample(2)]
    metrics, pred, gt = diagnostics.evaluate_constant_velocity(dataset)
    assert torch.equal(pred[0, :, 0], torch.tensor([1., 2., 3., 4., 5., 6.]))
    assert torch.equal(pred, gt)
    assert metrics['ADE'] == metrics['FDE'] == 0.
    assert metrics['samples'] == 2
    assert [metrics[f'L2@{i*.5:g}s'] for i in range(1, 7)] == [0.] * 6
    assert diagnostics.constant_velocity_metrics(dataset) == metrics
    assert torch.equal(diagnostics.constant_velocity_predictions(dataset), pred)


def test_constant_velocity_reports_same_metric_contract_on_turning_examples():
    item = sample(0)
    item['trajectory_gt'] = item['trajectory_gt'] + torch.tensor([3., 4.])
    metrics = diagnostics.constant_velocity_metrics([item])
    assert metrics['ADE'] == 5. and metrics['FDE'] == 5.
    assert metrics['loss_l1_xy'] == 7.
    with pytest.raises(ValueError, match='empty'):
        diagnostics.constant_velocity_metrics([])


def anchor(*, delta_v=0., delta_yaw=0., pose_scale=1.):
    item = sample(0)
    item['metadata']['audit_motion'] = {'speed_min_mps': 2., 'max_delta_speed_mps': delta_v,
                                       'max_delta_yaw_rad': delta_yaw, 'num_frames': 31,
                                       'source': 'future_sensor_speed_compass_audit_only_never_model_input'}
    item['trajectory_gt'] *= pose_scale
    return [item]


def test_anchor_checks_centimetre_geometry_on_independent_straight_steady_evidence():
    result = diagnostics.audit_constant_velocity_anchor(anchor())
    assert result['status'] == 'passed' and result['selected_samples'] == 1
    assert result['max_point_error_m'] == 0.
    assert result['criteria']['max_delta_speed_mps'] == .1
    assert result['criteria']['max_delta_yaw_rad'] == .01
    assert result['criteria']['max_point_error_m'] == .02


def test_anchor_detects_bad_geometry_despite_independent_steady_evidence():
    result = diagnostics.audit_constant_velocity_anchor(anchor(pose_scale=2.))
    assert result['status'] == 'failed' and result['max_point_error_m'] == 6.


@pytest.mark.parametrize('kwargs', [dict(delta_v=1.), dict(delta_yaw=.1)])
def test_turning_or_accelerating_examples_never_pass_centimetre_anchor(kwargs):
    result = diagnostics.audit_constant_velocity_anchor(anchor(**kwargs))
    assert result['status'] == 'insufficient' and result['selected_samples'] == 0
    assert result['max_point_error_m'] is None


def test_current_only_steady_motion_is_insufficient_without_future_audit_evidence():
    result = diagnostics.audit_constant_velocity_anchor([sample(0)])
    assert result['status'] == 'insufficient' and result['selected_samples'] == 0
    assert result['excluded']['missing_motion_evidence'] == 1


class ImagePlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, native, ego, command, nav):
        self.seen.append((native, ego.clone(), command.clone(), nav.clone()))
        x = native['pixel_values'].sum() + nav[:, 0]
        return torch.stack((x.expand(6), torch.zeros(6)), -1).unsqueeze(0)


def collate(samples, *, seed=0):
    assert len(samples) == 1
    item = samples[0]
    return {'native': {'pixel_values': torch.stack(item['images'])},
            **{k: item[k].unsqueeze(0) for k in ['ego', 'command', 'nav', 'trajectory_gt']}}


def test_image_swap_changes_all_four_images_and_preserves_all_nonimage_inputs():
    dataset = [sample(i, route=f'route_{i%2}') for i in range(4)]
    model = ImagePlanner(); model.train()
    report = diagnostics.evaluate_image_swap(model, dataset, collate, 'cpu')
    assert model.training and report['samples'] == 4 and report['cohort'] == 'val'
    assert report['cross_route_pairs'] == 4
    assert report['mean_point_change_m'] > 0
    assert len({pair['donor_index'] for pair in report['pairs']}) == 4
    for pair in report['pairs']:
        assert pair['index'] != pair['donor_index']
        assert pair['route'] != pair['donor_route']
        original, swapped = model.seen[pair['index']*2:pair['index']*2+2]
        assert torch.equal(swapped[0]['pixel_values'], torch.stack(dataset[pair['donor_index']]['images']))
        assert torch.equal(original[1], swapped[1])
        assert torch.equal(original[2], swapped[2])
        assert torch.equal(original[3], swapped[3])
    assert sorted(report['per_route']) == ['route_0', 'route_1']
    assert report == diagnostics.evaluate_image_swap(ImagePlanner(), dataset, collate, 'cpu')


def test_single_route_swap_is_honest_about_train_cohort_and_cross_route_limitation():
    dataset = [sample(i, split='train') for i in range(3)]
    report = diagnostics.evaluate_image_swap(ImagePlanner(), dataset, collate, 'cpu', max_samples=2)
    assert report['cohort'] == 'train' and report['samples'] == 2
    assert report['cross_route_pairs'] == 0 and report['different_route_available'] is False
    report = diagnostics.evaluate_image_swap(ImagePlanner(), dataset[:1], collate, 'cpu')
    assert report['status'] == 'insufficient' and report['mean_point_change_m'] is None


def test_stratified_visualizations_cover_scene_and_command_before_repeating():
    dataset = [sample(i, command=4, scene='Straight') for i in range(20)]
    dataset += [sample(20, command=1, scene='Turn'), sample(21, command=2, scene='Turn'),
                sample(22, command=4, scene='Junction')]
    indices = diagnostics.select_visualization_indices(dataset)
    assert len(indices) == len(set(indices)) == 10
    assert {20, 21, 22}.issubset(indices)
    assert diagnostics.select_visualization_indices(dataset) == indices
    assert diagnostics.select_visualization_indices(dataset[:3]) == [0, 1, 2]
    assert diagnostics.select_visualization_indices([]) == []


def test_capped_image_swap_includes_other_routes_when_available():
    dataset = [sample(i, route='route_a') for i in range(6)] + [sample(6, route='route_b')]
    report = diagnostics.evaluate_image_swap(ImagePlanner(), dataset, collate, 'cpu', max_samples=2)
    assert report['different_route_available'] is True
    assert report['cross_route_pairs'] == 2
    assert {pair['route'] for pair in report['pairs']} == {'route_a', 'route_b'}
