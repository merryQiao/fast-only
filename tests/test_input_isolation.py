"""Exercise the real annotation -> Dataset -> local AutoProcessor input boundary."""
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fast_only.common import NativeCollator
from fast_only.data import FastDataset, collect_windows, normalization_stats


MODEL_PATH = Path(__file__).resolve().parents[1] / 'models/Qwen3.5-0.8B'


def create_route(root, n=40):
    route = root / 'Straight_Town01_Route1_Weather1_fixture'
    (route / 'anno').mkdir(parents=True)
    (route / 'camera/rgb_front').mkdir(parents=True)
    for frame in range(n):
        raw = {'timestamp': frame / 10., 'x': frame*.2, 'y': 0.,
               'theta': math.pi/2, 'speed': 2., 'command_near': 4,
               'x_command_near': frame*.2 + 8., 'y_command_near': -2.,
               'throttle': .4, 'should_brake': False}
        write_annotation(route, frame, raw)
        Image.new('RGB', (64, 64), (frame, 30, 80)).save(route/'camera/rgb_front'/f'{frame:05d}.png')
    return route


def write_annotation(route, frame, raw):
    with gzip.open(route/'anno'/f'{frame:05d}.json.gz', 'wt') as stream:
        json.dump(raw, stream)


def read_annotation(route, frame):
    with gzip.open(route/'anno'/f'{frame:05d}.json.gz', 'rt') as stream:
        return json.load(stream)


def dataset_for_row(tmp_path, name, row, stats_path):
    manifest = tmp_path / f'{name}.jsonl'
    manifest.write_text(json.dumps(dict(row, split='train')) + '\n')
    return FastDataset(manifest, stats_path)


def explicit_inputs(batch):
    return {**{f'native.{key}': value for key, value in batch['native'].items()},
            **{key: batch[key] for key in ['ego', 'nav', 'command']}}


def test_future_annotations_and_images_cannot_change_actual_processor_model_inputs(tmp_path):
    route = create_route(tmp_path / 'data')
    original_rows, _ = collect_windows(route)
    original = original_rows[0]
    assert original['frame'] == 3
    assert [Path(p).stem for p in original['images']] == ['00000', '00001', '00002', '00003']
    assert all(int(Path(p).stem) <= original['frame'] for p in original['images'])
    fixed_train_stats = normalization_stats(original_rows)
    stats_path = tmp_path / 'fixed_train_stats.json'
    stats_path.write_text(json.dumps(fixed_train_stats))
    before_dataset = dataset_for_row(tmp_path, 'before', original, stats_path)
    # Use the installed local AutoProcessor, including chat template, image patching,
    # masks and grid generation. A fake collator would not test this boundary.
    collator = NativeCollator(str(MODEL_PATH), policy='original_pad', brightness=.1)
    before = collator([before_dataset[0]], seed=42)
    future = [read_annotation(route, frame) for frame in range(4, 40)]
    for frame, shuffled in zip(range(4, 40), reversed(future), strict=True):
        changed = dict(shuffled, timestamp=frame/10., x=500. + shuffled['x']*2,
                       y=25., speed=60., theta=1., command_near=1,
                       x_command_near=-900., y_command_near=300.,
                       throttle=-100., should_brake=True)
        write_annotation(route, frame, changed)
        Image.new('RGB', (96, 64), (255, frame, 0)).save(route/'camera/rgb_front'/f'{frame:05d}.png')
    changed_rows, _ = collect_windows(route)
    changed = next(row for row in changed_rows if row['frame'] == 3)
    after_dataset = dataset_for_row(tmp_path, 'after', changed, stats_path)
    after = collator([after_dataset[0]], seed=42)
    assert json.loads(stats_path.read_text()) == fixed_train_stats
    assert changed['images'] == original['images']
    assert not torch.equal(before['trajectory_gt'], after['trajectory_gt'])
    old_inputs, new_inputs = explicit_inputs(before), explicit_inputs(after)
    assert old_inputs.keys() == new_inputs.keys()
    assert {'native.input_ids', 'native.pixel_values', 'native.image_grid_thw',
            'native.attention_mask', 'ego', 'nav', 'command'} <= old_inputs.keys()
    for key, old in old_inputs.items():
        new = new_inputs[key]
        assert old.dtype == new.dtype and old.shape == new.shape, key
        assert old.contiguous().numpy().tobytes() == new.contiguous().numpy().tobytes(), key


def test_ego_acceleration_is_backward_speed_difference_and_future_cannot_change_it(tmp_path):
    route = create_route(tmp_path)
    previous = read_annotation(route, 2); previous['speed'] = 3.
    current = read_annotation(route, 3); current['speed'] = 3.7
    write_annotation(route, 2, previous); write_annotation(route, 3, current)
    for frame in range(4, 40):
        raw = read_annotation(route, frame); raw['speed'] = 999.
        raw['acceleration'] = [999., 999., 999.]
        write_annotation(route, frame, raw)
    row = collect_windows(route)[0][0]
    np.testing.assert_allclose(row['ego'], [3.7, 7., 0.], atol=1e-10)


@pytest.mark.parametrize('previous_yaw,current_yaw', [(math.pi-.01, -math.pi+.02),
                                                      (-math.pi+.01, math.pi-.02)])
def test_ego_yaw_rate_wraps_compass_boundary_and_uses_left_positive(tmp_path, previous_yaw, current_yaw):
    route = create_route(tmp_path)
    previous = read_annotation(route, 2); previous['theta'] = previous_yaw + math.pi/2
    current = read_annotation(route, 3); current['theta'] = current_yaw + math.pi/2
    write_annotation(route, 2, previous); write_annotation(route, 3, current)
    dyaw = math.atan2(math.sin(current_yaw-previous_yaw), math.cos(current_yaw-previous_yaw))
    row = collect_windows(route)[0][0]
    assert row['ego'][2] == pytest.approx(-dyaw/.1, abs=1e-10)
    assert abs(row['ego'][2]) == pytest.approx(.3)
