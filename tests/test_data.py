import gzip
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def api():
    from fast_only import data
    return data


def route_fixture(root, name='Scene_Town01_Route1_Weather1_demo', n=50, timestamps=True):
    route = root / name
    (route / 'anno').mkdir(parents=True)
    (route / 'camera/rgb_front').mkdir(parents=True)
    for frame in range(n):
        raw = {'x': frame * 0.5, 'y': 0., 'theta': math.pi / 2,
               'speed': 5. + frame * .1, 'command_near': 4,
               'x_command_near': frame * .5 + 10., 'y_command_near': -2.,
               'throttle': 99, 'should_brake': 99,
               'acceleration': [999, 999, 999], 'angular_velocity': [999, 999, 999]}
        if timestamps:
            raw['timestamp'] = frame / 10.
        with gzip.open(route / 'anno' / f'{frame:05}.json.gz', 'wt') as f:
            json.dump(raw, f)
        Image.new('RGB', (8, 6), (frame, 10, 20)).save(route / 'camera/rgb_front' / f'{frame:05}.jpg')
    return route


def test_world_to_ego_carla_forward_and_left():
    transform = api().world_to_ego
    np.testing.assert_allclose(transform([[12, 3], [10, 1], [10, 5]], [10, 3], 0),
                               [[2, 0], [0, 2], [0, -2]])
    np.testing.assert_allclose(transform([[10, 5], [12, 3]], [10, 3], math.pi / 2),
                               [[2, 0], [0, 2]], atol=1e-7)
    np.testing.assert_allclose(transform([[10, 1], [8, 3]], [10, 3], -math.pi / 2),
                               [[2, 0], [0, 2]], atol=1e-7)
    with pytest.raises(ValueError):
        transform([[float('nan'), 1]], [0, 0], 0)


def test_commands_have_seven_safe_embedding_indices():
    encode = api().encode_command
    assert [encode(v) for v in [-1, 1, 2, 3, 4, 5, 6, None]] == [0, 1, 2, 3, 4, 5, 6, 0]
    with pytest.raises(ValueError):
        encode(7)


def test_windows_are_causal_and_gt_is_future_only(tmp_path):
    route_fixture(tmp_path)
    rows, report = api().collect_windows(tmp_path, allow_frame_index_timing=False, nav_mode='route_command')
    assert len(rows) == 17
    row = rows[0]
    assert row['frame'] == 3 and row['timestamp'] == .3
    assert [Path(x).stem for x in row['images']] == ['00000', '00001', '00002', '00003']
    np.testing.assert_allclose(row['ego'], [5.3, 1., 0], atol=1e-6)
    np.testing.assert_allclose(row['nav'], [10, 2], atol=1e-6)
    np.testing.assert_allclose(row['trajectory_gt'], [[2.5*i, 0] for i in range(1, 7)], atol=1e-6)
    assert 'throttle' not in row and 'should_brake' not in row
    # Modifying future pose/speed/controls must affect only the label.
    p = next(tmp_path.glob('*/anno/00008.json.gz'))
    with gzip.open(p, 'rt') as f:
        raw = json.load(f)
    raw.update(x=999, speed=999, theta=2, should_brake=-999)
    with gzip.open(p, 'wt') as f:
        json.dump(raw, f)
    after = api().collect_windows(tmp_path, nav_mode='route_command')[0][0]
    for key in ['ego', 'nav', 'command', 'images']:
        assert after[key] == row[key]
    assert after['trajectory_gt'] != row['trajectory_gt']


def test_missing_frames_and_irregular_timing_are_rejected(tmp_path):
    route = route_fixture(tmp_path)
    (route / 'anno/00025.json.gz').unlink()
    rows, report = api().collect_windows(tmp_path)
    assert rows == [] and report['rejected']['missing_frame'] > 0


def test_timing_fallback_is_explicit(tmp_path):
    route_fixture(tmp_path, timestamps=False)
    rows, report = api().collect_windows(tmp_path)
    assert rows == [] and report['rejected']['missing_timestamp'] > 0
    rows, report = api().collect_windows(tmp_path, allow_frame_index_timing=True)
    assert len(rows) == 17
    assert rows[0]['timing_source'] == 'frame_index_10hz_explicit_fallback'


def test_timestamp_gap_rejects_whole_required_window(tmp_path):
    route = route_fixture(tmp_path)
    p = route / 'anno/00025.json.gz'
    with gzip.open(p, 'rt') as f:
        raw = json.load(f)
    raw['timestamp'] += .02
    with gzip.open(p, 'wt') as f:
        json.dump(raw, f)
    rows, report = api().collect_windows(tmp_path)
    assert rows == [] and report['rejected']['irregular_timing'] > 0


def test_weather_repetitions_share_route_group_split():
    names = [f'Scene_Town01_Route{i}_Weather{w}_demo' for i in range(12) for w in [1, 2]]
    split = api().split_routes(names, seed=42)
    assert set(split.values()) == {'train', 'val', 'test'}
    for i in range(12):
        assert split[names[2*i]] == split[names[2*i+1]]
    assert api().split_routes(list(reversed(names)), seed=42) == split
    assert api().split_routes(names[:2], smoke_train_only=True) == dict.fromkeys(names[:2], 'train')


def test_stats_use_train_only_and_dataset_contract(tmp_path):
    import torch
    route_fixture(tmp_path / 'data')
    rows, _ = api().collect_windows(tmp_path / 'data')
    for row in rows:
        row['split'] = 'train'
    train = rows[:2]
    val = dict(rows[2], split='val', ego=[10000, 10000, 10000], nav=[10000, 10000])
    stats = api().normalization_stats(train + [val])
    np.testing.assert_allclose(stats['ego_mean'], np.mean([x['ego'] for x in train], axis=0))
    manifest = tmp_path / 'train.jsonl'
    manifest.write_text('\n'.join(json.dumps(x) for x in train))
    statfile = tmp_path / 'stats.json'
    statfile.write_text(json.dumps(stats))
    ds = api().FastDataset(manifest, statfile)
    sample = ds[0]
    assert len(sample['images']) == 4 and sample['images'][0].mode == 'RGB'
    assert sample['ego'].dtype == torch.float32 and sample['ego'].shape == (3,)
    assert sample['nav'].shape == (2,) and sample['command'].dtype == torch.long
    assert sample['trajectory_gt'].shape == (6, 2)
    assert sample['metadata']['frame'] == 3


def make_tar():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name, content in [('route/anno/00000.json.gz', b'anno'),
                              ('route/camera/rgb_front/00000.jpg', b'rgb'),
                              ('route/camera/rgb_back/00000.jpg', b'forbidden'),
                              ('route/anno/00001.json.gz', b'skip')]:
            member = tarfile.TarInfo(name); member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return buf.getvalue()


def test_bounded_stream_extracts_only_requested_front_and_annotations(tmp_path):
    from tools.download_smoke import extract_subset, LimitedReader
    payload = make_tar()
    reader = LimitedReader(io.BytesIO(payload), len(payload))
    result = extract_subset(reader, tmp_path, start_frame=0, num_frames=1)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()) == [
        'route/anno/00000.json.gz', 'route/camera/rgb_front/00000.jpg']
    assert result['extracted_files'] == 2 and reader.bytes_read <= len(payload)
    with pytest.raises(RuntimeError, match='budget'):
        extract_subset(LimitedReader(io.BytesIO(payload), 20), tmp_path / 'limited', 0, 1)


def test_supervision_uses_true_ego_pose_but_inputs_use_only_sensors(tmp_path):
    route = route_fixture(tmp_path)
    for frame, p in enumerate(sorted((route / 'anno').glob('*'))):
        with gzip.open(p, 'rt') as f:
            raw = json.load(f)
        raw['bounding_boxes'] = [{'class': 'ego_vehicle', 'location': [frame, 0., 0.], 'rotation': [0., 0., 0.]}]
        # Simulate noisy GPS; this affects sensor navigation, never the label origin.
        raw['x'] += math.sin(frame)
        with gzip.open(p, 'wt') as f:
            json.dump(raw, f)
    rows, report = api().collect_windows(tmp_path, nav_mode='route_command')
    np.testing.assert_allclose(rows[0]['trajectory_gt'], [[5*i, 0.] for i in range(1, 7)])
    assert rows[0]['label_pose_source'] == 'official_ego_vehicle_pose_supervision_only'
    assert rows[0]['ego'] == pytest.approx([5.3, 1, 0])
    assert report['pose_validation']['checked_frames'] == 50


def test_legacy_navigation_zero_remains_explicit_opt_in(tmp_path):
    route_fixture(tmp_path)
    row = api().collect_windows(tmp_path, nav_mode='zero')[0][0]
    assert row['nav'] == [0., 0.]
    assert row['nav_source'] == 'explicit_zero_target_fallback'


def test_dataset_rejects_validation_statistics_and_nonfinite_means(tmp_path):
    manifest = tmp_path / 'empty.jsonl'; manifest.write_text('')
    p = tmp_path / 'stats.json'
    stats = {'split': 'val', 'ego_mean': [0., 0., 0.], 'ego_std': [1., 1., 1.], 'nav_mean': [0., 0.], 'nav_std': [1., 1.]}
    p.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='TRAIN'):
        api().FastDataset(manifest, p)
    stats['split'] = 'train'; stats['ego_mean'][0] = float('inf')
    p.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match='normalization'):
        api().FastDataset(manifest, p)


def test_manifest_builder_forbids_fallback_outside_train_only_smoke(tmp_path):
    from tools.build_fast_manifest import build
    with pytest.raises(ValueError, match='smoke-train-only'):
        build(tmp_path / 'unused', tmp_path / 'out', allow_frame_index_timing=True)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('custom_group', [None, 'shared_geography_A'])
def test_manifest_records_actual_geographic_split_group(tmp_path, custom_group):
    from tools.build_fast_manifest import build
    route = route_fixture(tmp_path / 'data')
    group_map = {route.name: custom_group} if custom_group else None
    build(tmp_path / 'data', tmp_path / 'out', smoke_train_only=True,
          geographic_groups=group_map, max_train_windows=2)
    rows = [json.loads(line) for line in (tmp_path / 'out/train.jsonl').read_text().splitlines()]
    expected = custom_group or 'Town01_Route1'
    assert {row['geographic_group'] for row in rows} == {expected}
    split_report = json.loads((tmp_path / 'out/splits.json').read_text())
    assert split_report['geographic_groups'][route.name] == expected


def test_downloader_pins_listing_and_archive_to_resolved_revision(tmp_path, monkeypatch):
    from tools import download_smoke as module
    revision = 'a'*40
    archive = 'route.tar.gz'
    payload = make_tar()
    metadata_requests, archive_requests = [], []

    def fetch(url, remaining):
        metadata_requests.append(url)
        if url == f'https://huggingface.co/api/datasets/{module.REPO}':
            result = json.dumps({'sha': revision}).encode()
        elif '/tree/' in url:
            result = json.dumps([{'path': archive, 'size': len(payload)}]).encode()
        elif url == module.SOURCE_URL:
            result = b'frame_rate=10.0'
        else:
            raise AssertionError(f'Unexpected metadata request {url}')
        assert remaining >= len(result)
        return result, len(result), ''

    class Response:
        raw = io.BytesIO(payload)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass

    def get(url, **kwargs):
        archive_requests.append(url)
        return Response()

    monkeypatch.setattr(module, 'fetch_small', fetch)
    monkeypatch.setattr(module.requests, 'get', get)
    report = module.download(tmp_path, num_frames=1)
    assert metadata_requests[0] == f'https://huggingface.co/api/datasets/{module.REPO}'
    assert f'/tree/{revision}?' in metadata_requests[1]
    assert archive_requests == [f'https://huggingface.co/datasets/{module.REPO}/resolve/{revision}/{archive}']
    assert report['dataset_revision'] == revision
    assert report['dataset_revision_pinned_at_download'] is True


def test_varied_scenarios_still_populate_all_route_splits():
    names = [f'{scene}_Town0{i+1}_Route{i}_Weather1_demo' for i, scene in enumerate(['Turn','Brake','Cross','Yield'])]
    split = api().split_routes(names, seed=42)
    assert sorted(split.values()) == ['test', 'train', 'train', 'val']


def test_default_navigation_uses_current_official_world_route_target(tmp_path):
    route_fixture(tmp_path)
    row = api().collect_windows(tmp_path)[0][0]
    assert row['nav'] == pytest.approx([8.6, 2])
    assert row['nav_source'] == 'official_annotation_near_route_target'
    assert row['navigation_metadata']['target_fields'] == ['x_command_near', 'y_command_near']
    assert row['navigation_metadata']['future_expert_positions_used'] is False


def test_navigation_missing_near_target_is_rejected_never_zero_or_x_target(tmp_path):
    route = route_fixture(tmp_path)
    for p in (route / 'anno').glob('*'):
        raw = api().read_annotation(p)
        raw.pop('x_command_near')
        raw.update(x_target=999, y_target=999)
        with gzip.open(p, 'wt') as f:
            json.dump(raw, f)
    rows, report = api().collect_windows(tmp_path)
    assert not rows
    assert report['rejected']['invalid_navigation'] == 17


def test_nominal_simulation_time_requires_explicit_source_evidence(tmp_path):
    route_fixture(tmp_path, timestamps=False)
    with pytest.raises(ValueError, match='source_audit'):
        api().collect_windows(tmp_path, timing_policy='nominal_simulation_time_10hz')


def test_nominal_clock_is_distinct_from_unverified_fallback(tmp_path):
    from fast_only.navigation import load_source_audit
    route_fixture(tmp_path, timestamps=False)
    evidence = load_source_audit('outputs/fast_only/audit_v2/source_audit.json')
    rows, report = api().collect_windows(tmp_path, timing_policy='nominal_simulation_time_10hz', source_audit=evidence)
    assert len(rows) == 17
    assert rows[0]['timestamp'] == .3
    assert rows[0]['timing_source'] == 'nominal_simulation_time_10hz'
    assert rows[0]['timing_metadata']['absolute_timestamp_available'] is False
    assert rows[0]['timing_metadata']['hz'] == 10


def test_downloader_hashes_exact_archive_stream(tmp_path):
    from tools.download_smoke import LimitedReader, extract_subset
    import hashlib
    payload = make_tar()
    reader = LimitedReader(io.BytesIO(payload), len(payload))
    extract_subset(reader, tmp_path, num_frames=1)
    assert reader.sha256 == hashlib.sha256(payload).hexdigest()


def test_audit_motion_is_future_only_diagnostic_metadata(tmp_path):
    route_fixture(tmp_path)
    row = api().collect_windows(tmp_path)[0][0]
    motion = row['audit_motion']
    assert motion['num_frames'] == 31
    assert motion['speed_min_mps'] == pytest.approx(5.3)
    assert motion['max_delta_speed_mps'] == pytest.approx(3.)
    assert motion['max_delta_yaw_rad'] == 0.
    assert 'audit_only' in motion['source']


def test_builder_config_selects_real_navigation_and_nominal_clock(tmp_path):
    from tools.build_fast_manifest import build_from_config
    import yaml
    route_fixture(tmp_path / 'data', timestamps=False)
    config = {'data': {'root': str(tmp_path / 'data'), 'manifest_dir': str(tmp_path / 'out'),
               'navigation_mode': 'annotation_near_route_target',
               'timing_policy': 'nominal_simulation_time_10hz',
               'source_audit': 'outputs/fast_only/audit_v2/source_audit.json',
               'smoke_train_only': True, 'split_seed': 123}}
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    report = build_from_config(path)
    assert report['nav_mode'] == 'annotation_near_route_target'
    assert report['split_seed'] == 123
    assert report['timing_sources'] == {'nominal_simulation_time_10hz': 50}
    assert report['saved_windows']['train'] == 17


def test_discovery_can_reuse_legacy_route_by_symlink_without_mutation(tmp_path):
    route = route_fixture(tmp_path / 'old')
    combined = tmp_path / 'combined'; combined.mkdir()
    (combined / route.name).symlink_to(route, target_is_directory=True)
    assert [r.name for r in api().discover_routes(combined)] == [route.name]


def test_inspector_reports_real_navigation_and_nominal_limitations(tmp_path):
    from tools.inspect_b2d import inspect
    route_fixture(tmp_path, timestamps=False)
    report = inspect(tmp_path, timing_policy='nominal_simulation_time_10hz',
                     source_audit='outputs/fast_only/audit_v2/source_audit.json')
    assert report['windows'] == 17
    assert report['source_audit']['navigation']['verified'] is True
    assert report['routes'][next(iter(report['routes']))]['image_dimensions'] == {'8x6': 50}


def test_explicit_geographic_group_holdout_is_assigned_before_windows(tmp_path):
    from tools.build_fast_manifest import build
    names = [f'{scene}_Town{town}_Route{i}_Weather1_demo' for i,(scene,town) in enumerate([('Turn','11'),('Junction','11'),('Green','12'),('Red','13'),('Hazard','10')])]
    for name in names:
        route_fixture(tmp_path/'data',name=name)
    groups = {name: name.split('_')[1] for name in names}
    assignments = {'Town11':'train','Town12':'val','Town13':'train','Town10':'test'}
    report = build(tmp_path/'data',tmp_path/'out', geographic_groups=groups, group_splits=assignments)
    assert report['saved_windows'] == {'train':51,'val':4,'test':4}
    split = json.loads((tmp_path/'out/splits.json').read_text())
    assert split['group_splits'] == assignments
    assert all(split['routes'][name] == assignments[group] for name,group in groups.items())
    with pytest.raises(ValueError, match='group_splits'):
        build(tmp_path/'data',tmp_path/'bad',geographic_groups=groups,group_splits={'Town11':'train'})


def test_town_holdout_keeps_towns_sealed_and_default_other_towns_train(tmp_path):
    from tools.build_fast_manifest import build
    names = [f'Scene{i}_Town{town}_Route{i}_Weather1_demo' for i,town in enumerate(['11','11','12','13','06'])]
    for name in names:
        route_fixture(tmp_path/'data',name=name)
    report = build(tmp_path/'data',tmp_path/'out',geographic_group_policy='town_holdout',
                   val_towns=['Town12'],test_towns=['Town06'])
    assert report['saved_windows'] == {'train':51,'val':4,'test':4}
    split = json.loads((tmp_path/'out/splits.json').read_text())
    assert split['geographic_group_policy'] == 'town_holdout'
    assert split['assignment_method'] == 'explicit_town_holdout_other_towns_train'
    assert set(split['geographic_groups'].values()) == {'Town11','Town12','Town13','Town06'}
    assert split['routes'][names[0]] == split['routes'][names[1]] == split['routes'][names[3]] == 'train'
    with pytest.raises(ValueError,match='overlap'):
        build(tmp_path/'data',tmp_path/'bad',geographic_group_policy='town_holdout',val_towns=['Town12'],test_towns=['Town12'])


def test_navigation_extrinsic_correction_rotates_with_sensor_heading(tmp_path):
    route = route_fixture(tmp_path)
    for p in (route/'anno').glob('*'):
        raw = api().read_annotation(p)
        raw.update(x=1.,y=2.,theta=math.pi,x_command_near=3.,y_command_near=12.)
        with gzip.open(p,'wt') as f:
            json.dump(raw,f)
    center = api().collect_windows(tmp_path, navigation_origin='ego_center_from_gnss_extrinsic',gnss_mount_xy_m=[-1.4,0])[0][0]
    legacy = api().collect_windows(tmp_path, nav_mode='route_command')[0][0]
    np.testing.assert_allclose(center['nav'],[8.6,2],atol=1e-8)
    np.testing.assert_allclose(legacy['nav'],[10,2],atol=1e-8)
    assert center['navigation_metadata']['origin'] == 'ego_center_from_gnss_extrinsic'
    assert center['navigation_metadata']['gnss_mount_xy_m'] == [-1.4,0]
    assert center['navigation_metadata']['privileged_ego_pose_used'] is False


def test_v2_builder_validates_policy_before_opening_data(tmp_path):
    from tools.build_fast_manifest import build_from_config
    import yaml
    route_fixture(tmp_path/'data')
    config={'schema_version':2,'data':{'root':str(tmp_path/'data'),'manifest_dir':str(tmp_path/'out'),
                                     'navigation_mode':'zero','smoke_train_only':True}}
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='data'):
        build_from_config(path)
    assert not (tmp_path/'out').exists()
