"""Create route-group splits, contiguous windows and TRAIN normalization."""
import argparse
import json
from pathlib import Path
import re

import yaml

import numpy as np
from fast_only.data import collect_windows, discover_routes, geographic_group, normalization_stats, split_routes
from fast_only.navigation import load_source_audit


def build(data_root, output_dir, allow_frame_index_timing=False, smoke_train_only=False, max_train_windows=None,
          seed=42, train_stride=1, eval_stride=5, nav_mode='annotation_near_route_target', geographic_groups=None, timing_policy='measured_timestamp', source_audit=None, require_official_label_pose=False, group_splits=None, geographic_group_policy='route_identity', val_towns=None, test_towns=None, navigation_origin='ego_center_from_gnss_extrinsic', gnss_mount_xy_m=(-1.4,0.)):
    if allow_frame_index_timing and not smoke_train_only:
        raise ValueError('--allow-frame-index-timing requires --smoke-train-only; formal manifests require actual timestamps')
    if nav_mode == 'zero' and not smoke_train_only:
        raise ValueError('Formal manifests require real navigation; zero is legacy smoke-only')
    evidence = load_source_audit(source_audit) if isinstance(source_audit, (str, Path)) else source_audit
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    routes = discover_routes(data_root)
    names = [p.name for p in routes]
    # Decide geographic holdouts BEFORE any sliding windows exist.
    if geographic_group_policy not in ('route_identity', 'town_holdout'):
        raise ValueError('Invalid geographic_group_policy')
    if geographic_group_policy == 'town_holdout':
        if group_splits is not None or geographic_groups is not None:
            raise ValueError('town_holdout must not be combined with custom group_splits/geographic_groups')
        val_towns, test_towns = set(val_towns or []), set(test_towns or [])
        if val_towns & test_towns:
            raise ValueError('VAL and TEST town assignments overlap')
        geographic_groups = {}
        for name in names:
            match = re.search(r'Town\d+(?:HD)?', name)
            if not match:
                raise ValueError(f'town_holdout cannot identify Town for route: {name}')
            geographic_groups[name] = match.group()
        group_splits = {town: 'val' if town in val_towns else 'test' if town in test_towns else 'train'
                        for town in set(geographic_groups.values())}

    if group_splits is not None:
        groups = {n: (geographic_groups or {}).get(n, geographic_group(n)) for n in names}
        if set(groups.values())-set(group_splits) or set(group_splits.values())-{'train','val','test'}:
            raise ValueError('group_splits must assign every geographic group to train/val/test')
        splits = {n: group_splits[g] for n,g in groups.items()}
    else:
        splits = split_routes(names, seed=seed, smoke_train_only=smoke_train_only, geographic_groups=geographic_groups)
    split_report = dict(seed=seed, routes=splits, group_splits=group_splits,
                        geographic_group_policy=geographic_group_policy,
                        val_towns=sorted(val_towns or []), test_towns=sorted(test_towns or []),
                        assignment_method='explicit_town_holdout_other_towns_train' if geographic_group_policy == 'town_holdout' else 'explicit_group_splits' if group_splits is not None else 'seeded_geographic_group_shuffle',
                        geographic_groups={n: (geographic_groups or {}).get(n, geographic_group(n)) for n in names},
                        smoke_train_only=smoke_train_only,
                        limitation=None if geographic_group_policy == 'town_holdout' else 'Town+Route groups join weather/date duplicates; geographic overlap across distinct Route IDs is not provably excluded without an explicit group map.')
    (output / 'splits.json').write_text(json.dumps(split_report, indent=2))
    rows, report = collect_windows(data_root, allow_frame_index_timing=allow_frame_index_timing,
                                  route_splits=splits, train_stride=train_stride, eval_stride=eval_stride, nav_mode=nav_mode,
                                  geographic_groups=geographic_groups, timing_policy=timing_policy, source_audit=evidence,
                                  require_official_label_pose=require_official_label_pose,
                                  navigation_origin=navigation_origin, gnss_mount_xy_m=gnss_mount_xy_m)
    report['available_windows_before_cap'] = len(rows)
    if max_train_windows is not None:
        if max_train_windows < 1:
            raise ValueError('max_train_windows must be positive')
        training = [r for r in rows if r['split'] == 'train']
        selected = np.linspace(0, len(training)-1, min(max_train_windows, len(training)), dtype=int)
        rows = [training[int(i)] for i in selected] + [r for r in rows if r['split'] != 'train']
    for split in ('train', 'val', 'test'):
        cohort = [r for r in rows if r['split'] == split]
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in cohort))
    report['saved_windows'] = {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}
    report['nav_mode'] = nav_mode
    report['allow_frame_index_timing'] = allow_frame_index_timing
    report['split_seed'] = seed
    report['geographic_group_policy'] = geographic_group_policy
    report['geographic_groups'] = split_report['geographic_groups']
    report['val_towns'] = split_report['val_towns']
    report['test_towns'] = split_report['test_towns']
    report['group_splits'] = group_splits
    report['source_audit_sha256'] = (evidence or {}).get('audit_sha256')
    report['split_assigned_before_window_construction'] = True
    report['test_usage'] = 'schema_geometry_timing_audit_only; no_test_metric_tuning'
    report['require_official_label_pose'] = require_official_label_pose
    provenance = Path(data_root) / 'download_provenance.json'
    if provenance.exists():
        report['download_provenance'] = json.loads(provenance.read_text())
        report['download_provenance'].pop('files', None)
    report['per_download_provenance'] = []
    for prov in sorted(Path(data_root).rglob('download_provenance.json')):
        value = json.loads(prov.read_text())
        value.pop('files', None); value.pop('file_sha256', None)
        report['per_download_provenance'].append(value)
    (output / 'build_report.json').write_text(json.dumps(report, indent=2))
    if not any(r['split'] == 'train' for r in rows):
        raise ValueError('No valid TRAIN windows: inspect build_report.json; missing timestamps require explicit --allow-frame-index-timing for smoke only')
    stats = normalization_stats(rows)
    stats.update(manifest=str((output / 'train.jsonl').resolve()), nav_mode=nav_mode,
                 ego_dimensions=['speed_m_s', 'backward_acceleration_m_s2', 'backward_yaw_rate_left_rad_s'])
    (output / 'stats.json').write_text(json.dumps(stats, indent=2))
    return report


def build_from_config(config_path):
    """Use the same data policy as training; command-line defaults cannot mask it."""
    config = yaml.safe_load(Path(config_path).read_text())
    data = config['data']
    if config.get('schema_version') == 2:
        from fast_only.configuration import validate_data_policy
        validate_data_policy(data)
    groups = data.get('geographic_groups')
    if isinstance(groups, str):
        groups = json.loads(Path(groups).read_text())
    return build(data_root=data['root'], output_dir=data['manifest_dir'],
        allow_frame_index_timing=data.get('allow_frame_index_timing', False),
        smoke_train_only=data.get('smoke_train_only', False),
        max_train_windows=data.get('max_train_windows'), seed=data.get('split_seed', 42),
        train_stride=data.get('train_stride', 1), eval_stride=data.get('eval_stride', 5),
        nav_mode=data.get('navigation_mode', 'annotation_near_route_target'),
        geographic_groups=groups, group_splits=data.get('group_splits'),
        geographic_group_policy=data.get('geographic_group_policy', 'route_identity'),
        val_towns=data.get('val_towns'), test_towns=data.get('test_towns'),
        timing_policy=data.get('timing_policy', 'measured_timestamp'),
        source_audit=data.get('source_audit'),
        navigation_origin=data.get('navigation_origin', 'ego_center_from_gnss_extrinsic'),
        gnss_mount_xy_m=data.get('gnss_mount_xy_m', [-1.4,0]),
        require_official_label_pose=data.get('require_official_label_pose', not data.get('smoke_train_only', False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='Training YAML; data policies are read from its data mapping')
    p.add_argument('--data-root', default='data/smoke')
    p.add_argument('--output-dir', default='outputs/fast_only/data')
    p.add_argument('--allow-frame-index-timing', action='store_true')
    p.add_argument('--smoke-train-only', action='store_true')
    p.add_argument('--max-train-windows', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-stride', type=int, default=1)
    p.add_argument('--eval-stride', type=int, default=5)
    p.add_argument('--nav-mode', choices=['zero', 'route_command', 'annotation_near_route_target'], default='annotation_near_route_target')
    p.add_argument('--timing-policy', choices=['measured_timestamp', 'nominal_simulation_time_10hz'], default='measured_timestamp')
    p.add_argument('--source-audit')
    p.add_argument('--navigation-origin', default='ego_center_from_gnss_extrinsic')
    p.add_argument('--gnss-mount-xy-m', nargs=2, type=float, default=[-1.4,0])
    p.add_argument('--geographic-group-policy', choices=['route_identity','town_holdout'], default='route_identity')
    p.add_argument('--val-towns', nargs='*')
    p.add_argument('--test-towns', nargs='*')
    p.add_argument('--require-official-label-pose', action='store_true')
    p.add_argument('--geographic-groups', help='JSON mapping route directory name to shared geographic identity')
    a = vars(p.parse_args())
    config = a.pop('config')
    if config:
        print(json.dumps(build_from_config(config), indent=2))
        return
    if a['geographic_groups']:
        a['geographic_groups'] = json.loads(Path(a['geographic_groups']).read_text())
    report = build(**a)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
