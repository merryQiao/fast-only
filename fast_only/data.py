"""Leakage-safe Bench2Drive v0.0.4 front-camera observations and future labels.

Official `theta` is IMU compass radians, hence CARLA yaw = theta - pi/2.
CARLA world XY is left handed (positive yaw turns right). Our output is x
forward, y left, so the CARLA local lateral coordinate is explicitly negated.
Only speed and current/past compass contribute to the ego input; privileged
bounding boxes can validate label geometry but never become model features.
"""
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from fast_only.navigation import NAVIGATION_METADATA, validate_nominal_clock

FUTURE_OFFSETS = (5, 10, 15, 20, 25, 30)
TIMING_FALLBACK = 'frame_index_10hz_explicit_fallback'
SOURCE_URL = 'https://github.com/Thinklab-SJTU/Bench2Drive/blob/0.0.4/tools/data_collect.py'


def world_to_ego(points, origin, yaw_radians):
    """CARLA world XY -> x-forward/y-left, with CARLA yaw in radians."""
    points, origin = np.asarray(points, dtype=np.float64), np.asarray(origin, dtype=np.float64)
    if points.shape[-1] != 2 or origin.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(origin).all() or not np.isfinite(yaw_radians):
        raise ValueError('Expected finite XY points, XY origin, and radian yaw')
    delta = points - origin
    c, s = math.cos(yaw_radians), math.sin(yaw_radians)
    return np.stack((c*delta[..., 0] + s*delta[..., 1], s*delta[..., 0] - c*delta[..., 1]), axis=-1)


def encode_command(command):
    """RoadOption VOID/unknown -> 0; LEFT..CHANGERIGHT keep official 1..6."""
    if command is None or command == -1:
        return 0
    value = int(command)
    if value != command or value not in range(7):
        raise ValueError(f'Invalid RoadOption: {command}')
    return value


def read_annotation(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def discover_routes(root):
    root = Path(root).resolve()
    paths = ([root] if (root / 'anno').is_dir() else [p.parent for p in root.rglob('anno') if p.is_dir()])
    # pathlib.rglob deliberately does not follow directory symlinks. Direct
    # route links let audits reuse immutable legacy raw data without copying.
    if not (root / 'anno').is_dir():
        paths += [p for p in root.iterdir() if p.is_dir() and (p / 'anno').is_dir()]
    return sorted(set(p for p in paths if (p / 'camera/rgb_front').is_dir()))


def geographic_group(route_name):
    """Weather/date duplicates of the same Town+Route cannot cross splits."""
    town = re.search(r'Town\d+', route_name)
    route = re.search(r'Route\d+', route_name)
    return f'{town.group()}_{route.group()}' if town and route else route_name


def split_routes(route_names, seed=42, smoke_train_only=False, geographic_groups=None):
    names = sorted(set(route_names))
    if smoke_train_only:
        return dict.fromkeys(names, 'train')
    groups = {}
    for name in names:
        group = (geographic_groups or {}).get(name, geographic_group(name))
        groups.setdefault(group, []).append(name)
    # Assign entire geographic groups globally. Per-scenario rounding puts
    # every route in TRAIN when a small audit has one route per scenario.
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    n = len(group_names)
    n_val = max(1, round(n * .1)) if n >= 3 else 0
    n_test = max(1, round(n * .1)) if n >= 3 else 0
    result = {}
    for idx, group in enumerate(group_names):
        split = 'val' if idx < n_val else 'test' if idx < n_val + n_test else 'train'
        for member in groups[group]:
            result[member] = split
    return result


def annotation_timestamp(raw, frame, allow_frame_index_timing, timing_policy='measured_timestamp'):
    for field in ('timestamp', 'timestamp_seconds', 'elapsed_seconds'):
        if field in raw and isinstance(raw[field], (int, float)) and np.isfinite(raw[field]):
            return float(raw[field]), field
    if timing_policy == 'nominal_simulation_time_10hz':
        return frame / 10., timing_policy
    if allow_frame_index_timing:
        return frame / 10., TIMING_FALLBACK
    raise ValueError('missing_timestamp')


def pose(raw):
    position = np.array([raw['x'], raw['y']], dtype=np.float64)
    compass = float(raw['theta'])
    if not np.isfinite(position).all() or not np.isfinite(compass):
        raise ValueError('invalid_pose')
    if abs(compass) > 2*math.pi + .01:
        raise ValueError('invalid_compass_units')
    return position, compass - math.pi/2


def validate_privileged_pose(raw):
    """Check sensor-derived pose against ego box ONLY for coordinate auditing."""
    boxes = raw.get('bounding_boxes', [])
    if isinstance(boxes, dict):
        boxes = list(boxes.values())
    ego = next((b for b in boxes if b.get('class') == 'ego_vehicle'), None)
    if ego is None:
        return None
    position, yaw = pose(raw)
    location = np.asarray(ego['location'][:2], dtype=np.float64)
    box_yaw = math.radians(ego['rotation'][2])
    yaw_error = abs(math.atan2(math.sin(yaw-box_yaw), math.cos(yaw-box_yaw)))
    position_error = float(np.linalg.norm(position-location))
    # GPS is noisy and mounted 1.4m behind the vehicle origin. Its position
    # difference is an audit statistic, not an assertion of identical origins.
    if yaw_error > .02:
        raise ValueError('pose_convention_mismatch')
    # Matrix row 0 points forward and row 1 points right in CARLA.
    matrix_error = None
    if 'world2ego' in ego:
        matrix = np.asarray(ego['world2ego'], dtype=np.float64)
        matrix_yaw = math.atan2(matrix[0, 1], matrix[0, 0])
        matrix_error = abs(math.atan2(math.sin(yaw-matrix_yaw), math.cos(yaw-matrix_yaw)))
        if matrix_error > .02:
            raise ValueError('world2ego_convention_mismatch')
    return {'position_error_m': position_error, 'yaw_error_rad': yaw_error, 'matrix_yaw_error_rad': matrix_error}


def label_pose(raw):
    """Privileged ego origin for labels ONLY; sensor pose if unavailable."""
    boxes = raw.get('bounding_boxes', [])
    if isinstance(boxes, dict):
        boxes = list(boxes.values())
    ego = next((b for b in boxes if b.get('class') == 'ego_vehicle'), None)
    if ego is None:
        position, yaw = pose(raw)
        return position, yaw, 'sensor_pose_fallback'
    position = np.asarray(ego['location'][:2], dtype=np.float64)
    yaw = math.radians(ego['rotation'][2])
    if not np.isfinite(position).all() or not np.isfinite(yaw):
        raise ValueError('invalid_label_pose')
    return position, yaw, 'official_ego_vehicle_pose_supervision_only'


def collect_windows(root, allow_frame_index_timing=False, route_splits=None, train_stride=1, eval_stride=5, nav_mode='annotation_near_route_target', geographic_groups=None, timing_policy='measured_timestamp', source_audit=None, require_official_label_pose=False, navigation_origin='ego_center_from_gnss_extrinsic', gnss_mount_xy_m=(-1.4, 0.)):
    """Construct only contiguous route-local windows; return rows and audit counts."""
    if nav_mode not in ('annotation_near_route_target', 'route_command', 'zero'):
        raise ValueError('Invalid navigation mode')
    if timing_policy not in ('measured_timestamp', 'nominal_simulation_time_10hz'):
        raise ValueError('Invalid timing_policy')
    timing_metadata = validate_nominal_clock(source_audit) if timing_policy == 'nominal_simulation_time_10hz' else {'clock': 'measured_annotation_timestamp', 'absolute_timestamp_available': True}
    mount = np.asarray(gnss_mount_xy_m, dtype=np.float64)
    if nav_mode == 'annotation_near_route_target':
        if navigation_origin != 'ego_center_from_gnss_extrinsic':
            raise ValueError('Official navigation requires ego_center_from_gnss_extrinsic')
        if mount.shape != (2,) or not np.isfinite(mount).all() or not np.allclose(mount, [-1.4,0], atol=1e-9, rtol=0):
            raise ValueError('gnss_mount_xy_m differs from verified official GNSS extrinsic [-1.4,0]')
    nav_metadata = dict(NAVIGATION_METADATA)
    if nav_mode == 'route_command':
        nav_metadata.update(origin='current_sensor_GPS_xy', mode='explicit_legacy_GNSS_origin')
    rows, rejected = [], Counter()
    timing_sources, pose_checks = Counter(), []
    route_summary = {}
    for route in discover_routes(root):
        annotations = {}
        for path in sorted((route / 'anno').glob('*.json*')):
            frame = int(path.name.split('.')[0])
            try:
                raw = read_annotation(path)
                position, yaw = pose(raw)
                timestamp, timing_source = annotation_timestamp(raw, frame, allow_frame_index_timing, timing_policy)
                speed = float(raw['speed'])
                if not np.isfinite(speed):
                    raise ValueError('invalid_speed')
                check = validate_privileged_pose(raw)
                if check is not None:
                    pose_checks.append(check)
                label_position, label_yaw, label_source = label_pose(raw)
                if require_official_label_pose and label_source != 'official_ego_vehicle_pose_supervision_only':
                    raise ValueError('missing_official_label_pose')
                annotations[frame] = dict(raw=raw, position=position, yaw=yaw, timestamp=timestamp,
                                          timing_source=timing_source, speed=speed,
                                          label_position=label_position, label_yaw=label_yaw, label_source=label_source)
                timing_sources[timing_source] += 1
            except (ValueError, KeyError, TypeError) as error:
                rejected[str(error).strip("'")] += 1
        images = {int(p.stem): p.resolve() for p in (route / 'camera/rgb_front').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        if not annotations:
            route_summary[route.name] = {'valid_annotations': 0, 'windows': 0}
            continue
        split = route_splits[route.name] if route_splits is not None else 'train'
        stride = train_stride if split == 'train' else eval_stride
        if stride < 1:
            raise ValueError('stride must be positive')
        before = len(rows)
        for frame in range(min(annotations)+3, max(annotations)-29, stride):
            required = list(range(frame-3, frame+31))
            if any(f not in annotations for f in required):
                rejected['missing_frame'] += 1; continue
            if any(f not in images for f in range(frame-3, frame+1)):
                rejected['missing_image'] += 1; continue
            seq = [annotations[f] for f in required]
            if len({x['timing_source'] for x in seq}) != 1 or not np.allclose(np.diff([x['timestamp'] for x in seq]), .1, atol=1e-4, rtol=0):
                rejected['irregular_timing'] += 1; continue
            current, previous = annotations[frame], annotations[frame-1]
            dt = current['timestamp'] - previous['timestamp']
            dyaw = math.atan2(math.sin(current['yaw']-previous['yaw']), math.cos(current['yaw']-previous['yaw']))
            ego = [current['speed'], (current['speed']-previous['speed'])/dt, -dyaw/dt]
            raw = current['raw']
            try:
                command = encode_command(raw.get('command_near'))
                if nav_mode in ('route_command', 'annotation_near_route_target'):
                    nav = world_to_ego([raw['x_command_near'], raw['y_command_near']], current['position'], current['yaw']).tolist()
                    if nav_mode == 'annotation_near_route_target':
                        # GPS_world = ego_center_world + R(yaw) @ mount.
                        # In x-forward/y-left, vehicle-center target =
                        # GPS-relative target + [mount_x, -mount_y].
                        nav = (np.asarray(nav) + [mount[0], -mount[1]]).tolist()
                    nav_source = 'official_annotation_near_route_target'
                else:
                    nav, nav_source = [0., 0.], 'explicit_zero_target_fallback'
            except (ValueError, KeyError, TypeError):
                rejected['invalid_navigation'] += 1; continue
            if len({x['label_source'] for x in seq}) != 1:
                rejected['mixed_label_pose_sources'] += 1; continue
            gt = world_to_ego([annotations[frame+o]['label_position'] for o in FUTURE_OFFSETS], current['label_position'], current['label_yaw'])
            audit_seq = [annotations[f] for f in range(frame, frame+31)]
            audit_motion = dict(
                speed_min_mps=min(x['speed'] for x in audit_seq),
                max_delta_speed_mps=max(abs(x['speed']-current['speed']) for x in audit_seq),
                max_delta_yaw_rad=max(abs(math.atan2(math.sin(x['yaw']-current['yaw']), math.cos(x['yaw']-current['yaw']))) for x in audit_seq),
                num_frames=len(audit_seq), source='future_sensor_speed_compass_audit_only_never_model_input')
            weather_match = re.search(r'Weather(\d+)', route.name)
            rows.append(dict(route=route.name, geographic_group=(geographic_groups or {}).get(route.name, geographic_group(route.name)),
                             scene=route.name.split('_Town')[0],
                             weather=int(weather_match.group(1)) if weather_match else None,
                             frame=frame, timestamp=current['timestamp'], images=[str(images[f]) for f in range(frame-3, frame+1)],
                             ego=ego, command=command, nav=nav, trajectory_gt=gt.tolist(), split=split,
                             timing_source=current['timing_source'], nav_source=nav_source,
                             timing_metadata=timing_metadata if current['timing_source'] != TIMING_FALLBACK else {'clock': 'unverified_frame_index_fallback', 'absolute_timestamp_available': False},
                             navigation_metadata=nav_metadata if nav_mode != 'zero' else {'mode': 'explicit_legacy_zero'},
                             audit_motion=audit_motion,
                             ego_source='speed_sensor_backward_speed_difference_backward_compass_difference',
                             label_pose_source=current['label_source'],
                             coordinate_system='CARLA_world_to_x_forward_y_left; yaw=compass_rad-pi/2'))
        indices = sorted(annotations)
        route_rows = rows[before:]
        dimensions = Counter()
        for image_path in images.values():
            with Image.open(image_path) as image:
                dimensions[f'{image.width}x{image.height}'] += 1
        distances = [float(np.linalg.norm(np.diff([annotations[f]['label_position'] for f in indices], axis=0), axis=1).max())] if len(indices) > 1 else []
        dt_estimates = []
        for a,b in zip(indices, indices[1:]):
            if b-a != 1:
                continue
            first, last = annotations[a], annotations[b]
            if first['raw'].get('teleport') or last['raw'].get('teleport'):
                continue
            mean_speed = (first['speed']+last['speed'])/2
            distance = float(np.linalg.norm(last['label_position']-first['label_position']))
            if min(first['speed'], last['speed']) > 1 and distance/mean_speed < .3:
                dt_estimates.append(distance/mean_speed)
        clock_audit = {'num_moving_adjacent_pairs': len(dt_estimates),
            'speed_threshold_mps': 1., 'estimate_method': 'official_ego_displacement_divided_by_mean_adjacent_sensor_speed',
            'median_estimated_dt_seconds': float(np.median(dt_estimates)) if dt_estimates else None,
            'p05_p95_estimated_dt_seconds': np.quantile(dt_estimates,[.05,.95]).tolist() if dt_estimates else None,
            'median_abs_error_to_10hz_seconds': float(np.median(np.abs(np.asarray(dt_estimates)-.1))) if dt_estimates else None,
            'median_abs_error_to_20hz_seconds': float(np.median(np.abs(np.asarray(dt_estimates)-.05))) if dt_estimates else None,
            'limitation': 'Kinematic estimate validates nominal cadence, not absolute timestamps or per-sensor acquisition times.'}
        route_summary[route.name] = {'valid_annotations': len(annotations), 'windows': len(route_rows),
            'kinematic_clock_audit': clock_audit,
            'schema_keys': sorted(annotations[indices[0]]['raw']), 'image_dimensions': dict(dimensions),
            'annotation_range': [indices[0], indices[-1]],
            'annotation_gaps': [[a,b] for a,b in zip(indices, indices[1:]) if b-a != 1],
            'missing_front_images': sorted(set(indices)-set(images)),
            'max_label_displacement_per_saved_frame_m': max(distances, default=None),
            'nav_norm_m_range': [min((np.linalg.norm(r['nav']) for r in route_rows), default=None), max((np.linalg.norm(r['nav']) for r in route_rows), default=None)],
            'nav_forward_nonnegative_fraction': sum(r['nav'][0] >= 0 for r in route_rows)/len(route_rows) if route_rows else None,
            'commands': dict(Counter(r['command'] for r in route_rows)), 'split': split}

    report = dict(routes=route_summary, windows=len(rows), rejected=dict(rejected), timing_sources=dict(timing_sources),
                  pose_validation={'checked_frames': len(pose_checks),
                                   'max_position_error_m': max((p['position_error_m'] for p in pose_checks), default=None),
                                   'max_yaw_error_rad': max((p['yaw_error_rad'] for p in pose_checks), default=None)},
                  source=SOURCE_URL, navigation_contract=nav_metadata,
                  navigation_origin=nav_metadata['origin'], gnss_mount_xy_m=mount.tolist(),
                  navigation_verified=nav_mode in ('route_command','annotation_near_route_target'),
                  timing_policy=timing_policy, timing_metadata=timing_metadata,
                  timing_limitation='Frame-index fallback assumes 10Hz from official collector; actual timestamps unavailable and cannot be independently verified.' if TIMING_FALLBACK in timing_sources else None)
    return rows, report


def normalization_stats(rows):
    train = [r for r in rows if r['split'] == 'train']
    if not train:
        raise ValueError('Cannot compute normalization without TRAIN observations')
    stats = {'split': 'train', 'num_samples': len(train), 'epsilon': 1e-6}
    for key in ('ego', 'nav'):
        values = np.asarray([r[key] for r in train], dtype=np.float64)
        stats[f'{key}_mean'] = values.mean(0).tolist()
        std = values.std(0)
        stats[f'{key}_std'] = np.where(std < 1e-6, 1., std).tolist()
    return stats


class FastDataset(Dataset):
    def __init__(self, manifest_path, stats_path):
        self.rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
        self.stats = json.loads(Path(stats_path).read_text())
        if self.stats.get('split') != 'train':
            raise ValueError('Normalization statistics must originate from TRAIN')
        for key, dim in [('ego', 3), ('nav', 2)]:
            if np.shape(self.stats[f'{key}_mean']) != (dim,) or np.shape(self.stats[f'{key}_std']) != (dim,) or not np.isfinite(self.stats[f'{key}_mean']).all() or not np.isfinite(self.stats[f'{key}_std']).all() or np.any(np.asarray(self.stats[f'{key}_std']) <= 0):
                raise ValueError(f'Invalid {key} normalization statistics')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        for path in row['images']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        result = dict(images=images, command=torch.tensor(row['command'], dtype=torch.long),
                      trajectory_gt=torch.tensor(row['trajectory_gt'], dtype=torch.float32), metadata=row)
        for key in ('ego', 'nav'):
            value = (np.asarray(row[key])-self.stats[f'{key}_mean']) / self.stats[f'{key}_std']
            result[key] = torch.tensor(value, dtype=torch.float32)
        return result
