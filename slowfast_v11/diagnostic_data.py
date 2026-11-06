"""Causal observations and eligibility for bounded, offline KV diagnostics.

Only current/history raw fields construct model inputs. Labels remain on the
caller's manifest rows and are never copied into observations or pair metadata.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from numbers import Integral
from pathlib import Path
import pickle

from fast_navsim.data import _pose


class _Ineligible(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class DiagnosticPair:
    target_token: str
    source_token: str
    source_index: int
    source_route: str
    age_s: float  # Requested diagnostic bin; actual age is independently recorded.
    source_pose: tuple[float, float, float]
    current_pose: tuple[float, float, float]
    split: str
    actual_age_s: float
    source_timestamp_us: int
    current_timestamp_us: int


def _command(frame):
    try:
        values = [float(x) for x in frame['driving_command']]
    except (KeyError, TypeError, ValueError):
        raise _Ineligible('invalid_command') from None
    if len(values) != 4 or values.count(1.) != 1 or any(x not in (0., 1.) for x in values):
        raise _Ineligible('invalid_command')
    return values.index(1.)


def _timestamp(frame):
    value = frame['timestamp']
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _Ineligible('invalid_timestamp')
    return int(value)


def _raw_observation(frames, index, sensor_root):
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 3:
        raise _Ineligible('missing_history')
    if index >= len(frames):
        raise _Ineligible('missing_current')
    history = frames[index-3:index+1]
    current = history[-1]
    route = current['log_name']
    if any(frame['log_name'] != route for frame in history):
        raise _Ineligible('log_mismatch')
    command = _command(current)
    if any(_command(frame) != command for frame in history):
        raise _Ineligible('navigation_mismatch')
    timestamps = [_timestamp(frame) for frame in history]
    if any(abs(b-a-500_000) > 1000 for a, b in zip(timestamps, timestamps[1:])):
        raise _Ineligible('timestamp_gap')
    root = Path(sensor_root).resolve()
    paths = []
    identities = []
    for frame in history:
        relative = Path(frame['cams']['CAM_F0']['data_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise _Ineligible('invalid_image_path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise _Ineligible('invalid_image_path')
        if not path.is_file():
            raise _Ineligible('missing_image')
        stat = path.stat()
        identities.append((stat.st_dev, stat.st_ino))
        paths.append(str(path))
    if len(set(identities)) != 4:
        raise _Ineligible('duplicate_images')
    ego = [float(value) for value in current['ego_dynamic_state']]
    if len(ego) != 4 or not all(math.isfinite(value) for value in ego):
        raise _Ineligible('invalid_ego')
    pose = _pose(current)
    token = current['token']
    if not isinstance(token, str) or not token or not isinstance(route, str) or not route:
        raise _Ineligible('invalid_identity')
    metadata = dict(token=token, route=route, frame=int(index),
                    timestamp_us=timestamps[-1], pose=pose, images=paths)
    return metadata, ego, command


def observation_at(frames, index, sensor_root, stats):
    """Return four original RGB images, ego[4], scalar command, causal metadata."""
    import torch
    from PIL import Image

    if stats.get('source_split') != 'train':
        raise ValueError('Ego normalization requires TRAIN-only statistics')
    mean, std = stats['ego_mean'], stats['ego_std']
    if (len(mean) != 4 or len(std) != 4
            or not all(math.isfinite(float(x)) for x in (*mean, *std))
            or any(float(x) <= 0 for x in std)):
        raise ValueError('Ego statistics require four finite means and positive standard deviations')
    metadata, ego, command = _raw_observation(frames, index, sensor_root)
    images = []
    for path in metadata['images']:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return dict(images=images,
                ego=torch.tensor([(x-m)/s for x,m,s in zip(ego,mean,std)], dtype=torch.float32),
                command=torch.tensor(command, dtype=torch.long), metadata=metadata)


def _validate_rows(rows):
    tokens = set()
    route_splits = {}
    for row in rows:
        split = row.get('split')
        if split not in ('train', 'val'):
            raise ValueError('Diagnostics accept TRAIN/VAL splits only; no TEST access')
        token, route = row['token'], row['route']
        if token in tokens:
            raise ValueError('Duplicate diagnostic token')
        tokens.add(token)
        if route in route_splits and route_splits[route] != split:
            raise ValueError('Route leakage across diagnostic splits')
        route_splits[route] = split


def build_pairs(rows, frames_by_log, sensor_root, ages_s=(0., .5, 1., 1.5)):
    """Return genuine source pairs plus counts of each explicit exclusion reason.

    An age requests an index offset at 2 Hz, then validates the real observed
    timestamps within 1 ms. No future frame or target annotation is inspected.
    """
    rows = list(rows)
    _validate_rows(rows)
    ages = tuple(float(age) for age in ages_s)
    if (not ages or len(set(ages)) != len(ages)
            or any(not math.isfinite(age) or age < 0 or abs(age*2-round(age*2)) > 1e-9 for age in ages)):
        raise ValueError('Diagnostic ages must be distinct finite nonnegative multiples of 0.5s')
    report = {'by_age': {str(age): {'eligible': 0, 'excluded': {}, 'eligible_tokens': []} for age in ages},
              'rows': len(rows), 'timestamp_tolerance_us': 1000}
    pairs = []
    for age in ages:
        entry = report['by_age'][str(age)]
        for row in rows:
            try:
                frames = frames_by_log.get(row['route'])
                if frames is None:
                    raise _Ineligible('missing_log')
                index = row['frame']
                current, _, command = _raw_observation(frames, index, sensor_root)
                if (current['token'] != row['token'] or current['route'] != row['route']
                        or current['timestamp_us'] != row['timestamp_us'] or command != row['command']):
                    raise _Ineligible('row_mismatch')
                source_index = index - round(age*2)
                source, _, source_command = _raw_observation(frames, source_index, sensor_root)
                if any(frame['log_name'] != row['route'] for frame in frames[source_index-3:index+1]):
                    raise _Ineligible('log_mismatch')
                if source_command != command or any(_command(frame) != command for frame in frames[source_index-3:index+1]):
                    raise _Ineligible('navigation_mismatch')
                actual_us = current['timestamp_us']-source['timestamp_us']
                if actual_us < 0 or abs(actual_us-age*1_000_000) > 1000:
                    raise _Ineligible('age_mismatch')
                pairs.append(DiagnosticPair(row['token'],source['token'],source_index,row['route'],age,
                    source['pose'],current['pose'],row['split'],actual_us/1_000_000,
                    source['timestamp_us'],current['timestamp_us']))
                entry['eligible'] += 1
                entry['eligible_tokens'].append(row['token'])
            except _Ineligible as error:
                entry['excluded'][error.reason] = entry['excluded'].get(error.reason,0)+1
            except (KeyError, TypeError, ValueError, IndexError):
                reason = 'invalid_observation'
                entry['excluded'][reason] = entry['excluded'].get(reason,0)+1
    return pairs, report


def deterministic_mismatches(rows):
    """Split-local complete derangements, preferring other TRAIN logs.

    Same-log swaps require >=2 seconds separation. A split with no complete
    matching is explicitly unavailable and contributes no partial mapping.
    """
    rows = list(rows)
    _validate_rows(rows)
    mapping, report = {}, {}
    for split in ('train', 'val'):
        selected = sorted((row for row in rows if row['split'] == split), key=lambda row: row['token'])
        n = len(selected)
        if not n:
            report[split] = dict(available=False,reason='empty_split',count=0)
            continue
        candidates = []
        for target in selected:
            options = []
            for j,source in enumerate(selected):
                cross_log = target['route'] != source['route']
                if target['token'] != source['token'] and (cross_log or abs(target['timestamp_us']-source['timestamp_us']) >= 2_000_000):
                    options.append((j, int(cross_log and split == 'train')))
            candidates.append(options)

        @lru_cache(None)
        def assign(i, used):
            if i == n:
                return 0, ()
            best = None
            for j, preference in candidates[i]:
                if used & (1 << j):
                    continue
                rest = assign(i+1, used | (1 << j))
                if rest is not None:
                    result = preference+rest[0], (j,)+rest[1]
                    if best is None or result[0] > best[0]:
                        best = result
            return best

        result = assign(0,0)
        if result is None:
            report[split] = dict(available=False,reason='no_complete_derangement',count=n)
            continue
        mapping.update({target['token']:selected[j]['token'] for target,j in zip(selected,result[1])})
        report[split] = dict(available=True,count=n,cross_log_count=result[0])
    return mapping, report


def load_frames_by_log(rows):
    """Load only the distinct manifest-referenced trusted mini PKLs after SHA256 checks."""
    rows = list(rows)
    _validate_rows(rows)
    references = {}
    for row in rows:
        source = row['source']
        reference = (str(Path(source['metadata_path']).resolve()), source['metadata_sha256'])
        if row['route'] in references and references[row['route']] != reference:
            raise ValueError('Conflicting metadata identity for one route')
        references[row['route']] = reference
    blobs, result = {}, {}
    for route,(path,expected) in references.items():
        if path not in blobs:
            blobs[path] = Path(path).read_bytes()
        blob = blobs[path]
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Metadata SHA256 mismatch')
        frames = pickle.loads(blob)
        if not isinstance(frames,list) or not frames or frames[0].get('log_name') != route:
            raise ValueError('Metadata does not match requested route')
        result[route] = frames
    return result
