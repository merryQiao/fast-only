"""Validate frozen cross-input evidence and report CPU-only paired statistics.

Public pure APIs: collapse_cells(records, arm) returns one object per target
window; paired_effect(windows, coefficients, metric='ADE', seed=20261109,
reps=20000) contrasts its already donor-averaged errors, then bootstraps logs.
No model, tensor framework, GPU or network is used by this report.
"""
import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re

import numpy as np

ARMS = ('full_fast_only', 'full_correct_kv')
OFFSETS = (0, 1, 4, 7)
TIMES = tuple(f'L2_{i/2:g}s' for i in range(1, 9))
METRICS = ('ADE', 'FDE', 'heading_MAE', 'XY_change') + TIMES
CHECKPOINTS = {
    'full_fast_only': '3af11731a14c80b5d08f30c28c6c8cf75727657b83a13984db5b5fe920e41d89',
    'full_correct_kv': '5f08df489c645487bf3183b8b0dfdc06ea691ba8a205ab2e9b131e662a8fd558',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recording_group(route):
    return re.sub(r'_\d+_\d+$', '', route)


def input_cells(arm):
    require(arm in ARMS, 'Unknown frozen input arm')
    if arm == ARMS[0]:
        return [(i, None, 'original' if i == 0 else 'swapped') for i in OFFSETS]
    return [(i, k, 'A' if i == k == 0 else 'B' if i == 0 else 'C' if k == 0
             else 'D_same' if i == k else 'D_independent') for i in OFFSETS for k in OFFSETS]


def poses(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Expected eight finite XY/heading poses') from error
    require(array.shape == (8, 3) and array.dtype.kind in 'fiu' and np.isfinite(array).all(),
            'Expected eight finite XY/heading poses')
    return array.astype(np.float64)


def pose_errors(prediction, target, baseline):
    """Compute errors before any donor reduction; headings are in radians."""
    prediction, target, baseline = poses(prediction), poses(target), poses(baseline)
    distance = np.linalg.norm(prediction[:, :2] - target[:, :2], axis=-1)
    angle = prediction[:, 2] - target[:, 2]
    result = dict(ADE=float(distance.mean()), FDE=float(distance[-1]),
                  heading_MAE=float(np.abs(np.arctan2(np.sin(angle), np.cos(angle))).mean()),
                  XY_change=float(np.linalg.norm(prediction[:, :2] - baseline[:, :2], axis=-1).mean()),
                  L2_by_time=distance.tolist())
    result.update(zip(TIMES, map(float, distance)))
    return result


def average_metrics(values):
    values = list(values)
    require(bool(values), 'Cannot average an empty metric cell')
    return {key: np.mean([value[key] for value in values], axis=0).tolist() for key in values[0]}


def subtract_metrics(left, right):
    return {key: (np.asarray(left[key]) - np.asarray(right[key])).tolist() for key in left}


def collapse_cells(records, arm):
    """Validate full per-window cells and average donor ERRORS within a target.

    The ``cells`` dictionary contains grouped metrics. ``offsets`` retains all
    four or sixteen source pairs as ``'image_offset,kv_offset'`` keys.
    ``matched_rescue`` retains D-C with the *same image donor* before reduction.
    Metadata and GT must be fixed across all interventions on each target.
    Full protocol and canonical-token authentication is performed by load_run.
    """
    expected = {(i, k): name for i, k, name in input_cells(arm)}
    groups = defaultdict(list)
    required = {'arm', 'token', 'route', 'command', 'speed_mps', 'image_offset', 'kv_offset',
                'cell', 'image_source_token', 'kv_source_token', 'prediction', 'target',
                'image_feature_relative_l2'}
    for row in records:
        require(isinstance(row, Mapping) and required <= row.keys(), 'Incomplete factorial record')
        require(isinstance(row['token'], str) and bool(row['token']), 'Invalid target token')
        groups[row['token']].append(row)
    require(bool(groups), 'No target windows')
    windows = []
    for token, rows in groups.items():
        indices = [(row['image_offset'], row['kv_offset']) for row in rows]
        require(len(indices) == len(set(indices)) and set(indices) == set(expected),
                'Incomplete or duplicate factorial cells')
        first = rows[0]
        require(isinstance(first['route'], str) and bool(first['route']), 'Invalid recording route')
        require(isinstance(first['speed_mps'], Real) and not isinstance(first['speed_mps'], bool)
                and np.isfinite(first['speed_mps']) and first['speed_mps'] >= 0, 'Invalid current speed')
        target = poses(first['target'])
        source_labels = {0: token}
        for row in rows:
            i, k = row['image_offset'], row['kv_offset']
            require(type(i) is int and (k is None or type(k) is int), 'Invalid source offset type')
            require(row['arm'] == arm and row['cell'] == expected[i, k], 'Factorial arm/cell label differs')
            require(all(row[key] == first[key] for key in ('route', 'command', 'speed_mps'))
                    and np.array_equal(poses(row['target']), target), 'Target GT/current metadata differs within window')
            value = row['image_feature_relative_l2']
            require(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)
                    and value >= 0 and (i != 0 or value == 0), 'Invalid visual feature difference')
            for offset, label in ((i, row['image_source_token']), (k, row['kv_source_token'])):
                if offset is None:
                    require(label is None, 'Fast-only KV source must be null')
                    continue
                require(isinstance(label, str) and bool(label), 'Invalid source token')
                require(offset not in source_labels or source_labels[offset] == label,
                        'Inconsistent source for a donor offset')
                source_labels[offset] = label
        require(len(set(source_labels.values())) == 4, 'Different offsets require different donor sources')
        lookup = {(row['image_offset'], row['kv_offset']): row for row in rows}
        baseline = lookup[0, None if arm == ARMS[0] else 0]['prediction']
        errors = {key: pose_errors(row['prediction'], row['target'], baseline) for key, row in lookup.items()}
        grouped = {name: average_metrics(errors[i, k] for i, k, label in input_cells(arm) if label == name)
                   for name in dict.fromkeys(expected.values())}
        matched = {}
        if arm == ARMS[1]:
            for label in ('D_same', 'D_independent'):
                matched[label] = {}
                for image in OFFSETS[1:]:
                    pairs = [(i, k) for i, k, name in input_cells(arm) if name == label and i == image]
                    require(len(pairs) == (1 if label == 'D_same' else 2), 'Unpaired D-C image donors')
                    matched[label][str(image)] = average_metrics(
                        subtract_metrics(errors[i, k], errors[image, 0]) for i, k in pairs)
                direct = subtract_metrics(grouped[label], grouped['C'])
                paired = average_metrics(matched[label].values())
                require(all(np.allclose(direct[key], paired[key], rtol=1e-12, atol=1e-12) for key in direct),
                        'Same-image D-C pairing disagrees with balanced cell contrast')
        windows.append(dict(token=token, log=recording_group(first['route']), route=first['route'],
                            command=first['command'], speed_mps=float(first['speed_mps']), cells=grouped,
                            offsets={f'{i},{k}': errors[i, k] for i, k in expected}, matched_rescue=matched))
    return windows


def paired_effect(windows, coefficients, metric='ADE', seed=20261109, reps=20000):
    """Contrast target-window errors, then resample paired recording means.

    Every represented recording receives equal bootstrap weight. The separately
    returned window-weighted mean describes its actual cohort composition.
    Intervals condition on the fixed targets/checkpoints/donor maps and contain
    neither training-seed nor donor-selection uncertainty. One eligible log has
    no interval (ci95=None); two or three eligible logs carry a fragility caveat.
    """
    require(isinstance(coefficients, Mapping) and bool(coefficients), 'Specify a nonempty cell contrast')
    require(all(isinstance(c, Real) and not isinstance(c, bool) and np.isfinite(c)
                for c in coefficients.values()), 'Contrast coefficients must be finite')
    require(isinstance(seed, Integral) and not isinstance(seed, bool) and seed >= 0,
            'Bootstrap seed must be a nonnegative integer')
    require(isinstance(reps, Integral) and not isinstance(reps, bool) and reps > 0,
            'Bootstrap replicates must be a positive integer')
    per_log = defaultdict(list); seen = set(); effects = []
    for window in windows:
        token, log = window['token'], window['log']
        require(token not in seen, 'Duplicate paired target window')
        seen.add(token)
        try:
            values = [window['cells'][name][metric] for name in coefficients]
        except KeyError as error:
            raise ValueError('Missing paired metric/cell') from error
        require(all(isinstance(value, Real) and np.isfinite(value) for value in values),
                'Paired metric must be a finite scalar')
        value = float(sum(c * value for c, value in zip(coefficients.values(), values)))
        effects.append(value); per_log[log].append(value)
    require(bool(effects), 'No paired windows in this cohort')
    logs = sorted(per_log)
    means = np.array([np.mean(per_log[log]) for log in logs])
    interval = None
    unavailable_reason = None
    caution = None
    if len(logs) < 2:
        unavailable_reason = 'fewer than two eligible recording groups; descriptive means only'
    else:
        rng = np.random.default_rng(seed)
        bootstrap = means[rng.integers(len(logs), size=(reps, len(logs)))].mean(1)
        interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
        if len(logs) <= 3:
            caution = f'Only {len(logs)} eligible recording groups; conditional bootstrap interval is fragile.'
    return dict(window_count=len(effects), log_count=len(logs), window_weighted_mean=float(np.mean(effects)),
                equal_log_mean=float(means.mean()), per_log_means=dict(zip(logs, means.tolist())),
                per_log_window_counts={log: len(per_log[log]) for log in logs},
                ci95=interval, ci95_unavailable_reason=unavailable_reason, interval_caution=caution, metric=metric,
                coefficients=dict(coefficients), analysis_seed=int(seed), bootstrap_replicates=int(reps),
                bootstrap_unit='target recording group; paired effects averaged within log',
                uncertainty_scope='conditional on fixed observed targets, donor maps and frozen checkpoints; '
                                  'no donor-selection or training-seed uncertainty')


RUNNER_SOURCE_FILES = (
    'tools/diagnose_cross_inputs.py', 'slowfast_v11/cross_inputs.py', 'slowfast_v11/model.py',
    'slowfast_v11/attention.py', 'tools/direct_full_common.py', 'slowfast_v11/full_training.py',
    'slowfast_v11/vision_protocol.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_run(run_root, *, project_root=None):
    """Authenticate protocol, source receipts and all 5,120 numeric records.

    Original checkpoint binaries and images are deliberately unnecessary. The
    authenticated runner has already rehashed/reloaded the pinned checkpoints;
    this CPU report independently checks every numeric replay and prior forecast.
    """
    import yaml
    root = Path(run_root).resolve()
    project = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    plan = read_json(root/'protocol.json'); config = plan['config']
    fixed = dict(schema='frozen_cross_inputs_v1', arms=list(ARMS), donor_recording_offsets=[1, 4, 7],
                 donor_rule='sorted_recording_groups_cyclic_offset_same_manifest_index', cartesian_joint_inputs=True,
                 val_windows=256, recording_groups=8, raw_replay_targets_per_recording=1,
                 raw_replay_image_sources=list(OFFSETS), metadata=[0., 0., 0., 0., 1., 1.], moving_speed_mps=1.,
                 high_speed_mps=10., practical_ade_reference_m=.02, analysis_seed=20261109,
                 bootstrap_replicates=20000, max_allocated_gib=32, new_training_updates=0, new_response_body_bytes=0,
                 checkpoint_sha256=CHECKPOINTS, vision_precision_policy='native_fp32_visual_rope')
    require(plan.get('schema') == 'frozen_cross_inputs_protocol_v1' and
            all(config.get(key) == value for key, value in fixed.items()), 'Frozen protocol/config differs')
    config_path = project/'configs/slowfast_cross_inputs.yaml'
    require(sha256(config_path) == plan['config_sha256'] and yaml.safe_load(config_path.read_text()) == config,
            'Protocol configuration file/hash differs')
    source_hashes = plan['source_code_sha256']
    require(set(source_hashes) == set(RUNNER_SOURCE_FILES), 'Runner source provenance is incomplete')
    for name, expected in source_hashes.items():
        require(sha256(project/name) == expected, f'Runner source hash differs: {name}')
    cache_path = project/config['cache']/'manifest.json'
    source_root = project/config['run_root']
    input_files = {'protocol.json': sha256(root/'protocol.json'), str(config_path): sha256(config_path)}
    pinned_paths = {'cache_manifest': cache_path, 'prior_diagnostic': source_root/'correct_checkpoint_diagnostics.json'}
    for arm in ARMS:
        for name, filename in [('result', 'result.json'), ('verification', 'verification.json'),
                               ('predictions', 'final_predictions.json')]:
            pinned_paths[arm+'_'+name] = source_root/arm/filename
    require(set(plan['hashes']) == set(pinned_paths), 'Protocol pinned file coverage differs')
    for key, path in pinned_paths.items():
        actual = sha256(path)
        require(actual == plan['hashes'][key], f'Pinned input file hash differs: {key}')
        input_files[str(path)] = actual
    manifest = read_json(cache_path)
    require(all(manifest.get(name) is True for name in ('completed', 'source_frozen', 'vision_native_prediction_all_exact')),
            'Canonical cache acceptance is incomplete')
    canonical = [row for row in manifest['rows'] if row['split'] == 'val']
    tokens = [row['token'] for row in canonical]
    groups = defaultdict(list)
    for row in canonical:
        groups[recording_group(row['route'])].append(row['token'])
    require(len(tokens) == len(set(tokens)) == 256 and len(groups) == 8 and
            {len(values) for values in groups.values()} == {32}, 'Expected canonical VAL256/8x32 logs')
    targets = [{key: row[key] for key in ('token', 'route', 'command', 'ego')} for row in canonical]
    require(plan['tokens'] == tokens and plan['targets'] == targets, 'Protocol current target metadata differs')
    group_names = sorted(groups)
    donors = {str(offset): {token: groups[group_names[(index+offset) % 8]][position]
                           for index, name in enumerate(group_names) for position, token in enumerate(groups[name])}
              for offset in OFFSETS[1:]}
    require(plan['donors'] == donors and plan['raw_targets'] == [groups[name][0] for name in group_names],
            'Frozen donor map or first-window raw controls differ')
    expected_records = dict(full_fast_only=1024, full_correct_kv=4096)
    expected_exact = dict(full_fast_only=256, full_correct_kv=512)
    require(plan['expected_records'] == expected_records and plan['expected_saved_exact'] == expected_exact
            and plan['expected_raw_replays_per_arm'] == 32, 'Protocol record/replay counts differ')
    require(plan['cells'] == {arm: [dict(image_offset=i, kv_offset=k, cell=name)
                                   for i, k, name in input_cells(arm)] for arm in ARMS}, 'Protocol factorial cells differ')
    old_root_result = read_json(source_root/'result.json')
    input_files[str(source_root/'result.json')] = sha256(source_root/'result.json')
    require(old_root_result['diagnostics_file_sha256'] == plan['hashes']['prior_diagnostic'],
            'Prior diagnostic authentication differs')
    prior = read_json(pinned_paths['prior_diagnostic'])
    require(prior['donors'] == donors['4'], 'Prior offset4 mapping differs')
    prior_rows = [row for row in prior['records'] if row['mode'] == 'shuffled']
    prior_by_token = {row['token']: row for row in prior_rows}
    require(len(prior_rows) == len(prior_by_token) == 256 and set(prior_by_token) == set(tokens),
            'Prior offset4 forecast coverage differs')
    by_token = {row['token']: row for row in canonical}
    all_records = {}; results = {}; exact_count = 0; raw_count = 0
    for arm in ARMS:
        previous_result = read_json(source_root/arm/'result.json')
        verification = read_json(source_root/arm/'verification.json')
        require(previous_result.get('completed') is True and previous_result.get('arm') == arm
                and verification.get('passed') is True
                and previous_result.get('checkpoint_sha256') == verification.get('checkpoint_sha256') == CHECKPOINTS[arm]
                and verification.get('predictions_sha256') == plan['hashes'][arm+'_predictions'],
                'Pinned endpoint verification/checkpoint identity differs')
        previous_rows = read_json(source_root/arm/'final_predictions.json')
        previous = {row['token']: row for row in previous_rows}
        require(len(previous_rows) == len(previous) == 256 and set(previous) == set(tokens), 'Prior endpoint coverage differs')
        directory = root/arm; result = read_json(directory/'result.json')
        require(result.get('schema') == 'frozen_cross_inputs_arm_v1' and result.get('completed') is True
                and result.get('arm') == arm and result.get('records') == expected_records[arm]
                and result.get('exact_previous_forecasts') == expected_exact[arm]
                and result.get('raw_processor_replays_exact') == 32
                and result.get('checkpoint_sha256') == CHECKPOINTS[arm]
                and result.get('parameter_versions_unchanged') is True
                and result.get('inputs_and_sources_unchanged') is True
                and result.get('new_training_updates') == result.get('new_response_body_bytes') == 0,
                'Cross-input runner acceptance receipt differs')
        parameter_hash = result.get('all_parameter_sha256_before', '')
        require(isinstance(parameter_hash, str) and re.fullmatch('[0-9a-f]{64}', parameter_hash) is not None
                and parameter_hash == result.get('all_parameter_sha256_after'), 'Full parameter immutability proof differs')
        buffer_hash = result.get('all_buffer_sha256_before', '')
        require(isinstance(buffer_hash, str) and re.fullmatch('[0-9a-f]{64}', buffer_hash) is not None
                and buffer_hash == result.get('all_buffer_sha256_after'), 'Full buffer immutability proof differs')
        rope = result.get('native_visual_rope')
        require(isinstance(rope, Mapping) and rope.get('after_dtype') == 'torch.float32'
                and rope.get('before_dtype') in ('torch.bfloat16', 'torch.float32'),
                'Native FP32 visual RoPE receipt missing or differs')
        peak = result.get('peak_allocated_gib')
        require(isinstance(peak, Real) and np.isfinite(peak) and 0 <= peak <= 32, 'GPU memory receipt exceeds fixed cap')
        for filename, key in [('predictions.jsonl', 'predictions_sha256'), ('raw_replays.json', 'raw_replays_sha256')]:
            actual = sha256(directory/filename)
            require(actual == result.get(key), f'Arm evidence file hash differs: {filename}')
            input_files[str(directory/filename)] = actual
        require(result.get('protocol_sha256') == input_files['protocol.json'], 'Arm protocol hash differs')
        input_files[str(directory/'result.json')] = sha256(directory/'result.json')
        rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        require(len(rows) == expected_records[arm] and {row['token'] for row in rows} == set(tokens),
                'Full target/arm coverage differs')
        collapse_cells(rows, arm)  # All factorial completeness, labels, poses and within-target invariants.
        lookup = {(row['token'], row['image_offset'], row['kv_offset']): row for row in rows}
        for row in rows:
            token = row['token']; target = by_token[token]; i, k = row['image_offset'], row['kv_offset']
            source = lambda offset: token if offset == 0 else donors[str(offset)][token]
            require(row['route'] == target['route'] and row['command'] == target['command']
                    and row['speed_mps'] == math.hypot(*target['ego'][:2])
                    and np.array_equal(poses(row['target']), poses(target['trajectory_gt'])),
                    'Record current metadata or GT differs from protocol/canonical target')
            require(row['image_source_token'] == source(i) and
                    row['kv_source_token'] == (None if k is None else source(k)), 'Record donor source label differs')
            reference = previous[token] if i == 0 and k in (None, 0) else prior_by_token[token] if i == 0 and k == 4 else None
            if reference is not None:
                require(np.array_equal(poses(row['prediction']), poses(reference['prediction']))
                        and np.array_equal(poses(row['target']), poses(reference['target'])),
                        'Actual previous endpoint/offset4 forecast is not exactly reproduced')
                exact_count += 1
        raw_rows = read_json(directory/'raw_replays.json')
        raw_keys = [(row['target_token'], row['image_source_token']) for row in raw_rows]
        raw_expected = {(token, token if image == 0 else donors[str(image)][token])
                        for token in plan['raw_targets'] for image in OFFSETS}
        require(len(raw_rows) == len(set(raw_keys)) == 32 and set(raw_keys) == raw_expected,
                'Raw replay controls are incomplete or duplicated')
        for row in raw_rows:
            token = row['target_token']
            image = next(i for i in OFFSETS if (token if i == 0 else donors[str(i)][token]) == row['image_source_token'])
            cached = lookup[token, image, None if arm == ARMS[0] else 0]['prediction']
            require(row.get('arm') == arm and row.get('max_abs_difference') == 0
                    and row.get('target_ego_preserved') is True and row.get('target_command_preserved') is True
                    and np.array_equal(poses(row['raw_prediction']), poses(cached))
                    and np.array_equal(poses(row['cached_prediction']), poses(cached)),
                    'Actual raw/cached Processor predictions differ')
            require(len(row.get('image_original_sizes', [])) == len(row.get('image_audit', [])) == 4,
                    'Raw four-frame audit is incomplete')
            raw_count += 1
        all_records[arm] = rows; results[arm] = result
    require(exact_count == 768 and raw_count == 64, 'Combined numeric replay acceptance differs')
    acceptance = dict(records_verified=sum(map(len, all_records.values())), saved_exact_predictions_verified=exact_count,
                      raw_replays_verified=raw_count, all_factorial_sources_and_current_metadata_verified=True,
                      all_buffer_immutability_verified=True, vision_precision_policy='native_fp32_visual_rope',
                      checkpoint_files_rehashed=False,
                      checkpoint_identity_scope='pinned endpoint verification and authenticated runner load_result_checkpoint; '
                                                'no independent checkpoint binary reload in CPU report',
                      whole_joint_execution='all 16 cells: same six fusion layers, metadata [0,0,0,0,1,1]; '
                                            'authenticated runner source and frozen protocol',
                      parameter_and_input_immutability='actual runner before/after parameter and buffer hashes, native FP32 visual RoPE and tensor-version receipts')
    return dict(protocol=plan, records=all_records, acceptance=acceptance, runner_results=results,
                input_sha256=input_files, runner_source_sha256=source_hashes, run_root=str(root))


CONTRASTS = {
    'full_fast_only': {'visual_damage': {'swapped': 1., 'original': -1.}},
    'full_correct_kv': {
        'B_minus_A': {'B': 1., 'A': -1.},
        'C_minus_A': {'C': 1., 'A': -1.},
        'D_same_minus_C': {'D_same': 1., 'C': -1.},
        'D_independent_minus_C': {'D_independent': 1., 'C': -1.},
        'interaction_D_same_minus_C_minus_B_plus_A': {'D_same': 1., 'C': -1., 'B': -1., 'A': 1.},
    },
}


def cohorts(windows):
    """Prespecified current-speed cohorts and descriptive current metadata strata."""
    return {
        'all': list(windows),
        'moving_ge_1': [w for w in windows if w['speed_mps'] >= 1.],
        'high_ge_10': [w for w in windows if w['speed_mps'] >= 10.],
        **{f'command_{command}': [w for w in windows if w['command'] == command]
           for command in sorted({w['command'] for w in windows})},
        'speed_0_to_1': [w for w in windows if 0. <= w['speed_mps'] < 1.],
        'speed_1_to_5': [w for w in windows if 1. <= w['speed_mps'] < 5.],
        'speed_5_to_10': [w for w in windows if 5. <= w['speed_mps'] < 10.],
        'speed_ge_10': [w for w in windows if w['speed_mps'] >= 10.],
    }


def describe_metrics(windows, section):
    result = {}
    if not windows:
        return result
    for cell in windows[0][section]:
        values = [w[section][cell] for w in windows]
        by_log = defaultdict(list)
        for window in windows:
            by_log[window['log']].append(window[section][cell])
        per_log = {log: average_metrics(rows) for log, rows in sorted(by_log.items())}
        result[cell] = dict(window_weighted=average_metrics(values), equal_log=average_metrics(per_log.values()),
                            per_log=per_log)
    return result


def csv_write(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def render_figures(output, summary):
    """Render only the prespecified figures, with no data-driven selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 9})
    cohort_names = ['all', 'moving_ge_1', 'high_ge_10']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, arm, contrast, title in zip(axes, ARMS, ('visual_damage', 'C_minus_A'),
                                      ('Fast-only image sensitivity', 'Joint image sensitivity (correct KV)')):
        for index, name in enumerate(cohort_names):
            effect = summary['cohorts'][name][arm]['effects'].get(contrast, {}).get('ADE')
            if effect:
                value = effect['equal_log_mean']
                ax.bar(index, value, color='#4477aa')
                if effect['ci95'] is None:
                    ax.annotate('1 log; no CI', (index, value), xytext=(0, 4),
                                textcoords='offset points', ha='center', fontsize=7)
                else:
                    lo, hi = effect['ci95']
                    ax.vlines(index, lo, hi, color='black'); ax.hlines([lo, hi], index-.07, index+.07, color='black')
        ax.axhline(0, color='black', linewidth=.6); ax.axhline(.02, color='#aa7733', linestyle=':', label='20 mm reference')
        ax.set_xticks(range(3), ['All', 'Speed >=1', 'Speed >=10']); ax.set_ylabel('Paired ADE damage (m)')
        ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle('Equal-log means; conditional 95% intervals require >=2 logs (2-3 logs: fragile)')
    fig.tight_layout(); fig.savefig(output/'image_sensitivity.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    available = []
    for name in ('all', 'moving_ge_1'):
        offsets = summary['cohorts'][name][ARMS[1]]['offsets']
        available.append(np.array([[offsets[f'{i},{k}']['equal_log']['ADE'] for k in OFFSETS]
                                  for i in OFFSETS]) if offsets else None)
    maximum = max((a.max() for a in available if a is not None), default=1.)
    for ax, name, values in zip(axes, ('All', 'Current speed >=1 m/s'), available):
        if values is None:
            ax.set_title(name+' (empty)'); ax.axis('off'); continue
        picture = ax.imshow(values, vmin=0, vmax=maximum, cmap='viridis')
        for (i, k), value in np.ndenumerate(values):
            ax.text(k, i, f'{value:.3f}', ha='center', va='center', color='white' if value < maximum*.6 else 'black')
        ax.set_xticks(range(4), ['Target', '+1', '+4', '+7']); ax.set_yticks(range(4), ['Target', '+1', '+4', '+7'])
        ax.set_xlabel('KV source recording offset'); ax.set_ylabel('Four-frame image source recording offset')
        ax.set_title(name); fig.colorbar(picture, ax=ax, label='Equal-log ADE (m)')
    fig.tight_layout(); fig.savefig(output/'joint_4x4_ADE.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ('A', 'B', 'C', 'D_same', 'D_independent')
    for index, (name, label) in enumerate((('all', 'All'), ('moving_ge_1', 'Current speed >=1'))):
        table = summary['cohorts'][name][ARMS[1]]['cells']
        if table:
            ax.bar(np.arange(5)+(index-.5)*.36, [table[c]['equal_log']['ADE'] for c in labels], width=.36, label=label)
    ax.set_xticks(range(5), ['A', 'B', 'C', 'D same', 'D different']); ax.set_ylabel('Equal-log ADE (m)')
    ax.set_title('Joint input cells; donor errors averaged within target first'); ax.legend()
    fig.tight_layout(); fig.savefig(output/'joint_cells_ADE.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fast = summary['cohorts']['all'][ARMS[0]]['effects']['visual_damage']['ADE']
    logs = sorted(fast['per_log_means']); x = np.arange(len(logs))
    axes[0].bar(x, [fast['per_log_means'][log] for log in logs], color='#4477aa')
    axes[0].set_title('Fast-only image damage: swapped - original')
    for name, label, marker in [('C_minus_A', 'Image damage C-A', 'o'),
                                 ('D_same_minus_C', 'Correct-KV rescue D_same-C', 's'),
                                 ('D_independent_minus_C', 'Correct-KV rescue D_different-C', '^')]:
        effect = summary['cohorts']['all'][ARMS[1]]['effects'][name]['ADE']
        axes[1].plot(x, [effect['per_log_means'][log] for log in logs], marker=marker, label=label)
    axes[1].set_title('Joint paired image damage / correct-KV rescue'); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.6); ax.set_xticks(x, [f'Log {i+1}' for i in x], rotation=35)
        ax.set_ylabel('Within-log paired ADE difference (m)')
    fig.tight_layout(); fig.savefig(output/'per_log_damage_rescue.png'); plt.close(fig)


def chinese_report(summary):
    rows = ['# 冻结图像 × Slow KV 交叉干预报告', '',
            '本报告在固定 VAL256／8 个 recording group、固定 donor 映射和冻结检查点上作离线机制诊断。'
            '没有新增训练、下载、Head 或网络层。全部指标由保存的八步 XY／航向预测以 CPU NumPy 独立重算。', '',
            '每条预测先计算误差，再在同一目标窗口内平均 donor；随后计算窗口等权与日志等权结果。'
            '绝不先平均轨迹再计算损失，也不把三个 donor 当作独立样本。', '',
            '原 Fast-only 256、联合 A 256、旧 B(offset4) 256 条预测逐数组精确复现；'
            '64 个原始 Processor 回放的原始／缓存预测和当前结果逐数组相同。'
            '检查点身份与不可变性来自已认证 runner 的实际加载、参数及 buffer 前后 hash 和版本收据；'
            '原生视觉 RoPE 使用固定 FP32 buffer 精度策略。本 CPU 报告不读取模型二进制或原图。', '',
            '## 当前速度子集与权重', '',
            '|子集|窗口数|日志数|每日志窗口数|', '|---|---:|---:|---|']
    for name in ('all', 'moving_ge_1', 'high_ge_10'):
        cohort = summary['cohorts'][name][ARMS[1]]
        rows.append(f"|{name}|{cohort['window_count']}|{cohort['log_count']}|{json.dumps(cohort['per_log_window_counts'], ensure_ascii=False)}|")
    rows += ['', '子集由当前 Ego 速度定义：移动 ≥1 m/s、高速 ≥10 m/s；不使用未来 GT 或预测误差筛选。'
             '子集改变日志构成及每日志窗口数，窗口等权与日志等权均保存在 JSON／CSV。', '',
             '## 配对 ADE 差值', '',
             '正的 visual_damage 或 C−A 表示换图增加误差；B−A 表示正确图像下正确 KV 的收益；'
             'D−C 为同一图像 donor 下正确 KV 的补偿量，正值表示正确 KV 降低误差。'
             'D_independent 仅表示图像与 KV 来自不同 donor，并非独立随机抽样。', '',
             '|子集|模型／差值|窗口均值 mm|日志均值 mm|条件日志 bootstrap 95% 区间 mm|',
             '|---|---|---:|---:|---|']
    for cohort in ('all', 'moving_ge_1', 'high_ge_10'):
        for arm in ARMS:
            for name, metrics in summary['cohorts'][cohort][arm]['effects'].items():
                effect = metrics['ADE']
                if effect['ci95'] is None:
                    interval = '不提供区间（仅 1 个日志；只报告描述性均值）'
                else:
                    lo, hi = effect['ci95']
                    interval = f'[{lo*1000:.3f}, {hi*1000:.3f}]'
                    if effect['interval_caution']:
                        interval += f"（仅 {effect['log_count']} 个日志，区间不稳）"
                rows.append(f"|{cohort}|{arm}/{name}|{effect['window_weighted_mean']*1000:.3f}|{effect['equal_log_mean']*1000:.3f}|{interval}|")
    rows += ['', '交互量定义为 D_same−C−B+A；先用相同图像 donor 配对 D−C，'
             '再于窗口内平均。D_same 与不同 donor 的 D 分开报告。', '',
             '区间用固定种子 20261109、20,000 次日志有放回重采样，仅条件于固定目标日志和 donor 映射。'
             '仅一个日志时不提供区间，只保留描述性均值；只有 2–3 个日志时区间不稳，需结合逐日志数量与权重审慎阅读。'
             '它不包含 donor 选择或训练种子不确定性；不显著不等于等效。20 mm 只是沿用的描述性实用参考，'
             '不是选择实验、判定显著性或强制归类的阈值。', '',
             '## 四图像来源 × 四 KV 来源', '', '|图像 offset／KV offset|0|1|4|7|', '|---|---:|---:|---:|---:|']
    offsets = summary['cohorts']['all'][ARMS[1]]['offsets']
    for image in OFFSETS:
        rows.append('|'+str(image)+'|'+'|'.join(f"{offsets[f'{image},{kv}']['equal_log']['ADE']:.6f}" for kv in OFFSETS)+'|')
    rows += ['', '表中为日志等权 ADE（m）。完整逐窗口、逐 offset、逐日志、逐速度档和导航类结果包含 ADE、FDE、'
             '0.5–4 秒八个时距的 L2、周期航向 MAE（rad），以及相对同模型当前输入基线的平均 XY 轨迹变化（m）。', '',
             '## 分层数量与边界', '', '|分层|窗口数|日志数|', '|---|---:|---:|']
    for name, cohort in summary['cohorts'].items():
        if name.startswith(('command_', 'speed_')):
            value = cohort[ARMS[1]]
            rows.append(f"|{name}|{value['window_count']}|{value['log_count']}|")
    rows += ['', '导航 command=2 只有一个窗口，不作该类别的稳定结论。其他稀疏分层也仅描述实际数量和观测值。'
             '日志编号依排序后的 recording group 对应 `per_log_metrics.csv`。', '',
             '错配可能是 OOD。Slow KV 包含图像、Ego、导航及旧反馈，所以补偿不能直接归因于独立视觉语义。'
             '两个模型检查点不同，绝对分数差不能归因于某一路。固定观测日志不支持真实闭环效果、'
             '在线补偿可靠性、未见日志泛化或状态捷径的确证；本报告不强行套用预设机制模式。', '',
             '图表：`image_sensitivity.png`、`joint_4x4_ADE.png`、`joint_cells_ADE.png`、'
             '`per_log_damage_rescue.png`。输入 hash、源代码 provenance、runner 验收收据与报告代码 hash 见 `provenance.json`。', '']
    return '\n'.join(rows)


def run(run_root, output, *, project_root=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior evidence; use a new report directory')
    evidence = load_run(run_root, project_root=project_root)
    windows = {arm: collapse_cells(evidence['records'][arm], arm) for arm in ARMS}
    sliced = {arm: cohorts(values) for arm, values in windows.items()}
    summary = dict(schema='frozen_cross_inputs_report_v1', completed=True, cohorts={},
                   acceptance=evidence['acceptance'], analysis_seed=20261109, bootstrap_replicates=20000,
                   practical_ade_reference_m=.02, practical_reference_role='descriptive prior reference only; no selection or equivalence gate',
                   metric_units={name: 'radians' if name == 'heading_MAE' else 'metres' for name in METRICS},
                   limitations=['observed fixed VAL256/8 logs and fixed donor maps', 'conditional log bootstrap only; no donor-selection or training-seed uncertainty',
                                'mismatches can be OOD; KV contains images, ego, navigation and old feedback',
                                'different model checkpoints; absolute arm differences are not pathway attribution',
                                'no closed-loop, online reliability or unseen-log claim; nonsignificance is not equivalence'])
    cell_rows = []; offset_rows = []; offset_summary_rows = []; per_log_rows = []; effect_rows = []; per_log_effect_rows = []
    for arm, values in windows.items():
        for window in values:
            metadata = {key: window[key] for key in ('token', 'log', 'route', 'command', 'speed_mps')}
            for cell, metrics in window['cells'].items():
                cell_rows.append(dict(arm=arm, **metadata, cell=cell, **{key: metrics[key] for key in METRICS}))
            for pair, metrics in window['offsets'].items():
                image, kv = pair.split(',')
                offset_rows.append(dict(arm=arm, **metadata, image_offset=image, kv_offset=kv,
                                        **{key: metrics[key] for key in METRICS}))
    for name in sliced[ARMS[0]]:
        cohort_summary = {}
        if name.startswith('command_'):
            cohort_summary['stable_class_conclusion_supported'] = False
            cohort_summary['class_scope'] = 'descriptive fixed observed stratum; no stable class conclusion, especially command=2 N=1'
        for arm in ARMS:
            values = sliced[arm][name]; counts = dict(sorted(Counter(w['log'] for w in values).items()))
            table = describe_metrics(values, 'cells'); offsets = describe_metrics(values, 'offsets'); effects = {}
            for contrast, coefficients in CONTRASTS[arm].items():
                if not values:
                    continue
                effects[contrast] = {}
                for metric in METRICS:
                    effect = paired_effect(values, coefficients, metric=metric)
                    effects[contrast][metric] = effect
                    effect_rows.append(dict(cohort=name, arm=arm, contrast=contrast, metric=metric,
                                            window_count=effect['window_count'], log_count=effect['log_count'],
                                            window_weighted_mean=effect['window_weighted_mean'], equal_log_mean=effect['equal_log_mean'],
                                            ci95_low=None if effect['ci95'] is None else effect['ci95'][0],
                                            ci95_high=None if effect['ci95'] is None else effect['ci95'][1],
                                            ci95_unavailable_reason=effect['ci95_unavailable_reason'],
                                            interval_caution=effect['interval_caution']))
                    for log, value in effect['per_log_means'].items():
                        per_log_effect_rows.append(dict(cohort=name, arm=arm, contrast=contrast, metric=metric,
                                                        log=log, window_count=counts[log], mean=value))
            for cell, metrics in table.items():
                for log, means in metrics['per_log'].items():
                    per_log_rows.append(dict(cohort=name, arm=arm, cell=cell, log=log, window_count=counts[log],
                                             **{key: means[key] for key in METRICS}))
            for pair, metrics in offsets.items():
                image, kv = pair.split(',')
                for weighting in ('window_weighted', 'equal_log'):
                    offset_summary_rows.append(dict(cohort=name, arm=arm, image_offset=image, kv_offset=kv,
                                                    weighting=weighting, window_count=len(values), log_count=len(counts),
                                                    **{key: metrics[weighting][key] for key in METRICS}))
            cohort_summary[arm] = dict(window_count=len(values), log_count=len(counts), per_log_window_counts=counts,
                                       cells=table, offsets=offsets, effects=effects)
        summary['cohorts'][name] = cohort_summary
    output.mkdir(parents=True)
    provenance = {key: evidence[key] for key in ('acceptance', 'runner_results', 'input_sha256', 'runner_source_sha256', 'run_root')}
    provenance.update(report_source_sha256=sha256(__file__), protocol=evidence['protocol'])
    write_json(output/'summary.json', summary); write_json(output/'provenance.json', provenance)
    write_json(output/'window_metrics.json', windows)
    csv_write(output/'cell_metrics.csv', cell_rows, ['arm', 'token', 'log', 'route', 'command', 'speed_mps', 'cell', *METRICS])
    csv_write(output/'offset_metrics.csv', offset_rows, ['arm', 'token', 'log', 'route', 'command', 'speed_mps', 'image_offset', 'kv_offset', *METRICS])
    csv_write(output/'offset_summary.csv', offset_summary_rows, ['cohort', 'arm', 'image_offset', 'kv_offset', 'weighting', 'window_count', 'log_count', *METRICS])
    csv_write(output/'per_log_metrics.csv', per_log_rows, ['cohort', 'arm', 'cell', 'log', 'window_count', *METRICS])
    csv_write(output/'effects.csv', effect_rows, ['cohort', 'arm', 'contrast', 'metric', 'window_count', 'log_count', 'window_weighted_mean', 'equal_log_mean', 'ci95_low', 'ci95_high', 'ci95_unavailable_reason', 'interval_caution'])
    csv_write(output/'per_log_effects.csv', per_log_effect_rows, ['cohort', 'arm', 'contrast', 'metric', 'log', 'window_count', 'mean'])
    (output/'report_zh.md').write_text(chinese_report(summary))
    render_figures(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Completed cross-input run root containing protocol.json')
    parser.add_argument('--output', required=True, help='Fresh output directory')
    arguments = parser.parse_args()
    result = run(arguments.run, arguments.output)
    print(json.dumps(dict(completed=result['completed'], acceptance=result['acceptance']), indent=2))


if __name__ == '__main__':
    main()
