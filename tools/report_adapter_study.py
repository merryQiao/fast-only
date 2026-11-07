"""CPU-only analysis and portable evidence for the locked paired adapter study.

No model imports, checkpoint deserialization, network requests, or model fitting.
Binary model artifacts are authenticated by SHA256 and never copied to evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from slowfast_v11.adapter_statistics import paired_gain_matrix, paired_adapter_statistics
from tools.report_slowfast_readability import pose_metrics, window_ade, speed_bin, sha256, read_json, read_lines

SEEDS = (42, 43, 44)
ARMS = ('correct', 'reference')
MODES = ('correct', 'off', 'reference', 'shuffled')
TRAINING_REVISION = '22ced6a'
COHORT_PIN_REVISION = '61f323f'
TOLERANCE = 1e-5


def portable(value):
    """Remove machine-local paths, including paths embedded in error messages."""
    if isinstance(value, dict):
        return {portable(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, str):
        if value.startswith('/'):
            return '<local>/' + Path(value).name
        return re.sub(r'(?<![:/\w])/(?:[^\s,;\]\[{}"\']+/)+[^\s,;\]\[{}"\']*',
                      lambda match: '<local>/' + Path(match.group()).name, value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(path, expected):
    require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected), 'Missing artifact SHA256')
    actual = sha256(path)
    require(actual == expected, f'Artifact SHA256 differs: {Path(path).name}')
    return actual


def recording_group(name):
    return re.sub(r'_\d+_\d+$', '', name)


def validate_cohorts(rows):
    require(len({r['token'] for r in rows}) == len(rows), 'Duplicate cohort token')
    held = [r for r in rows if r['cohort'] == 'holdout']
    old = [r for r in rows if r['cohort'] != 'holdout']
    logs = sorted({r['route'] for r in held})
    old_groups = {recording_group(r['route']) for r in old}
    held_groups = {recording_group(log) for log in logs}
    require(not old_groups & held_groups, 'New/old recording group boundary overlaps')
    require(len(held) == 256 and len(logs) == len(held_groups) == 8
            and Counter(r['route'] for r in held) == Counter({log: 32 for log in logs}),
            'Fixed eight recording groups x32 windows coverage differs')
    require(all(r['split'] == 'val' for r in held), 'Holdout split boundary differs')
    require(Counter(r['cohort'] for r in old) == {'train': 256, 'new_val': 64, 'legacy_val': 4},
            'Historical TRAIN256/VAL68 coverage differs')
    require(all(r['split'] == ('train' if r['cohort'] == 'train' else 'val') for r in old),
            'Historical train/validation split differs')
    return dict(holdout_windows=256, holdout_recording_groups=logs,
                previous_recording_groups=sorted(old_groups), recording_groups_disjoint=True,
                old_counts=dict(Counter(r['cohort'] for r in old)))


def assemble_windows(rows, predictions, *, cohort):
    selected = [r for r in rows if r['cohort'] == cohort]
    require(selected and len({r['token'] for r in selected}) == len(selected), 'Cohort token coverage differs')
    expected = {(r['token'], mode) for r in selected for mode in MODES}
    lookups = {}
    for seed in SEEDS:
        for arm in ARMS:
            records = [r for r in predictions[str(seed)][arm] if r['cohort'] == cohort]
            lookup = {(r['token'], r['mode']): r for r in records}
            require(len(lookup) == len(records), 'Duplicate prediction condition')
            require(set(lookup) == expected, f'Prediction condition coverage differs: {seed}/{arm}/{cohort}')
            lookups[seed, arm] = lookup
    windows = []
    for row in selected:
        token = row['token']
        speed = float(np.hypot(*row['ego'][:2]))
        window = dict(token=token, cohort=cohort, route=row['route'],
                      recording_group=recording_group(row['route']), timestamp_us=row.get('timestamp_us'),
                      command=row['command'], speed_mps=speed, speed_bin=speed_bin(speed),
                      target=row['trajectory_gt'], predictions={}, ADE={})
        off = None
        for seed in SEEDS:
            window['predictions'][str(seed)] = {}
            window['ADE'][str(seed)] = {}
            for arm in ARMS:
                poses = {}
                for mode in MODES:
                    record = lookups[seed, arm][token, mode]
                    require(record['target'] == row['trajectory_gt'], 'Prediction target changed')
                    require(record['route'] == row['route'] and record['command'] == row['command']
                            and abs(record['speed_mps'] - speed) <= 1e-8, 'Prediction metadata changed')
                    poses[mode] = record['prediction']
                require(off is None or poses['off'] == off, 'Off Fast baselines differ across arms/seeds')
                off = poses['off']
                window['predictions'][str(seed)][arm] = poses
                window['ADE'][str(seed)][arm] = {mode: window_ade(poses[mode], row['trajectory_gt']) for mode in MODES}
        windows.append(window)
    return windows


def recompute_metric_error(records, stored):
    maximum = 0.
    for mode in MODES:
        selected = [r for r in records if r['mode'] == mode]
        actual = pose_metrics([r['prediction'] for r in selected], [r['target'] for r in selected])
        # Match every exported metric; the previous generic CPU utility omits this one.
        p = np.asarray([r['prediction'] for r in selected], dtype=np.float64)
        t = np.asarray([r['target'] for r in selected], dtype=np.float64)
        actual['loss_l1_xy'] = float(np.abs(p[..., :2] - t[..., :2]).sum(-1).mean())
        require(set(stored[mode]).issubset(actual), 'Unknown saved metric')
        for key, saved in stored[mode].items():
            error = abs(actual[key] - saved)
            require(np.isfinite(error), 'Nonfinite saved/recomputed metric')
            maximum = max(maximum, error)
    require(maximum <= TOLERANCE, 'Saved/recomputed metric disagreement exceeds1e-5')
    return maximum


def validate_source_tokens(predictions, donors, reference):
    for seed in SEEDS:
        for arm in ARMS:
            for row in predictions[str(seed)][arm]:
                expected = {'off': None, 'correct': row['token'], 'reference': reference,
                            'shuffled': donors[row['token']]}[row['mode']]
                require(row['source_token'] == expected, 'Saved source intervention/donor token differs')


def primary_analysis(windows):
    logs = sorted({r['route'] for r in windows})
    require(len(windows) == 256 and len(logs) == 8, 'Primary analysis needs the complete eight-log cohort')
    paired = [dict(seed=seed, log=r['route'], token=r['token'],
                   ade_correct=r['ADE'][str(seed)]['correct']['correct'],
                   ade_reference=r['ADE'][str(seed)]['reference']['reference'])
              for seed in SEEDS for r in windows]
    matrix = paired_gain_matrix(paired, seeds=SEEDS, logs=logs)
    result = paired_adapter_statistics(matrix, seeds=SEEDS, logs=logs, replicates=20_000,
                                       analysis_seed=20261107, threshold_m=.02)
    result.update(primary='ADE(reference-trained/reference-eval) - ADE(correct-trained/correct-eval)',
                  averaging='window mean within recording group, equal mean of eight groups, equal mean of three fusion seeds',
                  bootstrap='20,000 paired Cartesian resamples of seed and recording-group indices; percentile95%',
                  window_independence_assumed=False, full_model_training_seeds=False,
                  threshold_locked_before_metrics=True)
    result['per_seed_median_log_gain_m'] = np.median(matrix, axis=1).tolist()
    result['per_seed_positive_log_count'] = (matrix > 0).sum(axis=1).tolist()
    result['median_seed_log_cell_gain_m'] = float(np.median(matrix))
    return result


def stratify(windows):
    def summarize(selected, seed):
        metrics = {arm: {mode: pose_metrics([r['predictions'][str(seed)][arm][mode] for r in selected],
                                             [r['target'] for r in selected]) for mode in MODES} for arm in ARMS}
        paired = np.array([r['ADE'][str(seed)]['reference']['reference'] -
                           r['ADE'][str(seed)]['correct']['correct'] for r in selected])
        return dict(windows=len(selected), metrics=metrics,
                    primary_descriptive_window_mean_gain_m=float(paired.mean()),
                    secondary_correct_trained_content_gain_m={
                        mode: float(np.mean([r['ADE'][str(seed)]['correct'][mode] -
                                             r['ADE'][str(seed)]['correct']['correct'] for r in selected]))
                        for mode in ('off', 'reference', 'shuffled')})
    return {str(seed): dict(all=summarize(windows, seed),
                by_log={log: summarize([r for r in windows if r['route'] == log], seed)
                        for log in sorted({r['route'] for r in windows})},
                by_speed={speed: summarize([r for r in windows if r['speed_bin'] == speed], seed)
                          for speed in ('low', 'mid', 'high') if any(r['speed_bin'] == speed for r in windows)})
            for seed in SEEDS}


def validate_gradient_probes(probes):
    require([r['step'] for r in probes] == [1, 2, 128, 512], 'Missing fusion gradient probes')
    for row in probes:
        gradients = row['gradients']
        require(gradients and all(g['finite'] for g in gradients.values()), 'Nonfinite fusion gradient probe')
        # A zero-initialized residual intentionally blocks the upstream branches
        # on update1; the residual output itself must already receive gradients.
        require(all(g['nonzero'] for name, g in gradients.items()
                    if row['step'] != 1 or name.startswith('residual.')),
                'Missing nonzero fusion gradient probe')


def validate_training(root, training, rows):
    train_tokens = {r['token'] for r in rows if r['cohort'] == 'train'}
    require(len(train_tokens) == 256, 'TRAIN256 identity differs')
    logs, health = {}, {}
    require(training['completed'] and training['new_backbone_updates'] == training['new_reader_updates'] == 0
            and not training['holdout_used_for_selection'], 'Training/freeze boundary differs')
    for seed in SEEDS:
        pair = training['seeds'][str(seed)]
        require(pair['correct']['initial_digest'] == pair['reference']['initial_digest']
                and pair['correct']['order'] == pair['reference']['order'], 'Paired initialization/order differs')
        logs[str(seed)], health[str(seed)] = {}, {}
        for arm in ARMS:
            saved = pair[arm]
            sub = root / f'seed_{seed}' / arm
            authenticate(sub / 'final.pt', saved['checkpoint_sha256'])
            authenticate(sub / 'before_final.pt', saved['before_final_sha256'])
            entries = read_lines(sub / 'train.jsonl')
            require(len(entries) == 512 and [r['step'] for r in entries] == list(range(1, 513)), 'Wrong optimizer update budget')
            order = [token for r in entries for token in r['tokens']]
            require(all(len(r['tokens']) == 8 for r in entries) and order == saved['order']
                    and Counter(order) == Counter({token: 16 for token in train_tokens})
                    and saved['new_exposures'] == 4096, 'Training sampler/order/exposure budget differs')
            require(saved['completed'] and saved['frozen_head_unchanged'] and saved['frozen_features_unchanged']
                    and not saved['val_used_for_selection'], 'Frozen parameter/selection boundary differs')
            for step, r in enumerate(entries, 1):
                require(np.isfinite([r['loss'], r['grad_norm'], r['lr']]).all()
                        and r['loss'] >= 0 and r['grad_norm'] >= 0 and r['lr'] > 0
                        and r['windows_seen'] == step * 8 and r['epochs'] == step / 32,
                        'Invalid optimizer loss/gradient/LR/budget trace')
            probes = [r for r in entries if r.get('gradients')]
            validate_gradient_probes(probes)
            logs[str(seed)][arm] = entries
            health[str(seed)][arm] = dict(updates=512, batch_size=8, exposures=4096, epochs=16,
                    grad_norm_median=float(np.median([r['grad_norm'] for r in entries])),
                    clip_fraction=float(np.mean([r['grad_norm'] > 1 for r in entries])),
                    trainable_parameters=saved['trainable_parameters'],
                    gradient_probes=probes,
                    first_step_zero_upstream_gradients_expected_from_zero_residual=True)
    for arm in ARMS:
        identity = training['seed42_prior_exact'][arm]
        require(identity['weights']['passed'] and identity['old_VAL_predictions']['passed']
                and identity['weights']['max_abs_difference'] == identity['old_VAL_predictions']['max_abs_difference'] == 0,
                'Seed42 prior weights/old VAL predictions are not exact')
    return logs, health


def validate_config(protocol):
    c = protocol['config']
    fixed = dict(schema='adapter_paired_study_v1', train_seeds=list(SEEDS), arms=list(ARMS),
        train_size=256, holdout_logs=8, holdout_windows_per_log=32, steps=512, batch_size=8,
        warmup_steps=32, min_lr_ratio=.1, learning_rate=5e-4, weight_decay=.01, grad_clip=1.,
        curve_steps=[128, 256, 384, 512], fusion=dict(reader_width=768, hidden=256, gate_bias=-2.))
    require(all(c.get(k) == value for k, value in fixed.items()), 'Locked paired study configuration differs')
    a = c['analysis']
    require(a['bootstrap_replicates'] == 20_000 and a['analysis_seed'] == 20261107 and a['confidence'] == .95
            and a['practical_mean_gain_m'] == .02 and a['require_positive_CI_lower_bound']
            and a['bootstrap'] == 'paired_cartesian_seed_and_log'
            and protocol['holdout_never_in_optimizer'], 'Locked primary analysis differs')
    require(c['old_feature_bank_sha256'] == protocol['training_bank_sha256']
            and c['old_rows_sha256'] == protocol['training_rows_sha256'], 'Historical feature bank identity differs')
    return c


def validate_replay(verify):
    require(verify['passed'] and verify['new_network_bytes'] == 0, 'Reload/final-update verification failed')
    for seed in SEEDS:
        for arm in ARMS:
            check = verify['comparisons'][str(seed)][arm]
            require(check['conditions'] == 1024, 'Full holdout reload coverage differs')
            full = check['full_holdout_predictions']
            require(full['passed'] and full['max_abs_difference'] == 0, 'Holdout reload predictions are not exact')
            replay = check['actual_step_replay']
            require(set(replay) == {'fusion', 'optimizer', 'scheduler', 'rng', 'reference_prediction'}
                    and check['before'] == 511 and check['after'] == 512
                    and all(r['passed'] and r['max_abs_difference'] == 0 for r in replay.values()),
                    'Missing exact actual final-update replay')
    return True


def validate_deployment(deployment, expected_tokens):
    require(deployment['passed'] and deployment['no_slow_or_reader']
            and deployment['models_constructed'] == dict(Fast=1, Slow=0, Reader=0)
            and deployment['new_network_bytes'] == 0, 'Fast-only cold deployment boundary failed')
    checks = deployment['checks']
    require(len(checks) == 6 and Counter(r['seed'] for r in checks) == Counter({seed: 2 for seed in SEEDS}),
            'Need two original-image deployment checks for each of three seeds')
    tokens = None
    for seed in SEEDS:
        selected = [r for r in checks if r['seed'] == seed]
        current = {r['token'] for r in selected}
        require(len(current) == 2 and current == set(expected_tokens) and (tokens is None or current == tokens),
                'Cold deployment sample identity differs')
        require(all(r.get('exact', r.get('max_abs_difference') == 0) and r.get('max_abs_difference', 0) == 0 for r in selected),
                'Fixed-bundle/reference-source deployment parity failed')
        require(all(r['native_processor_exact'] and r['original_head_calls'] == 1
                    and len(r['visual_tokens']) == 4 and all(v > 0 for v in r['visual_tokens']) for r in selected),
                'Cold original-image/native-processor/head proof failed')
        tokens = current
    return True


def validate_extraction_retry(receipt):
    require(receipt['passed'] and receipt.get('comparisons'), 'Missing/failed extraction retry verification')
    def visit(value):
        if isinstance(value, dict):
            if 'passed' in value:
                require(value['passed'], 'Failed extraction retry artifact comparison')
            if 'max_abs_difference' in value:
                require(value['max_abs_difference'] == 0, 'Inexact extraction retry artifact comparison')
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(receipt['comparisons'])
    return True


def source_hashes(protocol, features, repo):
    historical = {}
    for name, expected in protocol['source_code_sha256'].items():
        blob = subprocess.run(['git', 'show', f'{TRAINING_REVISION}:{name}'], cwd=repo,
                              check=True, capture_output=True).stdout
        digest = hashlib.sha256(blob).hexdigest()
        require(digest == expected, f'Historical training runtime differs: {name}')
        historical[name] = digest
    for name, expected in features['source_code_sha256'].items():
        authenticate(repo / name, expected)
    names = ['tools/report_adapter_study.py', 'tools/train_adapter_study.py', 'tools/extract_adapter_study.py',
             'tools/prepare_adapter_study.py', 'tools/verify_adapter_study.py', 'tools/verify_fixed_adapter.py',
             'slowfast_v11/adapter_statistics.py', 'slowfast_v11/adapter_data.py',
             'slowfast_v11/adapter_features.py', 'slowfast_v11/fixed_adapter.py', 'tools/train_reader_fusion.py',
             'slowfast_v11/reader_fusion.py', 'configs/slowfast_adapter_study.yaml', 'configs/slowfast_adapter_cohort.json']
    return dict(training_runtime_revision=TRAINING_REVISION, training_runtime=historical,
                extraction_runtime=features['source_code_sha256'], current_reporting_and_evaluation={name: sha256(repo / name) for name in names},
                note='Historical training source authenticated from its Git blob; later evaluation/authentication hardening recorded separately.')


def validate_data(data, rows, features):
    plan, frozen = read_json(data / 'plan.json'), read_json(data / 'frozen_tokens.json')
    audit, budget = read_json(data / 'source_audit.json'), read_json(data / 'network_budget.json')
    requests = read_lines(data / 'network_requests.jsonl')
    pin_path = Path(__file__).resolve().parents[1] / 'configs/slowfast_adapter_cohort.json'
    authenticate(pin_path, features['cohort_pin_sha256'])
    pin = read_json(pin_path)
    committed_pin = subprocess.run(['git', 'show', f'{COHORT_PIN_REVISION}:configs/slowfast_adapter_cohort.json'],
        cwd=pin_path.parents[1], check=True, capture_output=True).stdout
    require(hashlib.sha256(committed_pin).hexdigest() == features['cohort_pin_sha256'],
            'Cohort pin differs from the before-camera Git snapshot')
    for name, key in [('plan.json', 'data_plan_sha256'), ('frozen_tokens.json', 'frozen_tokens_sha256'),
                      ('source_audit.json', 'source_audit_sha256'), ('holdout.jsonl', 'holdout_manifest_sha256')]:
        authenticate(data / name, features[key])
    held = [r for r in rows if r['cohort'] == 'holdout']
    require(frozen['tokens'] == [r['token'] for r in held] and frozen['frozen_before_camera_access']
            and frozen['plan_sha256'] == plan['plan_sha256'] == audit['plan_sha256']
            and plan['selection_before_image_availability'], 'Frozen cohort/token plan differs')
    require(features['data_plan_sha256'] == pin['plan_file_sha256']
            and plan['plan_sha256'] == pin['plan_content_sha256']
            and features['frozen_tokens_sha256'] == pin['tokens_file_sha256']
            and frozen['tokens'] == pin['tokens'] and pin['selection_before_camera_and_model_evaluation'],
            'Before-camera committed cohort pin differs')
    require(held == plan['rows']['val'] == read_lines(data / 'holdout.jsonl'), 'Frozen manifest/feature rows differ')
    require(not audit['full_archives_verified'] and plan['selection_audit']['early_log_portions'], 'Data authenticity/sampling scope differs')
    logs = sorted({r['route'] for r in held})
    require(logs == sorted(plan['splits']['holdout']) and len({plan['logs'][log]['shard'] for log in logs}) == 8,
            'Eight fixed log/shard identity differs')
    groups = {log: sorted([r for r in held if r['route'] == log], key=lambda r: r['timestamp_us']) for log in logs}
    donors = {r['token']: groups[logs[(i + 4) % 8]][j]['token'] for i, log in enumerate(logs) for j, r in enumerate(groups[log])}
    require(features['donors'] == donors and len(set(donors.values())) == 256, 'Frozen cross-recording ordinal donor mapping differs')
    require(all(r['response_body_bytes'] >= 0 for r in requests)
            and len({r['request_id'] for r in requests}) == len(requests), 'Invalid/duplicate network request accounting')
    used = sum(r['response_body_bytes'] for r in requests)
    metadata = sum(r['response_body_bytes'] for r in requests if r['kind'] == 'metadata')
    camera = sum(r['response_body_bytes'] for r in requests if r['kind'] == 'camera')
    require(used == budget['used'] == budget['new_response_body_bytes'] == features['new_network_response_body_bytes']
            and metadata == budget['metadata_response_body_bytes'] and camera == budget['camera_response_body_bytes']
            and used == metadata + camera and used <= budget['limit'] == 2_000_000_000
            and budget['cumulative_response_body_bytes'] == used + budget['inherited_previous_response_body_bytes'],
            'New/cumulative response-body byte accounting differs')
    authenticate(data / 'network_requests.jsonl', budget['request_ledger_sha256'])
    camera_urls = {r['url'] for r in requests if r['kind'] == 'camera'}
    require(len(camera_urls) == 8 and not budget['camera_cap_overrides']
            and all(sum(r['response_body_bytes'] for r in requests if r['kind'] == 'camera' and r['url'] == url) <= 225_000_000
                    for url in camera_urls), 'Camera archive count/cap accounting differs')
    amendment = read_json(data / 'metadata_prefix_amendment.json')
    failure = read_json(data / 'selection_failure_280000000.json')
    require(failure['available_recording_groups'] == 7 and failure['required_recording_groups'] == 8 and not failure['cohort_frozen']
            and amendment['requested_metadata_prefix_cap_bytes'] == budget['metadata_prefix_cap_bytes'] == 320_000_000
            and amendment['requested_metadata_new_body_cap_bytes'] == budget['metadata_new_cap_bytes'] == 200_000_000
            and metadata <= 200_000_000 and budget['camera_prefix_cap_bytes'] == 225_000_000,
            'Documented pre-camera metadata amendment differs')
    return dict(plan=plan, frozen_tokens=frozen, source_audit=audit, network_budget=budget,
                network_requests=requests, metadata_prefix_amendment=amendment, original_280MB_selection_failure=failure,
                before_camera_cohort_pin=dict(revision=COHORT_PIN_REVISION, file_sha256=features['cohort_pin_sha256'], content=pin))


def make_plots(output, primary, summary, training):
    matrix = np.asarray(primary['paired_gain_matrix_m'])
    labels = [f'L{i + 1}' for i in range(8)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout='constrained')
    limit = max(float(np.abs(matrix).max()), .02)
    image = axes[0].imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    for i in range(3):
        for j in range(8):
            axes[0].text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center', fontsize=8,
                         color='white' if abs(matrix[i,j]) > .6 * limit else 'black')
    axes[0].set(xticks=range(8), xticklabels=labels, yticks=range(3), yticklabels=SEEDS,
                title='Paired ADE gain (m)', xlabel='New recording group', ylabel='Fusion seed')
    fig.colorbar(image, ax=axes[0], label='Reference minus matched source (m)')
    axes[1].bar(labels, matrix.mean(0), color='C0')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].axhline(.02, color='C3', ls='--', label='Declared 2 cm threshold')
    axes[1].set(title='Mean of three paired fusion seeds', ylabel='ADE gain (m)')
    axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=.2)
    fig.savefig(output / 'paired_seed_log_gains.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    for ax, seed in zip(axes, SEEDS):
        for arm, offset in [('correct', -.18), ('reference', .18)]:
            values = [summary['holdout'][str(seed)]['all']['metrics'][arm][mode]['ADE'] for mode in MODES]
            ax.bar(np.arange(4) + offset, values, width=.36, label=arm + '-trained')
        ax.set(title=f'New holdout: fusion seed {seed}', ylabel='ADE (m)', xticks=range(4), xticklabels=MODES)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'four_source_conditions.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for seed in SEEDS:
        for arm in ARMS:
            entries = training[str(seed)][arm]; width = 20
            values = [r['loss'] for r in entries]
            axes[0].plot(range(width, 513), np.convolve(values, np.ones(width) / width, mode='valid'), label=f'{seed}/{arm}', lw=1)
    axes[0].set(title='TRAIN256 loss: 20-update moving mean', xlabel='Optimizer update', ylabel='Pose L1 (m/rad)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.2)
    axes[1].plot(range(1, 513), [r['lr'] for r in training['42']['correct']], color='C0')
    axes[1].set(title='Fixed 512-update learning-rate schedule', xlabel='Optimizer update', ylabel='AdamW learning rate')
    axes[1].grid(alpha=.2)
    fig.savefig(output / 'training_traces.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for ax, cohort in zip(axes, ('new_val', 'legacy_val')):
        for arm, offset in [('correct', -.18), ('reference', .18)]:
            values = [np.mean([summary['old_val'][cohort][str(seed)]['all']['metrics'][arm][mode]['ADE'] for seed in SEEDS]) for mode in MODES]
            ax.bar(np.arange(4) + offset, values, width=.36, label=arm + '-trained')
        ax.set(title=f'Historical diagnostic only: {cohort}', ylabel='Mean fusion-seed ADE (m)', xticks=range(4), xticklabels=MODES)
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(output / 'historical_val_diagnostics.png', dpi=160); plt.close(fig)


def write_document(document, output, primary, summary, verification, data, config):
    mean = primary['mean_gain_m']; lo, hi = primary['ci95_m']
    verdict = '达到' if primary['passes_primary_gate'] else '未达到'
    gate_label = 'PASS' if primary['passes_primary_gate'] else 'FAIL'
    interval_statement = '区间跨越零；' if lo <= 0 <= hi else ''
    upper_statement = '区间上界也低于预设的 20 mm 实用阈值。' if hi < .02 else ''
    primary_ade = {arm: float(np.mean([summary['holdout'][str(seed)]['all']['metrics'][arm][arm]['ADE']
                                     for seed in SEEDS])) for arm in ARMS}
    off_ade = summary['holdout']['42']['all']['metrics']['correct']['off']['ADE']
    table = ['| Seed | ' + ' | '.join(f'L{i+1}' for i in range(8)) + ' | Mean | Median | Positive logs |',
             '| --- | ' + ' | '.join(['---:'] * 11) + ' |']
    for i, seed in enumerate(SEEDS):
        table.append(f'| {seed} | ' + ' | '.join(f'{x:+.6f}' for x in primary['paired_gain_matrix_m'][i]) +
                     f' | {primary["per_seed_gain_m"][i]:+.6f} | {primary["per_seed_median_log_gain_m"][i]:+.6f} | {primary["per_seed_positive_log_count"][i]}/8 |')
    identities = '\n'.join(f'- L{i+1}: `{name}`' for i, name in enumerate(primary['logs']))
    conditions = ['| Fusion seed | Training source | Correct | Off | Reference | Shuffled |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed in SEEDS:
        for arm in ARMS:
            metrics = summary['holdout'][str(seed)]['all']['metrics'][arm]
            conditions.append(f'| {seed} | {arm} | ' + ' | '.join(f'{metrics[m]["ADE"]:.6f}' for m in MODES) + ' |')
    budget = data['network_budget']
    relative = Path(os.path.relpath(output, Path(document).parent)).as_posix()
    text = f'''# 固定特征 Fast 适配器与新日志配对实验

匹配来源适配器**{verdict}预先声明的主要判据**：平均 ADE 收益为 `{mean:+.6f} m`，种子与日志配对 bootstrap 的 95% 百分位区间为 `[{lo:+.6f}, {hi:+.6f}] m`。正收益定义为 reference 训练/reference 评估的 ADE 减去 correct 训练/correct 评估的 ADE。判据在读取新 holdout 指标前锁定：平均收益至少 `0.02 m`，**且**区间下界大于零。模型、阈值、训练日程和评估队列均未根据本次结果选择或调整。

Reader 条件分支的实用增量判据：**{gate_label}**。实际平均收益为 `{mean * 1000:+.2f} mm`，95% 区间为 `[{lo * 1000:+.2f}, {hi * 1000:+.2f}] mm`；{interval_statement}{upper_statement}三个种子平均的 correct 训练/correct 评估 ADE 为 `{primary_ade['correct']:.6f} m`，reference 训练/reference 评估为 `{primary_ade['reference']:.6f} m`；原始 Fast/off 为 `{off_ade:.6f} m`。固定 reference 基线已吸收大部分相对原始 Fast 的改善，本次证据未达到逐窗口 Reader 条件分支相对该基线所要求的额外收益。此主要比较本身不单独识别视觉信息或来源匹配语义的作用。

主基线是导出的固定来源 Fast+task-adapter，保留残差结构并只在原始 Fast 上在线计算；原始 Fast/off 为次要比较。[固定适配器使用说明](FIXED_ADAPTER_USAGE.md) 给出带哈希认证的加载与推理方法。读取本轮终点结果后，没有延长训练、追加日志/窗口或挑选种子。

## 主要配对证据

先在每个 recording group 内对窗口取均值，再等权平均八个组，最后等权平均三个融合训练种子。区间使用 20,000 次配对重采样：分别独立重采样三个种子索引和八个日志索引，对每次得到的笛卡尔子矩阵取均值，再取 2.5/97.5 百分位数（`analysis_seed=20261107`）。相邻窗口不作为独立重采样单位。跨种子平均后，日志收益中位数为 `{primary['median_log_gain_m']:+.6f} m`，正收益日志数为 `{primary['positive_log_count']}/8`。下表保留全部 3×8 配对效应，单位为米。

{chr(10).join(table)}

{identities}

![种子与日志的配对 ADE 收益]({relative}/paired_seed_log_gains.png)

## 四种来源条件：次要分析

下表报告完整 256 窗口新 holdout 上的 ADE。每个日志均有 32 个窗口，因此此处总体均值也等于日志等权均值。六个训练端点均评估 correct、off、reference、shuffled 四种条件。[分层汇总]({relative}/stratified_summary.json) 保留每个种子、每个日志和每个速度区间的 ADE、FDE、朝向及各预测时刻指标，以及 correct 训练臂的来源干预收益。速度区间为 low <1 m/s、mid 1–<10 m/s、high ≥10 m/s；这些切片仅作描述，不替代预先锁定的主要分析。

{chr(10).join(conditions)}

![四种来源条件]({relative}/four_source_conditions.png)

## 数据、冻结模型与配对预算

本实验是基于固定 OpenScene mini 版本的 NAVSIM-mini 诊断。新队列含八个此前未见的 recording group，每组固定取早期 32 个窗口。选择规则为：按名称排序符合条件的相机分片首日志，排除所有此前见过的 recording group（去除末尾数字分段范围后判断身份），取前八组及各组最早的 32 个有效 stride-five 锚点。全部 12 帧历史、当前及未来位姿均须有限且满足结构有效性；未来运动误差、图像是否存在及模型输出不参与选择。计划与 token 列表在相机访问前冻结。错误来源按排序后的八个日志偏移四位（模八），匹配同一按时间排序的窗口序号，形成跨 recording group 的双射；报告程序也独立重算并核对此映射。图像保留原始 1920×1080 JPEG，模型只输入 2 Hz 的四帧历史/当前图像，没有未来输入或重压缩。

初始 280,000,000 字节元数据前缀只得到七个合格的新 recording group。保留该失败记录后，操作性修订 1 使用原先 40,000,000 字节预留，将同一私有前缀扩展到 320,000,000 字节，并将新增元数据响应体额度提高到 200,000,000 字节。此修订发生在相机访问及模型评估前；八组规则、八个各 225,000,000 字节的相机额度、2,000,000,000 字节新增总上限、训练和分析规则保持固定，没有降为七组。实际新增响应体为 `{budget['new_response_body_bytes']:,}` 字节，其中元数据 `{budget['metadata_response_body_bytes']:,}`、相机 `{budget['camera_response_body_bytes']:,}`；继承此前响应体 `{budget['inherited_previous_response_body_bytes']:,}` 字节，累计 `{budget['cumulative_response_body_bytes']:,}` 字节。计数包含尝试与重试。证据记录前缀、成员及原始图像的来源与哈希，不宣称已认证完整归档。

六次融合训练复用同一已认证的历史 TRAIN256 特征与归一化统计，冻结 Fast、Slow、Reader 和原始任务 head。Fast SHA256 为 `{config['fast_checkpoint_sha256']}`；Reader 为 `{config['reader_checkpoint_sha256']}`；冻结 joint Slow 来源为 `{config['source_checkpoint_sha256']}`。新 Slow 的反馈来自保存的 Fast512 producer，不使用最新 Fast 或真值反馈。256 个新目标的来源布局全部一致，reference 特征恒定。种子 42/43/44 各自训练 correct 与 reference 两臂，同一种子内使用相同初始化与 4096 次有序抽样：512 次更新 × batch 8，即每个 TRAIN 窗口出现 16 次；AdamW 学习率 5e-4、weight decay .01、梯度裁剪 1、32 步 warmup、cosine 降至初始学习率的 .1。没有新增 backbone/Reader 更新、early stopping 或重复调参。

特征提取保留了一次真实失败与重试：第一次已完成全部 256 个窗口的冻结推理并保存五个完整输出，但在最后组装回执、读取 `len(audit['sources'])` 时，逐图像 audit 变量遮蔽了来源审计对象，导致回执生成失败。修复仅重命名两个变量（首次代码 `3edf0b1`，修复 `b1123a2`），未改变权重、数据或分析规则。随后重新执行同一冻结推理，并将第一次保留的特征 bank 张量、元数据及原生输入包与重试结果逐项精确比较；[提取重试回执]({relative}/extraction_retry.json) 保留失败原因、代码版本、原始文件哈希和比较结果，不复制原始 PT。新 holdout 指标在这次修复与重试前尚未计算，也未用于决定重试。

![固定训练轨迹]({relative}/training_traces.png)

## 固定适配器部署与复核

导出的 reference 基线包含融合权重及与认证 Fast checkpoint 绑定的常量 `R0[1,8,768]`。在线路径仍使用当前 Fast 图像、Ego 状态及导航输入，经原始 Fast、残差适配器及原始 head 输出；跳过逐窗口 Slow/Reader 推理及其视觉路径，也无需实时来源 KV。常量保留历史 Slow/Reader 先验，相应历史训练成本仍须计入。实际条件分支的 KV 同时包含视觉、Ego、导航和旧计划信息，因此本比较未单独隔离视觉增量。原始 Fast/off 是次要比较。实际冷启动 Fast-only 推理从原始图像重新处理输入，对每个融合种子检查相同的一个历史 TRAIN 样本和一个新 holdout 样本；全程未实例化 Slow 或 Reader，且与 reference 来源推理逐值完全一致。

六个融合训练均完成实际最后一步优化器更新重放，并完整重载各自 1024 个 holdout 来源条件的预测。种子 42 精确复现此前融合权重和全部 68 个历史验证窗口的预测。CPU 重算指标的最大绝对差为 `{verification['metric_recompute_max_abs_error']:.9g}`，上限为 1e-5。回归测试 `{verification['tests']['counts']['tests']}` 项通过，错误、失败和跳过均为零。历史训练代码通过 Git 版本 `{TRAINING_REVISION}` 的 blob 认证；当前提取、评估、验证及报告代码另存哈希。便携证据仅包含 JSON/JSONL、图表及哈希，不复制权重或原始图像。

便携 `result.json` 保留评估完成时的历史标记 `verified:false`；最终独立复核状态以[报告验证记录]({relative}/verification.json) 的 `passed:true` 及其中三类证明为准。

## 历史诊断与推断范围

此前的 `new_val`（64 窗口、两个日志）与 `legacy_val`（四窗口、一个日志）单列于[历史诊断汇总]({relative}/stratified_summary.json) 和独立预测文件中。它们用于历史诊断和种子 42 身份复核，不进入本次新队列的主要效应矩阵或区间。

![历史验证诊断]({relative}/historical_val_diagnostics.png)

本区间条件于这些固定的 Fast、Slow、Reader 模型，三个种子仅代表融合模块训练随机性，不覆盖 provider、backbone 或 Reader 训练种子的不确定性。mini 中八段日志的早期片段是受传输预算约束的小型便利样本，相邻窗口重叠及同一 recording group 内相关性仍然存在；日志间异质性完整保留在配对矩阵中。本实验不构成整段 episode 表现、官方 NAVSIM navtrain/navtest 分数、异步传递行为或闭环驾驶收益的证据。主要判据结果对应这一次固定队列，不据此扩展队列或重新调参。

随文提供便携[主要统计]({relative}/primary_analysis.json)、[验证记录]({relative}/verification.json)、[源代码哈希]({relative}/source_hashes.json)、[证据文件哈希]({relative}/SHA256.json)、原始预测、训练 JSONL 轨迹、数据来源及四张图表。
'''
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(text)


def run(root, output, document, verification, deployment, tests):
    root, output, document = Path(root), Path(output), Path(document)
    result, protocol = read_json(root / 'result.json'), read_json(root / 'protocol.json')
    training, features = read_json(root / 'training.json'), read_json(root / 'features.json')
    extraction_retry = read_json(root / 'extraction_retry.json')
    validate_extraction_retry(extraction_retry)
    require(result['completed'] and result['protocol'] == protocol == training['protocol']
            and result['training'] == training and result['features'] == features, 'Run artifact identity/completion differs')
    config = validate_config(protocol)
    require(not result['new_holdout_used_for_selection'] and result['new_backbone_updates'] == result['new_Reader_updates'] == 0,
            'Holdout selection/freeze boundary differs')
    require(features['completed'] and features['teacher_saved_prediction_exact'] and features['reference_constant']
            and features['old_train_prediction_exact'] and features['old_Reader_features_exact']
            and features['source_layout_exact_count'] == features['native_head_exact_count'] == 256
            and features['new_backbone_updates'] == features['new_reader_updates'] == 0,
            'Frozen feature/feedback/layout verification differs')
    require(all(features[key] == config[key] for key in ('old_feature_bank_sha256', 'old_rows_sha256',
                'fast_checkpoint_sha256', 'reader_checkpoint_sha256', 'source_checkpoint_sha256'))
            and features['config_sha256'] == protocol['config_sha256'], 'Feature model/data/config identity differs')
    authenticate(Path(__file__).resolve().parents[1] / 'configs/slowfast_adapter_study.yaml', protocol['config_sha256'])
    authenticate(root / 'rows.json', features['rows_sha256'])
    authenticate(root / 'frozen_features.pt', features['feature_bank_sha256'])
    rows = read_json(root / 'rows.json'); cohorts = validate_cohorts(rows)
    parent = Path(config['parent_run'])
    authenticate(parent / 'rows.json', config['old_rows_sha256'])
    authenticate(parent / 'frozen_features.pt', config['old_feature_bank_sha256'])
    require([r for r in rows if r['cohort'] != 'holdout'] == read_json(parent / 'rows.json'),
            'Historical TRAIN/VAL rows changed during holdout extension')
    data = validate_data(Path(config['data_dir']), rows, features)
    train_logs, health = validate_training(root, training, rows)
    verify, deploy = read_json(verification), read_json(deployment)
    validate_replay(verify)
    require(len(features['cold_tokens']) == 2 and len(set(features['cold_tokens'])) == 2,
            'Need exactly two authenticated cold native inputs')
    row_lookup = {r['token']: r for r in rows}
    require(Counter(row_lookup[token]['cohort'] for token in features['cold_tokens']) == {'train': 1, 'holdout': 1},
            'Cold proof needs one historical TRAIN and one new holdout input')
    validate_deployment(deploy, features['cold_tokens'])
    authenticate(root / 'cold_inputs.pt', features['cold_inputs_sha256'])
    authenticate(root / 'cold_expected.json', result['cold_expected_sha256'])
    authenticate(Path(__file__).resolve().parents[1] / 'tools/train_adapter_study.py', result['evaluation_source_sha256'])
    authenticate(Path(__file__).resolve().parents[1] / 'tools/verify_fixed_adapter.py', deploy['source_code_sha256'])
    predictions, old_predictions, maximum = {}, {}, 0.
    for seed in SEEDS:
        predictions[str(seed)], old_predictions[str(seed)] = {}, {}
        for arm in ARMS:
            sub = root / f'seed_{seed}' / arm
            path = sub / 'holdout_predictions.json'
            stored = result['holdout'][str(seed)][arm]
            authenticate(path, stored['predictions_sha256'])
            records = read_json(path)
            require(len(records) == stored['conditions'] == 1024 and all(r['cohort'] == 'holdout' for r in records),
                    'Holdout prediction count/cohort differs')
            maximum = max(maximum, recompute_metric_error(records, stored['summary']))
            predictions[str(seed)][arm] = records
            old = read_json(sub / 'predictions.json')
            require(len(old) == 272 and Counter(r['cohort'] for r in old) == {'new_val': 256, 'legacy_val': 16},
                    'Historical VAL prediction coverage differs')
            for cohort in ('new_val', 'legacy_val'):
                maximum = max(maximum, recompute_metric_error([r for r in old if r['cohort'] == cohort],
                    training['seeds'][str(seed)][arm]['summary'][cohort]))
            old_predictions[str(seed)][arm] = old
        bundle = result['fixed_bundles'][str(seed)]
        authenticate(root / f'seed_{seed}' / 'fixed_adapter.pt', bundle['sha256'])
        require(bundle['source_training_checkpoint'] == training['seeds'][str(seed)]['reference']['checkpoint_sha256'],
                'Fixed adapter source training checkpoint differs')
    windows = assemble_windows(rows, predictions, cohort='holdout')
    validate_source_tokens(predictions, features['donors'], min(r['token'] for r in rows if r['cohort'] == 'train'))
    old_windows = {cohort: assemble_windows(rows, old_predictions, cohort=cohort) for cohort in ('new_val', 'legacy_val')}
    primary = primary_analysis(windows)
    summary = dict(holdout=stratify(windows), old_val={cohort: stratify(values) for cohort, values in old_windows.items()})
    suites = list(ET.parse(tests).getroot().iter('testsuite'))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'errors', 'failures', 'skipped')}
    require(counts['tests'] > 0 and not any(counts[k] for k in ('errors', 'failures', 'skipped')), 'Regression test record is incomplete/failing')
    repo = Path(__file__).resolve().parents[1]
    hashes = source_hashes(protocol, features, repo)
    report_verify = dict(passed=True, matched_holdout_windows=256, matched_holdout_conditions=6144,
        metric_recompute_max_abs_error=maximum, optimizer_health=health, cohort_boundary=cohorts,
        paired_initialization_and_order=True, fixed_update_budget=True, historical_seed42_exact=training['seed42_prior_exact'],
        source_frozen=True, holdout_used_for_selection=False, tests=dict(counts=counts, xml_sha256=sha256(tests)),
        replay=verify, cold_fast_only_deployment=deploy, extraction_retry=extraction_retry,
        extraction_retry_sha256=sha256(root / 'extraction_retry.json'))
    # No write occurs until every mandatory authentication and validation has passed.
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh empty evidence directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('result.json', result), ('protocol.json', protocol), ('config.json', config),
                        ('training.json', training), ('features.json', features), ('verification.json', report_verify),
                        ('primary_analysis.json', primary), ('stratified_summary.json', summary),
                        ('holdout_window_metrics.json', windows), ('historical_val_window_metrics.json', old_windows),
                        ('source_hashes.json', hashes), ('reload_verification.json', verify), ('deployment.json', deploy)]:
        write_json(output / name, value)
    write_json(output / 'extraction_retry.json', extraction_retry)
    for name, value in data.items():
        if name == 'network_requests':
            (output / 'network_requests.jsonl').write_text(''.join(json.dumps(portable(r), allow_nan=False) + '\n' for r in value))
        else:
            write_json(output / f'{name}.json', value)
    for seed in SEEDS:
        for arm in ARMS:
            prefix = f'seed_{seed}_{arm}'
            write_json(output / f'{prefix}_holdout_predictions.json', predictions[str(seed)][arm])
            write_json(output / f'{prefix}_historical_val_predictions.json', old_predictions[str(seed)][arm])
            (output / f'{prefix}_train.jsonl').write_text(''.join(json.dumps(portable(r), allow_nan=False) + '\n' for r in train_logs[str(seed)][arm]))
    make_plots(output, primary, summary, train_logs)
    write_document(document, output, primary, summary, report_verify, data, config)
    index = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'SHA256.json'}
    index['document:' + document.name] = sha256(document)
    write_json(output / 'SHA256.json', index)
    print(json.dumps(dict(completed=True, primary=primary, tests=counts, portable_files=len(index)), allow_nan=False), flush=True)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--tests', '--test-xml', dest='tests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--document', required=True)
    args = parser.parse_args()
    run(args.run, args.output, args.document, args.verification, args.deployment, args.tests)


if __name__ == '__main__':
    main()
