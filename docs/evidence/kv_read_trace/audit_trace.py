"""Independent CPU-only stdlib/NumPy audit; imports no product model or metrics.

Run from repository root after run_01/result.json exists. Missing completion is a
clean waiting state, not a failed or partially accepted audit. No training occurs.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

LAYERS = [4, 8, 12, 16, 20, 24]
MODES = ['correct', 'reference', 'shuffled']
POSITION_GROUPS = ['image_0', 'image_1', 'image_2', 'image_3', 'native_other', 'ego', 'command', 'feedback', 'planning_queries']
FIELDS = ['rms_difference', 'relative_l2', 'max_abs_difference', 'equal_fraction']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def recording(route):
    return re.sub(r'_\d+_\d+$', '', route)


def dist(rows, values):
    a = np.asarray(values, dtype=np.float64)
    assert len(rows) == len(a) and np.isfinite(a).all()
    if not len(a):
        return {'count': 0}
    grouped = defaultdict(list)
    for row, value in zip(rows, a):
        grouped[recording(row['route'])].append(float(value))
    means = {g: float(np.mean(v)) for g, v in sorted(grouped.items())}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p10': float(np.quantile(a, .1)), 'p90': float(np.quantile(a, .9)),
            'min': float(a.min()), 'max': float(a.max()), 'zero_fraction': float((a == 0).mean()),
            'recordings': len(means), 'by_recording': means,
            'equal_recording_mean': float(np.mean(list(means.values())))}


def ade(p, t):
    return np.linalg.norm(np.asarray(p, dtype=np.float64)[..., :2] - np.asarray(t, dtype=np.float64)[..., :2], axis=-1).mean(-1)


def metrics(p, t):
    p, t = np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)
    delta = p-t
    xy = np.linalg.norm(delta[..., :2], axis=-1)
    yaw = np.abs(np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2])))
    out = {'ADE': float(xy.mean()), 'FDE': float(xy[:, -1].mean()),
           'loss_l1_pose': float(np.abs(delta).mean()), 'loss_l1_xy': float(np.abs(delta[..., :2]).sum(-1).mean()),
           'heading_mae_rad': float(yaw.mean()), 'heading_final_mae_rad': float(yaw[:, -1].mean()), 'samples': len(p)}
    out.update({f'L2@{(i+1)*.5:g}s': float(v) for i, v in enumerate(xy.mean(0))})
    return out


def select64(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[recording(row['route'])].append(row)
    selected = []
    for group in sorted(groups):
        buckets = defaultdict(deque)
        for row in sorted(groups[group], key=lambda r: r['token']):
            speed = row['speed_mps']
            buckets[(row['command'], 0 if speed < 1 else 1 if speed < 10 else 2)].append(row['token'])
        chosen = []
        while len(chosen) < 8:
            for key in sorted(buckets):
                if buckets[key]:
                    chosen.append(buckets[key].popleft())
                    if len(chosen) == 8:
                        break
        selected.extend(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/slowfast_kv_read_trace/run_01'))
    parser.add_argument('--prior', type=Path, default=Path('outputs/slowfast_direct_full/run_01/correct_checkpoint_diagnostics.json'))
    args = parser.parse_args()
    run, output = args.run, Path(__file__).resolve().parent
    if not (run/'result.json').exists():
        print('WAITING: result.json is not yet present; script prepared, no partial audit accepted.')
        return
    paths = {name: run/name for name in ['protocol.json', 'records.jsonl', 'interventions.jsonl', 'result.json', 'summary.json']}
    paths['prior_diagnostic'] = args.prior
    paths['cache_manifest'] = Path('outputs/slowfast_direct_full/cache_01/manifest.json')
    paths['input_config'] = Path('configs/slowfast_direct_full.yaml')
    paths['correct_endpoint_verification'] = Path('outputs/slowfast_direct_full/run_01/full_correct_kv/verification.json')
    paths['audit_script'] = Path(__file__)
    hashes = {name: {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}
              for name, path in paths.items()}
    protocol, result, summary = (read(paths[name]) for name in ['protocol.json', 'result.json', 'summary.json'])
    prior = read(args.prior)
    for name, field in [('protocol.json', 'protocol_sha256'), ('records.jsonl', 'records_sha256'),
                        ('interventions.jsonl', 'interventions_sha256'), ('summary.json', 'summary_sha256')]:
        assert hashes[name]['sha256'] == result[field], ('hash', name)
    for name, field in [('prior_diagnostic', 'prior_diagnostic_sha256'), ('cache_manifest', 'cache_sha256'), ('input_config', 'input_config_sha256')]:
        assert hashes[name]['sha256'] == protocol[field], ('hash', name)
    assert prior['passed'] and protocol['checkpoint_sha256'] == prior['checkpoint_sha256']
    assert protocol['cache_sha256'] == prior['cache_sha256']
    assert read(paths['correct_endpoint_verification'])['checkpoint_sha256'] == protocol['checkpoint_sha256']
    for name, expected in protocol['source_code_sha256'].items():
        path = Path(name)
        hashes['source:'+name] = {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}
        assert hashes['source:'+name]['sha256'] == expected, ('source code changed', name)
    assert result['completed'] and result['new_training_updates'] == protocol['new_training_updates'] == 0
    assert result['new_response_body_bytes'] == protocol['new_response_body_bytes'] == 0
    assert result['all_parameter_sha256_before'] == result['all_parameter_sha256_after'] == protocol['model_all_parameters_sha256']
    assert result['parameter_versions_unchanged'] and result['source_tensors_unchanged']
    assert result['exact_prior_predictions'] == result['context_24layers_all_rows_exact_natural_passes'] == 768
    assert result['context_checks_exclude_single_layer_interventions'] and result['scope_cleanup_baseline_rechecks'] == 64
    assert protocol['modes'] == MODES and protocol['layers_one_based'] == LAYERS
    rows = [json.loads(line) for line in paths['records.jsonl'].read_text().splitlines()]
    interventions = [json.loads(line) for line in paths['interventions.jsonl'].read_text().splitlines()]
    by_token = {r['token']: r for r in rows}
    assert len(rows) == len(by_token) == 256
    assert [r['token'] for r in rows] == protocol['selected_tokens']
    counts = Counter(recording(r['route']) for r in rows)
    assert len(counts) == 8 and set(counts.values()) == {32}
    selected = select64(rows)
    assert selected == protocol['intervention_tokens'] and len(set(selected)) == 64
    assert set(Counter(recording(by_token[t]['route']) for t in selected).values()) == {8}
    old = {(r['token'], r['mode']): r for r in prior['records'] if r['mode'] in MODES}
    assert len(old) == 768
    manifest_rows = {r['token']: r for r in read(paths['cache_manifest'])['rows'] if r['split'] == 'val'}
    assert set(manifest_rows) == set(by_token)
    mass_errors, local_correct_errors, relation_errors, invariant_errors, head_pose_errors = [], [], [], [], []
    for row in rows:
        token = row['token']
        assert row['command'] == manifest_rows[token]['command']
        assert abs(row['speed_mps'] - np.hypot(*manifest_rows[token]['ego'][:2])) < 1e-12
        assert np.array_equal(np.asarray(row['target'], np.float32), np.asarray(manifest_rows[token]['trajectory_gt'], np.float32))
        assert row['source_tokens'] == {'correct': token, 'reference': protocol['reference_token'], 'shuffled': protocol['donors'][token]}
        assert protocol['donors'] == prior['donors']
        for mode in MODES:
            assert np.array_equal(np.asarray(row['predictions'][mode]), np.asarray(old[token, mode]['prediction']))
            assert row['route'] == old[token, mode]['route']
            assert row['context_checks'][mode] == {'checked_layers': 24, 'all_rows_exact': True}
            assert [q['layer_one_based'] for q in row['probes'][mode]] == LAYERS
            for probe in row['probes'][mode]:
                ref = probe['float32_reference']
                assert list(ref['groups']) == POSITION_GROUPS
                mass_errors.extend([abs(ref['external_mass'] + ref['self_mass'] - 1),
                                    abs(sum(g['mass'] for g in ref['groups'].values()) - ref['external_mass']),
                                    abs(np.mean(ref['external_mass_per_query'])-ref['external_mass']),
                                    abs(np.mean(ref['external_mass_per_head'])-ref['external_mass'])])
                assert sum(g['tokens'] for g in ref['groups'].values()) == 8255
                assert all(g['visible_tokens'] == g['tokens'] for g in ref['groups'].values())
                for g in ref['groups'].values():
                    mass_errors.append(abs(g['mass']/g['tokens']-g['mass_per_token']))
                if mode == 'correct':
                    variants = probe['local_variants']
                    assert set(variants) == {'correct', 'reference', 'reference_k_only', 'reference_v_only', 'shuffled', 'shuffled_k_only', 'shuffled_v_only', 'zero_v', 'masked'}
                    for stage in ['pre_gate', 'post_gate', 'post_projection']:
                        local_correct_errors.append(variants['correct'][stage]['rms_difference'])
                    for variant in variants.values():
                        for stage in ['pre_gate', 'post_gate', 'post_projection']:
                            v = variant[stage]
                            relation_errors.append(abs(v['relative_l2'] - v['rms_difference']/max(v['rms_reference'], 1e-12)))
                else:
                    assert not probe['local_variants']
        for mode in MODES[1:]:
            baseline_pose = np.asarray(row['predictions']['correct'], dtype=np.float64)
            candidate_pose = np.asarray(row['predictions'][mode], dtype=np.float64)
            delta_pose = candidate_pose-baseline_pose
            saved_head = row['natural_differences'][mode]['head_stage_3']
            head_pose_errors.extend([abs(saved_head['rms_difference']-float(np.sqrt(np.mean(delta_pose**2)))),
                                     abs(saved_head['relative_l2']-float(np.linalg.norm(delta_pose)/max(np.linalg.norm(baseline_pose), 1e-12)))])
            for stage in ['layer_1', 'layer_2', 'layer_3', 'gate_4']:
                invariant_errors.append(row['natural_differences'][mode][stage]['rms_difference'])
    assert max(mass_errors) < 2e-6
    assert max(local_correct_errors) == max(invariant_errors) == 0
    assert max(relation_errors) < 2e-5
    assert max(head_pose_errors) < 2e-6
    expected_interventions = {(t, mode, layer) for t in selected for mode in ['reference', 'shuffled', 'masked'] for layer in LAYERS}
    actual = {(r['token'], r['mode'], r['layer_one_based']) for r in interventions}
    assert len(interventions) == len(actual) == 1152 and actual == expected_interventions
    assert result['single_layer_interventions'] == protocol['total_single_layer_interventions'] == 1152
    assert set(Counter(r['token'] for r in interventions).values()) == {18}
    effect_errors = defaultdict(list)
    for r in interventions:
        base = by_token[r['token']]
        assert r['route'] == base['route']
        assert np.array_equal(np.asarray(r['baseline_prediction']), np.asarray(base['predictions']['correct']))
        assert np.array_equal(np.asarray(r['target']), np.asarray(base['target']))
        assert r['source_token'] == (None if r['mode'] == 'masked' else base['source_tokens'][r['mode']])
        change = np.linalg.norm(np.asarray(r['prediction'])[:, :2] - np.asarray(r['baseline_prediction'])[:, :2], axis=-1)
        calculated = {'ADE_gain_vs_correct_m': float(ade(r['baseline_prediction'], r['target'])-ade(r['prediction'], r['target'])),
                      'mean_plan_xy_change_m': float(change.mean()), 'max_plan_xy_change_m': float(change.max())}
        for k, v in calculated.items():
            effect_errors[k].append(abs(r[k]-v))
            r[k] = v  # aggregate independent float64 values, not saved scalar effects
    assert max(max(v) for v in effect_errors.values()) < 3e-6
    cohorts = {'all': rows, 'moving_ge_1mps': [r for r in rows if r['speed_mps'] >= 1],
               'high_ge_10mps': [r for r in rows if r['speed_mps'] >= 10]}
    aggregate, summary_errors = {}, []
    for cohort, chosen in cohorts.items():
        if not chosen:
            continue
        def get(fn):
            return dist(chosen, [fn(r) for r in chosen])
        part = {'count': len(chosen), 'metrics': {}, 'natural_effects': {}, 'layers': {}, 'natural_stages': {}}
        for mode in MODES:
            part['metrics'][mode] = metrics([r['predictions'][mode] for r in chosen], [r['target'] for r in chosen])
            for k, v in part['metrics'][mode].items():
                summary_errors.append(abs(v-summary[cohort]['metrics'][mode][k]))
        for mode in MODES[1:]:
            changes = [float(np.linalg.norm(np.asarray(r['predictions'][mode])[:, :2]-np.asarray(r['predictions']['correct'])[:, :2], axis=-1).mean()) for r in chosen]
            gains = [float(ade(r['predictions']['correct'], r['target'])-ade(r['predictions'][mode], r['target'])) for r in chosen]
            part['natural_effects'][mode] = {'mean_plan_xy_change_m': dist(chosen, changes), 'ADE_gain_vs_correct_m': dist(chosen, gains),
                                            'task_better_windows': sum(x > 0 for x in gains), 'task_worse_windows': sum(x < 0 for x in gains)}
            part['natural_stages'][mode] = {stage: {field: get(lambda r, m=mode, s=stage, f=field: r['natural_differences'][m][s][f]) for field in FIELDS}
                                           for stage in chosen[0]['natural_differences'][mode]}
        for i, layer in enumerate(LAYERS):
            entry = {'modes': {}, 'fixed_current_Q': {}, 'source_difference': {}}
            for mode in MODES:
                probes = [r['probes'][mode][i] for r in chosen]
                values = {'external_mass': [p['float32_reference']['external_mass'] for p in probes],
                          'self_mass': [p['float32_reference']['self_mass'] for p in probes]}
                for field in ['pre_gate_rms', 'post_gate_rms', 'query_projection_rms', 'gate_mean', 'gate_rms', 'gate_fraction_lt_0_01', 'gate_fraction_lt_0_1', 'gate_fraction_gt_0_9']:
                    values['actual.'+field] = [p['actual'][field] for p in probes]
                for field, vals in values.items():
                    product_field = ('float32_reference.'+field) if field in ['external_mass', 'self_mass'] else field
                    d = dist(chosen, vals)
                    for stat in ['mean', 'median', 'equal_recording_mean']:
                        summary_errors.append(abs(d[stat]-summary[cohort]['probes'][mode][str(layer)][product_field][stat]))
                entry['modes'][mode] = {name: dist(chosen, vals) for name, vals in values.items()}
                for what in ['sdpa_reconstruction', 'projection_reconstruction']:
                    entry['modes'][mode][what] = {field: dist(chosen, [p['float32_reference'][what][field] for p in probes]) for field in ['rms_difference', 'relative_l2', 'max_abs_difference']}
                entry['modes'][mode]['native_full_vs_query_projection'] = {field: get(lambda r, f=field, m=mode, l=str(layer): r['projection_shape_audit'][m][l][f]) for field in ['rms_difference', 'relative_l2', 'max_abs_difference']}
                entry['modes'][mode]['position_groups'] = {name: {field: dist(chosen, [p['float32_reference']['groups'][name][field] for p in probes]) for field in ['mass', 'mass_per_token']}
                                                           for name in POSITION_GROUPS}
                entry['modes'][mode]['external_contribution_rms'] = {stage: dist(chosen, [p['float32_reference']['external'][stage] for p in probes]) for stage in ['pre_gate_rms', 'post_gate_rms', 'linear_projection_rms']}
            for variant in chosen[0]['probes']['correct'][i]['local_variants']:
                entry['fixed_current_Q'][variant] = {stage: {field: get(lambda r, v=variant, st=stage, f=field: r['probes']['correct'][i]['local_variants'][v][st][f]) for field in FIELDS}
                                                    for stage in ['pre_gate', 'post_gate', 'post_projection']}
                pre = np.array([r['probes']['correct'][i]['local_variants'][variant]['pre_gate']['rms_difference'] for r in chosen])
                post = np.array([r['probes']['correct'][i]['local_variants'][variant]['post_gate']['rms_difference'] for r in chosen])
                projected = np.array([r['probes']['correct'][i]['local_variants'][variant]['post_projection']['rms_difference'] for r in chosen])
                entry['fixed_current_Q'][variant]['post_over_pre_absolute_RMS_ratio'] = dist(chosen, np.divide(post, pre, out=np.zeros(len(pre)), where=pre > 0))
                entry['fixed_current_Q'][variant]['projection_over_post_absolute_RMS_ratio'] = dist(chosen, np.divide(projected, post, out=np.zeros(len(pre)), where=post > 0))
            for mode in MODES[1:]:
                entry['source_difference'][mode] = {kind: {field: get(lambda r, m=mode, k=kind, f=field: r['source_differences'][m][i][k][f])
                                                          for field in ['rms_difference', 'relative_l2', 'cosine']}
                                                   for kind in ['key', 'value']}
            part['layers'][str(layer)] = entry
        aggregate[cohort] = part
    assert max(summary_errors) < 3e-6
    single = {}
    for cohort, predicate in [('all_selected64', lambda r: True), ('moving_ge_1mps_selected', lambda r: by_token[r['token']]['speed_mps'] >= 1)]:
        single[cohort] = {}
        for mode in ['reference', 'shuffled', 'masked']:
            single[cohort][mode] = {}
            for layer in LAYERS:
                chosen = [r for r in interventions if r['mode'] == mode and r['layer_one_based'] == layer and predicate(r)]
                single[cohort][mode][str(layer)] = {field: dist(chosen, [r[field] for r in chosen]) for field in ['ADE_gain_vs_correct_m', 'mean_plan_xy_change_m', 'max_plan_xy_change_m']}
                if cohort == 'all_selected64':
                    for field, independent in single[cohort][mode][str(layer)].items():
                        saved = summary['single_layer_interventions'][f'{mode}_{layer}'][field]
                        for stat in ['mean', 'median', 'equal_recording_mean']:
                            summary_errors.append(abs(independent[stat]-saved[stat]))
    assert max(summary_errors) < 3e-6
    # Scalar norm summaries do not retain vector directions. Decomposition norms
    # need not add, and projection bias cannot be isolated from these summaries.
    limitations = [
        'Own-head Slow ADE is not a proxy for frozen KV quality; no Slow adaptation is evaluated or recommended by this audit.',
        'Nonzero source-swap activations establish content sensitivity at fixed current Q/FastKV/gate; they do not establish task benefit.',
        'Natural all-layer source changes cascade through earlier layers and change later Q/gates. Single-layer causal interventions also propagate downstream; neither equals local fixed-Q changes.',
        'Masked source columns remove the external branch and renormalize joint softmax; effects mix branch availability, softmax competition, content, and distribution shift.',
        'Nine positional source groups partition the columns, but contextualized group states mix upstream information. Attention to an image span is not isolated image-modality attribution.',
        'P(0) bias vector/RMS is not serialized. Bias-aware projection reconstruction and native-full/query-shape errors are audited; an isolated bias magnitude is unidentifiable here.',
        'Float32 softmax decomposition and native BF16 projection are numerical references, not bit-exact additive causal attributions. Contribution RMS values cannot be summed as scalar shares.',
        'Only eight previously observed recording groups; interventions64 is metadata-selected, not the full VAL256, and moving subsets can change group weights.',
    ]
    output_data = {'schema': 'independent_frozen_six_layer_trace_audit_v1', 'passed': True, 'libraries': ['Python stdlib', 'NumPy'],
                   'counts': {'windows': 256, 'recording_groups': dict(counts), 'exact_previous_forecasts': 768, 'intervention_windows': 64,
                              'single_layer_records': 1152, 'single_layer_records_per_token': 18, 'source_position_groups': POSITION_GROUPS},
                   'checks': {'all_record_hashes_verified': True, 'metadata_selection64_reproduced_exactly': True,
                              'max_joint_mass_accounting_error': max(mass_errors), 'max_local_correct_RMS_error': max(local_correct_errors),
                              'max_pre_read_natural_invariant_RMS_error': max(invariant_errors),
                              'max_relative_norm_identity_error': max(relation_errors),
                              'max_head_stage_3_vs_independent_pose_error': max(head_pose_errors),
                              'max_saved_effect_vs_independent_float64_error': {k: max(v) for k, v in effect_errors.items()},
                              'max_product_summary_vs_independent_float64_error': max(summary_errors),
                              'model_unchanged_claims': 'matching before/after runner digests and version/source immutability assertions; model tensors not independently loaded'},
                   'cohorts': aggregate, 'single_layer_interventions': single, 'interpretation_limits': limitations, 'inputs': hashes}
    assert all(sha(Path(v['path'])) == v['sha256'] for v in hashes.values())
    dest = output/'audit.json'
    dest.write_text(json.dumps(output_data, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'output': str(dest), 'counts': output_data['counts'], 'checks': output_data['checks']}, indent=2))


if __name__ == '__main__':
    main()
