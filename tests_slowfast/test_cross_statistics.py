"""CPU-only evidence for target-window and recording-paired cross-input statistics."""
import importlib
import math

import numpy as np
import pytest


def api():
    return importlib.import_module('tools.report_cross_inputs')


def cells(arm):
    offsets = (0, 1, 4, 7)
    if arm == 'full_fast_only':
        return [(i, None, 'original' if i == 0 else 'swapped') for i in offsets]
    return [(i, k, 'A' if i == k == 0 else 'B' if i == 0 else 'C' if k == 0
             else 'D_same' if i == k else 'D_independent') for i in offsets for k in offsets]


def records(arm='full_correct_kv', token='t', log='log_a', errors=None):
    result = []
    for image, kv, cell in cells(arm):
        error = (errors or {}).get((image, kv), 0.)
        prediction = np.zeros((8, 3)); prediction[:, 0] = error
        result.append(dict(arm=arm, token=token, route=log+'_00000_00010', command=0,
            speed_mps=2., image_offset=image, kv_offset=kv, cell=cell,
            image_source_token=token if image == 0 else f'{token}-donor-{image}',
            kv_source_token=None if kv is None else token if kv == 0 else f'{token}-donor-{kv}',
            image_feature_relative_l2=float(image != 0), prediction=prediction.tolist(),
            target=np.zeros((8, 3)).tolist()))
    return result


def test_averages_donor_errors_in_window_without_cancelling_predictions():
    arm = 'full_fast_only'
    rows = records(arm, errors={(1, None): 2., (4, None): -2., (7, None): 2.})
    windows = api().collapse_cells(rows, arm)
    assert len(windows) == 1
    assert windows[0]['cells']['swapped']['ADE'] == 2.
    assert windows[0]['cells']['swapped']['XY_change'] == 2.
    effect = api().paired_effect(windows, {'swapped': 1., 'original': -1.}, reps=500)
    assert effect['window_count'] == effect['log_count'] == 1
    assert effect['ci95'] == [2., 2.]


def test_donors_are_not_replicates_and_logs_have_equal_bootstrap_weight():
    arm = 'full_fast_only'; rows = []
    for token, log, damage in [('a', 'x', 0.), ('b', 'x', 0.), ('c', 'x', 0.), ('d', 'y', 4.)]:
        rows += records(arm, token, log, {(i, None): damage for i in (1, 4, 7)})
    windows = api().collapse_cells(rows, arm)
    result = api().paired_effect(windows, {'swapped': 1., 'original': -1.}, reps=2000)
    assert result['window_count'] == 4 and result['log_count'] == 2
    assert result['window_weighted_mean'] == 1.
    assert result['equal_log_mean'] == 2.
    assert result['per_log_means'] == {'x': 0., 'y': 4.}
    rng = np.random.default_rng(20261109)
    expected = np.array([0., 4.])[rng.integers(2, size=(2000, 2))].mean(1)
    assert result['ci95'] == pytest.approx(np.percentile(expected, [2.5, 97.5]))


def test_factorial_interaction_and_different_donor_rescue_match_image_first():
    errors = {(0, 0): 1.}
    for n, image in enumerate((1, 4, 7)):
        errors[0, image] = 3.
        errors[image, 0] = 4. * (n+1)
        for kv in (1, 4, 7):
            errors[image, kv] = errors[image, 0] + (6.+n if image == kv else 20.+n)
    windows = api().collapse_cells(records(errors=errors), 'full_correct_kv')
    result = api().paired_effect(windows, {'D_same': 1., 'C': -1., 'B': -1., 'A': 1.}, reps=50)
    assert result['equal_log_mean'] == 5.
    assert windows[0]['matched_rescue']['D_same']['1']['ADE'] == 6.
    assert windows[0]['matched_rescue']['D_independent']['7']['ADE'] == 22.
    assert api().paired_effect(windows, {'D_independent': 1., 'C': -1.}, reps=50)['equal_log_mean'] == 21.


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'cell', 'source', 'target', 'metadata', 'nan'])
def test_incomplete_duplicate_or_inconsistent_windows_are_rejected(problem):
    rows = records()
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows.append(dict(rows[0]))
    elif problem == 'cell': rows[1]['cell'] = 'C'
    elif problem == 'source': rows[1]['kv_source_token'] = 'wrong'
    elif problem == 'target': rows[1]['target'][0][0] = 1.
    elif problem == 'metadata': rows[1]['speed_mps'] = 3.
    else: rows[1]['prediction'][0][0] = float('nan')
    with pytest.raises(ValueError):
        api().collapse_cells(rows, 'full_correct_kv')


def test_periodic_heading_error_wraps_at_pi_and_xy_uses_target_arm_baseline():
    rows = records('full_fast_only', errors={(0, None): 1., (1, None): 4.})
    for row in rows:
        row['prediction'][0][2] = -math.pi + .02
        row['target'][0][2] = math.pi - .01
    window = api().collapse_cells(rows, 'full_fast_only')[0]
    assert window['offsets']['1,None']['heading_MAE'] == pytest.approx(.03/8)
    assert window['offsets']['1,None']['XY_change'] == 3.
    assert window['offsets']['0,None']['XY_change'] == 0.
    assert len(window['offsets']['1,None']['L2_by_time']) == 8


def test_different_donor_labels_cannot_hide_same_source_and_pairing_rejects_duplicate_windows():
    rows = records()
    for row in rows:
        if row['image_offset'] == 4: row['image_source_token'] = 't-donor-1'
        if row['kv_offset'] == 4: row['kv_source_token'] = 't-donor-1'
    with pytest.raises(ValueError): api().collapse_cells(rows, 'full_correct_kv')
    windows = api().collapse_cells(records(), 'full_correct_kv')
    with pytest.raises(ValueError): api().paired_effect(windows + windows, {'A': 1.}, reps=10)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import yaml
    module = api()
    project = tmp_path / 'project'; project.mkdir()
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/slowfast_cross_inputs.yaml').read_text())
    source_root = project / config['run_root']; cache = project / config['cache']; cache.mkdir(parents=True)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False))
    pins = {}
    for arm in ('full_fast_only', 'full_correct_kv'):
        path = source_root / arm / 'final.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arm.encode()); pins[arm] = module.sha256(path)
    monkeypatch.setattr(module, 'CHECKPOINTS', pins)
    config['checkpoint_sha256'] = pins
    config_path = project / 'configs/slowfast_cross_inputs.yaml'; config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))
    canonical = []
    for group in range(8):
        for index in range(32):
            canonical.append(dict(token=f'g{group}t{index}', route=f'log{group}_00000_00010',
                command=2 if (group, index) == (0, 0) else group % 2, ego=[2.+group, 0., 0., 0.],
                trajectory_gt=np.zeros((8, 3)).tolist(), split='val'))
    targets = [{key: row[key] for key in ('token', 'route', 'command', 'ego')} for row in canonical]
    tokens = [row['token'] for row in canonical]
    donors = {str(offset): {f'g{group}t{index}': f'g{(group+offset)%8}t{index}'
                           for group in range(8) for index in range(32)} for offset in (1, 4, 7)}
    write(cache/'manifest.json', dict(completed=True, source_frozen=True,
          vision_native_prediction_all_exact=True, rows=canonical))
    data = {}
    for arm in ('full_fast_only', 'full_correct_kv'):
        data[arm] = []
        for target in targets:
            group = int(target['token'][1])
            rows = records(arm, target['token'], f'log{group}',
                           {(i, k): float(i != 0)+float(k not in (None, 0)) for i, k, _ in cells(arm)})
            for row in rows:
                row['command'] = target['command']; row['speed_mps'] = math.hypot(*target['ego'][:2])
                row['image_source_token'] = target['token'] if row['image_offset'] == 0 else donors[str(row['image_offset'])][target['token']]
                row['kv_source_token'] = None if row['kv_offset'] is None else target['token'] if row['kv_offset'] == 0 else donors[str(row['kv_offset'])][target['token']]
            data[arm].extend(rows)
        previous = [row for row in data[arm] if row['image_offset'] == 0 and row['kv_offset'] in (None, 0)]
        write(source_root/arm/'final_predictions.json', previous)
        write(source_root/arm/'verification.json', dict(passed=True, checkpoint_sha256=pins[arm],
            predictions_sha256=module.sha256(source_root/arm/'final_predictions.json')))
        write(source_root/arm/'result.json', dict(completed=True, arm=arm, checkpoint_sha256=pins[arm]))
    prior = dict(donors=donors['4'], records=[dict(row, mode='shuffled') for row in data['full_correct_kv']
                                            if row['image_offset'] == 0 and row['kv_offset'] == 4])
    write(source_root/'correct_checkpoint_diagnostics.json', prior)
    write(source_root/'result.json', dict(diagnostics_file_sha256=module.sha256(source_root/'correct_checkpoint_diagnostics.json')))
    hashes = dict(cache_manifest=module.sha256(cache/'manifest.json'), prior_diagnostic=module.sha256(source_root/'correct_checkpoint_diagnostics.json'))
    for arm in data:
        for name, file in [('result', 'result.json'), ('verification', 'verification.json'), ('predictions', 'final_predictions.json')]:
            hashes[arm+'_'+name] = module.sha256(source_root/arm/file)
    source_hashes = {}
    for name in module.RUNNER_SOURCE_FILES:
        path = project/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        source_hashes[name] = module.sha256(path)
    plan = dict(schema='frozen_cross_inputs_protocol_v1', config=config, config_sha256=module.sha256(config_path),
        hashes=hashes, tokens=tokens, targets=targets, donors=donors, raw_targets=[f'g{g}t0' for g in range(8)],
        cells={arm:[dict(image_offset=i, kv_offset=k, cell=c) for i,k,c in cells(arm)] for arm in data},
        expected_records=dict(full_fast_only=1024, full_correct_kv=4096), expected_raw_replays_per_arm=32,
        expected_saved_exact=dict(full_fast_only=256, full_correct_kv=512), source_code_sha256=source_hashes)
    run_root = tmp_path/'run'; write(run_root/'protocol.json', plan)
    for arm, rows in data.items():
        directory = run_root/arm; directory.mkdir()
        (directory/'predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        raw = [dict(target_token=row['token'], arm=arm, image_source_token=row['image_source_token'],
            max_abs_difference=0., raw_prediction=row['prediction'], cached_prediction=row['prediction'],
            target_ego_preserved=True, target_command_preserved=True, image_original_sizes=[[10, 10]]*4,
            image_audit=[{}]*4) for row in rows if row['token'] in plan['raw_targets'] and row['kv_offset'] in (None, 0)]
        write(directory/'raw_replays.json', raw)
        write(directory/'result.json', dict(schema='frozen_cross_inputs_arm_v1', completed=True, arm=arm,
            records=len(rows), exact_previous_forecasts=plan['expected_saved_exact'][arm], raw_processor_replays_exact=32,
            all_parameter_sha256_before='a'*64, all_parameter_sha256_after='a'*64,
            parameter_versions_unchanged=True, inputs_and_sources_unchanged=True, checkpoint_sha256=pins[arm],
            predictions_sha256=module.sha256(directory/'predictions.jsonl'), raw_replays_sha256=module.sha256(directory/'raw_replays.json'),
            protocol_sha256=module.sha256(run_root/'protocol.json'), peak_allocated_gib=2.,
            new_training_updates=0, new_response_body_bytes=0))
    return run_root, project, write


def test_load_run_authenticates_complete_coverage_provenance_and_all_exact_arrays(saved_run):
    root, project, _ = saved_run
    evidence = api().load_run(root, project_root=project)
    assert len(evidence['records']['full_correct_kv']) == 4096
    assert evidence['acceptance']['saved_exact_predictions_verified'] == 768
    assert evidence['acceptance']['raw_replays_verified'] == 64
    assert evidence['acceptance']['checkpoint_files_rehashed'] is False


@pytest.mark.parametrize('problem', ['incomplete', 'file_hash', 'source_hash', 'raw_array', 'old_array', 'metadata', 'donor'])
def test_load_run_rejects_invalid_receipts_arrays_current_metadata_and_provenance(saved_run, problem):
    import json
    root, project, write = saved_run
    arm = root/'full_correct_kv'; result_path = arm/'result.json'
    result = json.loads(result_path.read_text())
    if problem == 'incomplete':
        result['completed'] = False
    elif problem == 'file_hash':
        result['predictions_sha256'] = '0'*64
    elif problem == 'source_hash':
        (project/api().RUNNER_SOURCE_FILES[0]).write_text('changed')
    elif problem == 'raw_array':
        path = arm/'raw_replays.json'; raw = json.loads(path.read_text())
        raw[0]['raw_prediction'][0][0] += .1; write(path, raw)
        result['raw_replays_sha256'] = api().sha256(path)
    elif problem == 'old_array':
        path = project/'outputs/slowfast_direct_full/run_01/full_correct_kv/final_predictions.json'
        previous = json.loads(path.read_text()); previous[0]['prediction'][0][0] += .1; write(path, previous)
    else:
        path = arm/'predictions.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
        if problem == 'metadata':
            for row in rows:
                if row['token'] == 'g0t0': row['speed_mps'] += 1.
        else:
            for row in rows:
                if row['token'] == 'g0t0' and row['image_offset'] == 1: row['image_source_token'] = 'g2t0'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); result['predictions_sha256'] = api().sha256(path)
    write(result_path, result)
    with pytest.raises(ValueError): api().load_run(root, project_root=project)


def test_report_writes_fixed_tables_figures_and_sparse_class_caveat(saved_run, tmp_path):
    import json
    root, project, _ = saved_run
    output = tmp_path/'report'
    summary = api().run(root, output, project_root=project)
    assert summary['completed'] is True
    assert summary['cohorts']['all']['full_correct_kv']['window_count'] == 256
    assert summary['cohorts']['command_2']['full_correct_kv']['window_count'] == 1
    assert summary['cohorts']['command_2']['stable_class_conclusion_supported'] is False
    assert (output/'summary.json').is_file() and (output/'report_zh.md').is_file()
    assert len(list(output.glob('*.png'))) == 4
    assert len((output/'offset_metrics.csv').read_text().splitlines()) == 1+20*256
    assert json.loads((output/'provenance.json').read_text())['acceptance']['raw_replays_verified'] == 64
    with pytest.raises(FileExistsError): api().run(root, output, project_root=project)
