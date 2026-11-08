"""Tamper controls for full-language endpoint, replay and paired report artifacts."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from fast_navsim.model import trajectory_metrics
from slowfast_v11.expanded_protocol import lr_factor
from tools.direct_full_common import read_config
from tools.train_direct_full import FIXED_SOURCE, build_draw_order, order_digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(tensors=0):
    return dict(passed=True, max_abs_difference=0., differing_fields=0,
                first_differences=[], tensor_count=tensors)


def prediction(row, error, **extra):
    import torch
    poses = [[error, 0., 0.]]*8
    metric = trajectory_metrics(torch.tensor([poses]), torch.tensor([row['trajectory_gt']]))
    return dict(token=row['token'], route=row['route'], split='val', command=row['command'],
                target=row['trajectory_gt'], prediction=poses, ADE=metric['ADE'],
                FDE=metric['FDE'], loss_l1_pose=metric['loss_l1_pose'], **extra)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    from tools import report_direct_full as report
    config = read_config()
    config['data_dir'] = str(tmp_path/'data')
    monkeypatch.setattr(report, 'read_config', lambda path: copy.deepcopy(config))
    train = [dict(token=FIXED_SOURCE if i == 0 else f'train-{i}', split='train', cohort='train',
                  route=f'train-recording-{i//43}_00000_00500', command=1, ego=[0.]*4,
                  trajectory_gt=[[0., 0., 0.]]*8) for i in range(512)]
    val = [dict(token=f'val-{i}', split='val', cohort='new_val', command=1, ego=[0.]*4,
                route=f'val-recording-{i//32}_00000_00500', trajectory_gt=[[0., 0., 0.]]*8)
           for i in range(256)]
    run = tmp_path/'run'
    cache = tmp_path/'cache'
    slow = [prediction(row, .7) for row in val]
    save(cache/'slow_predictions.json', slow)
    save(cache/'manifest.json', dict(completed=True, rows=train+val, fixed_source=FIXED_SOURCE,
                                    slow_predictions_sha256=sha(cache/'slow_predictions.json')))
    identity = dict(cache_sha256=sha(cache/'manifest.json'), init_sha256='shared-init', initial_digest='same')
    order = build_draw_order(train)
    endpoint = {}
    for arm, error in zip(config['arms'], [.6, .4, .5]):
        directory = run/arm
        directory.mkdir(parents=True)
        (directory/'final.pt').write_bytes(arm.encode())
        (directory/'before_final.pt').write_bytes((arm+'before').encode())
        records = [prediction(row, error, arm=arm, mode='trained', source_token=(
            None if arm == 'full_fast_only' else row['token'] if arm == 'full_correct_kv' else FIXED_SOURCE)) for row in val]
        endpoint[arm] = records
        save(directory/'final_predictions.json', records)
        save(directory/'result.json', dict(completed=True, arm=arm, identity=identity,
            checkpoint_sha256=sha(directory/'final.pt'), before_final_sha256=sha(directory/'before_final.pt'),
            model_digest='full-weights', optimizer_updates=1024, windows_seen=4096,
            equivalent_train_epochs=8, val_used_for_selection=False, draw_order_sha256=order_digest(order),
            scope=dict(language=dict(tensors=512), new=dict(tensors=17)),
            optimizer_state_parameters=525 if arm == 'full_fast_only' else 529))
        save(directory/'verification.json', dict(passed=True, arm=arm, samples=256,
            checkpoint_sha256=sha(directory/'final.pt'), model_digest='full-weights', full_state_keys=529,
            reference_prediction_check=exact(), predictions_sha256=sha(directory/'final_predictions.json'),
            metrics=report.metrics(records), val_used_for_selection=False))
        save(directory/'replay.json', dict(passed=True, arm=arm, replayed_updates=1, micro_batches=4,
            checkpoint_sha256=sha(directory/'final.pt'), checks={
                key: exact(529 if key == 'model' else (525 if arm == 'full_fast_only' else 529)*3
                           if key == 'optimizer' else 2 if key == 'rng' else 0)
                for key in ['model', 'optimizer', 'scheduler', 'rng', 'sampler', 'reference_predictions']}))
        save(directory/'protocol.json', dict(arm=arm, config=config, identity=identity,
            draw_order=order, draw_order_sha256=order_digest(order), fixed_source_token=FIXED_SOURCE,
            val_used_for_selection=False))
        logs = []
        for update in range(1, 1025):
            factor = lr_factor(update, 1024, 64, .1)
            logs.append(dict(update=update, arm=arm, tokens=order[(update-1)*4:update*4],
                draw_index=update*4, accumulation_index=0, windows_seen=update*4,
                equivalent_train_epochs=update/128, loss=.1, micro_losses=[.1]*4, grad_norm=.2,
                learning_rates=dict(language=1e-5*factor, new=5e-5*factor)))
        (directory/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in logs))
    reference_records = [prediction(row, .8, method=method) for row in val for method in [
        'original_fast', 'historical_fixed_seed_42', 'historical_fixed_seed_43', 'historical_fixed_seed_44']]
    reference = tmp_path/'references.json'
    save(reference, dict(records=reference_records, cache_manifest_sha256=identity['cache_sha256'],
        metrics={method: report.metrics([r for r in reference_records if r['method'] == method])
                 for method in {r['method'] for r in reference_records}}))
    donors = {row['token']:val[(index+128)%256]['token'] for index, row in enumerate(val)}
    diagnostic_records = []
    for row in val:
        for mode, error in [('correct', .4), ('off', .65), ('reference', .6), ('shuffled', .55)]:
            source = {'correct':row['token'], 'off':None, 'reference':FIXED_SOURCE, 'shuffled':donors[row['token']]}[mode]
            diagnostic_records.append(prediction(row, error, arm='full_correct_kv', mode=mode, source_token=source))
    diagnostics = tmp_path/'diagnostics.json'
    save(diagnostics, dict(passed=True, correct_endpoint_exact_count=256, donors=donors,
        cache_sha256=identity['cache_sha256'], checkpoint_sha256=sha(run/'full_correct_kv/final.pt'),
        records=diagnostic_records, metrics={mode:report.metrics([r for r in diagnostic_records if r['mode']==mode])
            for mode in ['correct', 'off', 'reference', 'shuffled']}))
    save(Path(config['data_dir'])/'network_budget.json', dict(phase='complete', used=123))
    return dict(report=report, config=config, run=run, cache=cache, reference=reference,
                diagnostics=diagnostics, output=tmp_path/'report.json', endpoint=endpoint)


def execute(a):
    return a['report'].run('unused', a['run'], a['cache'], a['reference'], a['diagnostics'], a['output'])


def mutate(path, change):
    data = json.loads(path.read_text())
    change(data)
    save(path, data)


def test_complete_three_arm_report_has_one_seed_eight_group_positive_gain(artifacts):
    execute(artifacts)
    result = json.loads(artifacts['output'].read_text())
    primary = result['primary_correct_vs_separately_trained_fixed']
    assert primary['seed_count'] == 1 and primary['log_count'] == 8
    assert primary['mean_gain_m'] == pytest.approx(.1, abs=1e-6)
    assert primary['ci95_m'] == pytest.approx([.1, .1], abs=1e-6)
    assert 'no training-seed uncertainty' in primary['seed_scope']


def test_stale_per_window_ade_cannot_change_paired_inference(artifacts):
    correct = copy.deepcopy(artifacts['endpoint']['full_correct_kv'])
    correct[0]['ADE'] = 100.
    with pytest.raises(ValueError, match='ADE|metric'):
        artifacts['report'].paired(correct, artifacts['endpoint']['full_fixed_kv'], artifacts['config'])


@pytest.mark.parametrize('damage', ['wrong_checkpoint', 'missing_mode', 'wrong_source', 'wrong_correct_prediction', 'stale_metric'])
def test_diagnostic_artifact_must_match_correct_checkpoint_and_all_conditions(artifacts, damage):
    def change(data):
        if damage == 'wrong_checkpoint':
            data['checkpoint_sha256'] = 'different'
        elif damage == 'missing_mode':
            data['records'].pop()
        elif damage == 'wrong_source':
            data['records'][3]['source_token'] = data['records'][3]['token']
        elif damage == 'wrong_correct_prediction':
            data['records'][0]['prediction'][0][0] += .1
        else:
            data['metrics']['correct']['ADE'] += .1
    mutate(artifacts['diagnostics'], change)
    with pytest.raises(ValueError, match='diagnostic|condition|source|metric|checkpoint|prediction'):
        execute(artifacts)


@pytest.mark.parametrize('damage', ['missing_check', 'failed_check', 'inexact_check', 'wrong_micro_count', 'wrong_model_count', 'partial_full_state', 'bad_reference_check'])
def test_partial_or_inexact_full_replay_and_reload_proofs_are_rejected(artifacts, damage):
    directory = artifacts['run']/'full_correct_kv'
    if damage in ('partial_full_state', 'bad_reference_check'):
        def change(data):
            if damage == 'partial_full_state':
                data['full_state_keys'] = 200
            else:
                data['reference_prediction_check']['passed'] = False
        mutate(directory/'verification.json', change)
    else:
        def change(data):
            if damage == 'missing_check':
                data['checks'].pop('optimizer')
            elif damage == 'failed_check':
                data['checks']['rng']['passed'] = False
            elif damage == 'inexact_check':
                data['checks']['model']['max_abs_difference'] = .1
            elif damage == 'wrong_model_count':
                data['checks']['model']['tensor_count'] = 200
            else:
                data['micro_batches'] = 1
        mutate(directory/'replay.json', change)
    with pytest.raises(ValueError, match='replay|state|reference|reload'):
        execute(artifacts)


@pytest.mark.parametrize('damage', ['cursor', 'duplicate_update', 'val_token', 'shuffle', 'partial_accumulation'])
def test_training_trace_must_contain_exact_train_only_accumulated_order(artifacts, damage):
    path = artifacts['run']/'full_fixed_kv/train.jsonl'
    data = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == 'cursor':
        data[9]['draw_index'] += 1
    elif damage == 'duplicate_update':
        data[9]['update'] = 9
    elif damage == 'val_token':
        data[9]['tokens'][0] = 'val-0'
    elif damage == 'shuffle':
        data[9]['tokens'].reverse()
    else:
        data[9]['accumulation_index'] = 1
    path.write_text(''.join(json.dumps(row)+'\n' for row in data))
    with pytest.raises(ValueError, match='train|TRAIN|cursor|order|update|accumulation'):
        execute(artifacts)


@pytest.mark.parametrize('damage', ['target', 'route', 'split', 'source'])
def test_endpoint_identity_is_checked_against_authenticated_val(artifacts, damage):
    directory = artifacts['run']/'full_correct_kv'
    def change(data):
        if damage == 'target':
            data[0]['target'][0][0] = 100.
        elif damage == 'route':
            data[0]['route'] = 'wrong-recording'
        elif damage == 'split':
            data[0]['split'] = 'train'
        else:
            data[0]['source_token'] = FIXED_SOURCE
    mutate(directory/'final_predictions.json', change)
    mutate(directory/'verification.json', lambda data:data.update(predictions_sha256=sha(directory/'final_predictions.json')))
    with pytest.raises(ValueError, match='target|recording|route|split|source|endpoint'):
        execute(artifacts)


@pytest.mark.parametrize('artifact', ['final.pt', 'before_final.pt'])
def test_report_authenticates_the_binary_checkpoints_behind_replay_proofs(artifacts, artifact):
    (artifacts['run']/'full_fast_only'/artifact).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint|SHA'):
        execute(artifacts)


@pytest.mark.parametrize('artifact', ['references', 'diagnostics'])
def test_descriptive_metrics_are_recomputed_from_complete_paired_records(artifacts, artifact):
    path=artifacts['reference'] if artifact=='references' else artifacts['diagnostics']
    def change(data):
        method='historical_fixed_seed_42' if artifact=='references' else 'shuffled'
        data['metrics'][method]['FDE']+=1.
    mutate(path,change)
    with pytest.raises(ValueError, match='metric'):
        execute(artifacts)
