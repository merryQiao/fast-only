"""Split-history launch contract; no native model construction or GPU work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'configs/navsim_v1_split_history_smoke.yaml'
FORMAL = ROOT / 'configs/navsim_v1_split_history_h200_2x8.yaml'


def test_split_history_frames_and_legacy_helpers_are_explicit():
    from fast_cluster.config import baseline_only, fast_frame_count, read_config, split_history

    config = read_config(SMOKE)
    assert split_history(config) and not baseline_only(config)
    assert config['train']['arms'] == ['fast_single', 'fast_history', 'slow_fast']
    assert [fast_frame_count(config, arm) for arm in config['train']['arms']] == [1, 4, 1]
    assert config['model']['slow_image_frames'] == 4
    legacy = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    original = read_config(ROOT / 'configs/navsim_v1_baseline_smoke.yaml')
    assert not split_history(legacy) and not split_history(original)
    assert not baseline_only(legacy) and baseline_only(original)
    assert fast_frame_count(legacy, 'slow_fast') == fast_frame_count(original, 'fast_only') == 4
    with pytest.raises(ValueError, match='arm'):
        fast_frame_count(config, 'fast_only')


def test_formal_split_config_uses_only_original_asset_roots(monkeypatch):
    from fast_cluster.config import read_config, validate_topology

    for name in ('FAST_BASE_MODEL', 'SLOW_BASE_MODEL', 'NAVSIM_MANIFEST_DIR', 'NAVSIM_SENSOR_ROOT'):
        monkeypatch.setenv(name, f'/relocated/{name.lower()}')
    monkeypatch.delenv('SLOW_SOURCE_BUNDLE', raising=False)
    config = read_config(FORMAL)
    assert set(config['assets']) == {'fast_model', 'slow_model'}
    assert config['train']['epochs'] == 8 and config['train']['seeds'] == [42, 43]
    assert config['source'] == dict(init='pretrained', seed=142, freeze_backbone=True, train_new_modules=True)
    assert config['data']['normalization'] == 'full_train_per_model'
    assert config['data']['validation_policy'] == 'official_default_train_val'
    smoke = read_config(SMOKE)
    for section in ('processor', 'model', 'source', 'vision', 'evaluation', 'reproducibility'):
        assert smoke[section] == config[section]
    formal_train, smoke_train = dict(config['train']), dict(smoke['train'])
    formal_train.pop('epochs'); smoke_train.pop('epochs')
    assert smoke_train == formal_train
    validate_topology(config, world_size=16, local_world_size=8)
    with pytest.raises(ValueError, match='topology'):
        validate_topology(config, world_size=1, local_world_size=1)


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('model', 'fast_image_frames', {'fast_single': 4, 'fast_history': 4, 'slow_fast': 1}),
    ('model', 'slow_image_frames', 1),
    ('source', 'freeze_backbone', False),
    ('source', 'train_new_modules', False),
    ('source', 'init', 'legacy_bundle'),
    ('train', 'source_feedback', 'frozen_source_teacher'),
    ('train', 'kv_dropout', .2),
    ('train', 'feedback_dropout', .2),
    ('vision', 'fast_execution', 'joint_temporal'),
    ('vision', 'runtime_history_cache', False),
    ('data', 'normalization', 'full_train_fast__preserved_source_slow'),
])
def test_split_protocol_rejects_silent_policy_changes(section, key, value):
    from fast_cluster.config import read_config, validate_config

    config = read_config(SMOKE)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_split_semantic_identity_relocates_but_never_skips_legacy_output(tmp_path):
    from fast_cluster.config import completed_run_matches, read_config, semantic_config

    split = read_config(SMOKE)
    moved = copy.deepcopy(split)
    moved['assets'] = dict(fast_model='/new/fast', slow_model='/new/slow')
    moved['data'].update(bundle='/new/data', sensor_root='/new/sensors', workers=8, prefetch_factor=4)
    assert semantic_config(split) == semantic_config(moved)
    assert semantic_config(split)['source']['seed'] == 142
    legacy = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    result = dict(completed=True, arm='slow_fast', seed=42)
    (tmp_path / 'result.json').write_text(json.dumps(result))
    identity = dict(semantic_config=semantic_config(legacy), arm='slow_fast', seed=42)
    (tmp_path / 'identity.json').write_text(json.dumps(identity))
    assert completed_run_matches(legacy, tmp_path, 'slow_fast', 42)
    with pytest.raises(ValueError, match='different.*protocol'):
        completed_run_matches(split, tmp_path, 'slow_fast', 42)
    identity['semantic_config'] = semantic_config(split)
    (tmp_path / 'identity.json').write_text(json.dumps(identity))
    assert completed_run_matches(moved, tmp_path, 'slow_fast', 42)


def test_split_formal_source_guard_requires_full_official_val_without_legacy_registry(monkeypatch):
    from fast_cluster import source_contract

    monkeypatch.setattr(source_contract, 'load_source_provenance',
                        lambda: pytest.fail('new pretrained source must not load legacy provenance'))
    config = {'schema': 'navsim_split_history_v2', 'execution': {'mode': 'formal'},
              'data': {'validation_policy': 'official_default_train_val'}}
    identity = {'kind': 'navsim_v1_full',
                'provenance': {'validation_policy': 'official_default_train_val'},
                'manifests': {'val': {'rows': 5, 'sha256': 'official-val'}},
                'artifacts': {'val_official.jsonl': {'rows': 5, 'sha256': 'official-val',
                                                    'purpose': 'report_only'}}}
    source_contract.validate_source_data(config, SimpleNamespace(identity=identity))
    identity['manifests']['val']['rows'] = 4
    with pytest.raises(ValueError, match='official.*VAL'):
        source_contract.validate_source_data(config, SimpleNamespace(identity=identity))


def test_split_preflight_authenticates_bases_and_stats_without_importing_task_models(tmp_path, monkeypatch):
    import sys
    import torch
    from fast_cluster.config import read_config
    from fast_cluster import data as data_module
    from tools import preflight_cluster

    def forbidden(*args, **kwargs):
        pytest.fail('identity preflight must not load task models or legacy task bundles')
    monkeypatch.setitem(sys.modules, 'fast_cluster.models',
                        SimpleNamespace(asset_manifest=forbidden, load_source=forbidden))
    config = read_config(SMOKE)
    for role in ('fast', 'slow'):
        root = tmp_path / role
        root.mkdir()
        (root / 'config.json').write_text('{}')
        (root / 'model.safetensors').write_bytes(role.encode())
        config['assets'][role + '_model'] = str(root)
    stats = dict(source_split='train', count=8, ego_mean=[0.] * 4, ego_std=[1.] * 4)
    data = SimpleNamespace(identity={'kind': 'smoke'}, stats=stats,
                           dataset=lambda split: range(8 if split == 'train' else 4))
    monkeypatch.setattr(data_module, 'PreparedData', lambda *args, **kwargs: data)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    result = preflight_cluster.run_preflight(config)
    assert set(result['source_assets']) == {'fast', 'slow'}
    assert result['source_bundle_sha256'] is None
    assert result['initialization'] == 'pretrained_split_history'
    assert result['source_initialization'] == config['source']
    assert result['model_statistics'] == dict(fast=stats, slow=stats)
    assert result['train_windows'] == 8 and result['val_windows'] == 4
    assert result['distributed_training_executed'] is False


def test_launcher_defaults_to_three_arms_and_rejects_legacy_completed_output(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from fast_cluster.config import read_config, semantic_config

    wrapper = tmp_path / 'launch_recorder'
    wrapper.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
if sys.argv[1] == '-c':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
with open(os.environ['CLUSTER_LAUNCH_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    wrapper.chmod(0o755)
    log = tmp_path / 'launches.jsonl'
    for name, value in dict(MASTER_ADDR='127.0.0.1', NODE_RANK='0',
                            FAST_BASE_MODEL='/local/fast', SLOW_BASE_MODEL='/local/slow',
                            NAVSIM_MANIFEST_DIR='/local/manifests', NAVSIM_SENSOR_ROOT='/local/sensors',
                            TRAIN_OUTPUT_ROOT=str(tmp_path / 'training'), CLUSTER_PYTHON=str(wrapper),
                            CLUSTER_LAUNCH_LOG=str(log), RESUME_MATRIX='0').items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('CLUSTER_CONFIG', raising=False)
    monkeypatch.delenv('SLOW_SOURCE_BUNDLE', raising=False)
    script = ROOT / 'scripts/train_navsim_v1_2x8.sh'
    subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True, text=True)
    subprocess.run(['bash', str(script)], check=True, capture_output=True, text=True, env=os.environ.copy())
    launches = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(launches) == 6
    actual = {(row[row.index('--arm') + 1], row[row.index('--seed') + 1]) for row in launches}
    assert actual == {(arm, seed) for arm in ('fast_single', 'fast_history', 'slow_fast') for seed in ('42', '43')}
    assert all('--nnodes=2' in row and '--nproc-per-node=8' in row for row in launches)
    assert all(row[row.index('--config') + 1] == 'configs/navsim_v1_split_history_h200_2x8.yaml' for row in launches)
    legacy = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    completed = tmp_path / 'training' / 'seed_42' / 'slow_fast'
    completed.mkdir(parents=True)
    (completed / 'result.json').write_text(json.dumps(dict(completed=True, arm='slow_fast', seed=42)))
    (completed / 'identity.json').write_text(json.dumps(dict(semantic_config=semantic_config(legacy), arm='slow_fast', seed=42)))
    monkeypatch.setenv('RESUME_MATRIX', '1')
    rejected = subprocess.run(['bash', str(script)], capture_output=True, text=True, env=os.environ.copy())
    assert rejected.returncode != 0
    assert 'different training protocol' in rejected.stderr
