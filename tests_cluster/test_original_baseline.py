"""Original Fast-only must not require any prior Slow experiment artifact."""
import copy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_original_fast_only_constructs_without_loading_slow_or_bundle(monkeypatch):
    from fast_cluster import models

    class Fast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
    fast = Fast()
    monkeypatch.setattr(models, '_new_fast', lambda config, device: (fast, {}, {}))
    monkeypatch.setattr(models, 'NavsimCollator', lambda *args, **kwargs: object())
    monkeypatch.setattr(models, 'asset_manifest', lambda path: {'model.safetensors': 'base-fast-hash'})
    def forbidden(*args, **kwargs):
        pytest.fail('Fast-only tried to access a previous Slow/teacher artifact')
    monkeypatch.setattr(models, 'load_source', forbidden)
    config = {'assets': {'fast_model': '/original/0.8B'}, 'processor': {},
              'train': {'arms': ['fast_only']}}
    stats = {'ego_mean': [0.]*4, 'ego_std': [1.]*4}
    backend = models.build_backend(config, stats, 'fast_only', 42, 'cpu')
    assert backend.provider is None
    assert backend.model_identity['assets'] == {'fast': {'model.safetensors': 'base-fast-hash'}}
    assert backend.model_identity['source_bundle'] is None
    assert backend.model_identity['source_stats'] is None
    assert backend.model_identity['feedback'] == 'none'


def test_baseline_only_config_needs_only_fast_base_and_official_val(monkeypatch):
    from fast_cluster.config import read_config, validate_config

    config = read_config(ROOT/'configs/navsim_v1_smoke.yaml')
    config['assets'] = {'fast_model': '/original/0.8B'}
    config['train'].update(arms=['fast_only'], source_feedback='none')
    config['data']['normalization'] = 'full_train_fast'
    validate_config(config)
    formal = copy.deepcopy(config)
    formal['execution'].update(mode='formal', expected_world_size=16, expected_local_world_size=8)
    formal['train']['epochs'] = 8
    formal['data']['validation_policy'] = 'official_default_train_val'
    validate_config(formal)


def test_original_baseline_rejects_legacy_filtered_val_without_reading_source_registry(monkeypatch):
    from types import SimpleNamespace
    from fast_cluster import source_contract

    monkeypatch.setattr(source_contract, 'load_source_provenance',
        lambda: pytest.fail('original baseline must not depend on old source provenance'))
    config = {'execution': {'mode': 'formal'}, 'train': {'arms': ['fast_only']},
              'data': {'validation_policy': 'official_default_train_val'}}
    identity = {'kind': 'navsim_v1_full',
        'provenance': {'validation_policy': 'official_default_train_val'},
        'manifests': {'val': {'rows': 3, 'sha256': 'full-official-val'}},
        'artifacts': {'val_official.jsonl': {'rows': 3, 'sha256': 'full-official-val',
                                           'purpose': 'report_only'}}}
    data = SimpleNamespace(identity=identity)
    source_contract.validate_source_data(config, data)
    filtered = copy.deepcopy(identity)
    filtered['manifests']['val'].update(rows=2, sha256='filtered-val')
    with pytest.raises(ValueError, match='official.*VAL'):
        source_contract.validate_source_data(config, SimpleNamespace(identity=filtered))
    filtered = copy.deepcopy(identity)
    filtered['provenance']['validation_policy'] = 'exclude_upstream_training_groups'
    with pytest.raises(ValueError, match='official.*VAL'):
        source_contract.validate_source_data(config, SimpleNamespace(identity=filtered))


def test_completed_legacy_run_is_not_silently_skipped_as_original_baseline(tmp_path):
    import json
    from fast_cluster.config import read_config, semantic_config, completed_run_matches

    legacy = read_config(ROOT/'configs/navsim_v1_smoke.yaml')
    original = read_config(ROOT/'configs/navsim_v1_baseline_smoke.yaml')
    (tmp_path/'result.json').write_text(json.dumps({'completed': True, 'arm': 'fast_only', 'seed': 42}))
    (tmp_path/'identity.json').write_text(json.dumps({'semantic_config': semantic_config(legacy),
                                                    'arm': 'fast_only', 'seed': 42}))
    assert completed_run_matches(legacy, tmp_path, 'fast_only', 42)
    with pytest.raises(ValueError, match='different.*protocol'):
        completed_run_matches(original, tmp_path, 'fast_only', 42)
    with pytest.raises(ValueError, match='different.*protocol'):
        completed_run_matches(legacy, tmp_path, 'fast_only', 43)
