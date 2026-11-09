import json
from types import SimpleNamespace

import pytest

from fast_cluster.source_contract import load_source_provenance, validate_source_data


def test_formal_source_guard_rejects_unexcluded_source_seen_val(tmp_path):
    source = load_source_provenance()
    config = {'execution': {'mode': 'formal'},
              'assets': {'source_checkpoint_sha256': source['source_checkpoint_sha256']},
              'data': {'validation_policy': 'exclude_upstream_training_groups'}}
    data = SimpleNamespace(bundle_dir=tmp_path, identity={'kind': 'navsim_v1_full',
        'provenance': {'source_provenance': source,
                       'validation_policy': 'exclude_upstream_training_groups'}})
    (tmp_path/'audit.json').write_text(json.dumps({'role_recording_groups': {
        'val': [source['train_recording_groups'][0]]}}))
    with pytest.raises(ValueError, match='source-trained'):
        validate_source_data(config, data)
    (tmp_path/'audit.json').write_text(json.dumps({'role_recording_groups': {'val': ['unseen_log_group']}}))
    validate_source_data(config, data)
    data.identity['provenance'].pop('source_provenance')
    with pytest.raises(ValueError, match='provenance'):
        validate_source_data(config, data)


def test_source_registry_identifies_all_six_contaminated_official_val_logs():
    from fast_cluster.data import load_v1_contract, recording_group
    source = load_source_provenance()
    groups = set(source['train_recording_groups'])
    excluded = [log for log in load_v1_contract()['val_logs'] if recording_group(log) in groups]
    assert len(excluded) == 6
    assert {recording_group(log) for log in excluded} == {'2021.06.08.12.54.54_veh-26'}


def test_smoke_identity_is_not_relabelled_as_formal():
    validate_source_data({'execution': {'mode': 'smoke'}}, None)
