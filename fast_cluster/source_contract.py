"""Require full official VAL for pretrained-only protocols and isolate legacy data.

The old Mini split differs from official V1 roles. All new TRAIN remains in
place in legacy runs; six official VAL logs from one already-seen recording group
are retained for reporting and excluded from legacy checkpoint selection. New
pretrained split-history and original baseline runs use all official default VAL.
"""
import json
from pathlib import Path
from .config import baseline_only, split_history


def load_source_provenance():
    path = Path(__file__).with_name('source_provenance.json')
    source = json.loads(path.read_text())
    if source.get('schema') != 'navsim_frozen_source_provenance_v1':
        raise ValueError('Frozen source provenance schema differs')
    from .data import recording_group
    if sorted(set(map(recording_group, source['train_logs']))) != source['train_recording_groups']:
        raise ValueError('Frozen source TRAIN group provenance differs')
    return source


def validate_source_data(config, data):
    if config['execution']['mode'] == 'smoke':
        return
    if baseline_only(config) or split_history(config):
        identity = data.identity
        provenance = identity.get('provenance', {})
        selected = identity.get('manifests', {}).get('val', {})
        official = identity.get('artifacts', {}).get('val_official.jsonl', {})
        if (identity.get('kind') != 'navsim_v1_full'
                or config['data'].get('validation_policy') != 'official_default_train_val'
                or provenance.get('validation_policy') != 'official_default_train_val'
                or provenance.get('source_provenance') is not None
                or official.get('purpose') != 'report_only'
                or not selected.get('rows') or selected.get('rows') != official.get('rows')
                or not selected.get('sha256') or selected.get('sha256') != official.get('sha256')):
            raise ValueError('Pretrained-only protocols require complete official default VAL; '
                             'prepare data with --official-default-val, without old source exclusions')
        return
    expected = load_source_provenance()
    provenance = data.identity.get('provenance', {})
    if (data.identity.get('kind') != 'navsim_v1_full'
            or config['assets']['source_checkpoint_sha256'] != expected['source_checkpoint_sha256']
            or config['data'].get('validation_policy') != 'exclude_upstream_training_groups'
            or provenance.get('validation_policy') != 'exclude_upstream_training_groups'
            or provenance.get('source_provenance') != expected):
        raise ValueError('Formal data must bind current frozen source provenance and exclude its TRAIN groups')
    audit = json.loads((Path(data.bundle_dir)/'audit.json').read_text())
    groups = audit['role_recording_groups']['val']
    if not groups or set(groups) & set(expected['train_recording_groups']):
        raise ValueError('Selection VAL contains source-trained recording groups')


def validate_bundle_provenance(bundle):
    """Bind log provenance to the manifest digest embedded in original source."""
    expected = load_source_provenance()
    if (bundle['source_checkpoint_sha256'] != expected['source_checkpoint_sha256']
            or bundle['upstream_data_identity'].get('train_manifest') != expected['train_manifest_sha256']):
        raise ValueError('Frozen source upstream TRAIN manifest differs from registered provenance')
    return expected
