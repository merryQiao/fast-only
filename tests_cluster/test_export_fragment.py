"""Export authenticated causal runtime rows without reading sensor JPEG bytes."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image


def fixture(tmp_path, gapped=False):
    from fast_cluster.data import prepare_smoke_data
    from tests_cluster.test_data import smoke_fixture, frames_for, legacy_row, write_jsonl
    root, train_path, val_path, train, val = smoke_fixture(tmp_path)
    if gapped:
        frames = frames_for(root, train[0]['route'], 28)
        train = [legacy_row(root, frames, 'train', index) for index in [3, 4, 6, 7, 9, 10, 12, 13]]
        write_jsonl(train_path, train)
    bundle = tmp_path/'bundle'
    prepare_smoke_data(train_path, val_path, root, bundle)
    return bundle, root, train, val


def reauthenticate_manifest(bundle, mutate):
    from fast_cluster.data import _digest
    path = bundle/'train.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = mutate(rows)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    identity_path = bundle/'identity.json'
    identity = json.loads(identity_path.read_text())
    identity['manifests']['train']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity.pop('identity_sha256')
    identity['identity_sha256'] = _digest(identity)
    identity_path.write_text(json.dumps(identity))


def test_export_is_input_only_preserves_sources_and_roundtrips_without_image_reads(tmp_path, monkeypatch):
    from tools.export_cluster_fragment import export_fragment
    from fast_cluster.data import PreparedData, SceneDataset
    from fast_cluster.async_runtime import load_fragment
    bundle, root, original, _ = fixture(tmp_path)
    identity = PreparedData(bundle, root, allow_smoke=True).identity
    real_open = Path.open

    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            pytest.fail('exporter opened sensor image bytes')
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    monkeypatch.setattr(Image, 'open', lambda *args, **kwargs: pytest.fail('exporter decoded an image'))
    monkeypatch.setattr(SceneDataset, '__getitem__', lambda *args: pytest.fail('exporter loaded annotated model sample'))
    out = tmp_path/'runtime'
    receipt = export_fragment(bundle, root, 'train', 4, out, allow_smoke=True)
    rows = load_fragment(out/'fragment.json')
    assert len(rows) == 4 and receipt['windows'] == 4
    assert receipt['bundle_identity_sha256'] == identity['identity_sha256']
    assert receipt['manifest']['sha256'] == identity['manifests']['train']['sha256']
    for exported, expected in zip(rows, original):
        assert exported['ego'] == expected['ego']
        assert exported['ego_pose'] == expected['ego_pose']
        assert exported['command'] == expected['command']
        assert exported['timestamp_us'] == expected['timestamp_us']
        assert exported['token'] == expected['token']
        assert [Path(p).resolve() for p in exported['image_paths']] == [Path(p).resolve() for p in expected['images']]
        assert not any(k == 'trajectory_gt' or k.startswith('future') for k in exported)
    assert (out/'sensors').is_symlink()
    assert receipt['runtime_only'] is True and receipt['quality_evaluated'] is False
    assert receipt['sensor_bytes_read'] == 0 and receipt['images_copied'] == 0


def test_missing_contiguous_sequence_fails_instead_of_filling_gaps(tmp_path):
    from tools.export_cluster_fragment import export_fragment
    bundle, root, _, _ = fixture(tmp_path, gapped=True)
    with pytest.raises(ValueError, match='no_contiguous_sequence.*longest=2'):
        export_fragment(bundle, root, 'train', 3, tmp_path/'runtime', allow_smoke=True)
    assert not (tmp_path/'runtime').exists()


@pytest.mark.parametrize('escape', ['parent', 'symlink'])
def test_export_paths_cannot_escape_sensor_root(tmp_path, escape):
    from tools.export_cluster_fragment import export_fragment
    bundle, root, rows, _ = fixture(tmp_path)
    if escape == 'parent':
        def change(rows):
            rows[0]['images'][0] = '../outside.jpg'
            return rows
        reauthenticate_manifest(bundle, change)
    else:
        selected = Path(rows[0]['images'][0])
        selected.unlink()
        outside = tmp_path/'outside.jpg'; outside.write_bytes(b'not opened')
        selected.symlink_to(outside)
    with pytest.raises(ValueError, match='image_path_escape'):
        export_fragment(bundle, root, 'train', 2, tmp_path/'runtime', allow_smoke=True)


def test_smoke_requires_explicit_flag_and_episode_filter_is_exact(tmp_path):
    from tools.export_cluster_fragment import export_fragment
    bundle, root, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='allow_smoke'):
        export_fragment(bundle, root, 'train', 2, tmp_path/'runtime')
    with pytest.raises(ValueError, match='no_contiguous_sequence'):
        export_fragment(bundle, root, 'train', 2, tmp_path/'runtime',
                        episode_id='different-log', allow_smoke=True)


def test_manifest_authentication_precedes_export(tmp_path):
    from tools.export_cluster_fragment import export_fragment
    bundle, root, _, _ = fixture(tmp_path)
    with (bundle/'train.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='manifest.*identity'):
        export_fragment(bundle, root, 'train', 2, tmp_path/'runtime', allow_smoke=True)


def test_selection_is_deterministic_by_log_and_time_not_manifest_order(tmp_path):
    from tools.export_cluster_fragment import export_fragment
    from fast_cluster.async_runtime import load_fragment
    bundle, root, original, _ = fixture(tmp_path)
    reauthenticate_manifest(bundle, lambda rows: list(reversed(rows)))
    out = tmp_path/'runtime'
    export_fragment(bundle, root, 'train', 3, out, allow_smoke=True)
    rows = load_fragment(out/'fragment.json')
    assert [row['token'] for row in rows] == [row['token'] for row in original[:3]]
