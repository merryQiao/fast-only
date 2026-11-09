import json
import pickle
import shutil
from pathlib import Path

import pytest
from PIL import Image


def api():
    from fast_cluster import data
    return data


def frames_for(root, log="2021.01.01.00.00.00_veh-1_00000_00099", count=14):
    frames = []
    for index in range(count):
        relative = f"{log}/CAM_F0/{index}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (17, 11), (index, 2, 3)).save(path)
        frames.append({"log_name": log, "token": f"{log}-token-{index}",
            "scene_token": "scene", "timestamp": index * 500_000,
            "roadblock_ids": ["route"], "ego2global_translation": [index, 0, 0],
            "ego2global_rotation": [1, 0, 0, 0],
            "ego_dynamic_state": [float(index), 0, 1, 0],
            "driving_command": [0, 1, 0, 0],
            "cams": {"CAM_F0": {"data_path": relative}}})
    return frames


def legacy_row(root, frames, split, index=3):
    from fast_navsim.data import build_window
    row = build_window(frames, index, split, root)
    row["ego_pose"] = [float(index), 0, 0]
    return row


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def smoke_fixture(tmp_path):
    root = tmp_path / "sensors"
    train_frames = frames_for(root, "2021.01.01.00.00.00_veh-1_00000_00099", 22)
    val_frames = frames_for(root, "2021.01.02.00.00.00_veh-1_00000_00099", 18)
    train = [legacy_row(root, train_frames, "train", index) for index in range(3, 11)]
    val = [legacy_row(root, val_frames, "val", index) for index in range(3, 7)]
    for row in val:
        row["ego"] = [1000, 1000, 1000, 1000]
    train_path, val_path = tmp_path / "train_source.jsonl", tmp_path / "val_source.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    return root, train_path, val_path, train, val


def test_pinned_official_contract_has_independent_default_roles():
    contract = api().load_v1_contract()
    assert contract["revision"] == "3e8291bfa89ff247231e0227778840cd0a036896"
    scene = contract["navtrain"]
    assert [scene[key] for key in ("num_history_frames", "num_future_frames", "frame_interval")] == [4, 10, 1]
    assert scene["has_route"] is True
    assert len(scene["tokens"]) == 103288
    assert len(contract["train_logs"]) == 978
    assert len(contract["val_logs"]) == 214
    assert not set(contract["train_logs"]) & set(contract["val_logs"])
    assert not set(scene["tokens"]) & set(contract["navtest"]["tokens"])


def test_official_scene_requires_ten_future_but_emits_eight(tmp_path):
    data = api()
    root = tmp_path / "sensors"
    frames = frames_for(root)
    row = data.build_scene_row(frames, 3, "train", root)
    assert row["trajectory_gt"] == [[float(i), 0.0, 0.0] for i in range(1, 9)]
    assert row["ego_pose"] == [3.0, 0.0, 0.0]
    assert row["images"] == [frames[i]["cams"]["CAM_F0"]["data_path"] for i in range(4)]
    with pytest.raises(ValueError, match="official_future_frames"):
        data.build_scene_row(frames[:12], 3, "train", root)


@pytest.mark.parametrize("mutation,reason", [
    ("route", "official_no_route"),
    ("gap", "timestamp_gap"),
    ("command", "invalid_command"),
    ("pose", "invalid_pose"),
    ("image", "missing_image"),
    ("escape", "image_path_escape"),
])
def test_scene_rejections_have_specific_auditable_reasons(tmp_path, mutation, reason):
    root = tmp_path / "sensors"
    frames = frames_for(root)
    if mutation == "route": frames[3]["roadblock_ids"] = []
    if mutation == "gap": frames[5]["timestamp"] += 1001
    if mutation == "command": frames[3]["driving_command"] = [1, 1, 0, 0]
    if mutation == "pose": frames[0]["ego2global_rotation"] = [0, 0, 0, 0]
    if mutation == "image": (root / frames[0]["cams"]["CAM_F0"]["data_path"]).unlink()
    if mutation == "escape": frames[0]["cams"]["CAM_F0"]["data_path"] = "../bad.jpg"
    with pytest.raises(ValueError, match=reason):
        api().build_scene_row(frames, 3, "train", root)


def test_formal_missing_logs_fails_with_explicit_audit(tmp_path):
    root = tmp_path / "sensors"
    root.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="missing_official_logs"):
        api().prepare_official_data(metadata, root, output)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["status"] == "failed"
    assert audit["missing_official_log_count"] == 1192
    assert not (output / "identity.json").exists()


def test_official_builder_does_not_accept_nonallowlisted_anchor(tmp_path):
    data = api()
    contract = data.load_v1_contract()
    root, metadata = tmp_path / "sensors", tmp_path / "metadata"
    metadata.mkdir()
    log = contract["train_logs"][0]
    frames = frames_for(root, log)
    with (metadata / f"{log}.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="missing_official_logs"):
        data.prepare_official_data(metadata, root, output)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["accepted_counts"] == {"train": 0, "val": 0}


def official_fixture(tmp_path, monkeypatch):
    """Small metadata tree at the contract-loader boundary, without downloads."""
    data = api()
    real = data.load_v1_contract()
    contract = {**real, "train_logs": real["train_logs"][:1], "val_logs": real["val_logs"][:1],
        "navtrain": {**real["navtrain"], "tokens": real["navtrain"]["tokens"][:2]}}
    monkeypatch.setattr(data, "load_v1_contract", lambda *args: contract)
    metadata, root = tmp_path / "metadata", tmp_path / "sensors"
    metadata.mkdir()
    for split, token in zip(("train", "val"), contract["navtrain"]["tokens"]):
        log = contract[f"{split}_logs"][0]
        frames = frames_for(root, log)
        frames[3]["token"] = token
        with (metadata / f"{log}.pkl").open("wb") as stream:
            pickle.dump(frames, stream)
    return data, metadata, root, contract


def test_official_complete_membership_builds_full_kind(tmp_path, monkeypatch):
    data, metadata, root, contract = official_fixture(tmp_path, monkeypatch)
    output = tmp_path / "bundle"
    receipt = data.prepare_official_data(metadata, root, output)
    assert receipt["kind"] == "navsim_v1_full"
    prepared = data.PreparedData(output, root)
    assert len(prepared.dataset("train")) == len(prepared.dataset("val")) == 1
    assert prepared.dataset("train").row(0)["token"] == contract["navtrain"]["tokens"][0]
    audit = json.loads((output / "audit.json").read_text())
    assert audit["observed_official_tokens"] == 2
    assert audit["missing_official_token_count"] == 0
    assert audit["rejection_fraction"] == 0
    assert str(tmp_path) not in json.dumps(prepared.identity)


def test_official_rejections_are_not_silently_excluded(tmp_path, monkeypatch):
    data, metadata, root, contract = official_fixture(tmp_path, monkeypatch)
    path = metadata / f"{contract['val_logs'][0]}.pkl"
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    frames[3]["roadblock_ids"] = []
    with path.open("wb") as stream:
        pickle.dump(frames, stream)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="rejection_fraction_exceeded"):
        data.prepare_official_data(metadata, root, output)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["rejected_counts"] == {"official_no_route": 1}
    assert audit["rejection_rates"] == {"official_no_route": 0.5}
    assert audit["rejection_fraction"] == 0.5
    assert audit["status"] == "failed"
    assert not (output / "identity.json").exists()


def test_official_missing_allowed_token_fails_even_with_rejection_allowance(tmp_path, monkeypatch):
    data, metadata, root, contract = official_fixture(tmp_path, monkeypatch)
    path = metadata / f"{contract['val_logs'][0]}.pkl"
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    frames[3]["token"] = "not-official"
    with path.open("wb") as stream:
        pickle.dump(frames, stream)
    with pytest.raises(ValueError, match="missing_official_tokens"):
        data.prepare_official_data(metadata, root, tmp_path / "bundle", max_rejection_fraction=0.99)


def test_smoke_bundle_is_portable_train_normalized_and_lazy(tmp_path):
    data = api()
    root, train_path, val_path, train, _ = smoke_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    receipt = data.prepare_smoke_data(train_path, val_path, root, bundle)
    assert receipt["kind"] == "smoke"
    with pytest.raises(ValueError, match="allow_smoke"):
        data.PreparedData(bundle, root)
    prepared = data.PreparedData(bundle, root, allow_smoke=True)
    assert prepared.identity["kind"] == "smoke"
    assert prepared.stats["count"] == 8
    assert prepared.stats["ego_mean"] == [6.5, 0, 1, 0]
    dataset = prepared.dataset("train")
    assert len(dataset) == 8
    assert not any(isinstance(value, list) and value and isinstance(value[0], dict) for value in vars(dataset).values())
    row = dataset.row(0)
    assert all(not Path(path).is_absolute() for path in row["images"])
    sample = dataset[(0, False)]
    assert sample["valid"] is False
    assert [image.size for image in sample["images"]] == [(17, 11)] * 4
    assert sample["ego"] == train[0]["ego"]
    assert sample["command"] == 1
    assert sample["ego_pose"] == [3.0, 0.0, 0.0]
    assert data.identity_collate([sample]) == [sample]
    reloaded = pickle.loads(pickle.dumps(dataset))
    assert reloaded.row(0) == row
    new_root = tmp_path / "relocated"
    shutil.copytree(root, new_root)
    assert data.PreparedData(bundle, new_root, allow_smoke=True).identity == prepared.identity
    assert data.PreparedData(bundle, new_root, allow_smoke=True).dataset("train")[0]["images"][0].size == (17, 11)


def test_content_identity_does_not_include_absolute_roots(tmp_path):
    data = api()
    root, train_path, val_path, _, _ = smoke_fixture(tmp_path)
    data.prepare_smoke_data(train_path, val_path, root, tmp_path / "bundle-a")
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    for source in [train_path, val_path]:
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        for row in rows:
            row["images"] = [str(moved / Path(path).relative_to(root)) for path in row["images"]]
        write_jsonl(source, rows)
    data.prepare_smoke_data(train_path, val_path, moved, tmp_path / "bundle-b")
    first = json.loads((tmp_path / "bundle-a/identity.json").read_text())
    second = json.loads((tmp_path / "bundle-b/identity.json").read_text())
    assert first == second
    assert str(tmp_path) not in json.dumps(first)


def test_smoke_rejects_cross_segment_recording_group_leak(tmp_path):
    data = api()
    root, train_path, val_path, train, val = smoke_fixture(tmp_path)
    for row in val:
        row["route"] = train[0]["route"].rsplit("_", 2)[0] + "_00100_00199"
    write_jsonl(val_path, val)
    with pytest.raises(ValueError, match="recording_group_overlap"):
        data.prepare_smoke_data(train_path, val_path, root, tmp_path / "bundle")


def test_manifest_tampering_is_detected_before_loading(tmp_path):
    data = api()
    root, train_path, val_path, _, _ = smoke_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    data.prepare_smoke_data(train_path, val_path, root, bundle)
    with (bundle / "train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="manifest.*identity"):
        data.PreparedData(bundle, root, allow_smoke=True)


def test_valid_jpeg_byte_mutation_fails_lazily_before_decode(tmp_path):
    data = api()
    root, train_path, val_path, train, _ = smoke_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    data.prepare_smoke_data(train_path, val_path, root, bundle)
    identity = (bundle / "identity.json").read_bytes()
    prepared = data.PreparedData(bundle, root, allow_smoke=True)
    dataset = prepared.dataset("train")
    assert dataset[0]["images"][0].size == (17, 11)
    Image.new("RGB", (17, 11), (90, 80, 70)).save(train[0]["images"][0])
    assert (bundle / "identity.json").read_bytes() == identity
    assert dataset.row(0)["token"] == train[0]["token"]
    with pytest.raises(ValueError, match="image.*identity"):
        dataset[0]


def test_preparation_rejects_header_valid_truncated_jpeg(tmp_path):
    root = tmp_path / "sensors"
    frames = frames_for(root)
    path = root / frames[0]["cams"]["CAM_F0"]["data_path"]
    path.write_bytes(path.read_bytes()[:-2])
    with Image.open(path) as image:
        image.verify()  # Header verification alone does not catch this corruption.
    with pytest.raises(ValueError, match="invalid_image"):
        api().build_scene_row(frames, 3, "train", root)


def test_source_training_group_is_excluded_only_from_selection_val(tmp_path, monkeypatch):
    data = api()
    real = data.load_v1_contract()
    excluded_log = next(log for log in real["val_logs"]
        if sum(data.recording_group(other) == data.recording_group(log) for other in real["val_logs"]) >= 2)
    excluded_group = data.recording_group(excluded_log)
    same_group = next(log for log in real["val_logs"]
        if log != excluded_log and data.recording_group(log) == excluded_group)
    clean_log = next(log for log in real["val_logs"] if data.recording_group(log) != excluded_group)
    val_logs = [excluded_log, same_group, clean_log]
    contract = {**real, "train_logs": real["train_logs"][:1], "val_logs": val_logs,
        "navtrain": {**real["navtrain"], "tokens": real["navtrain"]["tokens"][:4]}}
    monkeypatch.setattr(data, "load_v1_contract", lambda *args: contract)
    metadata, root = tmp_path / "metadata", tmp_path / "sensors"
    metadata.mkdir()
    for log, token in zip(contract["train_logs"] + val_logs, contract["navtrain"]["tokens"]):
        frames = frames_for(root, log)
        frames[3]["token"] = token
        with (metadata / f"{log}.pkl").open("wb") as stream:
            pickle.dump(frames, stream)
    provenance = {"schema": "navsim_frozen_source_provenance_v1",
        "source_checkpoint_sha256": "a" * 64, "train_manifest_sha256": "b" * 64,
        "train_logs": [excluded_log], "train_recording_groups": [excluded_group]}
    registry_path = tmp_path / "provenance.json"
    registry_path.write_text(json.dumps(provenance))
    output = tmp_path / "bundle"
    data.prepare_official_data(metadata, root, output, source_provenance=registry_path)
    prepared = data.PreparedData(output, root)
    assert len(prepared.dataset("train")) == 1
    assert len(prepared.dataset("val")) == 1
    assert prepared.dataset("val").row(0)["route"] == clean_log
    report_only = prepared.dataset("val_official")
    assert report_only.report_only is True
    assert len(report_only) == 3
    assert {report_only.row(index)["route"] for index in range(3)} == set(val_logs)
    identity = prepared.identity
    assert identity["provenance"]["source_provenance"] == provenance
    assert identity["provenance"]["validation_policy"] == "exclude_upstream_training_groups"
    assert identity["artifacts"]["val_official.jsonl"]["rows"] == 3
    assert identity["artifacts"]["val_official.jsonl"]["purpose"] == "report_only"
    audit = json.loads((output / "audit.json").read_text())
    assert audit["observed_official_tokens"] == 4
    assert audit["missing_official_token_count"] == 0
    assert audit["accepted_official_counts"] == {"train": 1, "val": 3}
    assert audit["accepted_counts"] == {"train": 1, "val": 1}
    assert audit["rejected_counts"] == {}
    assert audit["rejection_fraction"] == 0
    assert audit["role_logs"]["val"] == [clean_log]
    assert audit["role_recording_groups"]["val"] == [data.recording_group(clean_log)]
    exclusions = audit["source_exclusions"]
    assert exclusions["token_count"] == 2
    assert exclusions["log_count"] == 2
    assert exclusions["recording_group_count"] == 1
    assert exclusions["source_checkpoint_sha256"] == provenance["source_checkpoint_sha256"]
    assert exclusions["train_manifest_sha256"] == provenance["train_manifest_sha256"]
    assert prepared.stats["count"] == 1


def test_source_provenance_rejects_incomplete_group_definition(tmp_path):
    provenance = {"schema": "navsim_frozen_source_provenance_v1",
        "source_checkpoint_sha256": "a" * 64, "train_manifest_sha256": "b" * 64,
        "train_logs": ["2021.01.01_veh-1_00000_00100"], "train_recording_groups": []}
    with pytest.raises(ValueError, match="source_provenance.*groups"):
        api().prepare_official_data(tmp_path, tmp_path, tmp_path / "bundle", source_provenance=provenance)
