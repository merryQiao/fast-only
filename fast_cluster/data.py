"""Portable NAVSIM V1 data bundles and lazy, worker-safe JSONL datasets.

Preparation is offline. Official scene membership stays at four history and ten
future frames; the model consumes four original CAM_F0 images and eight targets.
The 1 ms timestamp tolerance is an explicit project policy, not an upstream rule.
"""
from array import array
from collections import Counter
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import mmap
from pathlib import Path
import pickle

from PIL import Image


NAVSIM_REVISION = "3e8291bfa89ff247231e0227778840cd0a036896"
CONTRACT_ROOT = Path(__file__).with_name("v1_contract")
TIMESTAMP_INTERVAL_US = 500_000
TIMESTAMP_TOLERANCE_US = 1000
HISTORY_TIMES = [-1.5, -1.0, -0.5, 0.0]
FUTURE_TIMES = [0.5 * index for index in range(1, 9)]
SCHEMA = "navsim_cluster_data_v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def recording_group(log_name):
    parts = log_name.rsplit("_", 2)
    return parts[0] if len(parts) == 3 and all(item.isdigit() for item in parts[1:]) else log_name


@lru_cache(maxsize=4)
def load_v1_contract(devkit_root=None):
    """Verify vendored config hashes, optionally checking a local official devkit.

The returned lists are immutable by convention. A local devkit supplies matching
source bytes only; it cannot replace the pinned token or role definitions.
"""
    import yaml

    identity = json.loads((CONTRACT_ROOT / "identity.json").read_text())
    if identity["revision"] != NAVSIM_REVISION:
        raise ValueError("NAVSIM contract revision differs")
    loaded = {}
    for filename, item in identity["files"].items():
        path = CONTRACT_ROOT / filename
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"NAVSIM contract source identity differs: {filename}")
        if devkit_root is not None:
            upstream = Path(devkit_root) / item["source_path"]
            if not upstream.is_file() or _sha256(upstream) != item["sha256"]:
                raise ValueError(f"local devkit contract identity differs: {item['source_path']}")
        if filename.endswith(".yaml"):
            loaded[filename] = yaml.safe_load(path.read_text())
    navtrain, navtest, split = loaded["navtrain.yaml"], loaded["navtest.yaml"], loaded["log_split.yaml"]
    if any(navtrain[key] != value for key, value in {
        "num_history_frames": 4, "num_future_frames": 10, "frame_interval": 1,
        "has_route": True, "max_scenes": None,
    }.items()):
        raise ValueError("unsupported NAVSIM V1 scene filter")
    train_logs = sorted(set(navtrain["log_names"]) & set(split["train_logs"]))
    val_logs = sorted(set(navtrain["log_names"]) & set(split["val_logs"]))
    if (len(train_logs), len(val_logs), len(set(navtrain["tokens"]))) != (978, 214, 103288):
        raise ValueError("NAVSIM V1 official membership differs")
    if set(train_logs) | set(val_logs) != set(navtrain["log_names"]):
        raise ValueError("unassigned NAVSIM navtrain logs")
    if set(navtrain["tokens"]) & set(navtest["tokens"]):
        raise ValueError("NAVSIM navtest token overlap")
    _check_role_leaks({"train": train_logs, "val": val_logs})
    return {"revision": NAVSIM_REVISION, "identity": identity, "navtrain": navtrain,
            "navtest": navtest, "train_logs": train_logs, "val_logs": val_logs}


def _check_role_leaks(logs):
    train, val = set(logs["train"]), set(logs["val"])
    if train & val:
        raise ValueError("log_overlap between TRAIN and VAL")
    if set(map(recording_group, train)) & set(map(recording_group, val)):
        raise ValueError("recording_group_overlap between TRAIN and VAL")


def _pose(frame):
    try:
        q = [float(value) for value in frame["ego2global_rotation"]]
        xy = [float(value) for value in frame["ego2global_translation"][:2]]
        norm = math.sqrt(sum(value * value for value in q))
        if len(q) != 4 or len(xy) != 2 or not all(map(math.isfinite, q + xy)) or norm < 1e-12:
            raise ValueError
        w, x, y, z = [value / norm for value in q]
        return [*xy, math.atan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("invalid_pose") from error


def _relative_image(path, sensor_root):
    root = Path(sensor_root).resolve()
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("image_path_escape")
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("image_path_escape") from error


def _image_record(relative, sensor_root, image_cache):
    if relative in image_cache:
        return
    path = Path(sensor_root) / relative
    if not path.is_file():
        raise ValueError("missing_image")
    try:
        blob = path.read_bytes()
        with Image.open(BytesIO(blob)) as image:
            size, mode = image.size, image.mode
            image.load()
        if min(size) <= 0:
            raise ValueError("invalid_image")
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError("invalid_image") from error
    image_cache[relative] = {"path": relative, "sha256": hashlib.sha256(blob).hexdigest(),
                             "bytes": len(blob), "width": size[0], "height": size[1], "mode": mode}


def _validate_row(row, sensor_root, image_cache):
    if row.get("split") not in ("train", "val"):
        raise ValueError("invalid_split")
    if not isinstance(row.get("token"), str) or not row["token"]:
        raise ValueError("invalid_token")
    if not isinstance(row.get("route"), str) or not row["route"]:
        raise ValueError("invalid_log")
    command = row.get("command")
    if type(command) is not int or command not in range(4):
        raise ValueError("invalid_command")
    if row.get("command_onehot") != [float(command == index) for index in range(4)]:
        raise ValueError("invalid_command")
    ego, target, pose = row.get("ego", []), row.get("trajectory_gt", []), row.get("ego_pose", [])
    if len(ego) != 4 or not all(math.isfinite(float(value)) for value in ego):
        raise ValueError("invalid_ego")
    if len(target) != 8 or any(len(item) != 3 for item in target):
        raise ValueError("invalid_target")
    if len(pose) != 3 or not all(math.isfinite(float(value)) for value in pose):
        raise ValueError("invalid_pose")
    if any(not math.isfinite(float(value)) for item in target for value in item):
        raise ValueError("invalid_target")
    if any(abs(item[2]) > math.pi for item in target):
        raise ValueError("invalid_target_heading")
    if (row.get("coordinate_frame") != "current_rear_axle_se2"
            or row.get("history_times_s") != HISTORY_TIMES or row.get("future_times_s") != FUTURE_TIMES):
        raise ValueError("invalid_model_timing_or_coordinates")
    history, future = row.get("history_timestamps_us", []), row.get("future_timestamps_us", [])
    timestamps = history + future
    if (len(history) != 4 or len(future) != 8 or any(type(value) is not int for value in timestamps)
            or row.get("timestamp_us") != history[3]):
        raise ValueError("invalid_timestamp")
    if any(abs(b - a - TIMESTAMP_INTERVAL_US) > TIMESTAMP_TOLERANCE_US for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("timestamp_gap")
    if len(row.get("images", [])) != 4:
        raise ValueError("invalid_image_count")
    row["images"] = [_relative_image(path, sensor_root) for path in row["images"]]
    for relative in row["images"]:
        _image_record(relative, sensor_root, image_cache)
    return row


def build_scene_row(frames, current_index, split, sensor_root, image_cache=None):
    """Build one official 14-frame-eligible scene with eight model targets.

    Token/log allowlisting is applied by prepare_official_data before this helper.
    Rejections use stable reason codes for the offline coverage audit.
    """
    if current_index < 3:
        raise ValueError("official_history_frames")
    if current_index + 10 >= len(frames):
        raise ValueError("official_future_frames")
    current = frames[current_index]
    if not current.get("roadblock_ids"):
        raise ValueError("official_no_route")
    window = frames[current_index - 3:current_index + 11]
    if any(frame.get("log_name") != current.get("log_name") for frame in window):
        raise ValueError("cross_log_scene")
    try:
        required = window[:12]
        poses = [_pose(frame) for frame in required]
        origin = poses[3]
        cosine, sine = math.cos(origin[2]), math.sin(origin[2])
        target = []
        for x, y, heading in poses[4:]:
            dx, dy = x - origin[0], y - origin[1]
            target.append([cosine * dx + sine * dy, -sine * dx + cosine * dy,
                           math.atan2(math.sin(heading - origin[2]), math.cos(heading - origin[2]))])
        command = [float(value) for value in current["driving_command"]]
        if len(command) != 4 or command.count(1.0) != 1 or any(value not in (0, 1) for value in command):
            raise ValueError("invalid_command")
        timestamps = [frame["timestamp"] for frame in required]
        row = {"dataset": "navsim", "split": split, "route": current["log_name"],
            "scene": current.get("scene_token", ""), "frame": current_index, "token": current["token"],
            "images": [frame["cams"]["CAM_F0"]["data_path"] for frame in required[:4]],
            "ego": [float(value) for value in current["ego_dynamic_state"]],
            "command": command.index(1.0), "command_onehot": command,
            "ego_pose": origin, "trajectory_gt": target,
            "timestamp_us": timestamps[3], "history_timestamps_us": timestamps[:4],
            "future_timestamps_us": timestamps[4:], "timestamp_unit": "microseconds",
            "history_times_s": HISTORY_TIMES, "future_times_s": FUTURE_TIMES,
            "coordinate_frame": "current_rear_axle_se2"}
    except KeyError as error:
        raise ValueError(f"missing_field:{error.args[0]}") from error
    return _validate_row(row, sensor_root, image_cache if image_cache is not None else {})


class _BundleWriter:
    def __init__(self, bundle_dir, sensor_root, preserve_official_val=False, excluded_val_groups=()):
        self.directory = Path(bundle_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "identity.json").exists():
            raise ValueError("prepared bundle already exists; choose a new output directory")
        self.sensor_root = Path(sensor_root).resolve()
        self.streams = {split: (self.directory / f"{split}.jsonl").open("w") for split in ("train", "val")}
        self.official_val_stream = ((self.directory / "val_official.jsonl").open("w")
            if preserve_official_val else None)
        self.official_counts = Counter({"train": 0, "val": 0})
        self.excluded_val_groups = set(excluded_val_groups)
        self.excluded_val_logs, self.excluded_val_tokens = set(), []
        self.counts = Counter({"train": 0, "val": 0})
        self.logs = {"train": set(), "val": set()}
        self.tokens = set()
        self.images = {}
        self.mean, self.m2 = [0.0] * 4, [0.0] * 4

    def add(self, row):
        row = _validate_row(row, self.sensor_root, self.images)
        if row["token"] in self.tokens:
            raise ValueError("duplicate_token")
        self.tokens.add(row["token"])
        split = row["split"]
        self.official_counts[split] += 1
        if split == "val" and self.official_val_stream is not None:
            self.official_val_stream.write(_canonical(row) + "\n")
            if recording_group(row["route"]) in self.excluded_val_groups:
                self.excluded_val_logs.add(row["route"])
                self.excluded_val_tokens.append(row["token"])
                return
        self.logs[split].add(row["route"])
        _check_role_leaks(self.logs)
        self.counts[split] += 1
        if split == "train":
            count = self.counts[split]
            for index, value in enumerate(row["ego"]):
                delta = value - self.mean[index]
                self.mean[index] += delta / count
                self.m2[index] += delta * (value - self.mean[index])
        self.streams[split].write(_canonical(row) + "\n")

    def close(self):
        for stream in self.streams.values():
            stream.close()
        if self.official_val_stream is not None:
            self.official_val_stream.close()

    def finish(self, kind, audit, provenance, receipt):
        self.close()
        if min(self.counts.values()) <= 0:
            raise ValueError("nonempty TRAIN and VAL are required")
        stats = {"ego_mean": self.mean, "ego_std": [
            math.sqrt(value / self.counts["train"]) if value / self.counts["train"] >= 1e-12 else 1.0
            for value in self.m2], "count": self.counts["train"], "source_split": "train",
            "features": ["vx", "vy", "ax", "ay"], "normalization_policy": "population_std; unit std for constant dimensions"}
        _write_json(self.directory / "stats.json", stats)
        with (self.directory / "images.jsonl").open("w") as stream:
            for relative in sorted(self.images):
                stream.write(_canonical(self.images[relative]) + "\n")
        audit.update(status="passed", accepted_counts=dict(self.counts),
            role_logs={split: sorted(logs) for split, logs in self.logs.items()},
            role_recording_groups={split: sorted(set(map(recording_group, logs))) for split, logs in self.logs.items()})
        _write_json(self.directory / "audit.json", audit)
        identity = {"schema": SCHEMA, "kind": kind, "navsim_revision": NAVSIM_REVISION,
            "model_contract": {"camera": "CAM_F0", "history_frames": 4, "target_shape": [8, 3],
                "frequency_hz": 2, "image_policy": "original_jpeg_full_resolution", "coordinate_frame": "current_rear_axle_se2",
                "timestamp_interval_us": TIMESTAMP_INTERVAL_US, "timestamp_tolerance_us": TIMESTAMP_TOLERANCE_US},
            "provenance": provenance,
            "manifests": {split: {"file": f"{split}.jsonl", "rows": self.counts[split],
                "sha256": _sha256(self.directory / f"{split}.jsonl")} for split in ("train", "val")},
            "artifacts": {name: {"file": name, "sha256": _sha256(self.directory / name)}
                for name in ("stats.json", "images.jsonl", "audit.json")}}
        if self.official_val_stream is not None:
            identity["artifacts"]["val_official.jsonl"] = {"file": "val_official.jsonl",
                "sha256": _sha256(self.directory / "val_official.jsonl"),
                "rows": self.official_counts["val"], "purpose": "report_only"}
        identity["identity_sha256"] = _digest(identity)
        _write_json(self.directory / "identity.json", identity)
        receipt.update(kind=kind, identity_sha256=identity["identity_sha256"],
            counts=dict(self.counts), unique_images=len(self.images),
            bundle_dir=str(self.directory.resolve()), sensor_root=str(self.sensor_root))
        _write_json(self.directory / "receipt.json", receipt)
        return receipt


def _source_provenance(value):
    if value is None:
        return None
    registry = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(registry, dict) or registry.get("schema") != "navsim_frozen_source_provenance_v1":
        raise ValueError("source_provenance schema differs")
    for key in ("source_checkpoint_sha256", "train_manifest_sha256"):
        digest = registry.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"source_provenance {key} differs")
    logs, groups = registry.get("train_logs"), registry.get("train_recording_groups")
    if (not isinstance(logs, list) or not logs or any(not isinstance(log, str) or not log for log in logs)
            or len(logs) != len(set(logs)) or not isinstance(groups, list)
            or set(groups) != set(map(recording_group, logs)) or len(groups) != len(set(groups))):
        raise ValueError("source_provenance training groups differ from training logs")
    return registry


def prepare_official_data(metadata_root, sensor_root, bundle_dir, devkit_root=None,
                         max_rejection_fraction=0.0, source_provenance=None):
    """Build full official TRAIN/VAL locally, failing missing coverage by default.

    Any nonzero allowed rejection fraction must be supplied explicitly and is
    recorded in the identity. Missing logs or unknown/missing official tokens
    always fail. Failed attempts retain audits but never publish identity.json.
    Optional source_provenance is the exact registry dict or a local JSON path.
    It removes source-trained recording groups only from selection VAL; every
    accepted official VAL row is preserved in report-only val_official.jsonl.
    """
    if not 0 <= max_rejection_fraction < 1:
        raise ValueError("max_rejection_fraction must be in [0,1)")
    source_registry = _source_provenance(source_provenance)
    contract = load_v1_contract(str(Path(devkit_root).resolve()) if devkit_root else None)
    root = Path(metadata_root).resolve()
    writer = _BundleWriter(bundle_dir, sensor_root, preserve_official_val=True,
        excluded_val_groups=source_registry["train_recording_groups"] if source_registry else ())
    all_logs = contract["train_logs"] + contract["val_logs"]
    missing = [log for log in all_logs if not (root / f"{log}.pkl").is_file()]
    audit = {"schema": "navsim_v1_preparation_audit_v1", "mode": "official",
        "status": "building", "missing_official_logs": sorted(missing),
        "missing_official_log_count": len(missing), "accepted_counts": dict(writer.counts),
        "max_rejection_fraction": max_rejection_fraction,
        "timestamp_policy": "project: consecutive required 12 frames must be 500000 +/- 1000 us",
        "official_filter": {key: contract["navtrain"][key] for key in
            ("num_history_frames", "num_future_frames", "frame_interval", "has_route", "max_scenes")}}
    if missing:
        writer.close()
        audit["status"] = "failed"
        _write_json(writer.directory / "audit.json", audit)
        raise ValueError(f"missing_official_logs: {len(missing)}; see {writer.directory / 'audit.json'}")
    allowed = set(contract["navtrain"]["tokens"])
    seen, rejected, metadata_identity = set(), Counter(), []
    failure = None
    with (writer.directory / "rejected.jsonl").open("w") as rejection_stream:
        try:
            for split in ("train", "val"):
                for log in contract[f"{split}_logs"]:
                    path = root / f"{log}.pkl"
                    metadata_identity.append({"log": log, "sha256": _sha256(path)})
                    with path.open("rb") as stream:
                        frames = pickle.load(stream)
                    if not isinstance(frames, list):
                        raise ValueError(f"invalid_metadata_log: {log}")
                    for index, frame in enumerate(frames):
                        token = frame.get("token")
                        if token not in allowed:
                            continue
                        if token in seen:
                            raise ValueError(f"duplicate_official_token: {token}")
                        seen.add(token)
                        try:
                            if frame.get("log_name") != log:
                                raise ValueError("metadata_log_name_mismatch")
                            row = build_scene_row(frames, index, split, sensor_root, writer.images)
                            row["source"] = {"upstream_split": "trainval", "metadata_sha256": metadata_identity[-1]["sha256"],
                                "navsim_revision": NAVSIM_REVISION, "official_scene_frames": 14}
                            writer.add(row)
                        except (ValueError, TypeError, KeyError, OverflowError) as error:
                            reason = str(error)
                            rejected[reason] += 1
                            rejection_stream.write(_canonical({"token": token, "route": log, "frame": index,
                                "split": split, "reason": reason}) + "\n")
        except Exception as error:
            failure = error
        finally:
            writer.close()
    missing_tokens = sorted(allowed - seen)
    fraction = sum(rejected.values()) / len(seen) if seen else 0.0
    audit.update(accepted_counts=dict(writer.counts), official_tokens=len(allowed), observed_official_tokens=len(seen),
        accepted_official_counts=dict(writer.official_counts),
        missing_official_token_count=len(missing_tokens), missing_official_tokens=missing_tokens,
        rejected_counts=dict(sorted(rejected.items())), rejected_windows=sum(rejected.values()),
        rejection_rates={reason: count / len(seen) for reason, count in sorted(rejected.items())},
        rejection_fraction=fraction, metadata_identity=metadata_identity,
        rejected_manifest_sha256=_sha256(writer.directory / "rejected.jsonl"))
    if source_registry is not None:
        groups = sorted(set(map(recording_group, writer.excluded_val_logs)))
        audit["source_exclusions"] = {"scope": "selection_val_only", "token_count": len(writer.excluded_val_tokens),
            "log_count": len(writer.excluded_val_logs), "recording_group_count": len(groups),
            "logs": sorted(writer.excluded_val_logs), "recording_groups": groups,
            "tokens": writer.excluded_val_tokens,
            "source_checkpoint_sha256": source_registry["source_checkpoint_sha256"],
            "train_manifest_sha256": source_registry["train_manifest_sha256"]}
    if failure is not None or missing_tokens or fraction > max_rejection_fraction:
        audit["status"] = "failed"
        if failure is not None:
            audit["error"] = str(failure)
        _write_json(writer.directory / "audit.json", audit)
        if failure is not None:
            raise ValueError(f"official_preparation_failed: {failure}; see audit.json") from failure
        reason = "missing_official_tokens" if missing_tokens else "rejection_fraction_exceeded"
        raise ValueError(f"{reason}; see {writer.directory / 'audit.json'}")
    provenance = {"official_contract": contract["identity"], "data_split": "trainval",
        "role_policy": "navtrain filter intersect official train_logs / val_logs; navtest held out",
        "max_rejection_fraction": max_rejection_fraction,
        "validation_policy": "exclude_upstream_training_groups" if source_registry is not None else "official_default_train_val"}
    receipt = {"metadata_root": str(root), "no_downloads": True, "official_source_coverage": True,
        "validation_policy": provenance["validation_policy"]}
    if source_registry is not None:
        provenance["source_provenance"] = source_registry
        receipt["source_provenance"] = source_registry
        if isinstance(source_provenance, (str, Path)):
            receipt["source_provenance_file"] = str(Path(source_provenance).resolve())
    return writer.finish("navsim_v1_full", audit, provenance, receipt)


def _smoke_pose(row, metadata_cache):
    if "ego_pose" in row:
        return row["ego_pose"]
    path = row.get("source", {}).get("metadata_path")
    if not path:
        raise ValueError("smoke_source_missing_ego_pose_and_metadata")
    if path not in metadata_cache:
        with Path(path).open("rb") as stream:
            metadata_cache[path] = pickle.load(stream)
    frame = metadata_cache[path][row["frame"]]
    if frame["token"] != row["token"]:
        raise ValueError("smoke_metadata_anchor_identity")
    return _pose(frame)


def prepare_smoke_data(train_manifest, val_manifest, sensor_root, bundle_dir, train_rows=8, val_rows=4):
    """Select a fixed prefix from existing mini manifests; never claim V1 coverage."""
    if train_rows != 8 or val_rows != 4:
        raise ValueError("smoke protocol requires exactly 8 TRAIN and 4 VAL rows")
    writer = _BundleWriter(bundle_dir, sensor_root)
    metadata_cache = {}
    source_revisions = set()
    try:
        for split, path, limit in (("train", train_manifest, train_rows), ("val", val_manifest, val_rows)):
            count = 0
            with Path(path).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    original = json.loads(line)
                    source = original.get("source", {})
                    if source.get("navsim_revision"):
                        source_revisions.add(source["navsim_revision"])
                    keys = ("dataset", "route", "scene", "frame", "token", "images", "ego", "command",
                        "command_onehot", "trajectory_gt", "history_times_s", "future_times_s", "timestamp_us",
                        "history_timestamps_us", "future_timestamps_us", "coordinate_frame", "timestamp_unit")
                    row = {key: original[key] for key in keys if key in original}
                    row.update(split=split, ego_pose=_smoke_pose(original, metadata_cache))
                    row["source"] = {key: source[key] for key in
                        ("navsim_revision", "openscene_revision", "upstream_split", "metadata_sha256") if key in source}
                    writer.add(row)
                    count += 1
                    if count == limit:
                        break
            if count != limit:
                raise ValueError(f"insufficient_smoke_rows: {split} needs {limit}")
    finally:
        writer.close()
    audit = {"schema": "navsim_v1_preparation_audit_v1", "mode": "smoke",
        "official_source_coverage": False, "rejected_counts": {}, "rejection_fraction": 0.0,
        "scope": "local inherited mini smoke; not official NAVSIM V1 train/val coverage"}
    provenance = {"source_navsim_revisions": sorted(source_revisions), "official_source_coverage": False,
        "selection": "first 8 TRAIN and first 4 VAL rows of existing disjoint-group mini manifests"}
    return writer.finish("smoke", audit, provenance, {"no_downloads": True,
        "train_source": str(Path(train_manifest).resolve()), "val_source": str(Path(val_manifest).resolve()),
        "official_source_coverage": False})


class SceneDataset:
    """JSONL offsets plus a per-process mmap; no materialized list of scene dicts."""
    def __init__(self, manifest, sensor_root, expected_rows=None, image_manifest_sha256=None, image_frames=4):
        if type(image_frames) is not int or image_frames not in (1, 4):
            raise ValueError("image_frames must be 1 or 4")
        self.image_frames = image_frames
        self.manifest = Path(manifest).resolve()
        self.sensor_root = Path(sensor_root).resolve()
        self.image_manifest_sha256 = image_manifest_sha256
        self._image_hashes = None
        self.offsets = array("Q")
        offset = 0
        with self.manifest.open("rb") as stream:
            for line in stream:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        if expected_rows is not None and len(self.offsets) != expected_rows:
            raise ValueError("manifest row count identity differs")
        self._stream = self._mapping = None

    def __len__(self):
        return len(self.offsets)

    def row(self, index):
        if self._mapping is None:
            self._stream = self.manifest.open("rb")
            self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
        start = self.offsets[index]
        end = self._mapping.find(b"\n", start)
        return json.loads(self._mapping[start:end if end != -1 else len(self._mapping)])

    def __getitem__(self, index):
        valid = True
        if isinstance(index, tuple):
            index, valid = index
        row = self.row(index)
        selected_images = row["images"][-self.image_frames:]
        if self._image_hashes is None:
            inventory = self.manifest.with_name("images.jsonl")
            if self.image_manifest_sha256 is not None and _sha256(inventory) != self.image_manifest_sha256:
                raise ValueError("image inventory identity differs")
            self._image_hashes = {}
            with inventory.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    self._image_hashes[item["path"]] = bytes.fromhex(item["sha256"])
        relative_paths = [_relative_image(path, self.sensor_root) for path in selected_images]
        paths = [self.sensor_root / path for path in relative_paths]
        images = []
        for relative, path in zip(relative_paths, paths):
            blob = path.read_bytes()
            if hashlib.sha256(blob).digest() != self._image_hashes.get(relative):
                raise ValueError(f"image content identity differs: {relative}")
            with Image.open(BytesIO(blob)) as image:
                images.append(image.convert("RGB"))
        return {"images": images, "image_paths": [str(path) for path in paths],
            "ego": row["ego"], "command": row["command"], "trajectory_gt": row["trajectory_gt"],
            "token": row["token"], "route": row["route"], "ego_pose": row["ego_pose"],
            "timestamp_us": row["timestamp_us"], "valid": bool(valid)}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_stream=None, _mapping=None, _image_hashes=None)
        return state

    def __del__(self):
        if getattr(self, "_mapping", None) is not None:
            self._mapping.close()
        if getattr(self, "_stream", None) is not None:
            self._stream.close()


class PreparedData:
    """Validate compact bundle identities once; open datasets lazily by split."""
    def __init__(self, bundle_dir, sensor_root, allow_smoke=False):
        self.bundle_dir = Path(bundle_dir).resolve()
        self.sensor_root = Path(sensor_root).resolve()
        self.identity = json.loads((self.bundle_dir / "identity.json").read_text())
        unsigned = {key: value for key, value in self.identity.items() if key != "identity_sha256"}
        if self.identity.get("schema") != SCHEMA or self.identity.get("identity_sha256") != _digest(unsigned):
            raise ValueError("data bundle identity differs")
        if self.identity.get("navsim_revision") != NAVSIM_REVISION:
            raise ValueError("data NAVSIM revision differs")
        if self.identity.get("kind") == "smoke":
            if not allow_smoke:
                raise ValueError("smoke bundle requires allow_smoke=True")
        elif self.identity.get("kind") != "navsim_v1_full":
            raise ValueError("unknown prepared data kind")
        for split, item in self.identity["manifests"].items():
            if item["file"] != f"{split}.jsonl" or _sha256(self.bundle_dir / item["file"]) != item["sha256"]:
                raise ValueError(f"manifest {split} identity differs")
        for name, item in self.identity["artifacts"].items():
            if item["file"] != name or Path(name).name != name or _sha256(self.bundle_dir / name) != item["sha256"]:
                raise ValueError(f"artifact {name} identity differs")
        self.stats = json.loads((self.bundle_dir / "stats.json").read_text())
        if self.stats.get("source_split") != "train" or self.stats.get("count") != self.identity["manifests"]["train"]["rows"]:
            raise ValueError("TRAIN normalization identity differs")

    def dataset(self, split, image_frames=4):
        report_only = split == "val_official"
        if report_only:
            item = self.identity["artifacts"].get("val_official.jsonl")
            if item is None or item.get("purpose") != "report_only":
                raise ValueError("report-only official VAL is unavailable")
        elif split in ("train", "val"):
            item = self.identity["manifests"][split]
        else:
            raise ValueError("only TRAIN and independent VAL are prepared; navtest is held out")
        dataset = SceneDataset(self.bundle_dir / item["file"], self.sensor_root, item["rows"],
            self.identity["artifacts"]["images.jsonl"]["sha256"], image_frames=image_frames)
        dataset.report_only = report_only
        return dataset


def identity_collate(samples):
    return samples
