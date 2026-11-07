"""Frozen, group-disjoint mini holdout; every consumed network body is audited.

No training/statistics writer is called. Partial prefixes authenticate retained
members only and never imply verification of a complete upstream archive.
"""
from collections import Counter
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import threading
import urllib.request
import uuid

from fast_navsim.acquire import ByteBudget, download_prefix_ranges, write_json
from fast_navsim.data import _pose, build_window, validate_row
from slowfast_v11.expanded_data import (
    REVISION, NAVSIM_REVISION, _plan_digest, structural_candidates, sha256_file,
    acquire_plan, load_completed_reports, verify_provenance_coverage,
)

BASE_PREFIX_BYTES = 120_000_000
BASE_PREFIX_SHA256 = 'c9884785c4750e18a36b0c725fb697eaa4563d734c1eff1a498b67f73b7fa902'
METADATA_PREFIX_CAP = 280_000_000
METADATA_NEW_CAP = 160_000_000
CAMERA_PREFIX_CAP = 225_000_000
CAMERA_RESERVATION = 8 * CAMERA_PREFIX_CAP
NEW_NETWORK_CAP = 2_000_000_000
METADATA_URL = f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/openscene-v1.1/openscene_metadata_mini.tgz'
BUILDER_PROTOCOL = 'fast_navsim.data.build_window; native 2Hz CAM_F0; four history/eight future; current rear axle SE2'


class CohortSelectionError(ValueError):
    def __init__(self, message, selection_audit):
        super().__init__(message)
        self.selection_audit = selection_audit


def recording_group(name):
    """A recording identity does not include its final numeric segment range."""
    return re.sub(r'_\d+_\d+$', '', name)


def holdout_candidates(frames):
    """Structural stride-five anchors with finite poses across all 12 frames."""
    candidates, excluded = structural_candidates(frames)
    counts = Counter(excluded)
    valid = []
    for index in candidates:
        try:
            for frame in frames[index-3:index+9]:
                q = frame['ego2global_rotation']
                xyz = frame['ego2global_translation']
                if len(q) != 4 or len(xyz) != 3 or not all(math.isfinite(float(v)) for v in list(q)+list(xyz)):
                    raise ValueError('Nonfinite pose')
                _pose(frame)
        except (ValueError, TypeError, KeyError, IndexError):
            counts['invalid_pose'] += 1
        else:
            valid.append(index)
    return valid, dict(counts)


def make_adapter_plan(retained, camera_index, seen_logs, *, sensor_root):
    """Sort eligible metadata names, exclude recording groups, take 8 x first32.

    Metadata-only selection completes before any sensor path is inspected.
    `train`/`new_val` are compatibility fields for the bounded camera helpers;
    all emitted rows belong to the new `holdout` cohort and validation split.
    """
    sensor_root = Path(sensor_root).resolve()
    records = {}
    for item in retained:
        name = Path(item['path']).stem
        if name in records:
            raise ValueError('Duplicate retained metadata log')
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Metadata SHA256 mismatch')
        records[name] = item
    first = {}
    for key, names in camera_index.items():
        if not names:
            continue
        name = names[0]
        if name in first:
            raise ValueError('Metadata log appears first in multiple camera shards')
        first[name] = int(key.rsplit('_', 1)[1])
    excluded = {}
    seen_groups = {recording_group(name) for name in seen_logs}
    groups = set(seen_groups)
    selected = []
    loaded = {}
    eligibility = {}
    for name in sorted(records):
        group = recording_group(name)
        if group in seen_groups:
            excluded[name] = 'previously_seen_recording_group'
            continue
        if name not in first:
            excluded[name] = 'not_first_in_camera_shard'
            continue
        if group in groups:
            excluded[name] = 'duplicate_recording_group'
            continue
        frames = pickle.loads(Path(records[name]['path']).read_bytes())
        if not isinstance(frames, list) or not frames or any(frame.get('log_name') != name for frame in frames):
            raise ValueError('Metadata log name differs from retained member')
        candidates, reasons = holdout_candidates(frames)
        eligibility[name] = dict(candidate_count=len(candidates), excluded_windows=reasons)
        if len(candidates) < 32:
            excluded[name] = 'fewer_than_32_structural_anchors'
            continue
        groups.add(group)
        selected.append(name)
        loaded[name] = (frames, candidates)
    chosen = selected[:8]
    selection_audit = dict(policy='sorted eligible first-in-shard names; first eight unique unseen recording groups; first32 structural stride5 anchors',
        seen_recording_groups=sorted(seen_groups),eligible_names=selected,
        excluded=excluded,eligibility=eligibility,required_recording_groups=8,
        available_recording_groups=len(selected),cohort_frozen=len(chosen)==8,
        pose_validity='all twelve history/current/future finite poses; no future motion magnitude or error selection',
        early_log_portions=True)
    if len(chosen) != 8:
        raise CohortSelectionError(f'Need eight new eligible first-in-shard recording groups; found {len(chosen)}',selection_audit)
    rows = []
    logs = {}
    for name in chosen:
        frames, candidates = loaded[name]
        item = records[name]
        indices = candidates[:32]
        logs[name] = dict(shard=first[name], recording_group=recording_group(name),
            metadata_path=item['path'], metadata_sha256=item['sha256'],
            selected_indices=indices, cohort='holdout', **eligibility[name])
        source = dict(navsim_revision=NAVSIM_REVISION, openscene_revision=REVISION,
            upstream_split='mini', metadata_path=item['path'], metadata_sha256=item['sha256'],
            sensor_root=str(sensor_root), builder_protocol=BUILDER_PROTOCOL,
            camera_archive=f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{first[name]}.tgz')
        for index in indices:
            row = build_window(frames, index, 'val', sensor_root, source)
            row['cohort'] = 'holdout'
            validate_row(row)
            rows.append(row)
    tokens = [row['token'] for row in rows]
    if len(set(tokens)) != 256 or len({logs[name]['shard'] for name in chosen}) != 8:
        raise ValueError('Holdout needs 256 unique tokens and eight distinct camera shards')
    plan = dict(schema='slowfast_adapter_data_plan_v1', openscene_revision=REVISION,
        navsim_revision=NAVSIM_REVISION, splits=dict(train=[], new_val=chosen, holdout=chosen),
        logs=logs, rows=dict(train=[], val=rows), sensor_root=str(sensor_root),
        required_images=sorted({path for row in rows for path in row['images']}),
        legacy_copies=[], metadata_records=[records[name] for name in chosen],
        cohort_counts=dict(holdout=256), selection_before_image_availability=True,
        selection_audit=selection_audit,
        token_manifest_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest())
    plan['plan_sha256'] = _plan_digest(plan)
    return plan


def freeze_adapter_plan(plan, output):
    if _plan_digest(plan) != plan.get('plan_sha256'):
        raise ValueError('Frozen plan identity differs')
    output = Path(output)
    tokens = dict(schema='slowfast_adapter_tokens_v1', plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'],
        tokens=[row['token'] for row in plan['rows']['val']], frozen_before_camera_access=True)
    for filename, content in [('plan.json', plan), ('frozen_tokens.json', tokens)]:
        path = output/filename
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError(f'Frozen {filename} identity differs; no cohort replacement')
    for filename, content in [('plan.json', plan), ('frozen_tokens.json', tokens)]:
        if not (output/filename).exists():
            write_json(output/filename, content)


def copy_authenticated_prefix(source, destination, *, expected_bytes=BASE_PREFIX_BYTES,
                              expected_sha256=BASE_PREFIX_SHA256):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.stat().st_size != expected_bytes or sha256_file(source) != expected_sha256:
        raise ValueError('Prior metadata prefix size/SHA256 identity differs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if source.stat().st_ino == destination.stat().st_ino and source.stat().st_dev == destination.stat().st_dev:
        raise ValueError('Private metadata prefix must not alias its source inode')
    with destination.open('rb') as stream:
        digest = hashlib.sha256(stream.read(expected_bytes)).hexdigest()
    if destination.stat().st_size < expected_bytes or digest != expected_sha256:
        raise ValueError('Private metadata prefix initial SHA256 differs')
    return dict(path=str(destination), inherited_source=str(source), bytes=expected_bytes,
                sha256=expected_sha256, full_archive_verified=False)


class AuditedOpener:
    """Record bytes/hash for every request, including failed streaming responses."""
    def __init__(self, opener, kind, on_request=None):
        self.opener, self.kind, self.on_request = opener, kind, on_request
        self.records = []
        self.lock = threading.Lock()

    def _finish(self, record, digest, error):
        record['response_body_sha256'] = digest.hexdigest()
        record['error'] = f'{type(error).__name__}: {error}' if error is not None else None
        with self.lock:
            self.records.append(record)
            if self.on_request:
                self.on_request(record)

    def __call__(self, request, timeout=90):
        if not 0 < timeout <= 90:
            raise ValueError('Request timeout must be bounded at90 seconds')
        record = dict(request_id=uuid.uuid4().hex, kind=self.kind, url=request.full_url,
            range=request.get_header('Range'), timeout_seconds=timeout,
            response_body_bytes=0, full_archive_verified=False)
        digest = hashlib.sha256()
        try:
            response = self.opener(request, timeout=timeout)
        except BaseException as error:
            self._finish(record, digest, error)
            raise
        record.update(status=response.status, content_range=response.headers.get('Content-Range'))
        parent = self
        class Body:
            status = response.status
            headers = response.headers
            pending_error = None
            def __enter__(self):
                return self
            def read(self, size=-1):
                if self.pending_error is not None:
                    raise self.pending_error
                try:
                    data = response.read(size)
                except http.client.IncompleteRead as error:
                    if not error.partial:
                        raise
                    # Surface already consumed bytes to BudgetReader before
                    # raising on its next read, so neither budget nor hash loses
                    # the bytes carried by an interrupted HTTP response.
                    data = error.partial
                    self.pending_error = error
                record['response_body_bytes'] += len(data)
                digest.update(data)
                return data
            def __exit__(self, error_type, error, traceback):
                try:
                    response.close()
                finally:
                    parent._finish(record, digest, error)
        return Body()


def extend_metadata_prefix(prefix, budget, *, metadata_bytes_used=0,
                           opener=urllib.request.urlopen, on_request=None,
                           prefix_cap=None, metadata_new_cap=None):
    """Extend the private prefix under the explicitly selected metadata cap.

    Failed bodies count against the160MB default (200MB explicit amendment);
    a retry that cannot finish within that allowance stops before another body.
    """
    prefix = Path(prefix)
    prefix_cap = METADATA_PREFIX_CAP if prefix_cap is None else prefix_cap
    metadata_new_cap = METADATA_NEW_CAP if metadata_new_cap is None else metadata_new_cap
    if (prefix_cap,metadata_new_cap) not in ((METADATA_PREFIX_CAP,METADATA_NEW_CAP),(320_000_000,200_000_000)):
        raise ValueError('Unsupported metadata cap amendment')
    size = prefix.stat().st_size
    if not BASE_PREFIX_BYTES <= size <= prefix_cap:
        raise ValueError('Metadata prefix is outside fixed inherited/maximum bounds')
    needed = prefix_cap-size
    allowance = metadata_new_cap-metadata_bytes_used
    if needed > allowance or budget.remaining < CAMERA_RESERVATION+needed:
        raise ValueError('Insufficient budget to reserve fixed metadata and eight camera prefixes')
    child = ByteBudget(min(allowance, budget.remaining-CAMERA_RESERVATION))
    audited = AuditedOpener(opener, 'metadata', on_request)
    try:
        report = download_prefix_ranges(METADATA_URL, prefix, child, prefix_cap,
            chunk_size=8_000_000, opener=audited, resume=True)
        report.update(inherited_prefix_bytes=BASE_PREFIX_BYTES,
            inherited_prefix_sha256=BASE_PREFIX_SHA256, new_body_allowance_bytes=metadata_new_cap,
            requests_detail=audited.records, complete=True)
        return report
    finally:
        budget.used += child.used


def write_adapter_prepared(plan, output, *, expected_size=(1920,1080), acquisition_reports=(),
                           network_budget=None, metadata_provenance=None, upstream_audit=None):
    """Validate original images and emit only the new holdout manifest/audit."""
    from PIL import Image
    if _plan_digest(plan) != plan.get('plan_sha256'):
        raise ValueError('Frozen plan identity differs')
    coverage = verify_provenance_coverage(plan, acquisition_reports) if acquisition_reports else None
    sources = []
    for filename in plan['required_images']:
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f'Missing fixed planned image: {path}')
        with Image.open(path) as image:
            if image.size != expected_size or image.format != 'JPEG':
                raise ValueError('Requires original-resolution CAM_F0 JPEGs')
        sources.append(dict(path=str(path), sha256=sha256_file(path), bytes=path.stat().st_size,
                            kind='retained_original_jpeg'))
    for item in plan['metadata_records']:
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Metadata SHA256 changed since planning')
        sources.append(dict(item, kind='retained_metadata'))
    for row in plan['rows']['val']:
        validate_row(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze_adapter_plan(plan, output)
    manifest = output/'holdout.jsonl'
    manifest.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in plan['rows']['val']))
    audit = dict(schema='slowfast_adapter_source_audit_v1', openscene_revision=REVISION,
        navsim_revision=NAVSIM_REVISION, plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'], sources=sources,
        network_budget=network_budget, metadata_provenance=metadata_provenance,
        upstream_audit=upstream_audit, provenance_coverage=coverage,
        acquisition_reports=list(acquisition_reports), full_archives_verified=False,
        protocol='NAVSIM-mini diagnosis; no official navtrain/navtest score',
        training_and_statistics='reuse authenticated original TRAIN256 and stats; no recomputation')
    write_json(output/'source_audit.json', audit)
    result = dict(counts=plan['cohort_counts'], plan_sha256=plan['plan_sha256'],
        manifest_path=str(manifest.resolve()), manifest_sha256=sha256_file(manifest),
        original_image_size_wh=list(expected_size), stats_recomputed=False, train_recomputed=False)
    write_json(output/'build_report.json', result)
    return result
