"""Prepare TRAIN512/fresh VAL256 with an exactly bounded, durable transfer audit.

Default is offline. --extend-metadata permits only the fixed320->520MB private
metadata extension and freezes the resulting cohort. Inspect its plan/token SHA
before invoking --acquire for those eight exact camera prefixes (no replacement).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import urllib.request

from fast_navsim.acquire import ByteBudget, write_json
from slowfast_v11.direct_data import (
    REVISION, BASE_PREFIX_BYTES, BASE_PREFIX_SHA256, METADATA_PREFIX_CAP,
    METADATA_NEW_CAP, CAMERA_PREFIX_CAP, NEW_NETWORK_CAP, METADATA_URL,
    INHERITED_CUMULATIVE_BYTES, NORMALIZER_SHA256, AuditedOpener,
    copy_authenticated_prefix, extend_metadata_prefix, make_direct_plan,
    freeze_direct_plan, write_direct_prepared, acquire_plan, load_completed_reports,
    verify_provenance_coverage, sha256_file, _plan_digest,
    verify_durable_prefix, download_durable_prefix, extract_durable_camera,
    load_direct_completed_reports,
)
from slowfast_v11.expanded_data import _retrying_opener
from tools.prepare_navsim_smoke import extract_complete_metadata
from tools.prepare_slowfast_expanded import persist_attempt_report


def _camera_url(shard):
    return (f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/'
        f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz')


RECOVERY_INITIAL_BYTES = {27:74_665_502,17:112_000_000}
RECOVERY_RESPONSE_CAPS = {_camera_url(shard):count+225_000_000
    for shard,count in RECOVERY_INITIAL_BYTES.items()}
DURABLE_RESPONSE_CAPS = dict(RECOVERY_RESPONSE_CAPS,**{_camera_url(27):478_692_042})
EXTENT_RESPONSE_CAPS = dict(DURABLE_RESPONSE_CAPS,**{_camera_url(27):482_692_042})


def _validate_amendment(amendment):
    expected = {'slowfast_direct_full_transport_recovery_v1':RECOVERY_RESPONSE_CAPS,
        'slowfast_direct_full_durable_transport_recovery_v1':DURABLE_RESPONSE_CAPS,
        'slowfast_direct_full_durable_extent_recovery_v1':EXTENT_RESPONSE_CAPS}.get(amendment.get('schema'))
    expected_extents = {_camera_url(27):229_000_000} if amendment.get('schema')=='slowfast_direct_full_durable_extent_recovery_v1' else {}
    if (expected is None or amendment.get('response_body_cap_overrides_by_url') != expected
            or amendment.get('archive_prefix_extent_cap_overrides_by_url',{}) != expected_extents
            or amendment.get('archive_prefix_extent_cap_bytes') != CAMERA_PREFIX_CAP
            or amendment.get('new_total_response_body_cap_bytes') != NEW_NETWORK_CAP
            or amendment.get('camera_total_response_body_cap_bytes') != 8*CAMERA_PREFIX_CAP):
        raise ValueError('Unsupported transport recovery amendment; fixed limits differ')


def record_transport_amendment(output, plan):
    """Pin the explicitly authorized two-URL retransmission exception pre-request.

    Each rescan still stops at compressed offset225MB. Only consumed response
    retransmission allowances change; the global2GB and camera1.8GB caps do not.
    """
    output = Path(output)
    path = output/'transport_recovery_amendment.json'
    if path.exists():
        amendment = json.loads(path.read_text());_validate_amendment(amendment)
        if amendment['plan_sha256'] != plan['plan_sha256'] or amendment['token_manifest_sha256'] != plan['token_manifest_sha256']:
            raise ValueError('Recovery amendment frozen cohort identity differs')
        return amendment
    budget_path = output/'network_budget.json'
    previous = json.loads(budget_path.read_text())
    records = [json.loads(line) for line in (output/'network_requests.jsonl').read_text().splitlines() if line.strip()]
    for shard, expected in RECOVERY_INITIAL_BYTES.items():
        report = json.loads((output/f'camera_{shard}_acquisition.json').read_text())
        charged = sum(r['response_body_bytes'] for r in records if r['kind']=='camera' and r.get('url')==_camera_url(shard))
        if (report.get('shard') != shard or report.get('url') != _camera_url(shard)
                or report.get('complete') is not False or not report.get('error')
                or report.get('response_body_bytes') != expected or charged != expected):
            raise ValueError('Initial failure report identity differs from authorized recovery')
    remaining = NEW_NETWORK_CAP-previous['used']
    if remaining < 2*CAMERA_PREFIX_CAP or previous['metadata_response_body_bytes'] != METADATA_NEW_CAP:
        raise ValueError('Recovery cannot reserve two original camera prefixes within unchanged global cap')
    snapshots = []
    for filename in ('camera_27_acquisition.json','camera_17_acquisition.json','source_audit.json','network_budget.json',
                     'network_requests.jsonl','camera_attempts.jsonl'):
        source = output/filename
        if not source.exists():
            continue
        target = source.with_name(source.stem+'_initial_failed'+source.suffix)
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError('Preserved initial failure snapshot identity differs')
        if not target.exists():
            shutil.copy2(source,target)
        snapshots.append(dict(path=str(target.resolve()),sha256=sha256_file(target)))
    amendment = dict(schema='slowfast_direct_full_transport_recovery_v1',
        event_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Explicit root approval: rescan only failed shards27/17; unchanged cohort, archive extents, global camera1.8GB and total2GB',
        explicit_cli_flag='--recover-failed-cameras',
        reason='Three zero-body TLS EOF retries exhausted for each failed URL; original helper retained JPEGs but not compressed prefix/gzip state',
        plan_sha256=plan['plan_sha256'],token_manifest_sha256=plan['token_manifest_sha256'],
        response_body_cap_overrides_by_url=dict(RECOVERY_RESPONSE_CAPS),
        initial_response_body_bytes_by_url={_camera_url(s):n for s,n in RECOVERY_INITIAL_BYTES.items()},
        archive_prefix_extent_cap_bytes=CAMERA_PREFIX_CAP,
        camera_total_response_body_cap_bytes=8*CAMERA_PREFIX_CAP,
        new_total_response_body_cap_bytes=NEW_NETWORK_CAP,
        maximum_additional_response_body_bytes=2*CAMERA_PREFIX_CAP,
        initial_new_response_body_bytes=previous['used'],
        maximum_resulting_new_response_body_bytes=previous['used']+2*CAMERA_PREFIX_CAP,
        preserved_failure_snapshots=snapshots,
        initial_request_ledger=dict(path=str((output/'network_requests_initial_failed.jsonl').resolve()),
            sha256=sha256_file(output/'network_requests_initial_failed.jsonl'),requests=len(records)))
    _validate_amendment(amendment)
    write_json(path,amendment)
    return amendment


def record_durable_transport_amendment(output, plan):
    """Pin the final225MB durable transfer and its exact starting ledger index."""
    output = Path(output)
    path = output/'transport_durable_amendment.json'
    if path.exists():
        amendment = json.loads(path.read_text());_validate_amendment(amendment)
        if amendment['plan_sha256'] != plan['plan_sha256'] or amendment['token_manifest_sha256'] != plan['token_manifest_sha256']:
            raise ValueError('Durable amendment frozen cohort identity differs')
        return amendment
    previous_path = output/'transport_recovery_amendment.json'
    previous_amendment = json.loads(previous_path.read_text());_validate_amendment(previous_amendment)
    budget = json.loads((output/'network_budget.json').read_text())
    records = [json.loads(line) for line in (output/'network_requests.jsonl').read_text().splitlines() if line.strip()]
    failed = json.loads((output/'camera_27_acquisition.json').read_text())
    charged = sum(r['response_body_bytes'] for r in records if r['kind']=='camera' and r.get('url')==_camera_url(27))
    if (failed.get('complete') is not False or failed.get('response_body_bytes') != 179_026_540
            or charged != 253_692_042 or budget['used'] != 1_770_942_916
            or budget['metadata_response_body_bytes'] != METADATA_NEW_CAP
            or NEW_NETWORK_CAP-budget['used'] < CAMERA_PREFIX_CAP):
        raise ValueError('Second failure/budget identity differs from final durable authorization')
    snapshots = []
    for filename in ('camera_27_acquisition.json','source_audit.json','network_budget.json',
                     'network_requests.jsonl','camera_attempts.jsonl','transport_recovery_amendment.json'):
        source = output/filename
        if not source.exists():
            continue
        target = source.with_name(source.stem+'_second_failed'+source.suffix)
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError('Second failure snapshot identity differs')
        if not target.exists():
            shutil.copy2(source,target)
        snapshots.append(dict(path=str(target.resolve()),sha256=sha256_file(target)))
    ledger_snapshot = output/'network_requests_second_failed.jsonl'
    amendment = dict(schema='slowfast_direct_full_durable_transport_recovery_v1',
        event_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Explicit root approval: only shard27 lifetime478692042; durable private prefix with exact saved-byte resume; archive extent225MB/global camera1.8GB/new total2GB unchanged',
        explicit_cli_flag='--durable-shard27-recovery',plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'],
        response_body_cap_overrides_by_url=dict(DURABLE_RESPONSE_CAPS),
        archive_prefix_extent_cap_bytes=CAMERA_PREFIX_CAP,
        camera_total_response_body_cap_bytes=8*CAMERA_PREFIX_CAP,new_total_response_body_cap_bytes=NEW_NETWORK_CAP,
        initial_new_response_body_bytes=budget['used'],maximum_additional_response_body_bytes=CAMERA_PREFIX_CAP,
        maximum_resulting_new_response_body_bytes=budget['used']+CAMERA_PREFIX_CAP,
        global_spare_bytes_after_full_prefix=NEW_NETWORK_CAP-budget['used']-CAMERA_PREFIX_CAP,
        previous_transport_amendment=dict(path=str(previous_path.resolve()),sha256=sha256_file(previous_path)),
        starting_ledger=dict(path=str(ledger_snapshot.resolve()),sha256=sha256_file(ledger_snapshot),request_count=len(records)),
        preserved_failure_snapshots=snapshots,request_chunk_cap_bytes=1_000_000,
        retained_write_policy='write+flush+fsync every <=65536-byte consumed block before next read',
        resume_policy='authenticate only responses after starting_ledger.request_count; offset equals actual saved prefix bytes')
    _validate_amendment(amendment)
    write_json(path,amendment)
    return amendment


def durable_stage_records(ledger, amendment):
    """Earlier discarded scans also started at0, so exclude their entire segment."""
    marker = amendment['starting_ledger']
    if sha256_file(marker['path']) != marker['sha256']:
        raise ValueError('Starting durable ledger snapshot SHA256 differs')
    count = marker['request_count']
    lines = ledger.path.read_bytes().splitlines(keepends=True)
    if len(lines) < count or hashlib.sha256(b''.join(lines[:count])).hexdigest() != marker['sha256']:
        raise ValueError('Starting durable ledger prefix identity differs')
    records = ledger.records[count:]
    if any(r['kind'] != 'camera' or r.get('url') != _camera_url(27) for r in records):
        raise ValueError('Unexpected request in final durable ledger segment')
    return records


def record_extent_transport_amendment(output, plan):
    """Authorize only4MB contiguous extension after recorded126/128 coverage."""
    output = Path(output);path = output/'transport_extent_amendment.json'
    if path.exists():
        amendment = json.loads(path.read_text());_validate_amendment(amendment)
        if amendment['plan_sha256'] != plan['plan_sha256'] or amendment['token_manifest_sha256'] != plan['token_manifest_sha256']:
            raise ValueError('Extent amendment cohort identity differs')
        return amendment
    previous_path = output/'transport_durable_amendment.json'
    prior = json.loads(previous_path.read_text());_validate_amendment(prior)
    budget = json.loads((output/'network_budget.json').read_text())
    coverage = json.loads((output/'camera_27_extent_failure.json').read_text())
    prefix = json.loads((output/'camera_27_durable_prefix.json').read_text())
    if (budget['used'] != 1_995_942_916 or prefix['prefix_bytes'] != CAMERA_PREFIX_CAP or prefix.get('complete') is not True
            or coverage['plan_sha256'] != plan['plan_sha256'] or coverage['token_manifest_sha256'] != plan['token_manifest_sha256']
            or coverage.get('planned_images') != 128 or coverage.get('verified_planned_images') != 126 or len(coverage['missing_members']) != 2):
        raise ValueError('225MB coverage failure identity differs from4MB authorization')
    snapshots = []
    for filename in ('camera_27_extent_failure.json','camera_27_durable_prefix.json','source_audit.json',
                     'network_budget.json','network_requests.jsonl','transport_durable_amendment.json'):
        source = output/filename;target = source.with_name(source.stem+'_225mb_extent_failed'+source.suffix)
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError('225MB failure snapshot identity differs')
        if not target.exists():
            shutil.copy2(source,target)
        snapshots.append(dict(path=str(target.resolve()),sha256=sha256_file(target)))
    amendment = dict(prior,schema='slowfast_direct_full_durable_extent_recovery_v1',
        event_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        authorization='Explicit root approval: only27 extent225MB->229MB, lifetime482692042; append saved prefix; global2GB/camera1.8GB unchanged',
        explicit_cli_flag='--extend-durable-shard27-229',
        response_body_cap_overrides_by_url=dict(EXTENT_RESPONSE_CAPS),
        archive_prefix_extent_cap_overrides_by_url={_camera_url(27):229_000_000},
        previous_transport_amendment=dict(path=str(previous_path.resolve()),sha256=sha256_file(previous_path)),
        initial_new_response_body_bytes=budget['used'],maximum_additional_response_body_bytes=4_000_000,
        maximum_resulting_new_response_body_bytes=budget['used']+4_000_000,
        global_spare_bytes_after_full_prefix=NEW_NETWORK_CAP-budget['used']-4_000_000,
        base_prefix=dict(bytes=prefix['prefix_bytes'],sha256=prefix['prefix_sha256']),
        preserved_extent_failure_snapshots=snapshots)
    _validate_amendment(amendment);write_json(path,amendment)
    return amendment


def _validate_record(record):
    digest = record.get('response_body_sha256')
    if (not isinstance(record.get('request_id'),str) or not record['request_id']
            or type(record.get('response_body_bytes')) is not int or record['response_body_bytes'] < 0
            or record.get('kind') not in ('metadata','camera') or not isinstance(digest,str)
            or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('Invalid request ledger accounting/hash')


class RequestLedger:
    """Append/fsync all attempts before reporting limits, including failed bodies."""
    def __init__(self, output, *, inherited_bytes, inherited_budget, transport_amendment=None):
        self.output = Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.output/'network_requests.jsonl'
        self.inherited_bytes, self.inherited_budget = inherited_bytes,inherited_budget
        self.response_caps = {}
        self.archive_extent_caps = {}
        self.transport_amendment = None
        previous_transport_amendment = None
        if transport_amendment is not None:
            amendment_path = Path(transport_amendment)
            amendment = json.loads(amendment_path.read_text());_validate_amendment(amendment)
            self.response_caps = amendment['response_body_cap_overrides_by_url']
            self.archive_extent_caps = amendment.get('archive_prefix_extent_cap_overrides_by_url',{})
            self.transport_amendment = dict(path=str(amendment_path.resolve()),sha256=sha256_file(amendment_path))
            previous_transport_amendment = amendment.get('previous_transport_amendment')
        if type(inherited_bytes) is not int or inherited_bytes < 0:
            raise ValueError('Invalid inherited cumulative byte count')
        self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()] if self.path.exists() else []
        for record in self.records:
            _validate_record(record)
        if len({r['request_id'] for r in self.records}) != len(self.records):
            raise ValueError('Duplicate request ledger identity')
        snapshot = self.output/'network_budget.json'
        if snapshot.exists():
            old = json.loads(snapshot.read_text())
            if (old.get('transport_amendment') is not None and old['transport_amendment'] != self.transport_amendment
                    and old['transport_amendment'] != previous_transport_amendment):
                raise ValueError('Existing transport recovery requires the same explicit amendment')
            if old['inherited_previous_response_body_bytes'] != inherited_bytes or old['inherited_budget'] != inherited_budget:
                raise ValueError('Inherited cumulative budget identity differs')
            if old['metadata_prefix_cap_bytes'] != METADATA_PREFIX_CAP or old['metadata_new_cap_bytes'] != METADATA_NEW_CAP or old['limit'] != NEW_NETWORK_CAP:
                raise ValueError('Fixed direct-full network cap identity differs')
            if old['used'] > sum(r['response_body_bytes'] for r in self.records):
                raise ValueError('Network snapshot charges bytes absent from durable request ledger')
        self.state('initialized')

    def append(self, record):
        _validate_record(record)
        with self.lock:
            if any(r['request_id'] == record['request_id'] for r in self.records):
                raise ValueError('Duplicate request ledger identity')
            with self.path.open('a') as stream:
                stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.records.append(record)
            state = self.state(record['kind'])
            print(json.dumps(dict(event='request_complete',kind=record['kind'],range=record.get('range'),
                response_body_bytes=record['response_body_bytes'],new_response_body_bytes=state['used'],
                cumulative_response_body_bytes=state['cumulative_response_body_bytes'],error=record.get('error'))),flush=True)

    def state(self, phase):
        with self.lock:
            metadata = [r for r in self.records if r['kind']=='metadata']
            cameras = [r for r in self.records if r['kind']=='camera']
            metadata_bytes = sum(r['response_body_bytes'] for r in metadata)
            camera_bytes = sum(r['response_body_bytes'] for r in cameras)
            used = metadata_bytes+camera_bytes
            per_camera = {}
            for record in cameras:
                if record.get('url'):
                    per_camera[record['url']] = per_camera.get(record['url'],0)+record['response_body_bytes']
            state = dict(schema='slowfast_direct_full_network_budget_v1',limit=NEW_NETWORK_CAP,
                used=used,phase=phase,new_response_body_bytes=used,
                inherited_previous_response_body_bytes=self.inherited_bytes,
                cumulative_response_body_bytes=self.inherited_bytes+used,inherited_budget=self.inherited_budget,
                metadata_response_body_bytes=metadata_bytes,camera_response_body_bytes=camera_bytes,
                metadata_requests=len(metadata),camera_requests=len(cameras),
                metadata_new_cap_bytes=METADATA_NEW_CAP,metadata_prefix_cap_bytes=METADATA_PREFIX_CAP,
                camera_prefix_cap_bytes=CAMERA_PREFIX_CAP,camera_response_body_bytes_by_url=per_camera,
                camera_response_body_cap_overrides_by_url=self.response_caps,
                archive_prefix_extent_cap_overrides_by_url=self.archive_extent_caps,
                transport_amendment=self.transport_amendment,
                camera_cap_overrides={},request_ledger_path=str(self.path.resolve()),
                request_ledger_sha256=sha256_file(self.path) if self.path.exists() else hashlib.sha256(b'').hexdigest(),
                full_archives_verified=False)
            with tempfile.NamedTemporaryFile(mode='w',dir=self.output,prefix='.network-budget-',delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(state,stream,sort_keys=True,indent=2,allow_nan=False)
                stream.write('\n');stream.flush();os.fsync(stream.fileno())
            try:
                os.replace(temporary,self.output/'network_budget.json')
            finally:
                temporary.unlink(missing_ok=True)
            if (used > NEW_NETWORK_CAP or metadata_bytes > METADATA_NEW_CAP
                    or camera_bytes > 8*CAMERA_PREFIX_CAP
                    or any(n > self.response_caps.get(url,CAMERA_PREFIX_CAP) for url,n in per_camera.items())):
                raise ValueError('Fixed response-body cap exceeded; every attempt preserved')
            return state


def make_camera_opener(ledger, opener=urllib.request.urlopen):
    """Reserve every ranged body against that URL's lifetime225MB allowance.

    The archive helper also budgets each worker. This guard additionally prevents
    rerunning an interrupted worker from resetting its per-archive allowance.
    Each selected URL has one worker; independent URLs have disjoint allowances.
    """
    def bounded(request, timeout=90):
        match = re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range') or '')
        if not match or int(match[2]) < int(match[1]):
            raise ValueError('Camera request must have a finite bounded Range')
        if int(match[2]) >= ledger.archive_extent_caps.get(request.full_url,CAMERA_PREFIX_CAP):
            raise ValueError('Camera Range exceeds unchanged compressed archive extent cap')
        requested = int(match[2])-int(match[1])+1
        with ledger.lock:
            consumed = sum(r['response_body_bytes'] for r in ledger.records
                if r['kind']=='camera' and r.get('url')==request.full_url)
        if requested > ledger.response_caps.get(request.full_url,CAMERA_PREFIX_CAP)-consumed:
            raise ValueError('Request exceeds remaining camera response-body allowance; no cap increase')
        return opener(request,timeout=timeout)
    return AuditedOpener(bounded,'camera',ledger.append)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _authenticate_plan(path):
    plan = _read_json(path)
    if _plan_digest(plan) != plan.get('plan_sha256') or plan.get('openscene_revision') != REVISION:
        raise ValueError('Previously seen plan identity/revision differs')
    return plan


def _authenticate_inputs(args):
    original_plan = _authenticate_plan(args.original_plan)
    promoted_plan = _authenticate_plan(args.promoted_plan)
    train = [json.loads(line) for line in Path(args.original_train_manifest).read_text().splitlines() if line.strip()]
    promoted = [json.loads(line) for line in Path(args.promoted_holdout_manifest).read_text().splitlines() if line.strip()]
    if train != original_plan['rows']['train'] or promoted != promoted_plan['rows']['val']:
        raise ValueError('Inherited manifest rows differ from their complete frozen plans')
    if sha256_file(args.normalizer_source) != NORMALIZER_SHA256:
        raise ValueError('Authenticated original checkpoint normalizer SHA256 differs')
    seen = {row['route'] for plan in (original_plan,promoted_plan) for rows in plan['rows'].values() for row in rows}
    seen.update(original_plan['logs']);seen.update(promoted_plan['logs'])
    upstream = _read_json(args.upstream_audit)
    if upstream['openscene_revision'] != REVISION:
        raise ValueError('Wrong pinned upstream revision')
    sources = []
    for filename in ('mini_camera_index.json','hf_camera_tree.json'):
        path = Path(args.source_dir)/filename
        expected = next(item['sha256'] for item in upstream['sources'] if Path(item['path']).name==filename)
        if sha256_file(path) != expected:
            raise ValueError('Authenticated source index SHA256 differs')
        sources.append(dict(path=str(path.resolve()),sha256=expected,kind='authenticated_source_index'))
    inherited_sources = []
    for audit_path, plan in ((args.original_audit,original_plan),(args.promoted_audit,promoted_plan)):
        audit = _read_json(audit_path)
        if audit['openscene_revision'] != REVISION or audit['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('Inherited audit/frozen plan identity differs')
        identity = dict(path=str(Path(audit_path).resolve()),sha256=sha256_file(audit_path))
        inherited_sources.extend(dict(item,upstream_audit=identity) for item in audit['sources'])
    previous = _read_json(args.previous_budget)
    if previous.get('cumulative_response_body_bytes') != INHERITED_CUMULATIVE_BYTES or previous.get('phase') != 'complete':
        raise ValueError('Inherited previous cumulative response-body accounting differs')
    if sha256_file(previous['request_ledger_path']) != previous['request_ledger_sha256']:
        raise ValueError('Inherited request ledger SHA256 differs')
    inherited = dict(path=str(Path(args.previous_budget).resolve()),sha256=sha256_file(args.previous_budget))
    metadata = _read_json(args.promoted_audit)['metadata_provenance']
    source_prefix = Path(args.metadata_prefix_source or metadata['path'])
    if (metadata['prefix_bytes'] != BASE_PREFIX_BYTES or metadata['prefix_sha256'] != BASE_PREFIX_SHA256
            or metadata.get('complete') is not True):
        raise ValueError('Inherited320MB metadata provenance identity differs')
    for path,kind in [(args.original_plan,'original_train_and_seen_plan'),(args.promoted_plan,'promoted_holdout_plan'),
                      (args.original_train_manifest,'original_train_manifest'),(args.promoted_holdout_manifest,'promoted_holdout_manifest'),
                      (args.original_audit,'original_source_audit'),(args.promoted_audit,'promoted_source_audit'),
                      (args.normalizer_source,'checkpoint_train256_normalizer'),(args.upstream_audit,'upstream_source_audit')]:
        sources.append(dict(path=str(Path(path).resolve()),sha256=sha256_file(path),kind=kind))
    required = {str(Path(image).resolve()) for row in train+promoted for image in row['images']}
    required.update(str(Path(row['source']['metadata_path']).resolve()) for row in train+promoted)
    inherited_sources = [r for r in inherited_sources if str(Path(r['path']).resolve()) in required]
    return dict(index=_read_json(Path(args.source_dir)/'mini_camera_index.json'),
        tree=_read_json(Path(args.source_dir)/'hf_camera_tree.json'),seen=sorted(seen),train=train,
        promoted=promoted,sources=sources,inherited_sources=inherited_sources,
        inherited_bytes=INHERITED_CUMULATIVE_BYTES,inherited_budget=inherited,source_prefix=source_prefix)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--extend-metadata',action='store_true')
    mode.add_argument('--acquire',action='store_true')
    parser.add_argument('--recover-failed-cameras',action='store_true',
        help='Explicitly approved retransmission allowance for initial TLS-failed shards27/17 only')
    parser.add_argument('--durable-shard27-recovery',action='store_true',
        help='Explicit final durable225MB prefix transfer; resume from authenticated saved bytes only')
    parser.add_argument('--extend-durable-shard27-229',action='store_true',help='Explicit225->229MB extension; unchanged2GB total')
    parser.add_argument('--data-root',default='data/navsim_direct_full')
    parser.add_argument('--output',default='outputs/slowfast_direct_full/data')
    parser.add_argument('--metadata-prefix-source',default=None,help='Default: authenticated previous audit path')
    parser.add_argument('--source-dir',default='outputs/slowfast_v11_expanded/source')
    parser.add_argument('--original-plan',default='outputs/slowfast_v11_expanded/data/plan.json')
    parser.add_argument('--promoted-plan',default='outputs/slowfast_adapter_study/data/plan.json')
    parser.add_argument('--original-train-manifest',default='outputs/slowfast_v11_expanded/data/train.jsonl')
    parser.add_argument('--promoted-holdout-manifest',default='outputs/slowfast_adapter_study/data/holdout.jsonl')
    parser.add_argument('--original-audit',default='outputs/slowfast_v11_expanded/data/source_audit.json')
    parser.add_argument('--promoted-audit',default='outputs/slowfast_adapter_study/data/source_audit.json')
    parser.add_argument('--normalizer-source',default='outputs/slowfast_v11_expanded/data/stats.json')
    parser.add_argument('--previous-budget',default='outputs/slowfast_adapter_study/data/network_budget.json')
    parser.add_argument('--upstream-audit',default='/inspire/ssd/project/sais-bio/public/meiruiqiao/fast-only/outputs/navsim/audit/source_audit.json')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args = parser.parse_args(argv)
    inputs = _authenticate_inputs(args)
    root, output = Path(args.data_root).resolve(),Path(args.output).resolve()
    prefix = root/'metadata_prefix.gz'
    prefix_record = copy_authenticated_prefix(inputs['source_prefix'],prefix,
        expected_bytes=BASE_PREFIX_BYTES,expected_sha256=BASE_PREFIX_SHA256)
    inputs['sources'].append(dict(prefix_record,kind='inherited_authenticated_metadata_prefix'))
    amendment = None
    amendment_path = None
    if sum((args.recover_failed_cameras,args.durable_shard27_recovery,args.extend_durable_shard27_229)) > 1:
        raise ValueError('Choose exactly one explicit transport recovery stage')
    if args.recover_failed_cameras or args.durable_shard27_recovery or args.extend_durable_shard27_229:
        if not args.acquire:
            raise ValueError('Transport recovery requires explicit --acquire')
        recovery_plan = _read_json(output/'plan.json');freeze_direct_plan(recovery_plan,output)
        if args.extend_durable_shard27_229:
            amendment = record_extent_transport_amendment(output,recovery_plan)
            amendment_path = output/'transport_extent_amendment.json'
        elif args.durable_shard27_recovery:
            amendment = record_durable_transport_amendment(output,recovery_plan)
            amendment_path = output/'transport_durable_amendment.json'
        else:
            amendment = record_transport_amendment(output,recovery_plan)
            amendment_path = output/'transport_recovery_amendment.json'
    ledger = RequestLedger(output,inherited_bytes=inputs['inherited_bytes'],inherited_budget=inputs['inherited_budget'],
        transport_amendment=amendment_path)
    budget = ByteBudget(NEW_NETWORK_CAP,ledger.state('initializing')['used'])
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(30):
            print(json.dumps(dict(event='progress',request_count=len(ledger.records),
                new_response_body_bytes=sum(r['response_body_bytes'] for r in ledger.records),
                metadata_prefix_bytes=prefix.stat().st_size)),flush=True)
    progress = threading.Thread(target=heartbeat,daemon=True);progress.start()
    provenance, plan, failure, selection_failure = None,None,None,None
    reports = {};phase = 'planning'
    try:
        if args.extend_metadata:
            if (output/'plan.json').exists() and prefix.stat().st_size != METADATA_PREFIX_CAP:
                raise ValueError('Cannot extend metadata after a different cohort was frozen')
            phase = 'metadata'
            try:
                extend_metadata_prefix(prefix,budget,metadata_bytes_used=ledger.state(phase)['metadata_response_body_bytes'],
                    on_request=ledger.append)
            finally:
                current = ledger.state(phase)
                provenance = dict(url=METADATA_URL,path=str(prefix),prefix_bytes=prefix.stat().st_size,
                    prefix_sha256=sha256_file(prefix),inherited_prefix_bytes=BASE_PREFIX_BYTES,
                    inherited_prefix_sha256=BASE_PREFIX_SHA256,
                    new_response_body_bytes=current['metadata_response_body_bytes'],requests=current['metadata_requests'],
                    requests_detail=[r for r in ledger.records if r['kind']=='metadata'],
                    full_archive_verified=False,complete=prefix.stat().st_size==METADATA_PREFIX_CAP)
                write_json(output/'metadata_acquisition.json',provenance)
        elif (output/'metadata_acquisition.json').exists():
            provenance = _read_json(output/'metadata_acquisition.json')
            if provenance['prefix_bytes'] != prefix.stat().st_size or provenance['prefix_sha256'] != sha256_file(prefix):
                raise ValueError('Saved metadata prefix provenance differs')
        if not args.acquire:
            retained = extract_complete_metadata(prefix,root/'metadata')
            write_json(output/'metadata_retained.json',dict(files=retained,full_archive_verified=False,
                prefix_path=str(prefix),prefix_sha256=sha256_file(prefix)))
            plan = make_direct_plan(retained,inputs['index'],inputs['train'],inputs['promoted'],inputs['seen'],
                sensor_root=root/'sensors/openscene-v1.1/sensor_blobs/mini',
                inherited_sources=inputs['inherited_sources'],upstream_sources=inputs['sources'])
            freeze_direct_plan(plan,output)
        else:
            if not (output/'plan.json').exists() or not (output/'frozen_tokens.json').exists():
                raise ValueError('Freeze plan and tokens before any camera access')
            plan = _read_json(output/'plan.json');freeze_direct_plan(plan,output)
            if (plan['rows']['train'][:256] != inputs['train']
                    or plan['rows']['train'][256:] != [dict(row,split='train',cohort='train') for row in inputs['promoted']]
                    or plan['upstream_sources'] != inputs['sources']):
                raise ValueError('Frozen inherited source identity differs before camera access')
            for item in plan['metadata_records']:
                if sha256_file(item['path']) != item['sha256']:
                    raise ValueError('Frozen metadata SHA256 changed before camera access')
        print(json.dumps(dict(event='frozen_plan',plan_sha256=plan['plan_sha256'],
            token_manifest_sha256=plan['token_manifest_sha256'],logs=plan['splits']['new_val'],
            counts=plan['cohort_counts'],required_new_images=len(plan['required_images']),
            inherited_images=len(plan['inherited_images']),fixed_source_token=plan['fixed_source_token']),indent=2),flush=True)
        if not args.acquire:
            phase = 'plan_frozen';return
        phase = 'cameras'
        reports = {r['shard']:r for r in load_direct_completed_reports(output,plan,inputs['tree'],archive_extent_caps=ledger.archive_extent_caps)}
        force = set(plan['splits']['new_val'])-{r['log'] for r in reports.values()}
        audited = make_camera_opener(ledger)
        def record(report):
            reports[report['shard']] = report
            report['requests_detail'] = [r for r in audited.records if r['url']==report['url']]
            if sum(r['response_body_bytes'] for r in report['requests_detail']) != report['response_body_bytes']:
                raise ValueError('Camera request/worker byte accounting differs')
            persist_attempt_report(output,report,aggregate_used=ledger.state(phase)['used'])
        if args.durable_shard27_recovery or args.extend_durable_shard27_229:
            expected_log = next(name for name,info in plan['logs'].items() if info['shard']==27)
            if force and force != {expected_log}:
                raise ValueError('Final durable recovery may fetch only frozen shard27')
            if force:
                durable_prefix = root/'camera_prefixes/camera_27_prefix.gz'
                verify_durable_prefix(durable_prefix,durable_stage_records(ledger,amendment))
                identity = next(item for item in inputs['tree'] if item['path'].endswith('/openscene_sensor_mini_camera_27.tgz'))
                bounded_retry, _ = _retrying_opener(audited,3,5.)
                extent_cap = ledger.archive_extent_caps.get(_camera_url(27),CAMERA_PREFIX_CAP)
                try:
                    download_durable_prefix(_camera_url(27),durable_prefix,budget,
                        expected_archive_bytes=identity['size'],opener=bounded_retry,prefix_cap=extent_cap,archive_extent_cap=extent_cap)
                finally:
                    stage_records = durable_stage_records(ledger,amendment)
                    verify_durable_prefix(durable_prefix,stage_records)
                    transfer = dict(url=_camera_url(27),path=str(durable_prefix.resolve()),
                        prefix_bytes=durable_prefix.stat().st_size if durable_prefix.exists() else 0,
                        prefix_sha256=sha256_file(durable_prefix) if durable_prefix.exists() else hashlib.sha256(b'').hexdigest(),
                        requests_detail=stage_records,new_response_body_bytes=sum(r['response_body_bytes'] for r in stage_records),
                        complete=durable_prefix.exists() and durable_prefix.stat().st_size==extent_cap,
                        full_archive_verified=False,transport_amendment=ledger.transport_amendment)
                    write_json(output/'camera_27_durable_prefix.json',transfer)
                durable_report = extract_durable_camera(plan,inputs['tree'],durable_prefix,
                    sensor_dest=root/'sensors',shard=27,transfer=transfer,archive_extent_cap=extent_cap)
                reports[27] = durable_report
                persist_attempt_report(output,durable_report,aggregate_used=ledger.state(phase)['used'])
        else:
            acquire_plan(plan,inputs['tree'],sensor_dest=root/'sensors',budget=budget,opener=audited,
                on_report=record,workers=args.workers,force_reverify_logs=force,
                archive_limit=CAMERA_PREFIX_CAP,max_request_attempts=3)
        verify_provenance_coverage(plan,list(reports.values()))
        result = write_direct_prepared(plan,output,normalizer_source=args.normalizer_source,
            acquisition_reports=[reports[k] for k in sorted(reports)],network_budget=ledger.state('complete'),
            metadata_provenance=provenance,transport_amendment=amendment)
        phase = 'complete'
        print(json.dumps(dict(event='prepared',counts=result['counts'],normalization=result['normalization'],
            plan_sha256=result['plan_sha256']),sort_keys=True),flush=True)
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        selection_failure = getattr(error,'selection_audit',None)
        if selection_failure is not None:
            write_json(output/'selection_failure.json',selection_failure)
        raise
    finally:
        stopped.set();state = ledger.state('failed' if failure else phase)
        if phase != 'complete' or failure:
            retained_path = output/'metadata_retained.json'
            retained_manifest = (dict(path=str(retained_path.resolve()),sha256=sha256_file(retained_path))
                if retained_path.exists() else None)
            write_json(output/'source_audit.json',dict(schema='slowfast_direct_full_source_audit_v1',
                openscene_revision=REVISION,sources=inputs['sources'],network_budget=state,
                metadata_provenance=provenance,network_requests=ledger.records,
                retained_metadata_manifest=retained_manifest,
                transport_amendment=amendment,
                acquisition_reports=list(reports.values()),full_archives_verified=False,complete=False,
                error=failure,selection_failure=selection_failure,
                plan_sha256=plan['plan_sha256'] if plan else None,
                token_manifest_sha256=plan['token_manifest_sha256'] if plan else None))


if __name__ == '__main__':
    main()
