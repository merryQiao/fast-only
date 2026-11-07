"""Prepare a frozen independent8x32 mini holdout under a2GB new-body cap.

Run --extend-metadata first (metadata network only), inspect the frozen plan,
then --acquire for its exact original JPEGs. Default mode is offline planning.
No original TRAIN rows or normalization statistics are rewritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request
import shutil

from fast_navsim.acquire import ByteBudget, write_json
from slowfast_v11.adapter_data import (
    REVISION, BASE_PREFIX_BYTES, BASE_PREFIX_SHA256, METADATA_PREFIX_CAP,
    METADATA_NEW_CAP, CAMERA_PREFIX_CAP, NEW_NETWORK_CAP, METADATA_URL,
    AuditedOpener, copy_authenticated_prefix, extend_metadata_prefix,
    make_adapter_plan, freeze_adapter_plan, write_adapter_prepared,
    acquire_plan, load_completed_reports, verify_provenance_coverage,
    sha256_file, _plan_digest,
)
from tools.prepare_navsim_smoke import extract_complete_metadata
from tools.prepare_slowfast_expanded import persist_attempt_report


class RequestLedger:
    """Durable append-only request accounting, including zero-body failures."""
    def __init__(self, output, *, inherited_bytes, inherited_budget, metadata_prefix_cap=METADATA_PREFIX_CAP):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.output/'network_requests.jsonl'
        self.inherited_bytes = inherited_bytes
        self.inherited_budget = inherited_budget
        if metadata_prefix_cap not in (280_000_000,320_000_000):
            raise ValueError('Unsupported metadata prefix cap')
        self.metadata_prefix_cap = metadata_prefix_cap
        self.metadata_new_cap = metadata_prefix_cap-BASE_PREFIX_BYTES
        self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()] if self.path.exists() else []
        for item in self.records:
            digest = item.get('response_body_sha256')
            if (type(item.get('response_body_bytes')) is not int or item['response_body_bytes'] < 0
                    or item.get('kind') not in ('metadata','camera') or not isinstance(digest,str)
                    or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Invalid request ledger accounting/hash')
        if len({item['request_id'] for item in self.records}) != len(self.records):
            raise ValueError('Duplicate request ledger identity')
        state_path = self.output/'network_budget.json'
        if state_path.exists():
            old = json.loads(state_path.read_text())
            if old.get('metadata_prefix_cap_bytes',METADATA_PREFIX_CAP) > metadata_prefix_cap:
                raise ValueError('Existing amended metadata budget requires explicit --metadata-prefix-cap320000000')
            if old['inherited_previous_response_body_bytes'] != inherited_bytes or old['inherited_budget'] != inherited_budget:
                raise ValueError('Inherited cumulative budget identity differs')
            # A crash can leave the snapshot behind its durable request ledger.
            if old['used'] > sum(item['response_body_bytes'] for item in self.records):
                raise ValueError('Network snapshot charges bytes absent from durable request ledger')
        self.state('initialized')

    def append(self, record):
        with self.lock:
            self._append(record)

    def _append(self, record):
        if any(item['request_id'] == record['request_id'] for item in self.records):
            raise ValueError('Request identity already recorded')
        with self.path.open('a') as stream:
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        state = self.state(record['kind'])
        print(json.dumps(dict(event='request_complete', kind=record['kind'],
            range=record.get('range'), response_body_bytes=record['response_body_bytes'],
            new_response_body_bytes=state['used'], error=record.get('error'))),flush=True)

    def state(self, phase):
        with self.lock:
            return self._state(phase)

    def _state(self, phase):
        metadata = [r for r in self.records if r['kind'] == 'metadata']
        cameras = [r for r in self.records if r['kind'] == 'camera']
        metadata_bytes = sum(r['response_body_bytes'] for r in metadata)
        camera_bytes = sum(r['response_body_bytes'] for r in cameras)
        used = metadata_bytes+camera_bytes
        state = dict(schema='slowfast_adapter_network_budget_v1', limit=NEW_NETWORK_CAP,
            used=used, phase=phase, new_response_body_bytes=used,
            inherited_previous_response_body_bytes=self.inherited_bytes,
            cumulative_response_body_bytes=self.inherited_bytes+used,
            inherited_budget=self.inherited_budget,
            metadata_response_body_bytes=metadata_bytes, camera_response_body_bytes=camera_bytes,
            metadata_requests=len(metadata), camera_requests=len(cameras),
            metadata_new_cap_bytes=self.metadata_new_cap, metadata_prefix_cap_bytes=self.metadata_prefix_cap,
            camera_prefix_cap_bytes=CAMERA_PREFIX_CAP, camera_cap_overrides={},
            request_ledger_path=str(self.path.resolve()),
            request_ledger_sha256=sha256_file(self.path) if self.path.exists() else hashlib.sha256(b'').hexdigest(),
            full_archives_verified=False)
        with tempfile.NamedTemporaryFile(mode='w',dir=self.output,prefix='.network-budget-',delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(state,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary,self.output/'network_budget.json')
        finally:
            temporary.unlink(missing_ok=True)
        if used > NEW_NETWORK_CAP or metadata_bytes > self.metadata_new_cap:
            raise ValueError('New response-body cap exceeded; preserved every attempt')
        return state


def record_metadata_amendment(output, prefix, *, requested_prefix_cap):
    """Persist the specific authorized280-to320MB amendment before networking."""
    output, prefix = Path(output), Path(prefix)
    path = output/'metadata_prefix_amendment.json'
    if requested_prefix_cap != 320_000_000:
        raise ValueError('Only the explicit320MB metadata amendment is supported')
    if path.exists():
        amendment = json.loads(path.read_text())
        if (amendment.get('requested_metadata_prefix_cap_bytes') != requested_prefix_cap
                or amendment.get('new_total_response_body_cap_bytes') != NEW_NETWORK_CAP):
            raise ValueError('Metadata amendment identity differs')
        return amendment
    failure_path, acquisition_path = output/'selection_failure.json', output/'metadata_acquisition.json'
    if not failure_path.exists() or not acquisition_path.exists():
        raise ValueError('Explicit320MB amendment requires preserved280MB cohort-failure evidence')
    failure, acquisition = json.loads(failure_path.read_text()),json.loads(acquisition_path.read_text())
    digest = sha256_file(prefix)
    if (prefix.stat().st_size != METADATA_PREFIX_CAP or acquisition['prefix_bytes'] != METADATA_PREFIX_CAP
            or acquisition['prefix_sha256'] != digest or acquisition.get('complete') is not True
            or failure.get('required_recording_groups') != 8 or failure.get('available_recording_groups') != 7
            or failure.get('cohort_frozen') is not False):
        raise ValueError('Prior280MB prefix or seven-group failure identity differs')
    snapshots = []
    for filename in ('metadata_acquisition.json','selection_failure.json','source_audit.json','network_budget.json'):
        source = output/filename
        if not source.exists():
            continue
        target = source.with_name(source.stem+f'_{METADATA_PREFIX_CAP}'+source.suffix)
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError('Prior amendment snapshot identity differs')
        if not target.exists():
            shutil.copy2(source,target)
        snapshots.append(dict(path=str(target.resolve()),sha256=sha256_file(target)))
    amendment = dict(schema='slowfast_adapter_metadata_amendment_v1',
        event_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        explicit_cli_flag='--metadata-prefix-cap 320000000',
        reason='Authorized operational extension after fixed280MB prefix yielded seven eligible groups; eight-group cohort and all selection/model/analysis rules remain fixed; no model metrics accessed',
        previous_metadata_prefix_cap_bytes=METADATA_PREFIX_CAP,previous_prefix_sha256=digest,
        requested_metadata_prefix_cap_bytes=requested_prefix_cap,
        requested_metadata_new_body_cap_bytes=200_000_000,new_total_response_body_cap_bytes=NEW_NETWORK_CAP,
        camera_prefix_cap_bytes=CAMERA_PREFIX_CAP,camera_archives=8,
        selection_failure_path=str(failure_path.resolve()),selection_failure_sha256=sha256_file(failure_path),
        previous_metadata_acquisition_sha256=sha256_file(acquisition_path),preserved_snapshots=snapshots)
    write_json(path,amendment)
    return amendment


def _authenticate_inputs(args):
    upstream = json.loads(Path(args.upstream_audit).read_text())
    if upstream['openscene_revision'] != REVISION:
        raise ValueError('Wrong pinned OpenScene revision')
    sources = []
    for filename in ('mini_camera_index.json','hf_camera_tree.json'):
        path = Path(args.source_dir)/filename
        expected = next(item['sha256'] for item in upstream['sources'] if Path(item['path']).name == filename)
        if sha256_file(path) != expected:
            raise ValueError('Copied source index SHA256 mismatch')
        sources.append(dict(path=str(path.resolve()),sha256=expected,kind='authenticated_source_index'))
    seen_path = Path(args.seen_plan)
    seen = json.loads(seen_path.read_text())
    if _plan_digest(seen) != seen.get('plan_sha256') or seen.get('openscene_revision') != REVISION:
        raise ValueError('Previously seen cohort plan identity differs')
    seen_logs = set(seen['logs'])
    seen_logs.update(row['route'] for rows in seen['rows'].values() for row in rows)
    sources.append(dict(path=str(seen_path.resolve()),sha256=sha256_file(seen_path),kind='previously_seen_cohort'))
    previous_path = Path(args.previous_budget)
    previous = json.loads(previous_path.read_text())
    if type(previous.get('used')) is not int or previous['used'] < 0:
        raise ValueError('Invalid inherited response-body accounting')
    inherited = dict(path=str(previous_path.resolve()),sha256=sha256_file(previous_path))
    return (json.loads((Path(args.source_dir)/'mini_camera_index.json').read_text()),
        json.loads((Path(args.source_dir)/'hf_camera_tree.json').read_text()),
        sorted(seen_logs),sources,previous['used'],inherited)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--extend-metadata',action='store_true')
    mode.add_argument('--acquire',action='store_true')
    parser.add_argument('--data-root',default='data/navsim_adapter_study')
    parser.add_argument('--output',default='outputs/slowfast_adapter_study/data')
    parser.add_argument('--metadata-prefix-source',default='data/navsim_expanded/metadata_prefix.gz')
    parser.add_argument('--source-dir',default='outputs/slowfast_v11_expanded/source')
    parser.add_argument('--seen-plan',default='outputs/slowfast_v11_expanded/data/plan.json')
    parser.add_argument('--previous-budget',default='outputs/slowfast_v11_expanded/network_budget.json')
    parser.add_argument('--upstream-audit',default='/inspire/ssd/project/sais-bio/public/meiruiqiao/fast-only/outputs/navsim/audit/source_audit.json')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--metadata-prefix-cap',type=int,choices=(280_000_000,320_000_000),default=280_000_000,
        help='Explicit operational amendment only:320MB metadata prefix/200MB new metadata; total remains2GB')
    args = parser.parse_args(argv)
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    index, tree, seen, sources, inherited_bytes, inherited = _authenticate_inputs(args)
    prefix = root/'metadata_prefix.gz'
    prefix_record = copy_authenticated_prefix(args.metadata_prefix_source,prefix)
    sources.append(dict(prefix_record,kind='inherited_authenticated_metadata_prefix'))
    amendment = None
    if args.metadata_prefix_cap == 320_000_000:
        amendment = record_metadata_amendment(output,prefix,requested_prefix_cap=args.metadata_prefix_cap)
    ledger = RequestLedger(output,inherited_bytes=inherited_bytes,inherited_budget=inherited,
        metadata_prefix_cap=args.metadata_prefix_cap)
    budget = ByteBudget(NEW_NETWORK_CAP,ledger.state('initializing')['used'])
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(30):
            print(json.dumps(dict(event='progress',new_response_body_bytes=sum(r['response_body_bytes'] for r in ledger.records),
                request_count=len(ledger.records),metadata_prefix_bytes=prefix.stat().st_size)),flush=True)
    progress = threading.Thread(target=heartbeat,daemon=True)
    progress.start()
    provenance = None
    reports = {}
    plan = None
    phase = 'planning'
    failure = None
    selection_failure = None
    try:
        if args.extend_metadata:
            if (output/'plan.json').exists() and prefix.stat().st_size != args.metadata_prefix_cap:
                raise ValueError('Cannot extend metadata after freezing a different cohort')
            phase = 'metadata'
            try:
                provenance = extend_metadata_prefix(prefix,budget,
                    metadata_bytes_used=ledger.state(phase)['metadata_response_body_bytes'],on_request=ledger.append,
                    prefix_cap=args.metadata_prefix_cap,metadata_new_cap=args.metadata_prefix_cap-BASE_PREFIX_BYTES)
            finally:
                current = ledger.state(phase)
                provenance = dict(url=METADATA_URL,path=str(prefix),prefix_bytes=prefix.stat().st_size,
                    prefix_sha256=sha256_file(prefix),inherited_prefix_bytes=BASE_PREFIX_BYTES,
                    inherited_prefix_sha256=BASE_PREFIX_SHA256,
                    new_response_body_bytes=current['metadata_response_body_bytes'],
                    requests=current['metadata_requests'],full_archive_verified=False,
                    requests_detail=[r for r in ledger.records if r['kind']=='metadata'],
                    complete=prefix.stat().st_size == args.metadata_prefix_cap,
                    metadata_prefix_amendment=amendment)
                write_json(output/'metadata_acquisition.json',provenance)
        elif (output/'metadata_acquisition.json').exists():
            provenance = json.loads((output/'metadata_acquisition.json').read_text())
            if provenance['prefix_bytes'] != prefix.stat().st_size or provenance['prefix_sha256'] != sha256_file(prefix):
                raise ValueError('Saved metadata prefix provenance differs')
            provenance['requests_detail'] = [r for r in ledger.records if r['kind']=='metadata']
        if not args.acquire:
            retained = extract_complete_metadata(prefix,root/'metadata')
            write_json(output/'metadata_retained.json',dict(files=retained,full_archive_verified=False,
                prefix_path=str(prefix),prefix_sha256=sha256_file(prefix)))
            plan = make_adapter_plan(retained,index,seen,sensor_root=root/'sensors/openscene-v1.1/sensor_blobs/mini')
            freeze_adapter_plan(plan,output)
        else:
            if not (output/'plan.json').exists() or not (output/'frozen_tokens.json').exists():
                raise ValueError('Freeze plan and token manifest before any camera access')
            plan = json.loads((output/'plan.json').read_text())
            freeze_adapter_plan(plan,output)
            for item in plan['metadata_records']:
                if sha256_file(item['path']) != item['sha256']:
                    raise ValueError('Frozen metadata SHA256 changed before camera access')
        print(json.dumps(dict(event='frozen_plan',plan_sha256=plan['plan_sha256'],
            token_manifest_sha256=plan['token_manifest_sha256'],logs=plan['splits']['holdout'],
            counts=plan['cohort_counts'],required_images=len(plan['required_images'])),indent=2),flush=True)
        if not args.acquire:
            phase = 'plan_frozen'
            return
        phase = 'cameras'
        reports = {r['shard']:r for r in load_completed_reports(output,plan,tree)}
        verified = {r['log'] for r in reports.values()}
        force = set(plan['splits']['holdout'])-verified
        audited = AuditedOpener(urllib.request.urlopen,'camera',ledger.append)
        def record(report):
            reports[report['shard']] = report
            report['requests_detail'] = [r for r in audited.records if r['url'] == report['url']]
            if sum(r['response_body_bytes'] for r in report['requests_detail']) != report['response_body_bytes']:
                raise ValueError('Camera request/worker byte accounting differs')
            persist_attempt_report(output,report,aggregate_used=ledger.state(phase)['used'])
        acquire_plan(plan,tree,sensor_dest=root/'sensors',budget=budget,opener=audited,
            on_report=record,workers=args.workers,force_reverify_logs=force,
            archive_limit=CAMERA_PREFIX_CAP,max_request_attempts=3)
        verify_provenance_coverage(plan,list(reports.values()))
        phase = 'complete'
        result = write_adapter_prepared(plan,output,acquisition_reports=[reports[k] for k in sorted(reports)],
            network_budget=ledger.state(phase),metadata_provenance=provenance,
            upstream_audit=dict(path=str(Path(args.upstream_audit).resolve()),sha256=sha256_file(args.upstream_audit)))
        print(json.dumps(dict(event='prepared',**result),sort_keys=True),flush=True)
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        selection_failure = getattr(error,'selection_audit',None)
        if selection_failure is not None:
            write_json(output/'selection_failure.json',selection_failure)
        raise
    finally:
        stopped.set()
        state = ledger.state('failed' if failure else phase)
        if phase != 'complete' or failure:
            # An incomplete audit is also an artifact: failures are never lost.
            retained_path = output/'metadata_retained.json'
            retained_manifest = (dict(path=str(retained_path),sha256=sha256_file(retained_path))
                if retained_path.exists() else None)
            audit = dict(schema='slowfast_adapter_source_audit_v1',openscene_revision=REVISION,
                sources=sources,network_budget=state,metadata_provenance=provenance,
                retained_metadata_manifest=retained_manifest,
                metadata_prefix_amendment=amendment,
                network_requests=ledger.records,acquisition_reports=list(reports.values()),
                full_archives_verified=False,complete=False,error=failure,
                selection_failure=selection_failure,
                plan_sha256=plan['plan_sha256'] if plan else None,
                token_manifest_sha256=plan['token_manifest_sha256'] if plan else None)
            write_json(output/'source_audit.json',audit)


if __name__ == '__main__':
    main()
