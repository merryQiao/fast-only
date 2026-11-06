"""Freeze expanded mini data selection before bounded original-CAM_F0 acquisition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from fast_navsim.acquire import ByteBudget,write_json
from slowfast_v11.expanded_data import make_expanded_plan,acquire_plan,write_prepared,sha256_file,load_completed_reports,verify_provenance_coverage,REVISION

BASE=Path('/inspire/ssd/project/sais-bio/public/meiruiqiao/fast-only')


def _append_attempt(output,report,aggregate_used,imported_from=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ledger=output/'camera_attempts.jsonl'
    digest=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    previous=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    if any(item['record_sha256']==digest for item in previous):return
    item=dict(record_sha256=digest,aggregate_response_body_bytes=aggregate_used,report=report)
    if imported_from is not None:item['imported_from']=str(imported_from)
    with ledger.open('a') as stream:
        stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def persist_attempt_report(output,report,*,aggregate_used):
    """Fsync prior/current attempt records before atomically replacing latest."""
    output=Path(output);latest=output/f'camera_{report["shard"]}_acquisition.json'
    if latest.exists():_append_attempt(output,json.loads(latest.read_text()),aggregate_used,latest.name)
    _append_attempt(output,report,aggregate_used)
    with tempfile.NamedTemporaryFile(mode='w',dir=output,prefix='.camera-report-',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    try:os.replace(temporary,latest)
    finally:temporary.unlink(missing_ok=True)


def import_attempt_history(output,*,aggregate_used,selected_shards):
    """Import known backup artifacts once; aggregate budget remains authoritative."""
    output=Path(output)
    for path in sorted(output.parent.rglob('camera_*_acquisition.json')):
        report=json.loads(path.read_text())
        if report.get('shard') in selected_shards:
            _append_attempt(output,report,aggregate_used,path.relative_to(output.parent))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--acquire',action='store_true')
    parser.add_argument('--data-root',default='data/navsim_expanded')
    parser.add_argument('--output',default='outputs/slowfast_v11_expanded/data')
    parser.add_argument('--metadata',default='outputs/slowfast_v11_expanded/metadata_retained.json')
    parser.add_argument('--source-dir',default='outputs/slowfast_v11_expanded/source')
    parser.add_argument('--budget',default='outputs/slowfast_v11_expanded/network_budget.json')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3,help='Independent bounded camera streams (default3)')
    parser.add_argument('--archive-cap-override',action='append',choices=('7:256000000',),default=[],
        help='Explicit operational amendment: shard 7 only may use a 256 MB prefix; aggregate remains 2 GB')
    parser.add_argument('--legacy-manifest',default=str(BASE/'outputs/navsim/data/val.jsonl'))
    parser.add_argument('--legacy-sensor-root',default=str(BASE/'data/navsim_smoke/sensors/openscene-v1.1/sensor_blobs/mini'))
    parser.add_argument('--upstream-audit',default=str(BASE/'outputs/navsim/audit/source_audit.json'))
    args=parser.parse_args()
    cap_overrides={int(shard):int(cap) for shard,cap in (value.split(':') for value in args.archive_cap_override)}
    root=Path(args.data_root).resolve();output=Path(args.output).resolve();source=Path(args.source_dir)
    retained=json.loads(Path(args.metadata).read_text())['files']
    index=json.loads((source/'mini_camera_index.json').read_text())
    tree=json.loads((source/'hf_camera_tree.json').read_text())
    upstream=json.loads(Path(args.upstream_audit).read_text())
    if upstream['openscene_revision']!=REVISION:raise ValueError('Wrong pinned OpenScene source revision')
    for filename in ('mini_camera_index.json','hf_camera_tree.json'):
        expected=next(item['sha256'] for item in upstream['sources'] if Path(item['path']).name==filename)
        if sha256_file(source/filename)!=expected:raise ValueError('Copied source index SHA256 mismatch')
    legacy=[json.loads(line) for line in Path(args.legacy_manifest).read_text().splitlines() if line.strip()]
    plan=make_expanded_plan(retained,index,legacy,sensor_root=root/'sensors/openscene-v1.1/sensor_blobs/mini',
                            legacy_sensor_root=args.legacy_sensor_root)
    output.mkdir(parents=True,exist_ok=True);plan_path=output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Existing frozen plan differs; use a new cohort destination')
    else:write_json(plan_path,plan)
    print(json.dumps({'plan_sha256':plan['plan_sha256'],'splits':plan['splits'],'counts':plan['cohort_counts'],
        'required_image_count':len(plan['required_images']),
        'selected_speed_ranges':{name:info['selected_speed_min_max'] for name,info in plan['logs'].items()}},indent=2),flush=True)
    if not args.acquire:return
    prior=json.loads(Path(args.budget).read_text())
    if prior['limit']!=2_000_000_000:raise ValueError('Expanded aggregate response-body budget is fixed at2GB')
    prior=dict(prior,archive_prefix_cap_policy=dict(default=225_000_000,
        overrides={str(shard):cap for shard,cap in cap_overrides.items()}))
    budget=ByteBudget(prior['limit'],prior['used'])
    expected={info['shard'] for info in plan['logs'].values()}
    import_attempt_history(output,aggregate_used=budget.used,selected_shards=expected)
    reports={report['shard']:report for report in load_completed_reports(output,plan,tree,archive_cap_overrides=cap_overrides)}
    verified_logs={report['log'] for report in reports.values()}
    force_reverify=set(plan['splits']['train']+plan['splits']['new_val'])-verified_logs
    def record(report):
        reports[report['shard']]=report;persist_attempt_report(output,report,aggregate_used=budget.used)
        write_json(args.budget,dict(prior,phase='cameras',used=budget.used))
    try:
        acquire_plan(plan,tree,sensor_dest=root/'sensors',budget=budget,on_report=record,workers=args.workers,
            force_reverify_logs=force_reverify,archive_cap_overrides=cap_overrides)
        if set(reports)!=expected or any(not report['complete'] for report in reports.values()):
            raise ValueError('Every selected camera archive requires completed acquisition provenance before manifests')
        coverage=verify_provenance_coverage(plan,list(reports.values()))
        if coverage['verified_images']!=len(plan['required_images']):raise ValueError('Exact frozen image provenance coverage required')
        result=write_prepared(plan,output,acquisition_reports=[reports[shard] for shard in sorted(reports)],network_bytes=budget.used,
            upstream_audit=dict(path=str(Path(args.upstream_audit).resolve()),sha256=sha256_file(args.upstream_audit)))
        print(json.dumps(result['counts']),flush=True)
    finally:
        write_json(args.budget,dict(prior,phase='cameras',used=budget.used))


if __name__=='__main__':main()
