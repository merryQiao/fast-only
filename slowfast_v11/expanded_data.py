"""Frozen metadata-first selection and bounded original-JPEG expanded mini data."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import ssl
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from fast_navsim.acquire import HTTPRangeReader,ByteBudget,extract_tar_prefix,write_json
from fast_navsim.data import _pose,build_window,write_manifests

REVISION='a76f840b65e972bc45e56c2adced897498e9a026'
NAVSIM_REVISION='0a380a9063d7162ec93d0f51e9990ebac585f720'
OLD_TRAIN_LOGS=('2021.06.09.14.58.55_veh-35_01894_02311','2021.06.09.14.03.17_veh-12_02584_02970')
LEGACY_VAL_LOG='2021.06.28.16.57.59_veh-26_00016_00484'
BINS=('low_speed','mid_speed','high_speed')


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _speed(frame):
    return math.hypot(*[float(v) for v in frame['ego_dynamic_state'][:2]])


def _bin(frame):
    speed=_speed(frame)
    return BINS[0] if speed<1. else BINS[1] if speed<10. else BINS[2]


def structural_candidates(frames,stride=5):
    """Fixed index anchors 3+5k; no image-existence or future-pose selection."""
    if stride!=5:raise ValueError('Expanded candidates use fixed stride five')
    candidates=[];excluded=Counter()
    for i in range(3,len(frames)-8,stride):
        reason=None
        try:
            span=frames[i-3:i+9];history=frames[i-3:i+1];route=frames[i]['log_name']
            timestamps=[int(f['timestamp']) for f in span]
            if any(abs(b-a-500_000)>1000 for a,b in zip(timestamps,timestamps[1:])):reason='timestamp_gap'
            elif any(f['log_name']!=route for f in span):reason='log_mismatch'
            commands=[list(f['driving_command']) for f in history]
            if reason is None and any(len(v)!=4 or sum(v)!=1 or any(x not in (0,1) for x in v) for v in commands):reason='invalid_command'
            if reason is None and any(v!=commands[-1] for v in commands):reason='navigation_mismatch'
            if reason is None:
                try:
                    for frame in history:_pose(frame)
                except (TypeError,ValueError,KeyError):reason='invalid_pose'
            ego=[float(v) for v in frames[i]['ego_dynamic_state']]
            if reason is None and (len(ego)!=4 or not all(math.isfinite(v) for v in ego)):reason='invalid_ego'
            paths=[Path(f['cams']['CAM_F0']['data_path']) for f in history]
            if reason is None and (len(set(paths))!=4 or any(p.is_absolute() or '..' in p.parts for p in paths)):reason='invalid_image_path'
        except (KeyError,TypeError,ValueError,IndexError):reason='malformed_frame'
        if reason:excluded[reason]+=1
        else:candidates.append(i)
    return candidates,dict(excluded)


def uniform_indices(candidates,count):
    if isinstance(count,bool) or not isinstance(count,int) or count<=0 or len(candidates)<count:
        raise ValueError('Insufficient candidates for fixed uniform selection')
    if count==1:return [candidates[len(candidates)//2]]
    return [candidates[round(i*(len(candidates)-1)/(count-1))] for i in range(count)]


def select_train_indices(frames,candidates,count=64,seed=42):
    """Current-speed quotas; deficits fill mid, then low, then high capacity."""
    if count!=64 or len(candidates)<64:raise ValueError('TRAIN requires exactly64 candidates without replacement')
    pools={name:[i for i in candidates if _bin(frames[i])==name] for name in BINS}
    desired=dict(zip(BINS,(16,24,24)))
    allocated={name:min(desired[name],len(pools[name])) for name in BINS}
    deficit=64-sum(allocated.values())
    for name in ('mid_speed','low_speed','high_speed'):
        add=min(deficit,len(pools[name])-allocated[name]);allocated[name]+=add;deficit-=add
    if deficit:raise ValueError('Insufficient candidates after explicit quota redistribution')
    selected=[]
    for name in BINS:
        # Independent deterministic sampling; no future target or image availability.
        rng=random.Random(f'{seed}:{frames[candidates[0]]["log_name"]}:{name}')
        selected.extend(rng.sample(pools[name],allocated[name]))
    selected.sort()
    report=dict(desired=desired,available={name:len(pools[name]) for name in BINS},selected=allocated,
                redistributed={name:allocated[name]-desired[name] for name in BINS},
                redistribution_order=['mid_speed','low_speed','high_speed'],
                selected_speed_min_max=[min(_speed(frames[i]) for i in selected),max(_speed(frames[i]) for i in selected)])
    return selected,report


def _plan_digest(plan):
    content={key:value for key,value in plan.items() if key!='plan_sha256'}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_expanded_plan(retained,camera_index,legacy_rows,*,sensor_root,legacy_sensor_root,seed=42):
    """Freeze all splits, tokens and paths before any new sensor-file checks."""
    if seed!=42:raise ValueError('Expanded selection seed is fixed at42')
    sensor_root=Path(sensor_root).resolve();legacy_sensor_root=Path(legacy_sensor_root).resolve()
    records={Path(item['path']).stem:item for item in retained}
    first={names[0]:int(key.rsplit('_',1)[1]) for key,names in camera_index.items()}
    loaded={};details={};new=[]
    for item in retained:
        name=Path(item['path']).stem
        if name==LEGACY_VAL_LOG or name not in first:continue
        if sha256_file(item['path'])!=item['sha256']:raise ValueError('Metadata SHA256 mismatch')
        frames=pickle.loads(Path(item['path']).read_bytes())
        candidates,excluded=structural_candidates(frames)
        if len(candidates)<64:
            if name in OLD_TRAIN_LOGS:raise ValueError('Old TRAIN log has fewer than64 structural candidates')
            continue
        loaded[name]=(frames,candidates)
        details[name]=dict(shard=first[name],candidate_count=len(candidates),excluded=excluded,
            candidate_speed_bins=dict(Counter(_bin(frames[i]) for i in candidates)),
            metadata_path=item['path'],metadata_sha256=item['sha256'])
        if name not in OLD_TRAIN_LOGS:new.append(name)
        if len(new)>=4 and all(name in loaded for name in OLD_TRAIN_LOGS):break
    if len(new)<4 or any(name not in loaded for name in OLD_TRAIN_LOGS):
        raise ValueError('Need both old TRAIN logs and four new eligible first-in-shard logs')
    chosen=new[:4];shuffled=sorted(chosen);random.Random(seed).shuffle(shuffled)
    splits=dict(train=list(OLD_TRAIN_LOGS)+shuffled[:2],new_val=shuffled[2:],legacy_val=[LEGACY_VAL_LOG])
    rows={'train':[],'val':[]};logs={}
    for cohort in ('train','new_val'):
        for name in splits[cohort]:
            frames,candidates=loaded[name];info=details[name]
            if cohort=='train':selected,selection=select_train_indices(frames,candidates,seed=seed)
            else:selected=uniform_indices(candidates,32);selection={'policy':'uniform temporal candidate positions','count':32}
            logs[name]=dict(info,selection=selection,selected_indices=selected,cohort=cohort,
                selected_command_counts={str(key):value for key,value in Counter(list(frames[i]['driving_command']).index(1) for i in selected).items()},
                selected_speed_min_max=[min(_speed(frames[i]) for i in selected),max(_speed(frames[i]) for i in selected)])
            source=dict(navsim_revision=NAVSIM_REVISION,openscene_revision=REVISION,upstream_split='mini',
                metadata_path=info['metadata_path'],metadata_sha256=info['metadata_sha256'],
                camera_archive=f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{info["shard"]}.tgz')
            for i in selected:
                row=build_window(frames,i,'train' if cohort=='train' else 'val',sensor_root,source)
                row['cohort']=cohort;rows[row['split']].append(row)
    if len(legacy_rows)!=4 or any(r['split']!='val' or r['route']!=LEGACY_VAL_LOG for r in legacy_rows):
        raise ValueError('Legacy VAL must be the four original rows from its excluded log')
    legacy_record=records[LEGACY_VAL_LOG]
    if sha256_file(legacy_record['path'])!=legacy_record['sha256']:raise ValueError('Legacy metadata SHA256 mismatch')
    copies={}
    for original in legacy_rows:
        if original['source']['metadata_sha256']!=legacy_record['sha256']:
            raise ValueError('Legacy metadata identity differs from original VAL')
        row=copy.deepcopy(original);row['cohort']='legacy_val';images=[]
        for original_path in original['images']:
            relative=Path(original_path).resolve().relative_to(legacy_sensor_root)
            target=str(sensor_root/relative);images.append(target)
            copies[target]=dict(source=str(Path(original_path).resolve()),target=target,sha256=sha256_file(original_path))
        row['images']=images;row['source']['metadata_path']=legacy_record['path'];rows['val'].append(row)
    plan=dict(schema='slowfast_expanded_data_plan_v1',split_seed=seed,openscene_revision=REVISION,
        selected_new_logs_archive_order=chosen,splits=splits,logs=logs,rows=rows,
        sensor_root=str(sensor_root),required_images=sorted({p for values in rows.values() for row in values for p in row['images']}),
        legacy_copies=sorted(copies.values(),key=lambda item:item['target']),
        metadata_records=[records[name] for name in list(splits['train'])+list(splits['new_val'])+[LEGACY_VAL_LOG]],
        selection_before_image_availability=True,cohort_counts={'train':256,'new_val':64,'legacy_val':4})
    plan['plan_sha256']=_plan_digest(plan)
    return plan


def _retrying_opener(opener,max_attempts,backoff_seconds):
    stats={'attempts':0,'pre_response_failures':0}
    def open_request(request,timeout):
        for attempt in range(max_attempts):
            stats['attempts']+=1
            try:return opener(request,timeout=timeout)
            except urllib.error.HTTPError as error:
                stats['pre_response_failures']+=1
                retry=error.code in (429,500,502,503,504)
                error.close()
                if not retry or attempt+1==max_attempts:raise
            except (ssl.SSLError,http.client.RemoteDisconnected,ConnectionError,TimeoutError,urllib.error.URLError):
                stats['pre_response_failures']+=1
                if attempt+1==max_attempts:raise
            if backoff_seconds:time.sleep(backoff_seconds*(2**attempt))
    return open_request,stats


def acquire_plan(plan,camera_tree,*,sensor_dest,budget,revision=REVISION,archive_limit=225_000_000,
                 opener=urllib.request.urlopen,on_report=None,workers=3,force_reverify_logs=None,
                 max_request_attempts=3,retry_backoff_seconds=.5,chunk_size=8_000_000):
    """Fetch frozen paths in parallel under separately reserved archive caps.

    Workers never mutate the parent budget or call on_report. The calling thread
    collects every child charge before returning or raising, including partial
    response reads and other workers still running after one archive fails.
    """
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=3:
        raise ValueError('Acquisition workers must be an integer from1 to3')
    if (isinstance(max_request_attempts,bool) or not isinstance(max_request_attempts,int)
            or not 1<=max_request_attempts<=3 or not 0<chunk_size<=8_000_000
            or not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds<0):
        raise ValueError('Unsupported bounded request retry/chunk policy')
    if revision!=REVISION or not 0<archive_limit<=225_000_000:raise ValueError('Unsupported archive revision/prefix cap')
    if 'plan_sha256' in plan and _plan_digest(plan)!=plan['plan_sha256']:raise ValueError('Frozen plan identity differs')
    sensor_root=Path(plan['sensor_root']).resolve();sensor_dest=Path(sensor_dest).resolve()
    if sensor_root!=sensor_dest/'openscene-v1.1/sensor_blobs/mini':raise ValueError('Sensor destination differs from frozen plan')
    for item in plan['legacy_copies']:
        if sha256_file(item['source'])!=item['sha256']:raise ValueError('Legacy image source SHA256 changed')
        target=Path(item['target']);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256_file(target)!=item['sha256']:raise ValueError('Legacy target SHA256 mismatch')
        if not target.exists():shutil.copy2(item['source'],target)
    jobs=[];force=set(force_reverify_logs or ())
    selected_logs=plan['splits']['train']+plan['splits']['new_val']
    if force-set(selected_logs):raise ValueError('Reverification logs must belong to the frozen cohort')
    for name in selected_logs:
        shard=plan['logs'][name]['shard']
        archive_path=f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz'
        identity=next(item for item in camera_tree if item['path']==archive_path)
        if archive_limit>=identity['size']:raise ValueError('Refuse a prefix cap that could request the complete archive')
        needed={str(Path(p).resolve()) for p in plan['required_images']
                if Path(p).relative_to(sensor_root).parts[0]==name and (name in force or not Path(p).is_file())}
        if not needed:continue
        members={'openscene-v1.1/sensor_blobs/mini/'+str(Path(p).relative_to(sensor_root)) for p in needed}
        url=f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{revision}/{archive_path}'
        jobs.append(dict(log=name,shard=shard,url=url,archive_identity=identity,
                         needed=needed,members=members,budget=ByteBudget(archive_limit)))
    reservation=len(jobs)*archive_limit
    if budget.remaining<reservation:
        raise ValueError(f'Insufficient aggregate byte budget to reserve {reservation} bytes for {len(jobs)} camera prefixes')

    def fetch(job):
        child=job['budget']
        request_opener,retry_stats=_retrying_opener(opener,max_request_attempts,retry_backoff_seconds)
        transport=HTTPRangeReader(job['url'],child,archive_limit,chunk_size=chunk_size,opener=request_opener)
        failure=None;verified={}
        def publish(retained):
            for item in retained:
                member=item['member']
                if member in verified:continue
                staged=Path(item['path']);target=sensor_dest/member
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if not target.is_file() or target.stat().st_size!=item['bytes'] or sha256_file(target)!=item['sha256']:
                        raise ValueError(f'SHA256 mismatch for existing planned JPEG: {member}')
                    verification='existing_match'
                else:
                    try:os.link(staged,target)
                    except FileExistsError:
                        if not target.is_file() or sha256_file(target)!=item['sha256']:
                            raise ValueError(f'SHA256 mismatch for concurrently installed JPEG: {member}')
                    verification='new_install'
                staged.unlink()
                verified[member]=dict(item,path=str(target),verification=verification)
            return False
        try:
            with tempfile.TemporaryDirectory(prefix='.camera-verify-',dir=sensor_dest.parent) as temporary:
                result=extract_tar_prefix(transport,temporary,ByteBudget(archive_limit+1_000_000),
                    lambda member:member in job['members'],len(job['members']),archive_limit+1_000_000,stop=publish)
                # extract_tar_prefix short-circuits its callback on the last
                # selected member, so publish that final member explicitly.
                publish(result['files'])
            if any(not Path(p).is_file() for p in job['needed']):
                raise ValueError(f'Missing fixed planned images after bounded prefix: {job["log"]}')
        except BaseException as error:
            failure=f'{type(error).__name__}: {error}'
        # A response may fail after consuming bytes but before HTTPRangeReader
        # commits its buffer/digest. Child.used is authoritative in that case.
        report=dict(log=job['log'],shard=job['shard'],url=job['url'],archive_identity=job['archive_identity'],
            attempt_id=uuid.uuid4().hex,response_body_bytes=child.used,requests=retry_stats['attempts'],
            request_attempts=retry_stats['attempts'],successful_responses=transport.requests,
            pre_response_failures=retry_stats['pre_response_failures'],chunk_size_bytes=chunk_size,
            reserved_cap_bytes=archive_limit,
            response_prefix_sha256=transport.sha.hexdigest() if child.used==transport.response_body_bytes else None,
            full_archive_verified=False,files=list(verified.values()),
            complete=failure is None,error=failure)
        return report

    reports=[];failures=[];accounted=set();interrupted=None
    executor=ThreadPoolExecutor(max_workers=workers)
    futures={executor.submit(fetch,job):job for job in jobs}
    def collect(future):
        report=future.result()
        budget.used+=futures[future]['budget'].used
        accounted.add(future);reports.append(report)
        if report['error']:failures.append(f'{report["log"]}: {report["error"]}')
        if on_report:
            try:on_report(report)
            except BaseException as error:failures.append(f'Acquisition report callback failed: {type(error).__name__}: {error}')
    try:
        for future in as_completed(futures):collect(future)
    except BaseException as error:
        interrupted=error
    finally:
        executor.shutdown(wait=True)
        for future in futures:
            if future not in accounted:collect(future)
    if interrupted is not None:raise interrupted
    if failures:raise RuntimeError('Bounded camera acquisition failed after accounting every worker: '+'; '.join(failures))
    if any(not Path(p).is_file() for p in plan['required_images']):raise ValueError('Missing fixed planned images; no substitution')
    return reports


def load_completed_reports(output,plan,camera_tree):
    """Restore completed local archive provenance after verifying its identities.

    The wire-prefix digest is checked for valid form; compressed camera prefixes
    are not stored for an independent rehash. Retained JPEG hashes are rehashed.
    Interrupted reports are preserved by the caller and never count as complete.
    """
    reports=[];sensor_root=Path(plan['sensor_root']).resolve()
    for name in plan['splits']['train']+plan['splits']['new_val']:
        shard=plan['logs'][name]['shard'];path=Path(output)/f'camera_{shard}_acquisition.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        if report.get('complete') is not True:continue
        archive_path=f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz'
        identity=next(item for item in camera_tree if item['path']==archive_path)
        url=f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/{archive_path}'
        digest=report.get('response_prefix_sha256');requests=report.get('requests');size=report.get('response_body_bytes')
        if (report.get('log')!=name or report.get('shard')!=shard or report.get('archive_identity')!=identity
                or report.get('url')!=url or report.get('full_archive_verified') is not False or report.get('error') is not None
                or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)
                or isinstance(requests,bool) or not isinstance(requests,int) or requests<=0
                or isinstance(size,bool) or not isinstance(size,int) or not 0<size<=225_000_000
                or not isinstance(report.get('files'),list) or not report['files']):
            raise ValueError(f'Invalid completed archive identity/accounting: {path.name}')
        required={p for p in plan['required_images'] if Path(p).relative_to(sensor_root).parts[0]==name}
        # Old interrupted/resumed acquisitions may claim completion after
        # downloading only previously missing members. They cannot establish
        # full-log provenance and must be rescanned, even if files now exist.
        if len(report['files'])!=len(required) or {item['path'] for item in report['files']}!=required:
            continue
        for item in report['files']:
            relative=Path(item['path']).relative_to(sensor_root)
            if (item['path'] not in required
                    or item['member']!='openscene-v1.1/sensor_blobs/mini/'+str(relative)
                    or not Path(item['path']).is_file() or Path(item['path']).stat().st_size!=item['bytes']
                    or sha256_file(item['path'])!=item['sha256']):
                raise ValueError(f'Completed archive retained-file identity differs: {path.name}')
        reports.append(report)
    return reports


def verify_provenance_coverage(plan,reports):
    """Require an exact verified camera/legacy union before dataset finalization."""
    sensor_root=Path(plan['sensor_root']).resolve();covered=set();seen_logs=set()
    expected_logs=set(plan['splits']['train']+plan['splits']['new_val'])
    for report in reports:
        name=report['log']
        required={p for p in plan['required_images'] if Path(p).relative_to(sensor_root).parts[0]==name}
        paths=[item['path'] for item in report['files']]
        if (name not in expected_logs or name in seen_logs or not report.get('complete')
                or report.get('error') is not None or len(paths)!=len(required) or set(paths)!=required):
            raise ValueError('Completed camera provenance coverage differs from frozen plan')
        for item in report['files']:
            path=Path(item['path'])
            if not path.is_file() or path.stat().st_size!=item['bytes'] or sha256_file(path)!=item['sha256']:
                raise ValueError('Completed camera provenance retained-file SHA256 mismatch')
        covered.update(paths);seen_logs.add(name)
    if seen_logs!=expected_logs:raise ValueError('Camera archive identity coverage differs from frozen plan')
    for item in plan['legacy_copies']:
        if item['target'] in covered or sha256_file(item['target'])!=item['sha256']:
            raise ValueError('Legacy image provenance coverage or SHA256 differs')
        covered.add(item['target'])
    if covered!=set(plan['required_images']):raise ValueError('Camera and legacy provenance coverage differs from frozen plan')
    return dict(verified_images=len(covered),camera_archives=len(seen_logs),legacy_images=len(plan['legacy_copies']))


def write_prepared(plan,output,*,expected_size=(1920,1080),acquisition_reports=(),network_bytes=0,upstream_audit=None):
    """Verify fixed images, write TRAIN256-only statistics and disjoint VAL cohorts."""
    from PIL import Image
    if _plan_digest(plan)!=plan['plan_sha256']:raise ValueError('Frozen plan identity differs')
    coverage=verify_provenance_coverage(plan,acquisition_reports) if acquisition_reports else None
    sources=[]
    for path in plan['required_images']:
        if not Path(path).is_file():raise ValueError(f'Missing fixed planned image: {path}')
        with Image.open(path) as image:
            if image.size!=expected_size or image.format!='JPEG':raise ValueError('Requires original-resolution CAM_F0 JPEGs')
        sources.append(dict(path=path,sha256=sha256_file(path),kind='retained_original_jpeg',bytes=Path(path).stat().st_size))
    for item in plan['metadata_records']:
        if sha256_file(item['path'])!=item['sha256']:raise ValueError('Metadata SHA256 changed since planning')
        sources.append(dict(item,kind='retained_metadata'))
    report=write_manifests({'train':plan['rows']['train'],'val':plan['rows']['val'],'test':[]},output,
        provenance=dict(openscene_revision=REVISION,plan_sha256=plan['plan_sha256'],expanded_mini=True))
    prefix_bytes=sum(report['response_body_bytes'] for report in acquisition_reports)
    if prefix_bytes>network_bytes:raise ValueError('Archive report bytes exceed aggregate network accounting')
    audit=dict(schema='slowfast_expanded_source_audit_v1',openscene_revision=REVISION,navsim_revision=NAVSIM_REVISION,
        sources=sources,plan_sha256=plan['plan_sha256'],network_response_body_bytes=network_bytes,
        completed_archive_response_body_bytes=prefix_bytes,other_charged_response_body_bytes=network_bytes-prefix_bytes,
        provenance_coverage=coverage,
        acquisition_reports=list(acquisition_reports),full_archives_verified=False,upstream_audit=upstream_audit)
    write_json(Path(output)/'source_audit.json',audit)
    report=dict(report,counts=plan['cohort_counts'],plan_sha256=plan['plan_sha256'],
        original_image_size_wh=list(expected_size),test_used=False,cohorts_pooled_for_metrics=False)
    write_json(Path(output)/'build_report.json',report)
    return report
