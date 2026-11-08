"""Expanded TRAIN512 plus a frozen, metadata-selected fresh VAL256.

Only roles change for the entire previous diagnostic cohort. The resumed model's
TRAIN256 normalizer is a checkpoint interface; TRAIN512 statistics are an audit.
The acquisition compatibility split names contain only fresh camera work.
"""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import uuid

from fast_navsim.acquire import ByteBudget, BudgetReader, download_prefix_ranges, extract_tar_prefix, write_json
from fast_navsim.data import compute_ego_stats, validate_row
from slowfast_v11.adapter_data import (
    AuditedOpener, copy_authenticated_prefix, make_adapter_plan, recording_group,
)
from slowfast_v11.expanded_data import (
    REVISION, NAVSIM_REVISION, _plan_digest, sha256_file, acquire_plan,
    load_completed_reports, verify_provenance_coverage,
)

BASE_PREFIX_BYTES = 320_000_000
BASE_PREFIX_SHA256 = '3f6d96cdcaf718720f076ac5b51320ced3ab724399d9f15cc1eac2dca8445eb8'
METADATA_PREFIX_CAP = 520_000_000
METADATA_NEW_CAP = 200_000_000
WHOLE_METADATA_BYTES = 534_344_190
CAMERA_PREFIX_CAP = 225_000_000
CAMERA_RESERVATION = 8 * CAMERA_PREFIX_CAP
NEW_NETWORK_CAP = 2_000_000_000
INHERITED_CUMULATIVE_BYTES = 3_403_085_681
NORMALIZER_SHA256 = '35dc682328e66c31f35c671ad2a6654031b31d7d6c22927377301cceaf687bb6'
SOURCE_CHECKPOINT_SHA256 = '448571f4d543913a4d93471781a1ecf4aa392078f2aa2dafa3816be43f9f0ae9'
FAST_CHECKPOINT_SHA256 = '02c3964a225efd79e1c06644cbf49228a38a612021f211258e8476e6606882b1'
METADATA_URL = f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/openscene-v1.1/openscene_metadata_mini.tgz'


def _tokens_digest(rows):
    tokens = {split: [row['token'] for row in values] for split, values in rows.items()}
    return hashlib.sha256(json.dumps(tokens,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def make_direct_plan(retained, camera_index, original_train, promoted_holdout, seen_logs,
                     *, sensor_root, inherited_sources=(), upstream_sources=()):
    """Promote all256 old holdout rows and select8x32 unused metadata windows.

    This function deliberately never inspects sensor availability. Source hashes
    already authenticated by the caller are included in the frozen identity.
    """
    if len(original_train) != 256 or len(promoted_holdout) != 256:
        raise ValueError('Need the complete original TRAIN256 and promoted holdout256')
    if any(row['split'] != 'train' for row in original_train) or any(row['split'] != 'val' or row.get('cohort') != 'holdout' for row in promoted_holdout):
        raise ValueError('Unexpected original training or diagnostic holdout roles')
    for row in list(original_train)+list(promoted_holdout):
        validate_row(row)
    old_groups = Counter(recording_group(row['route']) for row in original_train)
    promoted_groups = Counter(recording_group(row['route']) for row in promoted_holdout)
    if len(old_groups) != 4 or set(old_groups.values()) != {64} or len(promoted_groups) != 8 or set(promoted_groups.values()) != {32}:
        raise ValueError('Need original four x64 and promoted eight x32 recording groups')
    if set(old_groups) & set(promoted_groups):
        raise ValueError('Promoted recording group overlaps original TRAIN')
    old_tokens = [row['token'] for row in list(original_train)+list(promoted_holdout)]
    if len(set(old_tokens)) != 512:
        raise ValueError('Original and promoted tokens must be unique and disjoint')
    seen_groups = {recording_group(name) for name in seen_logs}
    if len(seen_groups) != 15 or not set(old_groups)|set(promoted_groups) <= seen_groups:
        raise ValueError('Must exclude all fifteen previously seen recording groups')
    # Existing helper enforces fixed stride, all12 finite poses, stable historic
    # command and timestamps, before constructing rows or inspecting images.
    plan = make_adapter_plan(retained,camera_index,seen_logs,sensor_root=sensor_root)
    train = copy.deepcopy(list(original_train))
    train.extend(dict(copy.deepcopy(row),split='train',cohort='train') for row in promoted_holdout)
    for row in plan['rows']['val']:
        row['cohort'] = 'new_val'
    for info in plan['logs'].values():
        info['cohort'] = 'new_val'
    plan['schema'] = 'slowfast_direct_full_data_plan_v1'
    plan['rows']['train'] = train
    plan['splits'] = dict(train=[],new_val=plan['splits']['new_val'])
    plan['training_recordings'] = sorted({row['route'] for row in train})
    plan['cohort_counts'] = dict(train=512,new_val=256)
    plan['inherited_images'] = sorted({path for row in train for path in row['images']})
    plan['inherited_sources'] = copy.deepcopy(list(inherited_sources))
    plan['upstream_sources'] = copy.deepcopy(list(upstream_sources))
    plan['role_promotion'] = dict(count=256,recording_groups=8,
        recordings=sorted({row['route'] for row in promoted_holdout}),
        tokens=[row['token'] for row in promoted_holdout],
        previous_split='val',previous_cohort='holdout',new_split='train',new_cohort='train',
        changed_fields=['split','cohort'],selection_based_on_model_outputs=False,
        entire_previous_cohort=True,previous_scores='historical only; no longer evaluation')
    plan['fixed_source_token'] = min(row['token'] for row in train)
    plan['normalizer_policy'] = 'preserve_source_checkpoint_train256_interface; expanded_train512_audit_only'
    plan['token_manifest_sha256'] = _tokens_digest(plan['rows'])
    all_tokens = old_tokens+[row['token'] for row in plan['rows']['val']]
    if len(set(all_tokens)) != 768:
        raise ValueError('Fresh validation token overlaps TRAIN')
    plan['plan_sha256'] = _plan_digest(plan)
    return plan


def freeze_direct_plan(plan, output):
    if _plan_digest(plan) != plan.get('plan_sha256') or _tokens_digest(plan['rows']) != plan['token_manifest_sha256']:
        raise ValueError('Frozen plan identity differs')
    output = Path(output)
    tokens = dict(schema='slowfast_direct_full_tokens_v1',plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'],
        tokens={split:[row['token'] for row in values] for split,values in plan['rows'].items()},
        fixed_source_token=plan['fixed_source_token'],frozen_before_camera_access=True)
    for filename, content in [('plan.json',plan),('frozen_tokens.json',tokens)]:
        path = output/filename
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError(f'Frozen {filename} identity differs; no cohort replacement')
    for filename, content in [('plan.json',plan),('frozen_tokens.json',tokens)]:
        if not (output/filename).exists():
            write_json(output/filename,content)


def extend_metadata_prefix(prefix, budget, *, metadata_bytes_used=0,
                           opener=urllib.request.urlopen, on_request=None):
    """Extend only the private320MB prefix to exactly520MB under a200MB cap."""
    prefix = Path(prefix)
    size = prefix.stat().st_size
    if not BASE_PREFIX_BYTES <= size <= METADATA_PREFIX_CAP < WHOLE_METADATA_BYTES:
        raise ValueError('Metadata prefix is outside its fixed inherited/maximum bounds')
    needed = METADATA_PREFIX_CAP-size
    allowance = METADATA_NEW_CAP-metadata_bytes_used
    if needed > allowance:
        raise ValueError('Failed attempts consumed the fixed metadata allowance; no cap increase')
    if budget.remaining < CAMERA_RESERVATION+needed:
        raise ValueError('Insufficient budget to reserve metadata and eight fixed camera prefixes')
    child = ByteBudget(min(allowance,budget.remaining-CAMERA_RESERVATION))
    audited = AuditedOpener(opener,'metadata',on_request)
    try:
        result = download_prefix_ranges(METADATA_URL,prefix,child,METADATA_PREFIX_CAP,
            chunk_size=8_000_000,opener=audited,resume=True)
        result.update(inherited_prefix_bytes=BASE_PREFIX_BYTES,inherited_prefix_sha256=BASE_PREFIX_SHA256,
            new_body_allowance_bytes=METADATA_NEW_CAP,requests_detail=audited.records,complete=True)
        return result
    finally:
        budget.used += child.used


def verify_durable_prefix(prefix, records):
    """Authenticate saved contiguous ranges; local reads never count as traffic."""
    prefix = Path(prefix)
    if not prefix.exists():
        if any(r['response_body_bytes'] for r in records):
            raise ValueError('Durable prefix is absent despite charged response bodies')
        return
    offset = 0
    with prefix.open('rb') as stream:
        for record in records:
            size = record['response_body_bytes']
            if not size:
                continue
            match = re.fullmatch(r'bytes=(\d+)-(\d+)',record.get('range') or '')
            if not match or int(match[1]) != offset or size > int(match[2])-offset+1:
                raise ValueError('Durable prefix ranges are not contiguous saved offsets')
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                data = stream.read(min(65536,remaining))
                if not data:
                    raise ValueError('Durable prefix SHA256/length differs from charged body')
                digest.update(data);remaining -= len(data)
            if digest.hexdigest() != record['response_body_sha256']:
                raise ValueError('Durable prefix SHA256 differs from charged response body')
            offset += size
        if stream.read(1) or offset != prefix.stat().st_size:
            raise ValueError('Durable prefix contains bytes absent from request ledger')


def download_durable_prefix(url, prefix, budget, *, expected_archive_bytes,
                            prefix_cap=CAMERA_PREFIX_CAP, chunk_size=1_000_000,
                            opener=urllib.request.urlopen, archive_extent_cap=CAMERA_PREFIX_CAP):
    """Persist/fsync each consumed block before asking for another response.

    A short or interrupted body resumes from actual saved bytes. The caller
    supplies the audited opener; only reads through BudgetReader charge traffic.
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    if (archive_extent_cap not in (CAMERA_PREFIX_CAP,229_000_000) or prefix.is_symlink()
            or not 0 < prefix_cap <= archive_extent_cap or not 0 < chunk_size <= 1_000_000 or prefix_cap >= expected_archive_bytes):
        raise ValueError('Unsupported durable private prefix bounds')
    offset = prefix.stat().st_size if prefix.exists() else 0
    if offset > prefix_cap or budget.remaining < prefix_cap-offset:
        raise ValueError('Insufficient remaining byte allowance for exact durable prefix')
    requests = 0
    initial = offset
    with prefix.open('ab') as output:
        while offset < prefix_cap:
            end = min(offset+chunk_size,prefix_cap)-1
            request = urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}'})
            received = 0
            with opener(request,timeout=90) as response:
                requests += 1
                expected = f'bytes {offset}-{end}/{expected_archive_bytes}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('Durable prefix response Content-Range differs from exact pinned range/size')
                reader = BudgetReader(response,budget,end-offset+1)
                while reader.used < end-offset+1:
                    data = reader.read(min(65536,end-offset+1-reader.used))
                    if not data:
                        break
                    output.write(data);output.flush();os.fsync(output.fileno())
                    received += len(data)
            if not received:
                raise RuntimeError('Empty durable ranged response before fixed prefix end')
            offset += received
    return dict(url=url,path=str(prefix.resolve()),prefix_bytes=prefix.stat().st_size,
        prefix_sha256=sha256_file(prefix),new_response_body_bytes=prefix.stat().st_size-initial,
        requests=requests,complete=prefix.stat().st_size==prefix_cap,full_archive_verified=False)


def extract_durable_camera(plan, camera_tree, prefix, *, sensor_dest, shard, transfer,
                           archive_extent_cap=CAMERA_PREFIX_CAP):
    """Extract only frozen members from an authenticated local compressed prefix."""
    prefix = Path(prefix)
    if ('plan_sha256' in plan and _plan_digest(plan) != plan['plan_sha256']
            or transfer.get('complete') is not True or prefix.stat().st_size != transfer['prefix_bytes']
            or sha256_file(prefix) != transfer['prefix_sha256'] or prefix.stat().st_size > archive_extent_cap
            or archive_extent_cap not in (CAMERA_PREFIX_CAP,229_000_000)):
        raise ValueError('Durable prefix/plan SHA256 identity differs before extraction')
    sensor_dest = Path(sensor_dest).resolve();sensor_root = Path(plan['sensor_root']).resolve()
    if sensor_root != sensor_dest/'openscene-v1.1/sensor_blobs/mini':
        raise ValueError('Durable extraction destination differs from frozen plan')
    names = [name for name in plan['splits']['new_val'] if plan['logs'][name]['shard']==shard]
    if len(names) != 1:
        raise ValueError('Durable shard must identify one frozen recording')
    name = names[0]
    required = {p for p in plan['required_images'] if Path(p).relative_to(sensor_root).parts[0]==name}
    members = {'openscene-v1.1/sensor_blobs/mini/'+str(Path(p).relative_to(sensor_root)) for p in required}
    archive_path = f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz'
    identity = next(item for item in camera_tree if item['path']==archive_path)
    files = []
    sensor_dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.durable-camera-',dir=sensor_dest.parent) as temporary:
        with prefix.open('rb') as stream:
            extracted = extract_tar_prefix(stream,temporary,ByteBudget(archive_extent_cap+1_000_000),
                lambda member:member in members,len(members),archive_extent_cap+1_000_000)
        if {item['member'] for item in extracted['files']} != members:
            raise ValueError('Fixed durable prefix lacks planned images; no recording replacement')
        for item in extracted['files']:
            target = sensor_dest/item['member'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if not target.is_file() or target.stat().st_size != item['bytes'] or sha256_file(target) != item['sha256']:
                    raise ValueError('Existing planned JPEG SHA256 differs from durable archive')
                verification = 'existing_match'
            else:
                os.link(item['path'],target);verification = 'new_install'
            files.append(dict(item,path=str(target),verification=verification))
    requests = transfer.get('requests_detail',[])
    return dict(log=name,shard=shard,url=f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/{archive_path}',
        archive_identity=identity,attempt_id=uuid.uuid4().hex,
        response_body_bytes=transfer['prefix_bytes'],response_prefix_sha256=transfer['prefix_sha256'],
        requests=len(requests),request_attempts=len(requests),successful_responses=sum(r.get('status')==206 for r in requests),
        pre_response_failures=sum(r.get('status') is None for r in requests),reserved_cap_bytes=archive_extent_cap,
        requests_detail=requests,files=files,complete=True,error=None,full_archive_verified=False,
        durable_prefix=dict(path=str(prefix.resolve()),bytes=prefix.stat().st_size,sha256=sha256_file(prefix)),
        local_extraction_response_body_bytes=0)


def load_direct_completed_reports(output, plan, camera_tree, *, archive_extent_caps=None):
    """Use the legacy authenticator unless an explicit27 extent amendment applies."""
    caps = archive_extent_caps or {}
    if not caps:
        return load_completed_reports(output,plan,camera_tree)
    expected_url = f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_27.tgz'
    if caps != {expected_url:229_000_000} or _plan_digest(plan) != plan['plan_sha256']:
        raise ValueError('Unsupported direct completed-report extent amendment/plan')
    reports = []
    root = Path(plan['sensor_root'])
    for name in plan['splits']['train']+plan['splits']['new_val']:
        shard = plan['logs'][name]['shard'];path = Path(output)/f'camera_{shard}_acquisition.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report.get('complete') is not True:
            continue
        archive_path = f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz'
        identity = next(item for item in camera_tree if item['path']==archive_path)
        url = f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/{REVISION}/{archive_path}'
        cap = caps.get(url,CAMERA_PREFIX_CAP);digest = report.get('response_prefix_sha256','')
        if (report.get('log') != name or report.get('shard') != shard or report.get('url') != url
                or report.get('archive_identity') != identity or report.get('error') is not None
                or report.get('full_archive_verified') is not False or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or type(report.get('requests')) is not int or report['requests'] <= 0
                or type(report.get('response_body_bytes')) is not int or not 0 < report['response_body_bytes'] <= cap
                or report.get('reserved_cap_bytes',CAMERA_PREFIX_CAP) != cap or cap >= identity['size']):
            raise ValueError('Completed direct archive identity/accounting differs from approved extent')
        required = {p for p in plan['required_images'] if Path(p).relative_to(root).parts[0]==name}
        files = report.get('files',[])
        if len(files) != len(required) or {item['path'] for item in files} != required:
            raise ValueError('Completed direct archive lacks exact frozen image coverage')
        for item in files:
            path = Path(item['path'])
            if (item['member'] != 'openscene-v1.1/sensor_blobs/mini/'+str(path.relative_to(root))
                    or not path.is_file() or path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']):
                raise ValueError('Completed direct retained-image SHA256 differs')
        reports.append(report)
    return reports


def _authenticated_inherited_sources(plan):
    records = {}
    for item in plan['inherited_sources']:
        path = str(Path(item['path']).resolve())
        if path in records and records[path]['sha256'] != item['sha256']:
            raise ValueError('Conflicting inherited source SHA256')
        records[path] = item
    required = {str(Path(path).resolve()):'inherited_original_jpeg' for path in plan['inherited_images']}
    for row in plan['rows']['train']:
        path = str(Path(row['source']['metadata_path']).resolve())
        if path not in records or records[path]['sha256'] != row['source']['metadata_sha256']:
            raise ValueError('Inherited metadata SHA256 lacks authenticated source provenance')
        required[path] = 'inherited_metadata'
    sources = []
    for path, kind in sorted(required.items()):
        if path not in records or not records[path].get('upstream_audit'):
            raise ValueError('Missing authenticated inherited source provenance')
        item = records[path]
        if not Path(path).is_file() or Path(path).stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError('Inherited retained source SHA256/size changed')
        sources.append(dict(item,path=path,kind=kind))
    return sources


def write_direct_prepared(plan, output, *, normalizer_source,
                          normalizer_expected_sha256=NORMALIZER_SHA256,
                          expected_size=(1920,1080), acquisition_reports=(),
                          network_budget=None, metadata_provenance=None, transport_amendment=None):
    """Authenticate every image/metadata source; write rows and both stats IDs."""
    from PIL import Image
    if _plan_digest(plan) != plan.get('plan_sha256'):
        raise ValueError('Frozen plan identity differs')
    sources = _authenticated_inherited_sources(plan)
    coverage = verify_provenance_coverage(plan,acquisition_reports) if acquisition_reports else None
    for filename in plan['required_images']+plan['inherited_images']:
        path = Path(filename)
        if not path.is_file():
            raise ValueError(f'Missing fixed planned image: {path}')
        with Image.open(path) as image:
            if image.size != expected_size or image.format != 'JPEG':
                raise ValueError('Requires original-resolution CAM_F0 JPEGs')
        if filename in plan['required_images']:
            sources.append(dict(path=str(path),sha256=sha256_file(path),bytes=path.stat().st_size,
                                kind='retained_original_jpeg'))
    for item in plan['metadata_records']:
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Metadata SHA256 changed since planning')
        sources.append(dict(item,kind='retained_metadata'))
    for values in plan['rows'].values():
        for row in values:
            validate_row(row)
    source_path = Path(normalizer_source).resolve()
    if sha256_file(source_path) != normalizer_expected_sha256:
        raise ValueError('Checkpoint normalizer SHA256 identity differs')
    original_stats = json.loads(source_path.read_text())
    if original_stats != compute_ego_stats(plan['rows']['train'][:256]):
        raise ValueError('Checkpoint normalizer differs from original TRAIN256 statistics')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    selected_path = output/'stats.json'
    if selected_path.resolve() == source_path:
        raise ValueError('New normalizer path must not overwrite original statistics')
    if selected_path.exists() and selected_path.read_bytes() != source_path.read_bytes():
        raise ValueError('Existing selected normalizer identity differs; no overwrite')
    freeze_direct_plan(plan,output)
    if not selected_path.exists():
        shutil.copy2(source_path,selected_path)
    expanded_path = output/'train512_stats_audit.json'
    write_json(expanded_path,compute_ego_stats(plan['rows']['train']))
    normalization = dict(policy=plan['normalizer_policy'],shared_consumers=['fast','slow','feedback_teacher'],
        selected=dict(path=str(selected_path.resolve()),source_path=str(source_path),
            sha256=sha256_file(selected_path),count=256,selected_for_model_inputs=True,
            source_checkpoint_sha256=SOURCE_CHECKPOINT_SHA256,fast_checkpoint_sha256=FAST_CHECKPOINT_SHA256),
        expanded_train512=dict(path=str(expanded_path.resolve()),sha256=sha256_file(expanded_path),
            count=512,source_split='train',selected_for_model_inputs=False,purpose='distribution audit only'))
    manifests = {}
    for split, values in plan['rows'].items():
        path = output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in values))
        manifests[split] = dict(path=str(path.resolve()),sha256=sha256_file(path),count=len(values))
    retained_path = output/'metadata_retained.json'
    retained_manifest = (dict(path=str(retained_path.resolve()),sha256=sha256_file(retained_path))
        if retained_path.exists() else None)
    audit = dict(schema='slowfast_direct_full_source_audit_v1',openscene_revision=REVISION,
        navsim_revision=NAVSIM_REVISION,plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'],sources=sources,
        upstream_sources=plan['upstream_sources'],role_promotion=plan['role_promotion'],
        normalization=normalization,network_budget=network_budget,metadata_provenance=metadata_provenance,
        retained_metadata_manifest=retained_manifest,
        transport_amendment=transport_amendment,
        acquisition_reports=list(acquisition_reports),provenance_coverage=coverage,
        full_archives_verified=False,complete=True,
        protocol='bounded NAVSIM-mini full-language pilot; not official navtrain/navtest')
    write_json(output/'source_audit.json',audit)
    result = dict(counts=plan['cohort_counts'],plan_sha256=plan['plan_sha256'],
        token_manifest_sha256=plan['token_manifest_sha256'],fixed_source_token=plan['fixed_source_token'],
        manifests=manifests,normalization=normalization,role_promotion=plan['role_promotion'],
        original_image_size_wh=list(expected_size),train_recording_groups=12,val_recording_groups=8,
        selection_before_image_availability=True,expanded_stats_audit_only=True)
    write_json(output/'build_report.json',result)
    return result
