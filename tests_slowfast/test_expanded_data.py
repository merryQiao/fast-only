"""Freeze expanded cohort identities and paths before bounded JPEG acquisition."""
import copy
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

from PIL import Image
import pytest

from fast_navsim.acquire import ByteBudget
from fast_navsim.data import build_window
from slowfast_v11.expanded_data import (
    OLD_TRAIN_LOGS, LEGACY_VAL_LOG, structural_candidates, select_train_indices,
    uniform_indices, make_expanded_plan, acquire_plan, write_prepared,
)


def frames(log,count=500):
    return [dict(log_name=log,token=f'{log}-{i}',timestamp=1_000_000+i*500_000,
        driving_command=[0,1,0,0],ego_dynamic_state=[[0.,5.,12.][i%3],0.,0.,0.],
        ego2global_translation=[float(i),0.,0.],ego2global_rotation=[1.,0.,0.,0.],
        cams={'CAM_F0':{'data_path':f'{log}/CAM_F0/{i}.jpg'}}) for i in range(count)]


def fixture(tmp_path):
    old=tmp_path/'legacy';old.mkdir()
    legacy_data=frames(LEGACY_VAL_LOG)
    legacy=[build_window(legacy_data,i,'val',old) for i in (8,13,18,23)]
    for row in legacy:
        row['source']={'metadata_sha256':'will-set','metadata_path':'will-set'}
        for path in row['images']:
            p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(12,8)).save(p)
    names=[OLD_TRAIN_LOGS[0],'alpha','beta',LEGACY_VAL_LOG,'gamma',OLD_TRAIN_LOGS[1],'delta']
    retained=[];index={};all_frames={}
    for k,name in enumerate(names):
        data=frames(name);all_frames[name]=data
        blob=pickle.dumps(data);path=tmp_path/f'{name}.pkl';path.write_bytes(blob)
        retained.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
        index[f'openscene_sensor_mini_{k}']=[name,'second-log']
        if name==LEGACY_VAL_LOG:
            for row in legacy:row['source']=dict(metadata_path=str(path),metadata_sha256=retained[-1]['sha256'])
    return retained,index,legacy,all_frames,old


def plan_for(tmp_path):
    retained,index,legacy,data,old=fixture(tmp_path)
    plan=make_expanded_plan(retained,index,legacy,sensor_root=tmp_path/'new',legacy_sensor_root=old)
    return plan,retained,index,legacy,data,old


def test_candidates_use_fixed_stride_and_validate_history_not_future_annotations():
    data=frames('a',25)
    candidates,excluded=structural_candidates(data)
    assert candidates==[3,8,13] and excluded=={}
    for frame in data:frame['annotations']=object()
    assert structural_candidates(data)==(candidates,excluded)
    data[2]['driving_command']=[1,0,0,0]
    candidates,excluded=structural_candidates(data)
    assert candidates==[8,13] and excluded=={'navigation_mismatch':1}
    data[7]['ego2global_rotation']=[0.,0.,0.,0.]
    assert structural_candidates(data)[1]['invalid_pose']==1


def test_training_quota_shortage_is_explicit_and_fills_mid_before_low():
    data=frames('a',500);candidates=list(range(64))
    for i in candidates:data[i]['ego_dynamic_state']=[0. if i<20 else 5. if i<62 else 12.,0.,0.,0.]
    indices,report=select_train_indices(data,candidates)
    assert len(indices)==64 and len(set(indices))==64
    assert report['desired']=={'low_speed':16,'mid_speed':24,'high_speed':24}
    assert report['selected']=={'low_speed':20,'mid_speed':42,'high_speed':2}
    assert report['redistributed']=={'low_speed':4,'mid_speed':18,'high_speed':-22}
    assert indices==select_train_indices(data,candidates)[0]
    with pytest.raises(ValueError):select_train_indices(data,candidates[:63])


def test_uniform_validation_indices_cover_time_without_speed_selection():
    assert uniform_indices(list(range(100)),4)==[0,33,66,99]
    with pytest.raises(ValueError):uniform_indices([1,2],3)


def test_plan_freezes_split_tokens_and_exact_paths_without_existing_new_images(tmp_path):
    plan,retained,index,legacy,data,old=plan_for(tmp_path)
    assert len(plan['rows']['train'])==256 and len(plan['rows']['val'])==68
    assert set(OLD_TRAIN_LOGS)<=set(plan['splits']['train'])
    assert set(plan['splits']['train']).isdisjoint(plan['splits']['new_val'])
    assert LEGACY_VAL_LOG not in plan['splits']['train']+plan['splits']['new_val']
    assert {r['cohort'] for r in plan['rows']['val']}=={'new_val','legacy_val'}
    assert len([r for r in plan['rows']['val'] if r['cohort']=='legacy_val'])==4
    assert all(Path(p).is_relative_to(tmp_path/'new') for split in plan['rows'].values() for row in split for p in row['images'])
    assert not (tmp_path/'new').exists()
    expected={p for split in plan['rows'].values() for r in split for p in r['images']}
    assert set(plan['required_images'])==expected
    assert plan['plan_sha256']
    assert plan==json.loads(json.dumps(plan))
    assert plan==make_expanded_plan(retained,index,legacy,sensor_root=tmp_path/'new',legacy_sensor_root=old)


def test_plan_val_selection_ignores_speed_and_old_val_is_never_loaded(tmp_path):
    plan,retained,index,legacy,data,old=plan_for(tmp_path)
    for rec in retained:
        name=Path(rec['path']).stem
        if name in plan['splits']['new_val']:
            for frame in data[name]:frame['ego_dynamic_state']=[0.,0.,0.,0.]
            blob=pickle.dumps(data[name]);Path(rec['path']).write_bytes(blob)
            rec['sha256']=hashlib.sha256(blob).hexdigest()
        if name==LEGACY_VAL_LOG:
            Path(rec['path']).write_bytes(b'not-a-pickle')
            rec['sha256']=hashlib.sha256(b'not-a-pickle').hexdigest()
            for row in legacy:row['source']['metadata_sha256']=rec['sha256']
    newer=make_expanded_plan(retained,index,legacy,sensor_root=tmp_path/'new',legacy_sensor_root=old)
    for cohort in ('new_val','legacy_val'):
        assert [r['token'] for r in plan['rows']['val'] if r['cohort']==cohort]==[r['token'] for r in newer['rows']['val'] if r['cohort']==cohort]


def test_plan_rejects_bad_hash_and_insufficient_new_logs(tmp_path):
    retained,index,legacy,data,old=fixture(tmp_path)
    retained[0]['sha256']='wrong'
    with pytest.raises(ValueError,match='SHA256'):
        make_expanded_plan(retained,index,legacy,sensor_root=tmp_path/'new',legacy_sensor_root=old)
    retained[0]['sha256']=hashlib.sha256(Path(retained[0]['path']).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='four'):
        make_expanded_plan(retained[:4],index,legacy,sensor_root=tmp_path/'new',legacy_sensor_root=old)


def archive_blob(log):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz') as archive:
        for name in ('unused.jpg','1.jpg','2.jpg'):
            image=io.BytesIO();Image.new('RGB',(12,8)).save(image,format='JPEG');payload=image.getvalue()
            member=tarfile.TarInfo(f'openscene-v1.1/sensor_blobs/mini/{log}/CAM_F0/{name}')
            member.size=len(payload);archive.addfile(member,io.BytesIO(payload))
    # Real shard bodies exceed the streaming tar reader's first10KB read.
    return raw.getvalue()+b'\0'*12_000


class Response(io.BytesIO):
    def __init__(self,blob,start,end,total):
        super().__init__(blob[start:end+1]);self.status=206
        self.headers={'Content-Range':f'bytes {start}-{min(end,len(blob)-1)}/{total}'}
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def test_bounded_acquisition_keeps_only_frozen_paths_and_counts_every_byte(tmp_path):
    log='a';blob=archive_blob(log);requests=[]
    sensor=tmp_path/'sensors/openscene-v1.1/sensor_blobs/mini'
    required=[str(sensor/log/'CAM_F0'/name) for name in ('1.jpg','2.jpg')]
    plan={'splits':{'train':[log],'new_val':[]},'logs':{log:{'shard':5}},
          'required_images':required,'legacy_copies':[],'sensor_root':str(sensor)}
    tree=[{'path':'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_5.tgz','size':10_000_000}]
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'));requests.append((start,end))
        return Response(blob,start,end,10_000_000)
    budget=ByteBudget(100_000,100)
    reports=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,archive_limit=50_000,opener=opener)
    assert budget.used==100+len(blob)
    assert reports[0]['response_body_bytes']==len(blob)
    assert set(p.name for p in (sensor/log/'CAM_F0').iterdir())=={'1.jpg','2.jpg'}
    assert all(end<50_000 for start,end in requests)
    assert reports[0]['full_archive_verified'] is False


def test_missing_frozen_images_fail_without_substitution_or_complete_archive(tmp_path):
    sensor=tmp_path/'sensors/openscene-v1.1/sensor_blobs/mini'
    plan={'splits':{'train':['a'],'new_val':[]},'logs':{'a':{'shard':5}},
        'required_images':[str(sensor/'a/CAM_F0/missing.jpg')],'legacy_copies':[],'sensor_root':str(sensor)}
    tree=[{'path':'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_5.tgz','size':100}]
    with pytest.raises(ValueError,match='complete'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(1000),archive_limit=100)


def test_preparation_refuses_missing_planned_images_then_writes_train_only_stats(tmp_path):
    plan,*_=plan_for(tmp_path)
    with pytest.raises(ValueError,match='Missing'):
        write_prepared(plan,tmp_path/'out',expected_size=(12,8))
    for path in plan['required_images']:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(12,8)).save(p)
    report=write_prepared(plan,tmp_path/'out',expected_size=(12,8))
    stats=json.loads((tmp_path/'out/stats.json').read_text())
    assert stats['count']==256 and stats['source_split']=='train'
    assert report['counts']=={'train':256,'new_val':64,'legacy_val':4}
    assert len((tmp_path/'out/train.jsonl').read_text().splitlines())==256
    assert len((tmp_path/'out/val.jsonl').read_text().splitlines())==68
    assert (tmp_path/'out/source_audit.json').exists()


def parallel_fixture(tmp_path,count=3):
    sensor=tmp_path/'sensors/openscene-v1.1/sensor_blobs/mini'
    logs=[f'parallel-{i}' for i in range(count)]
    plan={'splits':{'train':logs,'new_val':[]},'logs':{name:{'shard':i} for i,name in enumerate(logs)},
        'required_images':[str(sensor/name/'CAM_F0/1.jpg') for name in logs],
        'legacy_copies':[],'sensor_root':str(sensor)}
    tree=[{'path':f'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{i}.tgz',
           'size':10_000_000} for i in range(count)]
    return plan,tree,{i:archive_blob(name) for i,name in enumerate(logs)}


def test_parallel_workers_overlap_and_main_thread_accounts_all_actual_bytes(tmp_path):
    import threading
    plan,tree,blobs=parallel_fixture(tmp_path,4)
    barrier=threading.Barrier(3);lock=threading.Lock();active=0;peak=0
    main_thread=threading.get_ident();callbacks=[]
    def opener(request,timeout):
        nonlocal active,peak
        shard=int(request.full_url.rsplit('_',1)[1].split('.')[0])
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        with lock:active+=1;peak=max(peak,active)
        try:
            if shard<3:barrier.wait(timeout=10)
            return Response(blobs[shard],start,end,10_000_000)
        finally:
            with lock:active-=1
    budget=ByteBudget(300_000,500)
    def report(item):callbacks.append((threading.get_ident(),item,budget.used))
    result=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
        archive_limit=50_000,opener=opener,on_report=report,workers=3)
    assert peak==3 and len(result)==len(callbacks)==4
    assert all(thread==main_thread for thread,item,used in callbacks)
    assert budget.used==500+sum(len(blob) for blob in blobs.values())
    assert [used for thread,item,used in callbacks]==sorted(used for thread,item,used in callbacks)
    assert all(Path(path).is_file() for path in plan['required_images'])


def test_parallel_failure_charges_partial_body_and_collects_other_workers(tmp_path):
    plan,tree,blobs=parallel_fixture(tmp_path)
    partial=b'charged-partial'
    class FailedBody(io.BytesIO):
        status=206
        headers={'Content-Range':'bytes 0-49999/10000000'}
        def __init__(self):self.first=True
        def read(self,size=-1):
            if self.first:self.first=False;return partial
            raise OSError('fixture interrupted response')
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def opener(request,timeout):
        shard=int(request.full_url.rsplit('_',1)[1].split('.')[0])
        if shard==0:return FailedBody()
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[shard],start,end,10_000_000)
    budget=ByteBudget(200_000,123);reports=[]
    with pytest.raises(RuntimeError,match='interrupted response'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
            archive_limit=50_000,opener=opener,on_report=reports.append,workers=3)
    assert budget.used==123+len(partial)+len(blobs[1])+len(blobs[2])
    assert len(reports)==3 and sum(report['complete'] for report in reports)==2
    failed=next(report for report in reports if not report['complete'])
    assert failed['response_body_bytes']==len(partial)
    assert failed['response_prefix_sha256'] is None
    assert all(Path(path).is_file() for path in plan['required_images'][1:])


def test_parallel_reservations_are_checked_before_any_network_body(tmp_path):
    plan,tree,blobs=parallel_fixture(tmp_path,2);called=[]
    def forbidden(*args,**kwargs):called.append(True);raise AssertionError('Must reserve first')
    budget=ByteBudget(100_000,1)
    with pytest.raises(ValueError,match='reserv'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
            archive_limit=50_000,opener=forbidden,workers=3)
    assert not called and budget.used==1


@pytest.mark.parametrize('workers',[0,4,True,1.5])
def test_acquisition_rejects_unbounded_or_invalid_worker_counts(tmp_path,workers):
    plan,tree,blobs=parallel_fixture(tmp_path)
    with pytest.raises(ValueError,match='worker'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(200_000),
            archive_limit=50_000,workers=workers)


def test_restart_retains_verified_complete_archive_provenance(tmp_path):
    from slowfast_v11.expanded_data import load_completed_reports
    plan,tree,blobs=parallel_fixture(tmp_path)
    def opener(request,timeout):
        shard=int(request.full_url.rsplit('_',1)[1].split('.')[0])
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[shard],start,end,10_000_000)
    first=copy.deepcopy(plan);first['splits']['train']=first['splits']['train'][:1]
    first['required_images']=first['required_images'][:1]
    reports=acquire_plan(first,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(100_000),
        archive_limit=50_000,opener=opener)
    output=tmp_path/'out';output.mkdir()
    (output/'camera_0_acquisition.json').write_text(json.dumps(reports[0]))
    # A prior interrupted attempt is not a completed archive identity.
    (output/'camera_1_acquisition.json').write_text(json.dumps({'complete':False}))
    prior=load_completed_reports(output,plan,tree)
    assert prior==reports
    budget=ByteBudget(300_000,123)
    current=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
        archive_limit=50_000,opener=opener,workers=3)
    merged={report['shard']:report for report in prior+current}
    assert set(merged)=={0,1,2}
    assert budget.used==123+len(blobs[1])+len(blobs[2])
    assert merged[0]['response_prefix_sha256']==reports[0]['response_prefix_sha256']
    for field,bad in [('requests',True),('response_prefix_sha256','invalid'),('archive_identity',{}),
                      ('url','https://wrong.example/archive'),('response_body_bytes',225_000_001)]:
        invalid=copy.deepcopy(reports[0]);invalid[field]=bad
        (output/'camera_0_acquisition.json').write_text(json.dumps(invalid))
        with pytest.raises(ValueError):load_completed_reports(output,plan,tree)


def test_reverification_covers_existing_members_without_replacing_them(tmp_path):
    plan,tree,blobs=parallel_fixture(tmp_path,1)
    sensor=Path(plan['sensor_root']);first=Path(plan['required_images'][0])
    first.parent.mkdir(parents=True);Image.new('RGB',(12,8)).save(first)
    before=first.read_bytes();mtime=first.stat().st_mtime_ns
    second=str(sensor/'parallel-0/CAM_F0/2.jpg');plan['required_images'].append(second)
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[0],start,end,10_000_000)
    reports=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(100_000),
        archive_limit=50_000,opener=opener,force_reverify_logs={'parallel-0'})
    assert {item['path'] for item in reports[0]['files']}==set(plan['required_images'])
    assert first.read_bytes()==before and first.stat().st_mtime_ns==mtime
    assert Path(second).is_file()


def test_reverification_detects_hash_mismatch_without_overwriting_existing_bytes(tmp_path):
    plan,tree,blobs=parallel_fixture(tmp_path,1);path=Path(plan['required_images'][0])
    path.parent.mkdir(parents=True);path.write_bytes(b'preserve-mismatching-local-bytes')
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[0],start,end,10_000_000)
    with pytest.raises(RuntimeError,match='SHA256 mismatch'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(100_000),
            archive_limit=50_000,opener=opener,force_reverify_logs={'parallel-0'})
    assert path.read_bytes()==b'preserve-mismatching-local-bytes'


def test_incomplete_report_coverage_forces_reverification_even_if_all_files_exist(tmp_path):
    from slowfast_v11.expanded_data import load_completed_reports
    plan,tree,blobs=parallel_fixture(tmp_path,1)
    plan['required_images'].append(str(Path(plan['sensor_root'])/'parallel-0/CAM_F0/2.jpg'))
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[0],start,end,10_000_000)
    report=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(100_000),
        archive_limit=50_000,opener=opener)[0]
    report['files'].pop();output=tmp_path/'out';output.mkdir()
    (output/'camera_0_acquisition.json').write_text(json.dumps(report))
    assert all(Path(path).exists() for path in plan['required_images'])
    assert load_completed_reports(output,plan,tree)==[]


def test_transient_pre_response_retry_keeps_byte_caps_and_records_attempts(tmp_path):
    import http.client
    import ssl
    plan,tree,blobs=parallel_fixture(tmp_path,1);calls=[]
    def opener(request,timeout):
        calls.append(request.headers['Range'])
        if len(calls)==1:raise ssl.SSLEOFError('fixture SSL EOF')
        if len(calls)==2:raise http.client.RemoteDisconnected('fixture disconnected')
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[0],start,end,10_000_000)
    budget=ByteBudget(100_000,123)
    report=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
        archive_limit=50_000,opener=opener,retry_backoff_seconds=0)[0]
    assert len(calls)==3 and len(set(calls))==1
    assert budget.used==123+len(blobs[0])
    assert report['request_attempts']==3 and report['pre_response_failures']==2
    assert report['response_prefix_sha256'] is not None


@pytest.mark.parametrize('code,attempts',[(403,1),(404,1),(429,3),(503,3)])
def test_retry_policy_never_retries_other_http_client_errors(tmp_path,code,attempts):
    import urllib.error
    plan,tree,blobs=parallel_fixture(tmp_path,1);calls=[];reports=[]
    def opener(request,timeout):
        calls.append(True)
        raise urllib.error.HTTPError(request.full_url,code,'fixture response',{},io.BytesIO())
    budget=ByteBudget(100_000,100)
    with pytest.raises(RuntimeError,match=str(code)):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,archive_limit=50_000,
            opener=opener,on_report=reports.append,retry_backoff_seconds=0)
    assert len(calls)==attempts and budget.used==100
    assert reports[0]['response_prefix_sha256'] is not None


def test_attempt_ledger_is_durable_before_latest_report_is_replaced(tmp_path):
    from tools.prepare_slowfast_expanded import persist_attempt_report
    output=tmp_path/'out';output.mkdir()
    old={'shard':5,'complete':False,'response_body_bytes':7}
    (output/'camera_5_acquisition.json').write_text(json.dumps(old))
    fresh={'shard':5,'complete':True,'response_body_bytes':20}
    persist_attempt_report(output,fresh,aggregate_used=127)
    ledger=[json.loads(line) for line in (output/'camera_attempts.jsonl').read_text().splitlines()]
    assert ledger[0]['report']==old and ledger[1]['report']==fresh
    assert ledger[1]['aggregate_response_body_bytes']==127
    assert json.loads((output/'camera_5_acquisition.json').read_text())==fresh


def test_final_provenance_requires_exact_camera_and_legacy_union(tmp_path):
    from slowfast_v11.expanded_data import verify_provenance_coverage
    plan,tree,blobs=parallel_fixture(tmp_path,1)
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'))
        return Response(blobs[0],start,end,10_000_000)
    reports=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(100_000),
        archive_limit=50_000,opener=opener)
    result=verify_provenance_coverage(plan,reports)
    assert result['verified_images']==1
    reports[0]['files']=[]
    with pytest.raises(ValueError,match='coverage'):verify_provenance_coverage(plan,reports)


def shard_seven_fixture(tmp_path,count=1):
    plan,tree,blobs=parallel_fixture(tmp_path,count)
    name=plan['splits']['train'][0];plan['logs'][name]['shard']=7
    tree[0]={'path':'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_7.tgz','size':1_000_000_000}
    if count>1:tree[1]['size']=1_000_000_000
    return plan,tree,blobs


def test_explicit_shard7_cap_override_has_bounded_child_and_restart_validation(tmp_path):
    from slowfast_v11.expanded_data import load_completed_reports
    plan,tree,blobs=shard_seven_fixture(tmp_path);ranges=[]
    def opener(request,timeout):
        start,end=map(int,request.headers['Range'].split('=')[1].split('-'));ranges.append((start,end))
        return Response(blobs[0],start,end,1_000_000_000)
    budget=ByteBudget(256_000_123,123)
    report=acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,
        opener=opener,archive_cap_overrides={7:256_000_000})[0]
    assert report['reserved_cap_bytes']==256_000_000
    assert budget.used==123+len(blobs[0])
    assert all(end<256_000_000 for start,end in ranges)
    output=tmp_path/'out';output.mkdir();(output/'camera_7_acquisition.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):load_completed_reports(output,plan,tree)
    assert load_completed_reports(output,plan,tree,archive_cap_overrides={7:256_000_000})==[report]


@pytest.mark.parametrize('overrides', [{6:256_000_000},{7:256_000_001},{7:225_000_000},
    {'7':256_000_000},{7:True},{True:256_000_000},{7.:256_000_000}])
def test_unsupported_archive_cap_overrides_are_rejected(tmp_path,overrides):
    plan,tree,blobs=shard_seven_fixture(tmp_path)
    with pytest.raises(ValueError,match='override'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(1_000_000_000),
            archive_cap_overrides=overrides)


def test_override_reservation_sums_each_pending_shard_cap_before_network(tmp_path):
    plan,tree,blobs=shard_seven_fixture(tmp_path,2);calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('Reserve first')
    budget=ByteBudget(481_000_000,1)
    with pytest.raises(ValueError,match='reserv'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=budget,opener=forbidden,
            archive_cap_overrides={7:256_000_000})
    assert not calls and budget.used==1


def test_shard7_override_still_cannot_request_a_complete_archive(tmp_path):
    plan,tree,blobs=shard_seven_fixture(tmp_path);tree[0]['size']=256_000_000
    with pytest.raises(ValueError,match='complete'):
        acquire_plan(plan,tree,sensor_dest=tmp_path/'sensors',budget=ByteBudget(1_000_000_000),
            archive_cap_overrides={7:256_000_000})
