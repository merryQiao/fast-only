"""Metadata-only promotion, frozen fresh validation and exact transfer evidence."""
import copy
import hashlib
import http.client
import io
import json
import pickle
from pathlib import Path

from PIL import Image
import pytest

from fast_navsim.acquire import ByteBudget
from fast_navsim.data import build_window, compute_ego_stats


def frames(name, count=340):
    return [dict(log_name=name, token=f'{name}-{i}', timestamp=i*500_000,
        driving_command=[0, 1, 0, 0], ego_dynamic_state=[float(i), 0., 0., 0.],
        ego2global_translation=[float(i), 0., 0.], ego2global_rotation=[1., 0., 0., 0.],
        cams={'CAM_F0': {'data_path': f'{name}/CAM_F0/{i}.jpg'}}) for i in range(count)]


def inputs(tmp_path):
    names=[f'2021.06.{i:02d}.12.00.00_veh-1_00000_00500' for i in range(1, 25)]
    retained=[]; index={}; train=[]; promoted=[]
    for shard, name in reversed(list(enumerate(names))):
        data=frames(name); blob=pickle.dumps(data); path=tmp_path/f'{name}.pkl';path.write_bytes(blob)
        retained.append(dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        index[f'openscene_sensor_mini_{shard}']=[name]
        if shard<4 or 7<=shard<15:
            rows=train if shard<4 else promoted
            source=dict(metadata_path=str(path),metadata_sha256=retained[-1]['sha256'])
            for k in range(64 if shard<4 else 32):
                row=build_window(data,3+5*k,'train' if shard<4 else 'val',tmp_path/'old-images',source)
                row['cohort']='train' if shard<4 else 'holdout';rows.append(row)
    return names, retained, index, train, promoted


def plan_for(tmp_path):
    from slowfast_v11.direct_data import make_direct_plan
    names, retained, index, train, promoted=inputs(tmp_path)
    plan=make_direct_plan(retained,index,train,promoted,names[:15],sensor_root=tmp_path/'sensors/openscene-v1.1/sensor_blobs/mini')
    return plan,names,retained,index,train,promoted


def test_promotes_entire_holdout_preserving_every_original_field_except_role(tmp_path):
    plan,names,retained,index,train,promoted=plan_for(tmp_path)
    assert plan['rows']['train'][:256]==train
    assert len(plan['rows']['train'])==512 and len(plan['rows']['val'])==256
    for before,after in zip(promoted,plan['rows']['train'][256:]):
        assert after==dict(before,split='train',cohort='train')
        assert before['split']=='val' and before['cohort']=='holdout'
    assert plan['role_promotion']['count']==256
    assert plan['role_promotion']['recording_groups']==8
    assert plan['role_promotion']['selection_based_on_model_outputs'] is False
    assert plan['fixed_source_token']==min(r['token'] for r in plan['rows']['train'])
    assert plan['cohort_counts']=={'train':512,'new_val':256}


def test_new_validation_excludes_all_fifteen_seen_groups_and_never_checks_images(tmp_path):
    from slowfast_v11.adapter_data import recording_group
    plan,names,*_=plan_for(tmp_path)
    assert plan['splits']['new_val']==names[15:23]
    assert plan['splits']['train']==[]  # acquisition helper fetches only fresh VAL
    assert len({recording_group(r['route']) for r in plan['rows']['train']})==12
    assert len({recording_group(r['route']) for r in plan['rows']['val']})==8
    assert len(plan['selection_audit']['seen_recording_groups'])==15
    assert all(r['cohort']=='new_val' and r['split']=='val' for r in plan['rows']['val'])
    assert all(info['selected_indices']==list(range(3,159,5)) for info in plan['logs'].values())
    assert not Path(plan['sensor_root']).exists()
    assert set(plan['required_images']).isdisjoint(plan['inherited_images'])


def test_rejects_incomplete_promotion_or_training_overlap_or_missing_seen_groups(tmp_path):
    from slowfast_v11.direct_data import make_direct_plan
    names,retained,index,train,promoted=inputs(tmp_path)
    kwargs=dict(sensor_root=tmp_path/'s')
    with pytest.raises(ValueError,match='256'):
        make_direct_plan(retained,index,train,promoted[:-1],names[:15],**kwargs)
    changed=copy.deepcopy(promoted);changed[0]['token']=train[0]['token']
    with pytest.raises(ValueError,match='token'):
        make_direct_plan(retained,index,train,changed,names[:15],**kwargs)
    with pytest.raises(ValueError,match='fifteen'):
        make_direct_plan(retained,index,train,promoted,names[:14],**kwargs)


def test_future_nonfinite_pose_changes_only_structural_eligibility_not_motion_policy(tmp_path):
    from slowfast_v11.direct_data import make_direct_plan
    names,retained,index,train,promoted=inputs(tmp_path)
    item=next(item for item in retained if Path(item['path']).stem==names[15])
    data=pickle.loads(Path(item['path']).read_bytes());data[9]['ego2global_translation'][0]=float('nan')
    blob=pickle.dumps(data);Path(item['path']).write_bytes(blob);item['sha256']=hashlib.sha256(blob).hexdigest()
    plan=make_direct_plan(retained,index,train,promoted,names[:15],sensor_root=tmp_path/'s')
    assert plan['logs'][names[15]]['selected_indices'][0]==13
    assert plan['logs'][names[15]]['excluded_windows']['invalid_pose']==2


def test_freezes_all_tokens_and_refuses_replacement(tmp_path):
    from slowfast_v11.direct_data import freeze_direct_plan
    plan,*_=plan_for(tmp_path);out=tmp_path/'out';freeze_direct_plan(plan,out)
    tokens=json.loads((out/'frozen_tokens.json').read_text())
    assert len(tokens['tokens']['train'])==512 and len(tokens['tokens']['val'])==256
    assert tokens['frozen_before_camera_access'] is True
    freeze_direct_plan(plan,out)
    changed=copy.deepcopy(plan);changed['rows']['val'][0]['token']='changed'
    with pytest.raises(ValueError,match='identity'):
        freeze_direct_plan(changed,out)


class Response(io.BytesIO):
    status=206
    def __init__(self,blob,start,end):
        super().__init__(blob[start:end+1]);self.headers={'Content-Range':f'bytes {start}-{end}/1000'}
    def close(self):
        super().close()


def small_caps(module,monkeypatch):
    for name,value in [('BASE_PREFIX_BYTES',10),('METADATA_PREFIX_CAP',30),('METADATA_NEW_CAP',20),('CAMERA_RESERVATION',40)]:
        monkeypatch.setattr(module,name,value)


def test_metadata_extension_accounts_attached_partial_error_bytes_without_changing_old_prefix(tmp_path,monkeypatch):
    import slowfast_v11.direct_data as module
    small_caps(module,monkeypatch)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789');reports=[]
    class Incomplete(Response):
        def read(self,size=-1):raise http.client.IncompleteRead(b'abcd',16)
    budget=ByteBudget(100)
    with pytest.raises(http.client.IncompleteRead):
        module.extend_metadata_prefix(prefix,budget,opener=lambda request,timeout:Incomplete(b'',10,29),on_request=reports.append)
    assert budget.used==4 and prefix.read_bytes()==b'0123456789abcd'
    assert reports[0]['response_body_bytes']==4 and reports[0]['error']
    assert reports[0]['response_body_sha256']==hashlib.sha256(b'abcd').hexdigest()
    with pytest.raises(ValueError,match='allowance'):
        module.extend_metadata_prefix(prefix,budget,metadata_bytes_used=5,opener=lambda *a,**k:pytest.fail('must stop before network'))


def test_metadata_exact_fixed_cap_reserves_cameras_and_refuses_changed_cap(tmp_path,monkeypatch):
    import slowfast_v11.direct_data as module
    small_caps(module,monkeypatch)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789')
    with pytest.raises(ValueError,match='reserv'):
        module.extend_metadata_prefix(prefix,ByteBudget(59),opener=lambda *a,**k:pytest.fail('reserve first'))
    budget=ByteBudget(60);reports=[]
    result=module.extend_metadata_prefix(prefix,budget,opener=lambda request,timeout:Response(b'0123456789abcdefghijklmnopqrst',10,29),on_request=reports.append)
    assert prefix.stat().st_size==30 and budget.used==20
    assert result['complete'] is True and len(reports)==1
    assert module.METADATA_PREFIX_CAP<module.WHOLE_METADATA_BYTES


def test_ledger_persists_all_failures_and_uses_previous_cumulative_not_previous_new(tmp_path,monkeypatch):
    from tools.prepare_direct_full import RequestLedger
    ledger=RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'x'})
    for n,kind in [(4,'metadata'),(0,'camera'),(7,'camera')]:
        ledger.append(dict(request_id=f'{kind}-{n}',kind=kind,response_body_bytes=n,
            response_body_sha256=hashlib.sha256(b'x'*n).hexdigest(),error='interrupted' if n<7 else None))
    state=ledger.state('failed')
    assert state['used']==11 and state['cumulative_response_body_bytes']==3_403_085_692
    assert state['metadata_prefix_cap_bytes']==520_000_000 and state['metadata_new_cap_bytes']==200_000_000
    assert state['camera_requests']==2
    assert RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'x'}).state('resumed')['used']==11
    with pytest.raises(ValueError,match='Inherited'):
        RequestLedger(tmp_path,inherited_bytes=1,inherited_budget={'sha256':'x'})


def test_ledger_saves_overrun_evidence_and_rejects_invalid_record_before_append(tmp_path,monkeypatch):
    import tools.prepare_direct_full as module
    monkeypatch.setattr(module,'NEW_NETWORK_CAP',10)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=100,inherited_budget={})
    with pytest.raises(ValueError,match='Invalid'):
        ledger.append(dict(request_id='bad',kind='camera',response_body_bytes=-1,response_body_sha256='x'))
    for identity,n in [('one',9),('two',2)]:
        record=dict(request_id=identity,kind='camera',response_body_bytes=n,response_body_sha256=hashlib.sha256(b'x'*n).hexdigest(),error='failed')
        if n==9:ledger.append(record)
        else:
            with pytest.raises(ValueError,match='cap'):ledger.append(record)
    assert len((tmp_path/'network_requests.jsonl').read_text().splitlines())==2
    assert json.loads((tmp_path/'network_budget.json').read_text())['used']==11


def install_sources(plan):
    sources=[]
    for image in plan['required_images']+plan['inherited_images']:
        path=Path(image);path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(12,8)).save(path)
        if image in plan['inherited_images']:
            sources.append(dict(path=image,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,kind='retained_original_jpeg',upstream_audit={'sha256':'authenticated-parent'}))
    for row in plan['rows']['train']:
        path=Path(row['source']['metadata_path'])
        if not any(s['path']==str(path) for s in sources):
            sources.append(dict(path=str(path),sha256=row['source']['metadata_sha256'],bytes=path.stat().st_size,kind='retained_metadata',upstream_audit={'sha256':'authenticated-parent'}))
    plan['inherited_sources']=sources
    from slowfast_v11.expanded_data import _plan_digest
    plan['plan_sha256']=_plan_digest(plan)


def test_prepared_writer_authenticates_promoted_images_and_preserves_checkpoint_normalizer(tmp_path):
    from slowfast_v11.direct_data import write_direct_prepared
    plan,*_=plan_for(tmp_path);install_sources(plan)
    source=tmp_path/'original-stats.json';source.write_text(json.dumps(compute_ego_stats(plan['rows']['train'][:256])))
    before=source.read_bytes();expected=hashlib.sha256(before).hexdigest();out=tmp_path/'out';out.mkdir()
    retained=out/'metadata_retained.json';retained.write_text(json.dumps({'files':plan['metadata_records']}))
    result=write_direct_prepared(plan,out,normalizer_source=source,normalizer_expected_sha256=expected,expected_size=(12,8))
    assert result['counts']=={'train':512,'new_val':256}
    assert len((out/'train.jsonl').read_text().splitlines())==512 and len((out/'val.jsonl').read_text().splitlines())==256
    expanded=json.loads((out/'train512_stats_audit.json').read_text())
    assert expanded['count']==512 and expanded['ego_mean']!=json.loads(before)['ego_mean']
    assert (out/'stats.json').read_bytes()==before and source.read_bytes()==before
    assert result['normalization']['selected']['sha256']==expected
    assert result['normalization']['expanded_train512']['selected_for_model_inputs'] is False
    assert set(result['normalization']['shared_consumers'])=={'fast','slow','feedback_teacher'}
    audit=json.loads((out/'source_audit.json').read_text())
    assert len([s for s in audit['sources'] if s['kind']=='inherited_original_jpeg'])==len(plan['inherited_images'])
    assert audit['role_promotion']['count']==256 and audit['full_archives_verified'] is False
    assert audit['normalization']==result['normalization']
    assert audit['retained_metadata_manifest']==dict(path=str(retained.resolve()),sha256=hashlib.sha256(retained.read_bytes()).hexdigest())


def test_prepared_writer_refuses_changed_promoted_jpeg_and_existing_different_stats(tmp_path):
    from slowfast_v11.direct_data import write_direct_prepared
    plan,*_=plan_for(tmp_path);install_sources(plan)
    source=tmp_path/'original-stats.json';source.write_text(json.dumps(compute_ego_stats(plan['rows']['train'][:256])))
    kwargs=dict(normalizer_source=source,normalizer_expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),expected_size=(12,8))
    target=Path(plan['inherited_images'][0]);before=target.read_bytes();target.write_bytes(before+b'tampered')
    with pytest.raises(ValueError,match='SHA256'):
        write_direct_prepared(plan,tmp_path/'out',**kwargs)
    target.write_bytes(before);out=tmp_path/'out';out.mkdir(exist_ok=True);(out/'stats.json').write_text('wrong')
    with pytest.raises(ValueError,match='normalizer'):
        write_direct_prepared(plan,out,**kwargs)


def test_camera_retries_cannot_reuse_a_new225mb_allowance_on_resume(tmp_path,monkeypatch):
    import urllib.request
    import tools.prepare_direct_full as module
    monkeypatch.setattr(module,'CAMERA_PREFIX_CAP',10)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=100,inherited_budget={})
    ledger.append(dict(request_id='partial',kind='camera',url='https://example.test/camera',
        response_body_bytes=7,response_body_sha256=hashlib.sha256(b'x'*7).hexdigest(),error='interrupted'))
    called=[]
    def network(request,timeout):
        called.append(True);return Response(b'0123456789',0,1)
    opener=module.make_camera_opener(ledger,network)
    request=urllib.request.Request('https://example.test/camera',headers={'Range':'bytes=0-3'})
    with pytest.raises(ValueError,match='remaining camera'):
        opener(request,timeout=90)
    assert not called and ledger.records[-1]['response_body_bytes']==0
    assert ledger.records[-1]['error']
    request=urllib.request.Request('https://example.test/camera',headers={'Range':'bytes=0-1'})
    with opener(request,timeout=90) as response:
        assert response.read()==b'01'
    assert ledger.state('camera')['camera_response_body_bytes']==9 and len(called)==1


def recovery_fixture(tmp_path):
    import tools.prepare_direct_full as module
    plan={'plan_sha256':'frozen-plan','token_manifest_sha256':'frozen-tokens'}
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'})
    ledger.append(dict(request_id='metadata',kind='metadata',response_body_bytes=200_000_000,
        response_body_sha256=hashlib.sha256(b'metadata').hexdigest(),error=None))
    for shard,n in [(27,74_665_502),(17,112_000_000)]:
        url=f'https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/a76f840b65e972bc45e56c2adced897498e9a026/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_{shard}.tgz'
        ledger.append(dict(request_id=str(shard),kind='camera',url=url,response_body_bytes=n,
            response_body_sha256=hashlib.sha256(str(shard).encode()).hexdigest(),error='URLError: TLS EOF'))
        (tmp_path/f'camera_{shard}_acquisition.json').write_text(json.dumps(dict(shard=shard,url=url,
            response_body_bytes=n,complete=False,error='URLError: TLS EOF')))
    (tmp_path/'source_audit.json').write_text(json.dumps({'complete':False,'error':'initial failed acquisition'}))
    return plan,ledger


def test_explicit_recovery_preserves_failure_evidence_and_pins_only_two_retransmission_caps(tmp_path):
    import tools.prepare_direct_full as module
    plan,ledger=recovery_fixture(tmp_path)
    original=(tmp_path/'camera_27_acquisition.json').read_bytes()
    amendment=module.record_transport_amendment(tmp_path,plan)
    assert amendment['archive_prefix_extent_cap_bytes']==225_000_000
    assert amendment['new_total_response_body_cap_bytes']==2_000_000_000
    assert sorted(amendment['response_body_cap_overrides_by_url'].values())==[299_665_502,337_000_000]
    assert (tmp_path/'camera_27_acquisition_initial_failed.json').read_bytes()==original
    assert amendment['plan_sha256']=='frozen-plan' and amendment['token_manifest_sha256']=='frozen-tokens'
    path=tmp_path/'transport_recovery_amendment.json'
    assert json.loads(path.read_text())==amendment
    assert module.record_transport_amendment(tmp_path,plan)==amendment
    resumed=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=path)
    assert resumed.state('recovery')['transport_amendment']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert resumed.state('recovery')['camera_response_body_cap_overrides_by_url']==amendment['response_body_cap_overrides_by_url']
    with pytest.raises(ValueError,match='explicit'):
        module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'})


def test_recovery_refuses_unapproved_overrides_and_changed_failure_identity(tmp_path):
    import tools.prepare_direct_full as module
    plan,ledger=recovery_fixture(tmp_path)
    report=tmp_path/'camera_17_acquisition.json';content=json.loads(report.read_text());content['response_body_bytes']+=1;report.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='failure'):
        module.record_transport_amendment(tmp_path,plan)
    content['response_body_bytes']-=1;report.write_text(json.dumps(content))
    amendment=module.record_transport_amendment(tmp_path,plan)
    url=next(iter(amendment['response_body_cap_overrides_by_url']));amendment['response_body_cap_overrides_by_url'][url]+=1
    path=tmp_path/'bad-amendment.json';path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError,match='amendment'):
        module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=path)


def test_recovery_opener_uses_amended_lifetime_cap_without_raising_archive_extent(tmp_path):
    import urllib.request
    import tools.prepare_direct_full as module
    plan,ledger=recovery_fixture(tmp_path);module.record_transport_amendment(tmp_path,plan)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=tmp_path/'transport_recovery_amendment.json')
    url=next(url for url in ledger.response_caps if url.endswith('_17.tgz'))
    opener=module.make_camera_opener(ledger,lambda request,timeout:Response(b'ab',0,1))
    # A complete225MB rescan is legal after the original112MB failed body.
    with opener(urllib.request.Request(url,headers={'Range':'bytes=0-224999999'}),timeout=90) as response:
        assert response.read()==b'ab'
    assert ledger.state('recovery')['camera_prefix_cap_bytes']==225_000_000
    with pytest.raises(ValueError,match='remaining camera'):
        opener(urllib.request.Request(url,headers={'Range':'bytes=0-224999999'}),timeout=90)
    with pytest.raises(ValueError,match='extent'):
        opener(urllib.request.Request(url,headers={'Range':'bytes=224999999-225000000'}),timeout=90)


def test_durable_prefix_fsyncs_partial_body_and_resumes_exact_saved_offsets(tmp_path):
    import slowfast_v11.direct_data as module
    blob=b'0123456789abcdefghij';prefix=tmp_path/'prefix.gz';records=[];calls=[]
    class Broken(Response):
        def read(self,size=-1):
            if self.tell():raise OSError('fixture TLS interrupted')
            return super().read(4)
    def broken(request,timeout):
        calls.append(request.get_header('Range'));return Broken(blob,0,7)
    budget=ByteBudget(100)
    with pytest.raises(OSError,match='interrupted'):
        module.download_durable_prefix('https://example.test/camera',prefix,budget,
            expected_archive_bytes=1000,prefix_cap=20,chunk_size=8,
            opener=module.AuditedOpener(broken,'camera',records.append))
    assert prefix.read_bytes()==b'0123' and budget.used==4 and records[0]['response_body_bytes']==4
    module.verify_durable_prefix(prefix,records)
    def healthy(request,timeout):
        calls.append(request.get_header('Range'));start,end=map(int,request.get_header('Range')[6:].split('-'))
        return Response(blob,start,end)
    result=module.download_durable_prefix('https://example.test/camera',prefix,budget,
        expected_archive_bytes=1000,prefix_cap=20,chunk_size=8,
        opener=module.AuditedOpener(healthy,'camera',records.append))
    assert calls==['bytes=0-7','bytes=4-11','bytes=12-19']
    assert prefix.read_bytes()==blob and budget.used==20 and result['complete'] is True
    assert sum(r['response_body_bytes'] for r in records)==20
    module.verify_durable_prefix(prefix,records)
    prefix.write_bytes(blob[:-1]+b'x')
    with pytest.raises(ValueError,match='SHA256'):
        module.verify_durable_prefix(prefix,records)


def test_durable_prefix_rejects_wrong_content_range_before_consuming_body(tmp_path):
    import slowfast_v11.direct_data as module
    prefix=tmp_path/'prefix.gz';records=[];budget=ByteBudget(100)
    raw=lambda request,timeout:Response(b'x'*20,1,7)
    with pytest.raises(ValueError,match='Content-Range'):
        module.download_durable_prefix('https://example.test/camera',prefix,budget,
            expected_archive_bytes=1000,prefix_cap=20,chunk_size=8,
            opener=module.AuditedOpener(raw,'camera',records.append))
    assert prefix.stat().st_size==0 and budget.used==0 and records[0]['response_body_bytes']==0


def test_local_durable_camera_extracts_only_frozen_images_and_authenticates_existing_files(tmp_path):
    import tarfile
    import slowfast_v11.direct_data as module
    sensor=tmp_path/'sensors';root=sensor/'openscene-v1.1/sensor_blobs/mini';name='frozen-log'
    paths=[root/name/'CAM_F0'/f'{i}.jpg' for i in range(2)]
    plan=dict(sensor_root=str(root),required_images=[str(p) for p in paths],legacy_copies=[],
        splits={'train':[],'new_val':[name]},logs={name:{'shard':27}})
    content=io.BytesIO()
    with tarfile.open(fileobj=content,mode='w:gz') as archive:
        for i in range(3):
            blob=io.BytesIO();Image.new('RGB',(12,8)).save(blob,format='JPEG');data=blob.getvalue()
            member=tarfile.TarInfo(f'openscene-v1.1/sensor_blobs/mini/{name}/CAM_F0/{i}.jpg');member.size=len(data)
            archive.addfile(member,io.BytesIO(data))
    prefix=tmp_path/'camera27.gz';prefix.write_bytes(content.getvalue())
    identity={'path':'openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_27.tgz','size':1_000_000_000}
    transfer=dict(prefix_bytes=prefix.stat().st_size,prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest(),complete=True,requests_detail=[])
    report=module.extract_durable_camera(plan,[identity],prefix,sensor_dest=sensor,shard=27,transfer=transfer)
    assert report['complete'] is True and len(report['files'])==2
    assert not (root/name/'CAM_F0/2.jpg').exists()
    module.verify_provenance_coverage(plan,[report])
    plan['plan_sha256']=module._plan_digest(plan)
    transfer['requests_detail']=[dict(response_body_bytes=prefix.stat().st_size,status=206)]
    report=module.extract_durable_camera(plan,[identity],prefix,sensor_dest=sensor,shard=27,transfer=transfer,archive_extent_cap=229_000_000)
    out=tmp_path/'reports';out.mkdir();(out/'camera_27_acquisition.json').write_text(json.dumps(report))
    caps={report['url']:229_000_000}
    assert len(module.load_direct_completed_reports(out,plan,[identity],archive_extent_caps=caps))==1
    with pytest.raises(ValueError,match='identity/accounting'):
        module.load_direct_completed_reports(out,plan,[identity])
    with pytest.raises(ValueError,match='amendment'):
        module.load_direct_completed_reports(out,plan,[identity],archive_extent_caps={report['url']:230_000_000})
    paths[0].write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA256'):
        module.extract_durable_camera(plan,[identity],prefix,sensor_dest=sensor,shard=27,transfer=transfer)


def second_failure_fixture(tmp_path):
    import tools.prepare_direct_full as module
    plan,ledger=recovery_fixture(tmp_path)
    for shard,n in [(3,160_000_000),(29,222_429_034),(0,168_065_481),(9,214_756_359),(19,152_000_000),(23,136_000_000)]:
        ledger.append(dict(request_id=f'complete-{shard}',kind='camera',url=module._camera_url(shard),
            response_body_bytes=n,response_body_sha256=hashlib.sha256(str(n).encode()).hexdigest(),error=None))
    module.record_transport_amendment(tmp_path,plan)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=tmp_path/'transport_recovery_amendment.json')
    for shard,n,complete in [(17,152_000_000,True),(27,179_026_540,False)]:
        ledger.append(dict(request_id=f'rescan-{shard}',kind='camera',url=module._camera_url(shard),
            range='bytes=0-'+str(n-1),response_body_bytes=n,response_body_sha256=hashlib.sha256(str(n).encode()).hexdigest(),error=None if complete else 'TLS EOF'))
        (tmp_path/f'camera_{shard}_acquisition.json').write_text(json.dumps(dict(shard=shard,url=module._camera_url(shard),response_body_bytes=n,complete=complete,error=None if complete else 'TLS EOF')))
    return plan,ledger


def test_final_durable_amendment_pins_starting_ledger_and_excludes_earlier_zero_offset_scans(tmp_path):
    import tools.prepare_direct_full as module
    plan,ledger=second_failure_fixture(tmp_path)
    amendment=module.record_durable_transport_amendment(tmp_path,plan)
    assert amendment['maximum_resulting_new_response_body_bytes']==1_995_942_916
    assert amendment['response_body_cap_overrides_by_url'][module._camera_url(27)]==478_692_042
    assert amendment['starting_ledger']['request_count']==len(ledger.records)
    assert (tmp_path/'camera_27_acquisition_second_failed.json').is_file()
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=tmp_path/'transport_durable_amendment.json')
    assert module.durable_stage_records(ledger,amendment)==[]
    record=dict(request_id='durable-new',kind='camera',url=module._camera_url(27),range='bytes=0-1',
        response_body_bytes=2,response_body_sha256=hashlib.sha256(b'ab').hexdigest(),error='TLS EOF')
    ledger.append(record)
    assert module.durable_stage_records(ledger,amendment)==[record]
    prefix=tmp_path/'durable.gz';prefix.write_bytes(b'ab')
    from slowfast_v11.direct_data import verify_durable_prefix
    verify_durable_prefix(prefix,module.durable_stage_records(ledger,amendment))
    saved=Path(amendment['starting_ledger']['path']);saved.write_bytes(saved.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='ledger'):
        module.durable_stage_records(ledger,amendment)


def test_four_mb_extent_amendment_keeps_global_cap_and_rejects_tampering(tmp_path):
    import tools.prepare_direct_full as module
    plan,ledger=second_failure_fixture(tmp_path);module.record_durable_transport_amendment(tmp_path,plan)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=tmp_path/'transport_durable_amendment.json')
    ledger.append(dict(request_id='durable225',kind='camera',url=module._camera_url(27),range='bytes=0-224999999',
        response_body_bytes=225_000_000,response_body_sha256='a'*64,error=None))
    (tmp_path/'camera_27_durable_prefix.json').write_text(json.dumps(dict(prefix_bytes=225_000_000,prefix_sha256='a'*64,complete=True)))
    (tmp_path/'camera_27_extent_failure.json').write_text(json.dumps(dict(plan_sha256=plan['plan_sha256'],token_manifest_sha256=plan['token_manifest_sha256'],planned_images=128,verified_planned_images=126,missing_members=['one','two'])))
    amendment=module.record_extent_transport_amendment(tmp_path,plan)
    assert amendment['maximum_resulting_new_response_body_bytes']==1_999_942_916
    assert amendment['new_total_response_body_cap_bytes']==2_000_000_000
    assert amendment['archive_prefix_extent_cap_overrides_by_url']=={module._camera_url(27):229_000_000}
    assert amendment['starting_ledger']['request_count']<len(ledger.records)
    path=tmp_path/'transport_extent_amendment.json'
    ledger=module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=path)
    assert ledger.archive_extent_caps[module._camera_url(27)]==229_000_000
    amendment['archive_prefix_extent_cap_overrides_by_url'][module._camera_url(27)]+=1
    bad=tmp_path/'bad-extent.json';bad.write_text(json.dumps(amendment))
    with pytest.raises(ValueError,match='amendment'):
        module.RequestLedger(tmp_path,inherited_bytes=3_403_085_681,inherited_budget={'sha256':'old'},transport_amendment=bad)
