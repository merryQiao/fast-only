"""Independent metadata-first holdout and bounded transfer provenance."""
import copy
import hashlib
import io
import json
import pickle
from pathlib import Path

from PIL import Image
import pytest

from fast_navsim.acquire import ByteBudget


def frames(name, count=190):
    return [dict(log_name=name, token=f'{name}-{i}', timestamp=i*500_000,
        driving_command=[0, 1, 0, 0], ego_dynamic_state=[float(i), 0., 0., 0.],
        ego2global_translation=[float(i), 0., 0.], ego2global_rotation=[1., 0., 0., 0.],
        cams={'CAM_F0': {'data_path': f'{name}/CAM_F0/{i}.jpg'}}) for i in range(count)]


def fixture(tmp_path):
    names=[f'2021.06.{i:02d}.12.00.00_veh-1_00000_00500' for i in range(1, 12)]
    retained=[]; index={}; data={}
    for shard, name in reversed(list(enumerate(names))):
        data[name]=frames(name); blob=pickle.dumps(data[name]); path=tmp_path/f'{name}.pkl'
        path.write_bytes(blob)
        retained.append(dict(path=str(path), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
        index[f'openscene_sensor_mini_{shard}']=[name, 'second-log']
    return names, retained, index, data


def plan_for(tmp_path):
    from slowfast_v11.adapter_data import make_adapter_plan
    names, retained, index, data=fixture(tmp_path)
    plan=make_adapter_plan(retained, index, [names[0]], sensor_root=tmp_path/'sensors/openscene-v1.1/sensor_blobs/mini')
    return plan, names, retained, index, data


def test_selects_sorted_unseen_groups_first_in_shard_first32_without_images(tmp_path):
    from slowfast_v11.adapter_data import recording_group
    plan, names, retained, index, data=plan_for(tmp_path)
    assert plan['splits']['holdout']==names[1:9]
    assert plan['splits']['train']==[] and plan['splits']['new_val']==names[1:9]
    assert plan['cohort_counts']=={'holdout':256}
    assert len(plan['rows']['val'])==256 and plan['rows']['train']==[]
    assert {r['split'] for r in plan['rows']['val']}=={'val'}
    assert {r['cohort'] for r in plan['rows']['val']}=={'holdout'}
    assert len({recording_group(r['route']) for r in plan['rows']['val']})==8
    assert all(info['selected_indices']==list(range(3, 159, 5)) for info in plan['logs'].values())
    assert not Path(plan['sensor_root']).exists()
    assert all(r['source']['sensor_root']==plan['sensor_root'] and r['source']['builder_protocol']
               for r in plan['rows']['val'])


def test_group_exclusion_and_duplicate_group_are_segment_independent(tmp_path):
    from slowfast_v11.adapter_data import make_adapter_plan, recording_group
    names, retained, index, data=fixture(tmp_path)
    other=names[1].rsplit('_',2)[0]+'_00700_00900'
    blob=pickle.dumps(frames(other)); path=tmp_path/f'{other}.pkl';path.write_bytes(blob)
    retained.append(dict(path=str(path),sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob)))
    index['openscene_sensor_mini_31']=[other]
    plan=make_adapter_plan(retained,index,[names[0].rsplit('_',2)[0]+'_00800_00900'],sensor_root=tmp_path/'s')
    assert plan['splits']['holdout']==names[1:9]
    assert recording_group(names[0])==recording_group(names[0].rsplit('_',2)[0]+'_00800_00900')
    assert plan['selection_audit']['excluded'][other]=='duplicate_recording_group'


def test_checks_all_twelve_pose_validity_before_selection_and_ignores_annotations():
    from slowfast_v11.adapter_data import holdout_candidates
    data=frames('a',25)
    assert holdout_candidates(data)==([3,8,13],{})
    for frame in data:frame['annotations']=object()
    assert holdout_candidates(data)==([3,8,13],{})
    data[9]['ego2global_translation'][0]=float('nan')
    indices, excluded=holdout_candidates(data)
    assert indices==[13] and excluded=={'invalid_pose':2}


def test_speed_and_future_motion_do_not_select_the_cohort(tmp_path):
    from slowfast_v11.adapter_data import make_adapter_plan
    plan,names,retained,index,data=plan_for(tmp_path)
    for item in retained:
        name=Path(item['path']).stem
        for frame in data[name]:
            frame['ego_dynamic_state']=[0.,0.,0.,0.]
            frame['ego2global_translation']=[0.,0.,0.]
        blob=pickle.dumps(data[name]);Path(item['path']).write_bytes(blob)
        item['sha256']=hashlib.sha256(blob).hexdigest()
    newer=make_adapter_plan(retained,index,[names[0]],sensor_root=plan['sensor_root'])
    assert [r['token'] for r in plan['rows']['val']]==[r['token'] for r in newer['rows']['val']]


def test_rejects_insufficient_first_logs_and_changed_metadata(tmp_path):
    from slowfast_v11.adapter_data import make_adapter_plan
    names,retained,index,data=fixture(tmp_path)
    with pytest.raises(ValueError,match='eight') as failure:
        make_adapter_plan(retained[:7],index,[],sensor_root=tmp_path/'s')
    assert len(failure.value.selection_audit['eligible_names'])==7
    assert failure.value.selection_audit['required_recording_groups']==8
    assert failure.value.selection_audit['cohort_frozen'] is False
    retained[0]['sha256']='bad'
    with pytest.raises(ValueError,match='SHA256'):
        make_adapter_plan(retained,index,[],sensor_root=tmp_path/'s')


def test_freezes_plan_and_tokens_and_refuses_modified_plan(tmp_path):
    from slowfast_v11.adapter_data import freeze_adapter_plan
    plan,*_=plan_for(tmp_path); out=tmp_path/'out'
    freeze_adapter_plan(plan,out)
    assert len(json.loads((out/'frozen_tokens.json').read_text())['tokens'])==256
    freeze_adapter_plan(plan,out)
    changed=copy.deepcopy(plan);changed['rows']['val'][0]['token']='changed'
    with pytest.raises(ValueError,match='identity'):
        freeze_adapter_plan(changed,out)


def test_authenticated_prefix_copy_is_private_and_refuses_changed_source(tmp_path):
    from slowfast_v11.adapter_data import copy_authenticated_prefix
    source=tmp_path/'old.gz';source.write_bytes(b'old-prefix');dest=tmp_path/'new.gz'
    record=copy_authenticated_prefix(source,dest,expected_bytes=10,expected_sha256=hashlib.sha256(b'old-prefix').hexdigest())
    assert record['sha256']==hashlib.sha256(b'old-prefix').hexdigest()
    with dest.open('ab') as stream:stream.write(b'extension')
    assert source.read_bytes()==b'old-prefix' and source.stat().st_ino!=dest.stat().st_ino
    source.write_bytes(b'bad-prefix')
    with pytest.raises(ValueError,match='SHA256'):
        copy_authenticated_prefix(source,dest,expected_bytes=10,expected_sha256=record['sha256'])


class Response(io.BytesIO):
    status=206
    def __init__(self,blob,start,end):
        super().__init__(blob[start:end+1]);self.headers={'Content-Range':f'bytes {start}-{end}/1000000000'}
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def test_metadata_extension_accounts_failed_partial_bodies_and_hashes(tmp_path,monkeypatch):
    import slowfast_v11.adapter_data as module
    monkeypatch.setattr(module,'BASE_PREFIX_BYTES',10)
    monkeypatch.setattr(module,'METADATA_PREFIX_CAP',30)
    monkeypatch.setattr(module,'METADATA_NEW_CAP',20)
    monkeypatch.setattr(module,'CAMERA_RESERVATION',40)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789');reports=[]
    class Broken(Response):
        def read(self,size=-1):
            if self.tell():raise OSError('interrupted fixture')
            return super().read(4)
    def opener(request,timeout):return Broken(b'0123456789abcdefghijklmnopqrst',10,29)
    budget=ByteBudget(100)
    with pytest.raises(OSError,match='interrupted'):
        module.extend_metadata_prefix(prefix,budget,opener=opener,on_request=reports.append)
    assert budget.used==4 and prefix.read_bytes()==b'0123456789abcd'
    assert len(reports)==1 and reports[0]['response_body_bytes']==4
    assert reports[0]['response_body_sha256']==hashlib.sha256(b'abcd').hexdigest()
    assert reports[0]['error'] and reports[0]['full_archive_verified'] is False


def test_metadata_reservation_fails_before_network(tmp_path,monkeypatch):
    import slowfast_v11.adapter_data as module
    monkeypatch.setattr(module,'BASE_PREFIX_BYTES',10)
    monkeypatch.setattr(module,'METADATA_PREFIX_CAP',30)
    monkeypatch.setattr(module,'METADATA_NEW_CAP',20)
    monkeypatch.setattr(module,'CAMERA_RESERVATION',40)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789');calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('reserve first')
    with pytest.raises(ValueError,match='reserv'):
        module.extend_metadata_prefix(prefix,ByteBudget(59),opener=forbidden)
    assert not calls


def test_metadata_preserves_bytes_attached_to_incomplete_read_errors(tmp_path,monkeypatch):
    import http.client
    import slowfast_v11.adapter_data as module
    monkeypatch.setattr(module,'BASE_PREFIX_BYTES',10)
    monkeypatch.setattr(module,'METADATA_PREFIX_CAP',30)
    monkeypatch.setattr(module,'METADATA_NEW_CAP',20)
    monkeypatch.setattr(module,'CAMERA_RESERVATION',40)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789');reports=[]
    class Incomplete(Response):
        def read(self,size=-1):raise http.client.IncompleteRead(b'abcd',16)
    def opener(request,timeout):return Incomplete(b'',10,29)
    budget=ByteBudget(100)
    with pytest.raises(http.client.IncompleteRead):
        module.extend_metadata_prefix(prefix,budget,opener=opener,on_request=reports.append)
    assert budget.used==4 and prefix.read_bytes()==b'0123456789abcd'
    assert reports[0]['response_body_bytes']==4
    assert reports[0]['response_body_sha256']==hashlib.sha256(b'abcd').hexdigest()


def test_holdout_writer_preserves_original_training_and_stats(tmp_path):
    from slowfast_v11.adapter_data import write_adapter_prepared
    plan,*_=plan_for(tmp_path); out=tmp_path/'out';out.mkdir()
    (out/'stats.json').write_text('original-stats');(out/'train.jsonl').write_text('original-train')
    with pytest.raises(ValueError,match='Missing'):
        write_adapter_prepared(plan,out,expected_size=(12,8))
    for image in plan['required_images']:
        path=Path(image);path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(12,8)).save(path)
    result=write_adapter_prepared(plan,out,expected_size=(12,8))
    assert result['counts']=={'holdout':256}
    assert len((out/'holdout.jsonl').read_text().splitlines())==256
    assert (out/'stats.json').read_text()=='original-stats'
    assert (out/'train.jsonl').read_text()=='original-train'
    assert json.loads((out/'source_audit.json').read_text())['full_archives_verified'] is False


def test_request_ledger_preserves_all_failures_and_exact_new_cumulative_totals(tmp_path):
    from tools.prepare_adapter_study import RequestLedger
    ledger=RequestLedger(tmp_path, inherited_bytes=123, inherited_budget={'path':'old','sha256':'oldhash'})
    for size, kind, error in [(4,'metadata','interrupted'),(0,'camera','503'),(7,'camera',None)]:
        ledger.append(dict(request_id=f'{size}-{kind}',kind=kind,response_body_bytes=size,
            response_body_sha256=hashlib.sha256(b'x'*size).hexdigest(),error=error))
    state=ledger.state('failed')
    assert state['used']==11 and state['new_response_body_bytes']==11
    assert state['cumulative_response_body_bytes']==134
    assert state['metadata_response_body_bytes']==4 and state['camera_response_body_bytes']==7
    assert state['metadata_requests']==1 and state['camera_requests']==2
    resumed=RequestLedger(tmp_path,inherited_bytes=123,inherited_budget={'path':'old','sha256':'oldhash'})
    assert resumed.state('failed')['used']==11 and len(resumed.records)==3
    assert json.loads((tmp_path/'network_budget.json').read_text())['used']==11


def test_ledger_rejects_cap_overruns_without_discarding_failed_attempt(tmp_path,monkeypatch):
    import tools.prepare_adapter_study as module
    monkeypatch.setattr(module,'NEW_NETWORK_CAP',10)
    ledger=module.RequestLedger(tmp_path,inherited_bytes=100,inherited_budget={})
    ledger.append(dict(request_id='one',kind='metadata',response_body_bytes=9,
        response_body_sha256=hashlib.sha256(b'x'*9).hexdigest(),error='failed'))
    with pytest.raises(ValueError,match='cap'):
        ledger.append(dict(request_id='two',kind='camera',response_body_bytes=2,
            response_body_sha256=hashlib.sha256(b'xx').hexdigest(),error='failed'))
    assert len((tmp_path/'network_requests.jsonl').read_text().splitlines())==2


def test_explicit_metadata_amendment_preserves_previous_evidence_before_extension(tmp_path,monkeypatch):
    import tools.prepare_adapter_study as module
    monkeypatch.setattr(module,'METADATA_PREFIX_CAP',10)
    prefix=tmp_path/'prefix.gz';prefix.write_bytes(b'0123456789')
    old=dict(prefix_bytes=10,prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest(),complete=True)
    failure=dict(required_recording_groups=8,available_recording_groups=7,cohort_frozen=False)
    (tmp_path/'metadata_acquisition.json').write_text(json.dumps(old))
    (tmp_path/'selection_failure.json').write_text(json.dumps(failure))
    amendment=module.record_metadata_amendment(tmp_path,prefix,requested_prefix_cap=320_000_000)
    assert amendment['previous_prefix_sha256']==old['prefix_sha256']
    assert amendment['new_total_response_body_cap_bytes']==2_000_000_000
    assert amendment['requested_metadata_prefix_cap_bytes']==320_000_000
    assert amendment['requested_metadata_new_body_cap_bytes']==200_000_000
    assert amendment['selection_failure_sha256']==hashlib.sha256((tmp_path/'selection_failure.json').read_bytes()).hexdigest()
    assert amendment['event_timestamp_utc'] and amendment['explicit_cli_flag']=='--metadata-prefix-cap 320000000'
    assert json.loads((tmp_path/'metadata_acquisition_10.json').read_text())==old
    assert json.loads((tmp_path/'selection_failure_10.json').read_text())==failure


def test_ledger_amendment_requires_explicit_new_limit_and_preserves_total_cap(tmp_path):
    from tools.prepare_adapter_study import RequestLedger
    ledger=RequestLedger(tmp_path,inherited_bytes=123,inherited_budget={},metadata_prefix_cap=320_000_000)
    state=ledger.state('amended')
    assert state['metadata_new_cap_bytes']==200_000_000
    assert state['metadata_prefix_cap_bytes']==320_000_000 and state['limit']==2_000_000_000
    before=(tmp_path/'network_budget.json').read_bytes()
    with pytest.raises(ValueError,match='explicit'):
        RequestLedger(tmp_path,inherited_bytes=123,inherited_budget={})
    assert (tmp_path/'network_budget.json').read_bytes()==before
