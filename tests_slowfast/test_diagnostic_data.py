"""Causal raw observations and real historical-cache diagnostic eligibility."""
import copy
import hashlib
import math
import pickle

from PIL import Image
import pytest
import torch

from slowfast_v11.diagnostic_data import (
    observation_at, build_pairs, deterministic_mismatches, load_frames_by_log,
)
from slowfast_v11.protocol import relative_source_pose


def frames(tmp_path, log='a', count=16):
    result = []
    for i in range(count):
        relative = f'{log}/{i}.jpg'
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 8), (i, 2, 3)).save(path)
        result.append(dict(token=f'{log}-{i}', timestamp=1_000_000+i*500_000,
            log_name=log, ego2global_translation=[10.,float(i),0.],
            ego2global_rotation=[math.cos(math.pi/4),0.,0.,math.sin(math.pi/4)],
            ego_dynamic_state=[float(i),2.,3.,4.], driving_command=[0,1,0,0],
            cams={'CAM_F0':{'data_path':relative}}, annotations={'future':'unused'}))
    return result


def row(data, i, split='train'):
    return dict(token=data[i]['token'], frame=i, route=data[i]['log_name'], split=split,
                timestamp_us=data[i]['timestamp'], command=1, trajectory_gt='NEVER READ')


STATS = {'source_split':'train','ego_mean':[1.,1.,1.,1.], 'ego_std':[2.,2.,2.,2.]}


def test_observation_is_unbatched_native_resolution_and_has_only_causal_fields(tmp_path):
    data=frames(tmp_path)
    sample=observation_at(data, 6, tmp_path, STATS)
    assert set(sample)=={'images','ego','command','metadata'}
    assert [im.size for im in sample['images']]==[(12,8)]*4
    assert sample['ego'].shape==(4,)
    torch.testing.assert_close(sample['ego'],torch.tensor([2.5,.5,1.,1.5]))
    assert sample['command'].shape==() and sample['command'].dtype==torch.long
    assert sample['command'].item()==1
    assert sample['metadata']['pose']==pytest.approx((10.,6.,math.pi/2))
    assert sample['metadata']['frame']==6
    assert sample['metadata']['timestamp_us']==4_000_000
    assert 'trajectory_gt' not in sample['metadata']


def test_future_annotation_and_image_mutation_cannot_change_source_inputs(tmp_path):
    data=frames(tmp_path)
    before=observation_at(data,6,tmp_path,STATS)
    for i,item in enumerate(data):
        item['annotations']={'future':object()}
        if i>6:
            item.update(timestamp=-1,ego2global_translation=None,ego_dynamic_state=None)
            item['cams']['CAM_F0']['data_path']='does/not/exist.jpg'
    after=observation_at(data,6,tmp_path,STATS)
    assert before['metadata']==after['metadata']
    torch.testing.assert_close(before['ego'],after['ego'])
    assert [im.tobytes() for im in before['images']]==[im.tobytes() for im in after['images']]


def test_pairs_use_actual_source_pose_in_current_frame_and_real_timestamp_age(tmp_path):
    data=frames(tmp_path);data[6]['timestamp']+=300
    pairs,report=build_pairs([row(data,6)], {'a':data},tmp_path)
    assert len(pairs)==4
    stale=next(p for p in pairs if p.age_s==1.5)
    assert stale.source_index==3 and stale.source_token=='a-3'
    assert stale.actual_age_s==pytest.approx(1.5003)
    assert relative_source_pose(stale.source_pose,stale.current_pose)==pytest.approx((-3.,0.,0.))
    assert report['by_age']['1.5']['eligible']==1
    assert report['by_age']['1.5']['excluded']=={}


def test_absent_history_is_counted_without_fresh_fallback(tmp_path):
    data=frames(tmp_path)
    pairs,report=build_pairs([row(data,3)],{'a':data},tmp_path)
    assert [(p.age_s,p.source_index) for p in pairs]==[(0.,3)]
    for age in ('.5','1.0','1.5'):
        entry=report['by_age'][str(float(age))]
        assert entry['eligible']==0 and entry['excluded']=={'missing_history':1}


@pytest.mark.parametrize('mutation,reason',[
    ('remove','missing_image'),('duplicate','duplicate_images'),('time','timestamp_gap'),
    ('navigation','navigation_mismatch'),('log','log_mismatch'),('path','invalid_image_path'),
])
def test_ineligible_history_has_explicit_reason(tmp_path,mutation,reason):
    data=frames(tmp_path)
    if mutation=='remove':(tmp_path/'a/3.jpg').unlink()
    if mutation=='duplicate':data[3]['cams']['CAM_F0']['data_path']='a/4.jpg'
    if mutation=='time':data[3]['timestamp']+=10_000
    if mutation=='navigation':data[3]['driving_command']=[1,0,0,0]
    if mutation=='log':data[3]['log_name']='other'
    if mutation=='path':data[3]['cams']['CAM_F0']['data_path']='../outside.jpg'
    pairs,report=build_pairs([row(data,6)],{'a':data},tmp_path,ages_s=(0.,))
    assert pairs==[] and report['by_age']['0.0']['excluded']=={reason:1}


def test_command_switch_between_source_and_target_excludes_cache(tmp_path):
    data=frames(tmp_path);data[5]['driving_command']=[1,0,0,0]
    pairs,report=build_pairs([row(data,8)],{'a':data},tmp_path,ages_s=(1.5,))
    assert not pairs
    assert report['by_age']['1.5']['excluded']=={'navigation_mismatch':1}


def test_pair_inputs_are_invariant_to_all_labels_and_future_fields(tmp_path):
    data=frames(tmp_path);target=row(data,6)
    before=build_pairs([target],{'a':data},tmp_path)
    target['trajectory_gt']=object()
    for i,item in enumerate(data):
        item['annotations']=object()
        if i>6:item.clear()
    assert before==build_pairs([target],{'a':data},tmp_path)


def test_row_identity_and_total_age_are_checked(tmp_path):
    data=frames(tmp_path);target=row(data,7);target['token']='wrong'
    pairs,report=build_pairs([target],{'a':data},tmp_path,ages_s=(0.,))
    assert not pairs and report['by_age']['0.0']['excluded']=={'row_mismatch':1}
    data=frames(tmp_path)
    for i in range(4,8):data[i]['timestamp']+=(i-3)*900
    pairs,report=build_pairs([row(data,7)],{'a':data},tmp_path,ages_s=(1.5,))
    assert not pairs and report['by_age']['1.5']['excluded']=={'age_mismatch':1}


@pytest.mark.parametrize('change', ['stats','ego','command'])
def test_observations_reject_malformed_numeric_inputs(tmp_path,change):
    data=frames(tmp_path);stats=copy.deepcopy(STATS)
    if change=='stats':stats['ego_std'][0]=0.
    if change=='ego':data[6]['ego_dynamic_state'][0]=float('nan')
    if change=='command':data[6]['driving_command']=[0,1,1,0]
    with pytest.raises(ValueError):observation_at(data,6,tmp_path,stats)


def test_mismatch_is_deterministic_bijective_and_split_preserving(tmp_path):
    a=frames(tmp_path,'a');b=frames(tmp_path,'b');v=frames(tmp_path,'v')
    rows=[row(a,6),row(a,7),row(b,6),row(b,7),row(v,3,'val'),row(v,4,'val'),
          row(v,8,'val'),row(v,9,'val')]
    mapping,report=deterministic_mismatches(rows)
    assert (mapping,report)==deterministic_mismatches(list(reversed(rows)))
    assert len(mapping)==8 and set(mapping)==set(mapping.values())
    by_token={r['token']:r for r in rows}
    for target,source in mapping.items():
        t,s=by_token[target],by_token[source]
        assert t['split']==s['split'] and target!=source
        if t['split']=='train':assert t['route']!=s['route']
        else:assert abs(t['timestamp_us']-s['timestamp_us'])>=2_000_000
    assert report['train']['available'] and report['val']['available']


def test_impossible_mismatch_is_explicit_and_never_partial(tmp_path):
    data=frames(tmp_path)
    mapping,report=deterministic_mismatches([row(data,3,'val'),row(data,4,'val')])
    assert mapping=={}
    assert report['val']['available'] is False
    assert report['val']['reason']=='no_complete_derangement'


def test_no_test_access_and_no_route_split_leakage(tmp_path):
    data=frames(tmp_path)
    with pytest.raises(ValueError,match='split'):build_pairs([row(data,6,'test')],{'a':data},tmp_path)
    with pytest.raises(ValueError,match='split'):
        deterministic_mismatches([row(data,6),row(data,7,'val')])


def test_metadata_loader_verifies_hash_and_deduplicates_routes(tmp_path):
    data=frames(tmp_path);blob=pickle.dumps(data);path=tmp_path/'meta.pkl';path.write_bytes(blob)
    source={'metadata_path':str(path),'metadata_sha256':hashlib.sha256(blob).hexdigest()}
    rows=[dict(row(data,i),source=source) for i in (6,7)]
    loaded=load_frames_by_log(rows)
    assert list(loaded)==['a'] and loaded['a'][6]['token']=='a-6'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):load_frames_by_log(rows)
