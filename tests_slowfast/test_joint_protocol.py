"""Predeclared balanced TRAIN-only live-KV experiment contract."""
from collections import Counter
import copy
from pathlib import Path

import pytest
import yaml

from slowfast_v11.joint_protocol import (
    validate_joint_config,balanced_plan,balanced_source_order,fresh_probe_steps,
)

CONFIG = Path(__file__).resolve().parents[1]/'configs/slowfast_v11_joint.yaml'


def config():
    return yaml.safe_load(CONFIG.read_text())


def rows():
    return [dict(token=f'm{i}',route='moving-log',split='train',ego=[1.+i,.1,0.,0.],
                 trajectory_gt=object()) for i in range(4)] + [
        dict(token=f'l{i}',route='slow-log',split='train',ego=[.01*i,0.,0.,0.],
             trajectory_gt=object()) for i in range(8)]


def test_checked_in_config_has_new_schema_and_predeclared_fixed_protocol():
    c=config()
    assert validate_joint_config(c) is None
    assert c['schema_version']=='slowfast_v11_joint'
    assert c['train_seed']==42 and 'seed' not in c
    assert c['stages']=={'slow_adapt':240,'fast_consumer':240}
    assert c['arms']==['no_kv','detached','joint']
    assert c['optimizer']['clip_per_model'] is True


@pytest.mark.parametrize('path,value',[
    ('schema_version','slowfast_v11_validation'),('train_seed',43),('train_seed',True),
    ('stages.slow_adapt',241),('stages.fast_consumer',120),('missing_fraction',.5),
    ('data.moving_fraction',.25),('data.moving_speed_mps',.5),('data.max_train_rows',13),
    ('data.ages_s',[0.,.5]),('arms',['joint','detached','no_kv']),('loss.slow_weight',0.),
    ('model.lora_rank',8),('model.lora_dropout',.1),('model.position_offset',128),
    ('processor.brightness',.1),('optimizer.clip_per_model',False),
    ('optimizer.schedule','cosine'),('optimizer.grad_clip',2.),
    ('reproducibility.warn_only',True),('reproducibility.deterministic_algorithms',False),
    ('reproducibility.cublas_workspace_config',':16:8'),('reproducibility.allow_tf32',True),
    ('feedback_policy','expert_trajectory'),('gradient_probe_count',3),
    ('train_curve_steps',[60,120,180,240]),('fast.device','cuda:1'),
    ('initial_checkpoint',''),('output',''),
])
def test_config_rejects_policy_drift(path,value):
    c=config();parts=path.split('.');node=c
    for part in parts[:-1]:node=node[part]
    node[parts[-1]]=value
    with pytest.raises(ValueError):validate_joint_config(c)


@pytest.mark.parametrize('mutation',['seed','extra_optimizer','missing_policy','missing_data','bool_steps'])
def test_config_rejects_unknown_missing_and_wrong_type_fields(mutation):
    c=config()
    if mutation=='seed':c['seed']=42
    if mutation=='extra_optimizer':c['optimizer']['single_clip']=True
    if mutation=='missing_policy':c.pop('reproducibility')
    if mutation=='missing_data':c['data'].pop('val_manifest')
    if mutation=='bool_steps':c['stages']['slow_adapt']=True
    with pytest.raises(ValueError):validate_joint_config(c)


def test_external_verified_paths_can_be_overridden_without_policy_changes(tmp_path):
    c=config()
    for field in ('train_manifest','val_manifest','stats','sensor_root'):c['data'][field]=str(tmp_path/field)
    for name in ('fast','slow'):c[name]['model_path']=str(tmp_path/name)
    c['initial_checkpoint']=str(tmp_path/'parent.pt');c['output']=str(tmp_path/'new')
    validate_joint_config(c)


def test_balanced_plan_has_exact_per_stratum_cache_counts_and_all_shared_targets():
    data=rows();c=config();plan=balanced_plan(data,[],c)
    assert len(plan)==240
    assert Counter(p['stratum'] for p in plan)=={'moving':120,'low_speed':120}
    assert Counter(p['mode'] for p in plan)=={'none':72,'fresh':168}
    for stratum in ('moving','low_speed'):
        assert Counter(p['mode'] for p in plan if p['stratum']==stratum)=={'none':36,'fresh':84}
    assert all(p['source_token']==p['target_token'] and p['split']=='train' and p['age_s']==0. for p in plan)
    assert all(set(p)=={'target_token','source_token','split','stratum','mode','age_s'} for p in plan)
    assert [p['target_token'] for p in plan]==balanced_source_order(data,240,42)
    assert plan==balanced_plan(data,[],c)


def test_target_cycles_do_not_replace_samples_within_either_stratum():
    data=rows();order=balanced_source_order(data,40,42)
    for prefix,pool_size in [('m',4),('l',8)]:
        selected=[token for token in order if token.startswith(prefix)]
        assert len(selected)==20
        for i in range(0,len(selected),pool_size):
            chunk=selected[i:i+pool_size]
            assert len(chunk)==len(set(chunk))
            if len(chunk)==pool_size:assert set(chunk)=={f'{prefix}{i}' for i in range(pool_size)}
    assert order==balanced_source_order(data,40,42)
    assert order!=balanced_source_order(data,40,43)


def test_plan_ignores_future_targets_pairs_and_row_iteration_order():
    data=rows();c=config();expected=balanced_plan(data,[],c)
    mutated=[{**r,'trajectory_gt':float('nan'),'future_images':object()} for r in reversed(data)]
    assert balanced_plan(mutated,[{'future':object(),'age_s':999}],c)==expected
    assert balanced_source_order(mutated,20,42)==balanced_source_order(data,20,42)


@pytest.mark.parametrize('mutation',['val','test','all_moving','all_low','nan','duplicate','cross_split_log'])
def test_sampling_rejects_split_leakage_bad_ego_and_empty_strata(mutation):
    data=rows()
    if mutation in ('val','test'):data[0]['split']=mutation
    if mutation=='all_moving':
        for r in data:r['ego']=[2.,0.,0.,0.]
    if mutation=='all_low':
        for r in data:r['ego']=[0.,0.,0.,0.]
    if mutation=='nan':data[0]['ego'][0]=float('nan')
    if mutation=='duplicate':data.append(data[0])
    if mutation=='cross_split_log':data.append(dict(data[0],token='val',split='val'))
    with pytest.raises(ValueError):balanced_source_order(data,20,42)
    with pytest.raises(ValueError):balanced_plan(data,[],config())


@pytest.mark.parametrize('steps',[0,1,21,-20,True,20.])
def test_source_order_rejects_invalid_or_unbalanced_step_count(steps):
    with pytest.raises(ValueError):balanced_source_order(rows(),steps,42)


def test_speed_threshold_uses_current_velocity_and_includes_exact_boundary():
    data=rows();data[0]['ego']=[.8,.6,99.,-99.]
    plan=balanced_plan(data,[],config())
    assert all(p['stratum']=='moving' for p in plan if p['target_token']=='m0')


def test_fresh_probes_are_fixed_quartiles_of_fresh_steps_including_late_replay_boundary():
    plan=balanced_plan(rows(),[],config())
    fresh=[i for i,p in enumerate(plan,1) if p['mode']=='fresh']
    probes=fresh_probe_steps(plan,4)
    assert probes==[fresh[41],fresh[83],fresh[125],fresh[167]]
    assert len(set(probes))==4 and probes==sorted(probes)
    assert all(plan[step-1]['mode']=='fresh' for step in probes)
    assert fresh_probe_steps(plan,1)==[fresh[-1]]


@pytest.mark.parametrize('count',[0,-1,True,4.,241])
def test_probes_reject_invalid_count_or_insufficient_fresh_steps(count):
    with pytest.raises(ValueError):fresh_probe_steps(balanced_plan(rows(),[],config()),count)
