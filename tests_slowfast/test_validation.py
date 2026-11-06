from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from slowfast_v11.model import FastWithSlowKV, SlowPrefill
from slowfast_v11.validation import (validate_config, set_stage, make_optimizer, sampling_plan,
    canonical_state, restore_weights, state_digest, paired_summary, perform_step)


def test_fixed_budget_config_rejects_formal_scale_and_silent_policy_changes():
    c=yaml.safe_load((Path(__file__).parents[1]/'configs/slowfast_v11_validation.yaml').read_text())
    validate_config(c)
    for section,key,value in [('stages','fast_consumer',8000),('processor','policy','resize'),
        ('model','position_offset',0),('data','ages_s',[0,0.05]),('optimizer','schedule','cosine')]:
        changed=deepcopy(c);changed[section][key]=value
        with pytest.raises(ValueError):validate_config(changed)


def test_sampling_schedule_never_uses_val_or_silently_falls_back():
    rows=[{'token':str(i),'split':'train'} for i in range(3)]
    pairs=[{'target_token':'0','source_token':'past','age_s':.5,'split':'train'}]
    modes=['none']*3+['fresh']*4+['stale']*3
    plan=sampling_plan(rows,pairs,20,42,modes)
    assert [x['mode'] for x in plan]==modes*2
    assert all(x['source_token']=='past' for x in plan if x['mode']=='stale')
    assert plan==sampling_plan(rows,pairs,20,42,modes)
    with pytest.raises(ValueError):sampling_plan(rows,[],20,42,modes)
    with pytest.raises(ValueError):sampling_plan(rows+[{'token':'v','split':'val'}],pairs,20,42,modes)


def test_stage_a_gradients_and_state_restore_do_not_depend_on_requires_grad(language_factory,native_input):
    slow=SlowPrefill(language_factory(64),lora_rank=2,lora_alpha=4,lora_dropout=0.,position_offset=128)
    set_stage(slow,'slow_inputs')
    assert not any(p.requires_grad for p in slow.backbone.parameters())
    result=slow(native_input,torch.ones(1,4),torch.tensor([1]))
    result.trajectory.square().mean().backward()
    assert slow.queries.grad is not None and slow.queries.grad.abs().sum()>0
    assert not any(p.grad is not None for p in slow.backbone.parameters())
    original=canonical_state(slow);digest=state_digest(original)
    set_stage(slow,'slow_lora')
    assert any(p.requires_grad for n,p in slow.backbone.named_parameters() if 'lora_' in n)
    assert not any(p.requires_grad for p in slow.backbone.visual.parameters())
    slow.requires_grad_(False)
    assert state_digest(canonical_state(slow))==digest
    with torch.no_grad():slow.queries.add_(1)
    restore_weights(slow,original)
    assert state_digest(canonical_state(slow))==digest
    with pytest.raises(ValueError):restore_weights(slow,{})


def test_optimizer_uses_declared_groups(language_factory):
    fast=FastWithSlowKV(language_factory(),lora_rank=2,lora_alpha=4,lora_dropout=0.,position_offset=128)
    set_stage(fast,'fast_warmup')
    opt=make_optimizer(fast,dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01))
    assert {group['lr'] for group in opt.param_groups}=={1e-4,5e-4}
    assert all(group['weight_decay']==.01 for group in opt.param_groups)


def test_real_tiny_optimizer_step_keeps_frozen_vision_and_updates_head(language_factory,native_input):
    fast=FastWithSlowKV(language_factory(),lora_rank=2,lora_alpha=4,lora_dropout=0.,position_offset=128)
    set_stage(fast,'fast_warmup')
    opt=make_optimizer(fast,dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01))
    before=fast.head[-1].weight.detach().clone()
    batch=dict(native=native_input,ego=torch.ones(1,4),command=torch.tensor([1]))
    result=perform_step(fast,opt,batch,torch.ones(1,8,3),1.)
    assert result['loss']>0 and result['grad_norm']>0
    assert not torch.equal(before,fast.head[-1].weight)
    assert all(p.grad is None for p in fast.backbone.visual.parameters())


def test_paired_reporting_uses_identical_tokens_for_stale_baseline():
    def record(token,mode,error):
        p=torch.zeros(8,3);p[:,0]=error
        return dict(split='val',stage='after_consumer',token=token,mode=mode,
            age_s=.5 if mode=='stale' else 0.,prediction=p.tolist(),target=torch.zeros(8,3).tolist(),speed_mps=2.)
    rows=[record('a','none',1),record('b','none',100),record('a','fresh',.5),
          record('b','fresh',1000),record('a','stale',2)]
    summary=paired_summary(rows)
    paired=summary['after_consumer']['val']['stale@0.5']['paired_no_kv']
    assert paired['samples']==1 and paired['ADE']==1
    assert summary['after_consumer']['val']['stale@0.5']['delta_ADE_vs_no_kv']==1
    result=summary['after_consumer']['val']['stale@0.5']
    assert result['paired_fresh']['samples']==1 and result['paired_fresh']['ADE']==.5
    assert result['delta_ADE_vs_fresh']==1.5
