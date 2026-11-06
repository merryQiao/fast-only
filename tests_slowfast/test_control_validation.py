"""The matched-budget branch must use the exact saved TRAIN target schedule."""
import copy

import pytest

from tools.control_slowfast_v11 import control_targets, target_order_digest


def fixture():
    rows=[{'token':f't{i}','split':'train','route':'train-log'} for i in range(12)]
    rows.append({'token':'v0','split':'val','route':'val-log'})
    modes=['none']*3+['fresh']*4+['stale']*3
    plan=[]
    for i in range(120):
        token=f't{(i*7)%12}'
        mode=modes[i%10]
        plan.append(dict(target_token=token,split='train',mode=mode,
                         source_token='historical' if mode=='stale' else token,
                         age_s=.5 if mode=='stale' else 0.))
    return rows,plan


def test_control_preserves_saved_order_including_repeated_stale_targets():
    rows,plan=fixture();before=copy.deepcopy(plan)
    targets=control_targets(plan,rows)
    assert len(targets)==120
    assert targets[:12]==('t0','t7','t2','t9','t4','t11','t6','t1','t8','t3','t10','t5')
    assert targets[12:24]==targets[:12]
    assert plan==before
    assert target_order_digest(targets)==target_order_digest(list(targets))
    swapped=list(targets);swapped[0],swapped[1]=swapped[1],swapped[0]
    assert target_order_digest(swapped)!=target_order_digest(targets)


@pytest.mark.parametrize('length',[0,119,121,240])
def test_control_rejects_shortened_or_extended_update_budget(length):
    rows,plan=fixture();plan=(plan*2)[:length]
    with pytest.raises(ValueError,match='120'):control_targets(plan,rows)


@pytest.mark.parametrize('mutation',['val_target','missing_target','nontrain_step','unknown_mode',
    'wrong_block_mode','missing_mode','future_age','fresh_wrong_source','stale_without_source'])
def test_control_rejects_leakage_and_corrupted_consumer_schedule(mutation):
    rows,plan=fixture()
    if mutation=='val_target':plan[0]['target_token']='v0'
    if mutation=='missing_target':plan[0]['target_token']='not-present'
    if mutation=='nontrain_step':plan[0]['split']='val'
    if mutation=='unknown_mode':plan[0]['mode']='shuffled'
    if mutation=='wrong_block_mode':plan[0]['mode']='fresh'
    if mutation=='missing_mode':plan[0].pop('mode')
    if mutation=='future_age':plan[7]['age_s']=-.5
    if mutation=='fresh_wrong_source':plan[3]['source_token']='historical'
    if mutation=='stale_without_source':plan[7]['source_token']=None
    with pytest.raises(ValueError):control_targets(plan,rows)


def test_duplicate_row_identity_is_not_accepted_as_training_membership():
    rows,plan=fixture();rows.append(dict(rows[0],split='val'))
    with pytest.raises(ValueError):control_targets(plan,rows)


def test_cpu_control_uses_warmup_teacher_and_runs_only_saved_train_targets(tmp_path,monkeypatch):
    """Exercise the whole control branch without a pretrained model or CUDA."""
    import json
    import sys
    from types import SimpleNamespace
    import torch
    from slowfast_v11.validation import canonical_state, state_digest
    from tools.control_slowfast_v11 import run_control

    rows,plan=fixture()
    for row in rows:
        row.update(ego=[1.,0.,0.,0.],command=1,trajectory_gt=[[1.,0.,0.]]*8)
    class TinyFast(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Module();self.backbone.visual=torch.nn.Identity()
            self.head=torch.nn.Linear(4,24)
        def forward(self,native,ego,command):
            return self.head(ego).view(-1,8,3)
    class ForbiddenSlow(torch.nn.Module):
        def forward(self,*args,**kwargs):
            raise AssertionError('The matched control cannot run Slow')
    torch.manual_seed(42)
    teacher=canonical_state(TinyFast())
    poison={name:torch.full_like(value,1000.) for name,value in teacher.items()}
    identities={'val_manifest':'fixture'}
    config={'seed':42,'fast':{'device':'cpu'},'slow':{'device':'cpu'},
        'stages':{'fast_warmup':80,'fast_consumer':120},
        'optimizer':dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01,grad_clip=1.,schedule='constant')}
    checkpoint=dict(schema='slowfast_validation_v1',stage='fast_consumer',step=120,
        config=config,feedback_producer=teacher,feedback_producer_digest=state_digest(teacher),
        fast=poison,assets={},data_identities=identities,sampler={'plan':plan,'next_index':120})
    source=tmp_path/'main.pt';torch.save(checkpoint,source)
    seen=[]
    class FakeExperiment:
        def __init__(self,c):
            self.c=c;self.rows=rows;self.train=rows[:-1];self.plan=plan
            self.fast=TinyFast();self.slow=ForbiddenSlow();self.assets={}
            self.data_identities=identities;self.audits={}
        def batch(self,token,which):
            seen.append((token,self.fast.training))
            return {'native':{},'ego':torch.tensor([[1.,0.,0.,0.]]),'command':torch.tensor([1])}
        def target(self,token,which):
            assert token!='v0', 'VAL cannot be used for a control optimizer update'
            return torch.tensor([[[1.,0.,0.]]*8])
        def log(self,message):
            pass
    monkeypatch.setitem(sys.modules,'tools.validate_slowfast_v11',SimpleNamespace(Experiment=FakeExperiment))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    output=tmp_path/'control'
    result=run_control(source,output)
    assert result['initial_fast_digest']==state_digest(teacher)
    assert result['final_fast_digest'] not in (state_digest(teacher),state_digest(poison))
    assert [token for token,training in seen if training]==list(control_targets(plan,rows))
    assert len([token for token,training in seen if not training])==len(rows)
    assert result['training_modes']=={'none':120}
    assert result['updates']['control_total_fast']==result['updates']['main_total_fast']==200
    assert result['counts']=={'train':12,'val':1}
    assert len(result['predictions'])==13
    training=[json.loads(line) for line in (output/'control_train.jsonl').read_text().splitlines()]
    assert len(training)==120 and training[-1]['loss']<training[0]['loss']
    with pytest.raises(FileExistsError):run_control(source,output)
