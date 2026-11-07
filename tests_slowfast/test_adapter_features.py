import pytest
import torch
from slowfast_v11.protocol import LayerKV,KVSnapshot


def test_snapshot_inputs_preserve_raw_head_values_and_exact_layout_guard():
    from slowfast_v11.adapter_features import snapshot_inputs,check_layout
    k=torch.arange(24,dtype=torch.float32).reshape(1,2,4,3)
    snap=KVSnapshot(tuple(LayerKV(k,k+1) for _ in range(6)),torch.zeros(3,1,4,dtype=torch.long),torch.ones(1,4,dtype=torch.bool))
    groups={'image':torch.tensor([True,True,False,False]),'context':torch.tensor([False,False,True,True])}
    b=snapshot_inputs(snap,groups)
    assert b['keys'].shape==(1,6,4,6)
    assert torch.equal(b['keys'][0,0],k[0].transpose(0,1).flatten(1))
    template={n:b[n].clone() for n in ['positions','group_ids','valid']}
    check_layout(b,template)
    for key in template:
        bad={**template,key:template[key].clone()}
        if key=='valid':bad[key][0,0]=False
        else:bad[key].flatten()[0]+=1
        with pytest.raises(ValueError,match='layout'):check_layout(b,bad)


def test_holdout_extends_source_content_without_touching_training_or_targets():
    from slowfast_v11.adapter_features import extend_bank
    old=dict(tokens=['train'],index={'train':0},fast=torch.randn(1,8,4),baseline=torch.randn(1,8,3),head={},
        reader={m:torch.randn(1,8,6) for m in ['correct','reference','shuffled']},
        sources={m:{'train':'train'} for m in ['correct','reference','shuffled']})
    rows=[dict(token='a',route='drive-a',cohort='holdout',split='val'),dict(token='b',route='drive-b',cohort='holdout',split='val')]
    f=torch.randn(2,8,4);r=torch.randn(2,8,6);p=torch.randn(2,8,3)
    out=extend_bank(old,rows,f,r,p,{'a':'b','b':'a'})
    assert torch.equal(out['fast'][0],old['fast'][0]) and torch.equal(out['fast'][1:],f)
    assert torch.equal(out['reader']['reference'][1:],old['reader']['reference'].expand(2,-1,-1))
    assert torch.equal(out['reader']['shuffled'][1:],r.flip(0))
    assert 'target' not in out and 'trajectory_gt' not in out
    with pytest.raises(ValueError):extend_bank(old,[dict(rows[0],split='train'),rows[1]],f,r,p,{'a':'b','b':'a'})
    with pytest.raises(ValueError):extend_bank(old,rows,f,r,p,{'a':'a','b':'a'})
