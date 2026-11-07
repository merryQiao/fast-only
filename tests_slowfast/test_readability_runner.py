from dataclasses import replace
import pytest
import torch
from slowfast_v11.protocol import LayerKV,KVSnapshot


def api():
    from tools import diagnose_slowfast_readability as module
    return module


def sample(value):
    x=torch.full((1,2,4,3),float(value))
    snap=KVSnapshot(tuple(LayerKV(x.clone(),x.clone()*2) for _ in range(6)),
                    torch.arange(4).view(1,1,4).expand(3,1,4),torch.ones(1,4,dtype=torch.bool))
    return snap,torch.ones(1,8,7)*value,{'visual':torch.tensor([1,1,0,0],dtype=torch.bool),'suffix':torch.tensor([0,0,1,1],dtype=torch.bool)}


def test_cache_keeps_raw_head_concatenation_and_target_layout_under_source_swap():
    a,features,groups=sample(1);b,other,_=sample(9)
    cache=api().FeatureCache(['a','b'],a,features,groups)
    cache.put('a',a,features,groups);cache.put('b',b,other,groups)
    batch,f=cache.batch(['a'],['b'])
    assert batch['keys'].shape==(1,6,4,6)
    assert torch.equal(batch['keys'],torch.ones_like(batch['keys'])*9)
    assert torch.equal(batch['values'],torch.ones_like(batch['values'])*18)
    assert torch.equal(f,other) and torch.equal(batch['positions'],a.position_ids)
    with pytest.raises(ValueError,match='layout'):
        cache2=api().FeatureCache(['c'],a,features,groups)
        cache2.put('c',replace(a,position_ids=a.position_ids+1),features,groups)


def test_train_scales_exclude_val_and_refuse_val_rows():
    a,features,groups=sample(2);b,other,_=sample(1e6)
    cache=api().FeatureCache(['a','b'],a,features,groups)
    cache.put('a',a,features,groups);cache.put('b',b,other,groups)
    train=[dict(token='a',split='train',cohort='train')]
    k,v=api().training_rms(cache,train)
    assert torch.equal(k,torch.ones(6)*2) and torch.equal(v,torch.ones(6)*4)
    with pytest.raises(ValueError,match='TRAIN'):
        api().training_rms(cache,[dict(token='b',split='val',cohort='new_val')])
