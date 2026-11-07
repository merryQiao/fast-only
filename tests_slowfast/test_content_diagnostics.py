from dataclasses import replace
import copy
import pytest
import torch

from slowfast_v11 import attention
from slowfast_v11.protocol import LayerKV, KVSnapshot


def api():
    import slowfast_v11.content_diagnostics as module
    return module


def snapshot(length=8):
    torch.manual_seed(14)
    layers=tuple(LayerKV(torch.randn(1,1,length,4),torch.randn(1,1,length,4)) for _ in range(6))
    return KVSnapshot(layers,torch.arange(length).view(1,1,-1).expand(3,1,-1),torch.ones(1,length,dtype=torch.bool))


def row(token,route,command,speed,cohort='new_val'):
    return dict(token=token,route=route,command=command,ego=[speed,0.,0.,0.],
                split='train' if cohort=='train' else 'val',cohort=cohort)


def test_contrast_donors_are_cohort_local_cross_log_and_gt_independent():
    rows=[row('a','one',1,5),row('b','two',1,20),row('c','two',0,4),
          row('d','three',0,10,'legacy_val'),row('e','three',0,1,'legacy_val'),
          row('f','train',2,100,'train')]
    mapping=api().contrast_donors(rows)
    assert mapping=={'a':'c','b':'a','c':'a','d':'e','e':'d'}
    changed=copy.deepcopy(rows)
    for r in changed:r.update(trajectory_gt=[[100,5,1]]*8,model_error=-999)
    assert api().contrast_donors(list(reversed(changed)))==mapping


def test_contrast_donors_fail_instead_of_cross_cohort_fallback():
    with pytest.raises(ValueError,match='donor'):
        api().contrast_donors([row('a','one',1,1)])


def test_transplant_uses_all_six_donor_layers_retains_layout_and_owns_zero_values():
    a=snapshot();b=replace(a,layers=tuple(LayerKV(z.key+2,z.value+3) for z in a.layers))
    for zero in [False,True]:
        result=api().transplant_snapshot(a,b,zero_values=zero)
        assert torch.equal(result.position_ids,a.position_ids)
        assert torch.equal(result.valid_mask,a.valid_mask)
        for old,new,donor in zip(a.layers,result.layers,b.layers):
            assert torch.equal(new.key,donor.key)
            assert torch.equal(new.value,torch.zeros_like(donor.value) if zero else donor.value)
            assert torch.equal(old.value+3,donor.value)
    with pytest.raises(ValueError,match='layout'):
        api().transplant_snapshot(a,replace(b,position_ids=b.position_ids+1))
    with pytest.raises(ValueError,match='layout'):
        api().transplant_snapshot(a,snapshot(9))


def test_groups_cover_each_native_image_and_suffix_exactly():
    native=dict(input_ids=torch.tensor([[7,91,91,8,91,91,8,91,91,8,91,91,9]]),
                attention_mask=torch.ones(1,13,dtype=torch.long),image_grid_thw=torch.tensor([[1,2,4]]*4))
    groups=api().source_groups(native,91,2)
    assert [int(groups[f'image_{i}'].sum()) for i in range(4)]==[2]*4
    assert int(groups['feedback'].sum())==8 and int(groups['planning_queries'].sum())==8
    assert torch.equal(torch.stack(list(groups.values())).sum(0),torch.ones(31,dtype=torch.long))


def test_difference_reports_replaced_values_for_every_layer_and_group():
    a=snapshot();b=replace(a,layers=tuple(LayerKV(z.key+1,z.value*2) for z in a.layers))
    groups={'first':torch.tensor([1,1,0,0,0,0,0,0],dtype=torch.bool),
            'rest':torch.tensor([0,0,1,1,1,1,1,1],dtype=torch.bool)}
    result=api().kv_difference(a,b,groups)
    assert len(result)==6
    for r in result:
        assert not r['key']['equal'] and not r['value']['equal']
        assert r['key']['rms_difference']==pytest.approx(1.)
        assert r['value']['relative_l2']==pytest.approx(1.)
        assert r['groups']['first']['tokens']==2


def tensors():
    q=torch.zeros(1,1,1,2);k=torch.zeros(1,1,2,2);v=torch.ones_like(k)*2
    sk=torch.zeros(1,1,2,2);sv=torch.ones_like(sk)*6
    fm=torch.ones(1,2,dtype=torch.bool);sm=torch.ones(1,2,dtype=torch.bool)
    return q,k,v,sk,sv,fm,sm


def test_probe_preserves_output_and_measures_softmax_mass_and_value_contribution():
    args=tensors();plain=attention.planning_attention(*args)
    original=attention.planning_attention
    with torch.no_grad(),api().attention_probe({'external':torch.ones(2,dtype=torch.bool)}) as stats:
        actual=attention.planning_attention(*args)
    assert attention.planning_attention is original and torch.equal(actual,plain)
    assert stats[0]['external_mass']==pytest.approx(.5)
    assert stats[0]['groups']['external']['mass']==pytest.approx(.5)
    assert stats[0]['external_value_norm']==pytest.approx(3*2**.5)


def test_masked_columns_differ_from_zero_v_and_are_invariant_to_external_content():
    q,k,v,sk,sv,fm,sm=tensors()
    with torch.no_grad():
        with api().attention_probe({'external':sm[0]},mask_external=True) as stats:
            masked=attention.planning_attention(q,k,v,sk,sv,fm,sm)
        with api().attention_probe({'external':sm[0]},mask_external=True):
            changed=attention.planning_attention(q,k,v,sk+999,sv-999,fm,sm)
        with api().attention_probe({'external':sm[0]}) as zero_stats:
            zero=attention.planning_attention(q,k,v,sk,torch.zeros_like(sv),fm,sm)
    assert torch.equal(masked,changed) and torch.equal(masked,torch.ones_like(masked)*2)
    assert torch.equal(zero,torch.ones_like(zero))
    assert stats[0]['external_mass']==0. and zero_stats[0]['external_mass']==pytest.approx(.5)
    assert zero_stats[0]['external_value_norm']==0.


def test_probe_rejects_grad_and_nested_calls_and_restores_after_exception():
    original=attention.planning_attention;groups={'external':torch.ones(2,dtype=torch.bool)}
    with pytest.raises(RuntimeError,match='no_grad'):
        with api().attention_probe(groups):pass
    with torch.no_grad(),pytest.raises(RuntimeError,match='sentinel'):
        with api().attention_probe(groups):
            with pytest.raises(RuntimeError,match='nested'):
                with api().attention_probe(groups):pass
            raise RuntimeError('sentinel')
    assert attention.planning_attention is original


def test_probe_operates_through_real_six_layer_bridge_and_keeps_context_unchanged(language_factory):
    lm=language_factory().language_model.eval();bridge=attention.KVBridge(lm,position_offset=128)
    x=torch.randn(1,16,32);p=torch.arange(16).view(1,1,-1).expand(3,1,-1);m=torch.ones(1,16,dtype=torch.long)
    def forward(positions):
        return lm(inputs_embeds=x,position_ids=positions,attention_mask=m,use_cache=False).last_hidden_state
    with torch.no_grad():
        with bridge.capture(p,m) as call:forward(p)
        plain=forward(p+128)
        with bridge.fuse(call.snapshot,p+128,m,torch.tensor([True])):
            active=forward(p+128)
        with api().attention_probe({'all':m[0].bool()}) as stats:
            with bridge.fuse(call.snapshot,p+128,m,torch.tensor([True])):
                measured=forward(p+128)
        assert torch.equal(active,measured) and len(stats)==6
        assert torch.equal(plain[:,:-8],measured[:,:-8])
        with api().attention_probe({'all':m[0].bool()},mask_external=True):
            with bridge.fuse(call.snapshot,p+128,m,torch.tensor([True])):
                masked=forward(p+128)
        assert torch.allclose(plain,masked,atol=1e-6,rtol=1e-6)


def test_summary_slices_clear_slow_advantage_and_keeps_cohorts_separate():
    target=[[0.,0.,0.]]*8
    def poses(x):return [[x,0.,0.]]*8
    records=[dict(token='a',cohort='new_val',route='one',speed_mps=2.,target=target,
                  slow_prediction=poses(.5),predictions={'none':poses(1.),'correct':poses(.7),'masked':poses(.9)}),
             dict(token='b',cohort='legacy_val',route='two',speed_mps=0.,target=target,
                  slow_prediction=poses(.45),predictions={'none':poses(.5),'correct':poses(.6),'masked':poses(.5)})]
    summary=api().summarize_records(records)
    assert summary['new_val']['clear_slow_advantage']['count']==1
    assert summary['legacy_val']['clear_slow_advantage']['count']==0
    paired=summary['new_val']['all']['modes']['correct']
    assert paired['gain_ADE_vs_masked_m']==pytest.approx(.2)
    assert paired['ADE']==pytest.approx(.7)
