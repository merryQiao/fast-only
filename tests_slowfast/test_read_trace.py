"""CPU semantics and native bridge checks for the scoped KV read probe."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slowfast_v11 import attention
from slowfast_v11.protocol import KVSnapshot, LayerKV


def api():
    from slowfast_v11 import read_trace
    return read_trace


def snapshot(length=3, dim=4):
    generator=torch.Generator().manual_seed(91)
    layers=tuple(LayerKV(torch.randn(1,1,length,dim,generator=generator),
                         torch.randn(1,1,length,dim,generator=generator)) for _ in attention.LAYERS)
    positions=torch.arange(length).view(1,1,-1).expand(3,1,-1)
    return KVSnapshot(layers,positions,torch.ones(1,length,dtype=torch.bool))


def simple_attention():
    q=torch.zeros(1,2,1,2);k=torch.zeros(1,1,2,2);v=torch.ones_like(k)*2
    sk=torch.zeros(1,1,2,2);sv=torch.ones_like(sk)*6
    mask=torch.ones(1,2,dtype=torch.bool)
    groups={'first':torch.tensor([True,False]),'second':torch.tensor([False,True])}
    return (q,k,v,sk,sv,mask,mask),groups


def test_float32_reference_splits_one_softmax_and_reconstructs_sdpa():
    args,groups=simple_attention()
    result=api()._float32_reference(*args,groups)
    actual=attention.planning_attention(*args)
    assert result['external_mass']==pytest.approx(.5)
    assert result['self_mass']==pytest.approx(.5)
    assert torch.allclose(result['external'],torch.full_like(actual,3.))
    assert torch.allclose(result['self'],torch.full_like(actual,1.))
    assert torch.allclose(result['combined'],actual)
    assert torch.equal(sum(result['group_contributions'].values()),result['external'])
    assert result['groups']['first']['mass']==pytest.approx(.25)


def test_reference_reports_head_masses_without_hiding_a_selective_head():
    args,groups=simple_attention();q,k,v,sk,sv,fm,sm=args
    q[0,1,0,0]=2**.5*torch.tensor(3.).log();sk[:,:,:,0]=1.
    result=api()._float32_reference(q,k,v,sk,sv,fm,sm,groups)
    assert result['external_mass_per_head']==pytest.approx([.5,.75])
    assert result['self_mass_per_head']==pytest.approx([.5,.25])
    assert result['groups']['first']['mass_per_head']==pytest.approx([.25,.375])


def test_zero_external_values_keep_the_denominator_while_masking_removes_it():
    args,groups=simple_attention();q,k,v,sk,sv,fm,sm=args
    zero=api()._float32_reference(q,k,v,sk,torch.zeros_like(sv),fm,sm,groups)
    masked=api()._float32_reference(q,k,v,sk,sv,fm,torch.zeros_like(sm),groups)
    assert zero['external_mass']==pytest.approx(.5)
    assert masked['external_mass']==0.
    assert torch.equal(zero['combined'],torch.ones_like(q))
    assert torch.equal(masked['combined'],torch.ones_like(q)*2)


def test_reference_honors_fast_causality_gqa_and_invalid_source_columns():
    torch.manual_seed(3)
    q=torch.randn(1,4,3,4);k=torch.randn(1,2,5,4);v=torch.randn_like(k)
    sk=torch.randn(1,2,3,4);sv=torch.randn_like(sk)
    fm=torch.tensor([[False,True,True,True,True]]);sm=torch.tensor([[True,False,True]])
    groups={'source':torch.ones(3,dtype=torch.bool)}
    result=api()._float32_reference(q,k,v,sk,sv,fm,sm,groups)
    actual=attention.planning_attention(q,k,v,sk,sv,fm,sm)
    assert torch.allclose(actual,result['combined'],atol=5e-7,rtol=2e-6)
    assert result['groups']['source']['visible_tokens']==2
    assert result['external_mass']+result['self_mass']==pytest.approx(1.)


def test_reference_remains_fp32_inside_the_native_bf16_autocast_context():
    torch.manual_seed(19)
    args,groups=simple_attention()
    args=list(args);args[0]=torch.randn_like(args[0]);args[1]=torch.randn_like(args[1]);args[3]=torch.randn_like(args[3])
    outside=api()._float32_reference(*args,groups)
    with torch.autocast('cpu',dtype=torch.bfloat16):inside=api()._float32_reference(*args,groups)
    assert inside['combined'].dtype==torch.float32
    assert torch.equal(inside['combined'],outside['combined'])
    with pytest.raises(ValueError,match='partition'):
        api()._float32_reference(*args,{'overlap':torch.ones(2,dtype=torch.bool),'extra':torch.ones(2,dtype=torch.bool)})


class BiasedLowRankProjection(nn.Module):
    """A real affine projection with an extra low-rank forward branch."""
    def __init__(self,width):
        super().__init__()
        self.base=nn.Linear(width,width,bias=True)
        self.a=nn.Linear(width,2,bias=False);self.b=nn.Linear(2,width,bias=False)
        with torch.no_grad():self.base.bias.fill_(.37)

    def forward(self,x):
        return self.base(x)+self.b(self.a(x))


def test_affine_split_uses_sigmoid_gate_and_includes_projection_bias_once():
    torch.manual_seed(8)
    projection=BiasedLowRankProjection(4)
    external=torch.randn(1,2,4);own=torch.randn_like(external)
    gate=torch.randn_like(external).sigmoid()
    calls=[]
    handle=projection.register_forward_pre_hook(lambda *args:calls.append(True))
    with torch.no_grad():
        result=api()._contribution_stages(projection,external,gate)
        other=api()._contribution_stages(projection,own,gate)
        bias=projection.forward(torch.zeros_like(external))
        combined=projection.forward((external+own)*gate)
    handle.remove()
    assert calls==[]  # Direct forward bypasses the native bridge's parent hooks.
    assert torch.equal(result['post_gate'],external*gate)
    assert torch.allclose(result['linear_projection']+other['linear_projection']+bias,combined,atol=2e-7)
    assert not torch.allclose(result['linear_projection']+other['linear_projection']+2*bias,combined)


def test_swap_layer_changes_exactly_one_of_the_six_layers_and_retains_target_layout():
    correct=snapshot();donor=replace(correct,layers=tuple(LayerKV(z.key+3,z.value-2) for z in correct.layers))
    for index,layer in enumerate(attention.LAYERS):
        result=api().swap_layer(correct,donor,layer+1)
        assert result.position_ids is correct.position_ids and result.valid_mask is correct.valid_mask
        assert all(result.layers[i] is (donor.layers[i] if i==index else correct.layers[i]) for i in range(6))
    for bad in [0,1,3,25,True]:
        with pytest.raises(ValueError,match='layer'):api().swap_layer(correct,donor,bad)
    with pytest.raises(ValueError,match='layout'):
        api().swap_layer(correct,replace(donor,position_ids=donor.position_ids+1),4)
    with pytest.raises(ValueError,match='six'):
        api().swap_layer(replace(correct,layers=correct.layers[:1]),replace(donor,layers=donor.layers[:1]),4)


@pytest.fixture
def native_bridge(language_factory):
    torch.manual_seed(23)
    language=language_factory().language_model
    for index in attention.LAYERS:
        language.layers[index].self_attn.o_proj=BiasedLowRankProjection(32)
    model=nn.Module();model.language=language
    model.bridge=attention.KVBridge(language,position_offset=128)
    model.eval()
    x=torch.randn(1,12,32)
    positions=torch.arange(12).view(1,1,-1).expand(3,1,-1)
    mask=torch.ones(1,12,dtype=torch.long)
    def run(offset=128):
        return language(inputs_embeds=x,position_ids=positions+offset,attention_mask=mask,use_cache=False).last_hidden_state
    with torch.no_grad(),model.bridge.capture(positions,mask) as call:run(0)
    state=SimpleNamespace(model=model,snapshot=call.snapshot,positions=positions+128,mask=mask,run=run)
    yield state
    model.bridge.close()


def fused(state,snap=None):
    with state.model.bridge.fuse(snap or state.snapshot,state.positions,state.mask,torch.tensor([True])):
        return state.run()


def test_probe_returns_original_sdpa_object_and_native_output_and_cleans_up(native_bridge,monkeypatch):
    state=native_bridge;original=attention.planning_attention;original_returns=[];observed=[]
    def spy(*args,**kwargs):
        result=original(*args,**kwargs);original_returns.append(result);return result
    monkeypatch.setattr(attention,'planning_attention',spy)
    projected=[];handles=[]
    for layer in attention.LAYERS:
        handles.append(state.model.language.layers[layer].self_attn.o_proj.register_forward_pre_hook(
            lambda module,args:projected.append(args[0].detach().clone())))
    groups={'all':state.mask[0].bool()}
    with torch.no_grad():
        before=fused(state);original_returns.clear();projected.clear()
        with api().read_probe(state.model,groups) as probe:
            wrapper=attention.planning_attention
            def capture_return(*args,**kwargs):
                start=len(original_returns);out=wrapper(*args,**kwargs)
                observed.append(out is original_returns[start]);return out
            attention.planning_attention=capture_return
            measured=fused(state)
        assert attention.planning_attention is spy
        assert torch.equal(before,measured)
        assert observed==[True]*6 and [x.shape[1] for x in projected]==[12]*6
        assert [r['layer_one_based'] for r in probe.records]==[4,8,12,16,20,24]
        assert set(probe.tensors)=={f'{name}_{layer+1}' for layer in attention.LAYERS for name in
                                  ['pre_gate','gate','post_gate','query_projection']}
        for index,layer in enumerate(attention.LAYERS):
            assert torch.equal(probe.tensors[f'pre_gate_{layer+1}'],original_returns[index].transpose(1,2).flatten(2).float())
            assert torch.equal(probe.tensors[f'post_gate_{layer+1}'],projected[index][:,-8:].float())
            gate=probe.tensors[f'gate_{layer+1}'];stats=probe.records[index]['actual']
            assert stats['gate_mean_per_query']==pytest.approx(gate.mean((0,2)).tolist())
            assert stats['gate_mean_per_head']==pytest.approx(gate.view(1,8,2,16).mean((0,1,3)).tolist())
            assert stats['gate_fraction_lt_0_01']==pytest.approx(float((gate<.01).float().mean()))
            assert stats['gate_fraction_lt_0_1']==pytest.approx(float((gate<.1).float().mean()))
            assert stats['gate_fraction_gt_0_9']==pytest.approx(float((gate>.9).float().mean()))
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 and not t.requires_grad for t in probe.tensors.values())
        json.dumps(probe.records,allow_nan=False)
        assert torch.equal(fused(state),before)
    for handle in handles:handle.remove()


def test_identical_local_sources_have_zero_difference_at_every_stage(native_bridge):
    state=native_bridge;groups={'all':state.mask[0].bool()}
    parameters={name:(value.detach().clone(),value._version) for name,value in state.model.named_parameters()}
    with torch.no_grad():
        baseline=fused(state)
        with api().read_probe(state.model,groups,alternatives={'reference':state.snapshot,'shuffled':state.snapshot}) as probe:
            measured=fused(state)
        assert torch.equal(measured,baseline)
    assert all(torch.equal(value,parameters[name][0]) and value._version==parameters[name][1]
               for name,value in state.model.named_parameters())
    for row in probe.records:
        variants=row['local_variants']
        for name in ['correct','reference','reference_k_only','reference_v_only','shuffled','shuffled_k_only','shuffled_v_only']:
            for stage in ['pre_gate','post_gate','post_projection']:
                assert variants[name][stage]['equal']
                assert variants[name][stage]['relative_l2']==0.
        assert variants['zero_v']['pre_gate']['rms_difference']>0
        assert variants['masked']['pre_gate']['rms_difference']>0
        assert variants['zero_v']['external_columns_masked'] is False
        assert variants['masked']['external_columns_masked'] is True
        reference=row['float32_reference']
        assert len(reference['external_mass_per_head'])==2 and len(reference['self_mass_per_head'])==2
        assert len(reference['groups']['all']['mass_per_head'])==2
        assert reference['sdpa_reconstruction']['rms_difference']<1e-6
        assert reference['projection_reconstruction']['rms_difference']<1e-6
        assert reference['groups']['all']['contribution']==reference['external']


def test_local_counterfactuals_use_donor_raw_keys_with_the_current_source_rope(native_bridge,monkeypatch):
    state=native_bridge
    donor=replace(state.snapshot,layers=tuple(LayerKV(z.key.flip(2),z.value.flip(2)) for z in state.snapshot.layers))
    original=attention.planning_attention;used=[]
    def spy(q,k,v,sk,sv,fm,sm,dropout=0.):
        call=state.model.bridge._call
        used.append((q,k,v,sk,sv,sm,call.source_rope))
        return original(q,k,v,sk,sv,fm,sm,dropout)
    monkeypatch.setattr(attention,'planning_attention',spy)
    with torch.no_grad(),api().read_probe(state.model,{'all':state.mask[0].bool()},alternatives={'reference':donor}) as probe:
        fused(state)
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    assert len(used)==6*6  # actual, donor, K-only, V-only, zero-V, masked
    for index in range(6):
        actual,full,key_only,value_only,zero,masked=used[index*6:(index+1)*6]
        _,expected=apply_rotary_pos_emb(donor.layers[index].key,donor.layers[index].key,*actual[6])
        assert all(local[0] is actual[0] and local[1] is actual[1] and local[2] is actual[2]
                   for local in [full,key_only,value_only,zero,masked])
        assert torch.equal(full[3],expected) and torch.equal(key_only[3],expected)
        assert value_only[3] is actual[3] and value_only[4] is donor.layers[index].value
        assert key_only[4] is actual[4] and full[4] is donor.layers[index].value
        assert zero[5] is actual[5] and not zero[4].any()
        assert masked[4] is actual[4] and not masked[5].any()
    assert all(row['local_variants']['reference']['pre_gate']['rms_difference']>1e-5 for row in probe.records)
    assert all(row['local_variants']['reference_k_only']['pre_gate']['rms_difference']>1e-5 for row in probe.records)
    assert all(row['local_variants']['reference_v_only']['pre_gate']['rms_difference']>1e-5 for row in probe.records)


def test_mask_layer_changes_only_the_designated_external_column_mask(monkeypatch):
    args,_=simple_attention();original=attention.planning_attention;used=[]
    def spy(q,k,v,sk,sv,fm,sm,dropout=0.):
        used.append(sm.clone());return original(q,k,v,sk,sv,fm,sm,dropout)
    monkeypatch.setattr(attention,'planning_attention',spy)
    with torch.no_grad(),api().mask_layer(12):
        outputs=[attention.planning_attention(*args) for _ in attention.LAYERS]
    assert [bool(m.any()) for m in used]==[True,True,False,True,True,True]
    assert torch.equal(outputs[2],torch.ones_like(outputs[2])*2)
    assert torch.equal(outputs[0],torch.ones_like(outputs[0])*4)
    assert attention.planning_attention is spy


def test_scope_validation_non_reentrancy_and_exception_cleanup(native_bridge):
    state=native_bridge;groups={'all':state.mask[0].bool()};original=attention.planning_attention
    with pytest.raises(RuntimeError,match='no_grad'):
        with api().read_probe(state.model,groups):pass
    state.model.train()
    with torch.no_grad(),pytest.raises(RuntimeError,match='eval'):
        with api().read_probe(state.model,groups):pass
    state.model.eval()
    with torch.no_grad(),pytest.raises(RuntimeError,match='sentinel'):
        with api().read_probe(state.model,groups):
            with pytest.raises(RuntimeError,match='nested'):
                with api().read_probe(state.model,groups):pass
            with pytest.raises(RuntimeError,match='nested'):
                with api().mask_layer(4):pass
            raise RuntimeError('sentinel')
    assert attention.planning_attention is original and state.model.bridge._call is None
    with torch.no_grad(),pytest.raises(RuntimeError,match='six'):
        with api().read_probe(state.model,groups):pass
    assert attention.planning_attention is original
    with torch.no_grad(),pytest.raises(RuntimeError,match='sentinel'):
        with api().mask_layer(4):raise RuntimeError('sentinel')
    assert attention.planning_attention is original
    with torch.no_grad(),pytest.raises(RuntimeError,match='six'):
        with api().mask_layer(4):pass
    assert attention.planning_attention is original


@pytest.mark.parametrize('damage',['groups','positions','mask','dtype','dropout'])
def test_probe_rejects_mismatched_layout_and_dropout_without_leaking(native_bridge,damage):
    state=native_bridge;original=attention.planning_attention;groups={'all':state.mask[0].bool()}
    donor=state.snapshot
    if damage=='groups':groups={'short':torch.ones(1,dtype=torch.bool)}
    if damage=='positions':donor=replace(donor,position_ids=donor.position_ids+1)
    if damage=='mask':donor=replace(donor,valid_mask=~donor.valid_mask)
    if damage=='dtype':donor=replace(donor,layers=tuple(LayerKV(z.key.double(),z.value.double()) for z in donor.layers))
    with torch.no_grad(),pytest.raises(ValueError,match='layout|dropout'):
        with api().read_probe(state.model,groups,alternatives={'reference':donor}):
            if damage=='dropout':
                args,_=simple_attention();attention.planning_attention(*args,dropout=.1)
            else:fused(state)
    assert attention.planning_attention is original and state.model.bridge._call is None
