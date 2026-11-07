"""Small CPU geometry; all 24 decoder blocks and all six source caches remain."""
import pytest
import torch


def config():
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(hidden_size=32, intermediate_size=96, num_hidden_layers=24,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, rms_norm_eps=1e-6,
        rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=.5,
                             mrope_section=[2,1,1]))


def expert():
    from slowfast_v11.action_expert import ActionExpert
    torch.manual_seed(42)
    return ActionExpert(config(),width=32,ffn_width=96,query_heads=2,kv_heads=1,head_dim=16)


def data():
    generator=torch.Generator().manual_seed(8)
    return dict(keys=torch.randn(2,6,24,16,generator=generator,requires_grad=True),
                values=torch.randn(2,6,24,16,generator=generator,requires_grad=True),
                positions=torch.arange(24)[None,None].expand(3,2,-1).clone(),
                group_ids=torch.arange(24)[None].expand(2,-1)%9,
                valid=torch.ones(2,24,dtype=torch.bool),ego=torch.ones(2,4),
                command=torch.tensor([0,2]),metadata=torch.tensor([[0.,0.,0.,0.,1.,1.]]*2))


def test_full_24_block_expert_shapes_gradients_and_frozen_sources():
    model=expert();batch=data();before={k:v.detach().clone() for k,v in batch.items()}
    result=model(**batch)
    assert result.shape==(2,8,3) and torch.isfinite(result).all()
    assert len(model.blocks)==24 and all(p.requires_grad for p in model.parameters())
    result.square().mean().backward()
    for block in model.blocks:
        for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
            gradient=getattr(block,name).weight.grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert batch['keys'].grad is None and batch['values'].grad is None
    assert all(torch.equal(before[k],v) for k,v in batch.items())


def test_source_is_direct_native_rotated_k_and_unchanged_v_with_four_block_mapping(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb,repeat_kv
    import slowfast_v11.action_expert as module
    model=expert().eval();batch=data();captured=[]
    original=module.F.scaled_dot_product_attention
    def capture(q,k,v,*args,**kwargs):
        captured.append((k[:,:,:24].detach().clone(),v[:,:,:24].detach().clone()))
        return original(q,k,v,*args,**kwargs)
    monkeypatch.setattr(module.F,'scaled_dot_product_attention',capture)
    with torch.no_grad():model(**batch)
    assert len(captured)==24 and model.source_layer_mapping==tuple(i//4 for i in range(24))
    source_keys=batch['keys'].detach()
    cos,sin=model.rotary_emb(source_keys,batch['positions'])
    for index,(actual_k,actual_v) in enumerate(captured):
        layer=index//4
        raw=source_keys[:,layer].reshape(2,24,1,16).transpose(1,2)
        _,rotated=apply_rotary_pos_emb(raw,raw,cos,sin)
        raw_v=batch['values'][:,layer].detach().reshape(2,24,1,16).transpose(1,2)
        torch.testing.assert_close(actual_k,repeat_kv(rotated,2),atol=0.,rtol=0.)
        torch.testing.assert_close(actual_v,repeat_kv(raw_v,2),atol=0.,rtol=0.)


def test_own_positions_match_fast_reserved_suffix_and_native_partial_rope():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model=expert();batch=data()
    actual=model.expert_positions(batch['positions'])
    expected=batch['positions'][:,:,-18,None]+4096+torch.arange(11)[None,None]
    assert torch.equal(actual,expected)
    native=Qwen3_5TextRotaryEmbedding(config())
    actual_rope=model.rotary_emb(batch['keys'],actual)
    expected_rope=native(batch['keys'],expected)
    assert all(torch.equal(a,b) for a,b in zip(actual_rope,expected_rope))
    assert actual_rope[0].shape[-1]==8  # Only half of the 16-dimensional test head rotates.


def test_future_expert_queries_do_not_change_earlier_hidden_rows():
    model=expert().eval();batch=data()
    with torch.no_grad():
        before=model.forward_features(**batch)
        model.planning_queries[-1].add_(10.)
        after=model.forward_features(**batch)
    assert before.shape==(2,11,32)
    torch.testing.assert_close(before[:,:-1],after[:,:-1],atol=0.,rtol=0.)
    assert not torch.equal(before[:,-1],after[:,-1])


def test_source_content_matters_and_all_masked_source_is_supported():
    model=expert().eval();batch=data()
    with torch.no_grad():
        correct=model(**batch)
        changed=model(**dict(batch,values=batch['values']+5.))
        masked=model(**dict(batch,valid=torch.zeros_like(batch['valid'])))
        masked_changed=model(**dict(batch,valid=torch.zeros_like(batch['valid']),
                                     keys=batch['keys']+30.,values=batch['values']-50.))
        zero_v=model(**dict(batch,values=torch.zeros_like(batch['values'])))
    assert not torch.allclose(correct,changed,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(masked,masked_changed,atol=0.,rtol=0.)
    assert torch.isfinite(masked).all() and not torch.allclose(masked,zero_v,atol=1e-7,rtol=1e-7)


def test_constructor_accepts_owned_native_rope_module_and_rejects_geometry_drift():
    from slowfast_v11.action_expert import ActionExpert
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rope=Qwen3_5TextRotaryEmbedding(config())
    model=ActionExpert(rope=rope,width=32,ffn_width=96,query_heads=2,kv_heads=1,head_dim=16)
    assert model.rotary_emb is not rope
    assert model(**data()).shape==(2,8,3)
    with pytest.raises(ValueError):ActionExpert(config(),layers=12)
    with pytest.raises(ValueError):ActionExpert(config(),head_dim=32)
    model=expert();batch=data()
    with pytest.raises(ValueError):model(**dict(batch,keys=batch['keys'][:,:5]))
    with pytest.raises(ValueError):model(**dict(batch,command=torch.tensor([0,4])))
    with pytest.raises(ValueError):model(**dict(batch,valid=batch['valid'].float()))
    with pytest.raises(ValueError):model(**dict(batch,positions=batch['positions'][:,:,:17]))
