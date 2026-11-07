"""CPU evidence for the isolated native-versus-cropped SDPA diagnostic."""
import pytest
import torch
from torch.nn import functional as F

from tools.audit_slowfast_sdpa import (
    capture_attention, replay_native, attention_geometry, tensor_difference,
)


def test_cropped_bottom_right_mask_and_masked_prefix_preserve_full_attention_math():
    torch.manual_seed(4)
    q=torch.randn(1,4,12,8);k=torch.randn(1,2,12,8);v=torch.randn_like(k)
    full=F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    cases=attention_geometry(q,k,v,queries=3,external_tokens=7)
    assert set(cases)=={'crop_gqa','crop_repeat','crop_repeat_masked_external'}
    for args in cases.values():
        actual=F.scaled_dot_product_attention(**args)
        torch.testing.assert_close(actual,full[:,:,-3:],atol=1e-6,rtol=1e-6)
    mask=cases['crop_repeat_masked_external']['attn_mask']
    assert mask.shape==(1,1,3,19)
    assert not mask[...,:7].any()
    assert mask[0,0,0,7:].tolist()==[True]*10+[False]*2
    assert mask[0,0,2,7:].all()
    assert cases['crop_gqa']['enable_gqa'] is True
    assert cases['crop_repeat']['key'].shape[1]==4


def attention():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    config=Qwen3_5TextConfig(hidden_size=32,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation='sdpa'
    return Qwen3_5Attention(config,layer_idx=3).eval()


def test_scoped_capture_and_native_helper_reproduce_real_module_post_gate_exactly():
    torch.manual_seed(2);model=attention();hidden=torch.randn(1,12,32)
    angle=torch.arange(12)[:,None]*torch.tensor([.01,.02,.03,.04])[None]
    angle=torch.cat((angle,angle),-1)[None];position=(angle.cos(),angle.sin())
    original_sdpa=F.scaled_dot_product_attention
    with torch.no_grad():
        with capture_attention(model) as captured:
            model(hidden_states=hidden,position_embeddings=position,attention_mask=None)
        assert F.scaled_dot_product_attention is original_sdpa
        result=replay_native(model,captured)
    assert result['post_gate_exact'] and result['recomputed_rope_query_exact'] and result['recomputed_rope_key_exact']
    assert captured['query'].shape==(1,4,12,8) and captured['key'].shape==(1,2,12,8)
    assert captured['sdpa_options']['enable_gqa'] is True
    assert captured['sdpa_options']['is_causal'] is True
    assert result['post_gate_difference']['max_abs_difference']==0.


def test_capture_cleanup_after_exception_never_leaks_a_global_sdpa_patch():
    model=attention();original=F.scaled_dot_product_attention
    hooks=(len(model._forward_pre_hooks),len(model._forward_hooks),len(model.o_proj._forward_pre_hooks))
    with torch.no_grad(),pytest.raises(RuntimeError,match='sentinel'):
        with capture_attention(model):raise RuntimeError('sentinel')
    assert F.scaled_dot_product_attention is original
    assert hooks==(len(model._forward_pre_hooks),len(model._forward_hooks),len(model.o_proj._forward_pre_hooks))


def test_difference_reports_real_magnitude_and_rejects_layout_mismatch():
    a=torch.tensor([1.,2.]);b=torch.tensor([1.,3.])
    d=tensor_difference(a,b)
    assert d['max_abs_difference']==1. and d['rms_difference']==pytest.approx(2**-.5)
    assert d['relative_l2']==pytest.approx(1/(5**.5))
    assert not d['exact']
    with pytest.raises(ValueError):tensor_difference(a,b[:,None])
