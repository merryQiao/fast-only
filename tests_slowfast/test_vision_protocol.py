import importlib

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding


def api():
    return importlib.import_module('slowfast_v11.vision_protocol')


def visual():
    model=nn.Module()
    model.weight=nn.Parameter(torch.ones(2,3,dtype=torch.bfloat16),requires_grad=False)
    model.rotary_pos_emb=Qwen3_5VisionRotaryEmbedding(32)
    return model


def test_native_rope_recovers_original_frequencies_instead_of_widening_rounded_values():
    v=visual();native=v.rotary_pos_emb.inv_freq.clone();weight=v.weight.clone()
    assert 'rotary_pos_emb.inv_freq' not in v.state_dict()
    v.to(torch.bfloat16)
    assert not torch.equal(native,v.rotary_pos_emb.inv_freq.float())
    receipt=api().restore_native_visual_rope(v)
    assert receipt['before_dtype']=='torch.bfloat16' and receipt['after_dtype']=='torch.float32'
    assert receipt['changed'] and torch.equal(v.rotary_pos_emb.inv_freq,native)
    assert torch.equal(v.weight,weight) and v.weight.dtype==torch.bfloat16
    assert not v.weight.requires_grad and 'rotary_pos_emb.inv_freq' not in v.state_dict()


def test_already_native_preserves_buffer_identity_and_reports_no_change():
    v=visual();original=v.rotary_pos_emb.inv_freq
    receipt=api().restore_native_visual_rope(v)
    assert not receipt['changed'] and v.rotary_pos_emb.inv_freq is original
    assert receipt['before_sha256']==receipt['after_sha256']


def test_buffer_identity_catches_precision_change_invisible_to_parameter_state():
    v=visual();parameter_before=v.weight.clone();before=api().buffer_digest(v)
    v.to(torch.bfloat16)
    assert torch.equal(parameter_before,v.weight)
    assert api().buffer_digest(v)!=before
    api().restore_native_visual_rope(v)
    assert api().buffer_digest(v)==before


def test_unknown_rotary_implementation_requires_explicit_alignment():
    v=nn.Module();v.rotary_pos_emb=nn.Identity()
    with pytest.raises(ValueError):api().restore_native_visual_rope(v)
