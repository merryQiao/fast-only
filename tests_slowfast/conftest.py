"""Small native 24-layer fixtures: six full-attention layers, unequal LM widths."""
from unittest.mock import patch
import pytest
import torch


@pytest.fixture
def language_factory():
    def build(width=32):
        from transformers import Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5Config, Qwen3_5Model
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native
        text = Qwen3_5TextConfig(
            vocab_size=64, hidden_size=width, intermediate_size=width*2, num_hidden_layers=24,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=['full_attention' if i%4==3 else 'linear_attention' for i in range(24)])
        text.linear_num_key_heads = text.linear_num_value_heads = 2
        text.linear_key_head_dim = text.linear_value_head_dim = 16
        text.rope_parameters = {'rope_type':'default', 'rope_theta':10000,
                               'partial_rotary_factor':.5, 'mrope_section':[2,1,1]}
        vision = Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,
            num_heads=2,out_hidden_size=width,patch_size=16,spatial_merge_size=2,num_position_embeddings=16)
        config = Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict(),
            image_token_id=61,video_token_id=62,vision_start_token_id=60,vision_end_token_id=63)
        config._attn_implementation = 'sdpa'
        with patch.multiple(native,FusedRMSNormGated=None,causal_conv1d_fn=None,chunk_gated_delta_rule=None):
            return Qwen3_5Model(config)
    return build


@pytest.fixture
def native_input():
    ids=torch.tensor([[2,60,61,63,3,60,61,63,4,60,61,63,5,60,61,63,6]])
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
        'mm_token_type_ids':(ids==61).long(),'image_grid_thw':torch.tensor([[1,2,2]]*4),
        'pixel_values':torch.randn(16,3*2*16*16)}
