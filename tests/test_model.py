import pytest
import torch

from fast_only.model import FastPlanner, append_positions, planning_loss
from fast_only.common import trajectory_metrics


def tiny_backbone():
    from unittest.mock import patch
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5Model
    from transformers.models.qwen3_5 import modeling_qwen3_5
    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=['linear_attention', 'full_attention'],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                         'partial_rotary_factor': 1., 'mrope_section': [2, 3, 3]},
    )
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                num_heads=2, out_hidden_size=32, patch_size=16,
                                spatial_merge_size=2, num_position_embeddings=16)
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                           image_token_id=61, video_token_id=62,
                           vision_start_token_id=60, vision_end_token_id=63)
    # CUDA-only optional kernels are installed on this host. Explicitly choose
    # the upstream PyTorch reference implementations for this CPU unit test;
    # real-weight CUDA P0 tests use the installed accelerated implementations.
    with patch.multiple(modeling_qwen3_5, FusedRMSNormGated=None,
                        causal_conv1d_fn=None, chunk_gated_delta_rule=None):
        return Qwen3_5Model(config)


def test_append_positions_continues_after_native_mrope_and_handles_padding():
    pos = torch.tensor([[[0, 1, 7, 0]], [[0, 1, 5, 0]], [[0, 1, 6, 0]]])
    mask = torch.tensor([[1, 1, 1, 0]])
    result = append_positions(pos, mask, 9)
    assert torch.equal(result[..., :4], pos)
    assert result.shape == (3, 1, 13)
    assert result[..., 4:].tolist() == [[[8, 9, 10, 11, 12, 13, 14, 15, 16]]] * 3


def test_planning_loss_reduces_sum_xy_then_mean_points():
    pred = torch.ones(2, 6, 2, requires_grad=True)
    assert planning_loss(pred, torch.zeros_like(pred)).item() == 2.
    planning_loss(pred, torch.zeros_like(pred)).backward()
    assert torch.isfinite(pred.grad).all()


def test_metrics_euclidean_not_coordinate_l1():
    gt = torch.zeros(2, 6, 2)
    pred = gt.clone(); pred[..., 0] = 3.; pred[..., 1] = 4.
    metrics = trajectory_metrics(pred, gt)
    assert metrics['ADE'] == 5.
    assert metrics['FDE'] == 5.
    assert metrics['L2@0.5s'] == 5.
    assert metrics['loss_l1_xy'] == 7.


def test_native_four_images_queries_and_gradients():
    torch.manual_seed(42)
    planner = FastPlanner(tiny_backbone(), lora_rank=2, lora_alpha=4, lora_dropout=0.)
    # Four native single-token images, each with grid (1,2,2), in chronological order.
    ids = torch.tensor([[2, 60, 61, 63, 3, 60, 61, 63, 4, 60, 61, 63, 5, 60, 61, 63, 6]])
    mm = (ids == 61).long()
    grid = torch.tensor([[1, 2, 2]] * 4)
    native = {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
              'mm_token_type_ids': mm, 'image_grid_thw': grid,
              'pixel_values': torch.randn(16, 3 * 2 * 16 * 16)}
    planner.eval()
    with torch.no_grad():
        reference = planner.backbone(**native, use_cache=False).last_hidden_state
    captured = []
    handle = planner.backbone.language_model.register_forward_hook(
        lambda _m, _args, out: captured.append(out.last_hidden_state))
    prediction = planner(native, torch.ones(1, 3), torch.tensor([4]), torch.ones(1, 2))
    handle.remove()
    # Appended planning tokens cannot change any native observation hidden state.
    assert torch.allclose(captured[0][:, :ids.shape[1]], reference, atol=1e-5, rtol=1e-5)
    assert prediction.shape == (1, 6, 2)
    assert planner.last_layout['query_indices'] == list(range(ids.shape[1] + 3, ids.shape[1] + 9))
    assert planner.last_layout['image_tokens_per_frame'] == [1] * 4
    planning_loss(prediction, torch.ones_like(prediction)).backward()
    report = planner.gradient_report()
    assert report['vision_frozen']
    assert report['vision_has_gradient'] is False
    for key in ['ego', 'command', 'nav', 'queries', 'head', 'full_attention_lora', 'gdn_layer_ffn_lora']:
        assert report[key]['finite'] and report[key]['nonzero'], (key, report)


def test_padding_must_be_on_left_and_query_positions_ignore_padding():
    planner = FastPlanner(tiny_backbone(), lora_rank=2, lora_alpha=4, lora_dropout=0.)
    with pytest.raises(ValueError, match='left'):
        planner({'input_ids': torch.tensor([[2, 0]]), 'attention_mask': torch.tensor([[1, 0]])},
                torch.ones(1, 3), torch.tensor([1]), torch.ones(1, 2))


def test_formal_lora_master_parameters_are_float32_with_bf16_frozen_backbone():
    planner = FastPlanner(tiny_backbone().to(torch.bfloat16), lora_rank=2,
                           lora_alpha=4, lora_parameter_dtype='float32')
    adapters = [p for name, p in planner.named_parameters() if 'lora_' in name]
    assert adapters and all(p.dtype == torch.float32 for p in adapters)
    assert planner.backbone.get_input_embeddings().weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in planner.backbone.visual.parameters())
