import pytest
import torch
from fast_navsim.model import pose_loss
from slowfast_v11.model import FastWithSlowKV, SlowPrefill


def planners(factory):
    kw=dict(lora_rank=2,lora_alpha=4,lora_dropout=0.,position_offset=128)
    return FastWithSlowKV(factory(),**kw).eval(), SlowPrefill(factory(64),**kw).eval()


def nonzero(params):
    gradients=[p.grad for p in params if p.grad is not None]
    return bool(gradients) and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().sum()>0 for g in gradients)


def test_planner_feedback_metadata_and_fast_loss_reaches_slow(language_factory,native_input):
    fast,slow=planners(language_factory)
    ego=torch.ones(1,4);cmd=torch.tensor([1])
    candidate=fast(native_input,ego,cmd).detach()
    result=slow(native_input,ego,cmd,feedback=candidate)
    assert result.trajectory.shape==(1,8,3)
    assert len(result.snapshot.layers)==6
    metadata=torch.tensor([[.5,1.,0.,0.,1.,1.]])
    prediction=fast(native_input,ego,cmd,snapshot=result.snapshot,metadata=metadata)
    for layer in result.snapshot.layers:
        layer.key.retain_grad();layer.value.retain_grad()
    pose_loss(prediction,torch.ones_like(prediction)).backward()
    for layer in result.snapshot.layers:
        assert nonzero([layer.key]) and nonzero([layer.value])
    assert nonzero([p for n,p in slow.named_parameters() if 'lora_' in n])
    assert nonzero(slow.feedback_encoder.parameters())
    assert nonzero(fast.metadata_encoder.parameters())
    assert all(p.grad is None for p in slow.backbone.visual.parameters())
    assert all(p.grad is None for n,p in slow.backbone.named_parameters() if 'lora_' not in n)


def test_masked_snapshot_uses_same_no_cache_token_as_absent(language_factory,native_input):
    fast,slow=planners(language_factory);ego=torch.ones(1,4);cmd=torch.tensor([0])
    snap=slow(native_input,ego,cmd).snapshot.inference_copy()
    with torch.no_grad():
        absent=fast(native_input,ego,cmd)
        disabled=fast(native_input,ego,cmd,snapshot=snap,metadata=torch.ones(1,6),enabled=torch.tensor([False]))
    assert torch.equal(absent,disabled)
    assert fast.last_layout['trajectory_shape']==[8,3]
    assert fast.last_layout['suffix_order']==['ego','command','cache_metadata','planning_queries_x8']
    with pytest.raises(ValueError,match='metadata'):
        fast(native_input,ego,cmd,snapshot=snap)
    with pytest.raises(ValueError,match='feedback'):
        slow(native_input,ego,cmd,feedback=torch.zeros(1,8,2))


def test_frozen_slow_body_still_trains_new_inputs(language_factory,native_input):
    _,slow=planners(language_factory)
    slow.backbone.requires_grad_(False)
    out=slow(native_input,torch.ones(1,4),torch.tensor([2]))
    pose_loss(out.trajectory,torch.ones_like(out.trajectory)).backward()
    assert nonzero([slow.queries])
    assert nonzero(slow.ego_mlp.parameters())
    assert nonzero([slow.no_feedback])
    assert not any(p.grad is not None for p in slow.backbone.parameters())
