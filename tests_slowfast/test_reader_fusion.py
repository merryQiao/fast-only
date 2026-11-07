"""Semantic contracts for an optional residual before the original single head."""
import pytest
import torch
from torch import nn


def setup():
    from slowfast_v11.reader_fusion import ResidualFusion
    torch.manual_seed(42)
    fusion=ResidualFusion(fast_width=12,reader_width=18,hidden=16)
    head=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,7),nn.GELU(),nn.Linear(7,3))
    head.requires_grad_(False)
    f=torch.randn(2,8,12,requires_grad=True)
    r=torch.randn(2,8,18,requires_grad=True)
    return fusion,head,f,r


def test_new_residual_is_exactly_zero_and_keeps_current_features():
    fusion,head,f,r=setup()
    out=fusion(f,r)
    assert torch.equal(out,f) and torch.equal(head(out),head(f))


def test_frozen_head_passes_gradients_to_fusion_not_sources():
    fusion,head,f,r=setup()
    initial={n:p.clone() for n,p in head.named_parameters()}
    loss=head(fusion(f,r)).square().mean();loss.backward()
    assert f.grad is None and r.grad is None
    assert all(p.grad is None for p in head.parameters())
    assert fusion.residual.weight.grad is not None and fusion.residual.weight.grad.abs().sum()>0
    opt=torch.optim.AdamW(fusion.parameters(),lr=.01);opt.step();opt.zero_grad(set_to_none=True)
    head(fusion(f,r)).square().mean().backward()
    assert fusion.combine.weight.grad.abs().sum()>0
    assert fusion.gate.weight.grad.abs().sum()>0
    assert all(torch.equal(p,initial[n]) for n,p in head.named_parameters())


def test_off_path_does_not_call_reader_or_install_hook():
    from slowfast_v11.reader_fusion import fused_head
    fusion,head,f,r=setup()
    def forbidden():raise AssertionError('disabled Reader was called')
    before=len(head._forward_pre_hooks)
    with fused_head(head,fusion,forbidden,enabled=False):
        assert len(head._forward_pre_hooks)==before
        assert torch.equal(head(f),head(f))
    assert len(head._forward_pre_hooks)==before


def test_scoped_fusion_calls_original_head_once_and_cleans_after_failure():
    from slowfast_v11.reader_fusion import fused_head
    fusion,head,f,r=setup();seen=[]
    handle=head.register_forward_hook(lambda module,args,out:seen.append(1))
    with torch.no_grad():fusion.residual.weight.normal_(std=.01)
    expected=head(fusion(f,r));seen.clear()
    with fused_head(head,fusion,lambda:r,enabled=True):observed=head(f)
    assert torch.equal(observed,expected) and len(seen)==1
    assert len(head._forward_pre_hooks)==0
    with pytest.raises(RuntimeError,match='abort'):
        with fused_head(head,fusion,lambda:r,enabled=True):raise RuntimeError('abort')
    assert len(head._forward_pre_hooks)==0
    handle.remove()


def test_wrong_source_does_not_replace_current_fast_features():
    fusion,head,f,r=setup()
    with torch.no_grad():fusion.residual.weight.normal_(std=.1)
    delta1=fusion(f,r)-f
    delta2=fusion(f,r.flip(0))-f
    assert not torch.equal(delta1,delta2)
    assert torch.isfinite(delta1).all()
    assert not torch.equal(fusion(f,r),fusion(f.flip(0),r))


def test_capture_reader_features_preserves_prediction_and_detaches():
    from slowfast_v11.reader_fusion import reader_features
    from slowfast_v11.readability import KVReader
    torch.manual_seed(4);reader=KVReader(kv_width=8,hidden=4).eval().requires_grad_(False)
    batch=dict(keys=torch.randn(2,6,9,8),values=torch.randn(2,6,9,8),positions=torch.zeros(3,2,9,dtype=torch.long),
               group_ids=torch.arange(9).expand(2,-1),valid=torch.ones(2,9,dtype=torch.bool))
    expected=reader(**batch)
    features,prediction=reader_features(reader,batch)
    assert features.shape==(2,8,24) and not features.requires_grad
    assert torch.equal(prediction,expected) and len(reader.head._forward_pre_hooks)==0
