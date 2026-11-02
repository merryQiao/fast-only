import math
import pytest
import torch

from fast_navsim.model import NavsimPlanner, pose_loss, trajectory_metrics
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('baseline_model_fixture',Path(__file__).parents[1]/'tests/test_model.py')
_fixture=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
tiny_backbone=_fixture.tiny_backbone


def native_input():
    ids=torch.tensor([[2,60,61,63,3,60,61,63,4,60,61,63,5,60,61,63,6]])
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
            'mm_token_type_ids':(ids==61).long(),'image_grid_thw':torch.tensor([[1,2,2]]*4),
            'pixel_values':torch.randn(16,3*2*16*16)}


def test_pose_loss_matches_official_mean_including_heading_and_all_eight_points():
    pred=torch.ones(2,8,3,requires_grad=True);target=torch.zeros_like(pred)
    pred.data[...,2]=4
    assert pose_loss(pred,target).item()==2
    pose_loss(pred,target).backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad!=0).all()
    with pytest.raises(ValueError):pose_loss(torch.zeros(1,6,2),torch.zeros(1,6,2))


def test_metrics_do_not_mix_heading_radians_into_xy_distance():
    pred=torch.zeros(1,8,3);gt=torch.zeros_like(pred)
    pred[...,0]=3;pred[...,1]=4;pred[...,2]=-math.pi+.01;gt[...,2]=math.pi-.01
    m=trajectory_metrics(pred,gt)
    assert m['ADE']==5 and m['FDE']==5 and m['L2@4s']==5
    assert m['heading_mae_rad']==pytest.approx(.02,abs=1e-6)
    assert m['loss_l1_pose']>4
    with pytest.raises(ValueError):trajectory_metrics(pred.fill_(float('nan')),gt)


def test_native_planner_has_eight_se2_queries_and_no_continuous_target():
    model=NavsimPlanner(tiny_backbone(),lora_rank=2,lora_alpha=4,lora_dropout=0.)
    batch=native_input();model.eval()
    with torch.no_grad():reference=model.backbone(**batch,use_cache=False).last_hidden_state
    captured=[]
    h=model.backbone.language_model.register_forward_hook(lambda m,a,o:captured.append(o.last_hidden_state))
    pred=model(batch,torch.ones(1,4),torch.tensor([1]));h.remove()
    assert pred.shape==(1,8,3)
    assert torch.allclose(captured[0][:,:17],reference,atol=1e-5,rtol=1e-5)
    assert model.last_layout['query_indices']==list(range(19,27))
    assert model.last_layout['continuous_target_token'] is False
    assert not any('nav_mlp' in n for n,p in model.named_parameters())
    pose_loss(pred,torch.ones_like(pred)).backward()
    report=model.gradient_report()
    assert report['vision_frozen'] and not report['vision_has_gradient']
    for key in ['ego','command','queries','head','full_attention_lora','gdn_layer_ffn_lora']:
        assert report[key]['finite'] and report[key]['nonzero'],key
    with pytest.raises(ValueError,match='target'):model(batch,torch.ones(1,4),torch.tensor([1]),torch.zeros(1,2))
