import math
from dataclasses import replace
import pytest
import torch
from slowfast_v11.protocol import (
    LayerKV, KVSnapshot, SourceMetadata, admit_snapshot, cache_metadata, relative_source_pose)


def test_snapshot_copy_owns_storage_and_training_transfer_keeps_autograd():
    x=torch.randn(1,1,3,4,requires_grad=True)
    snap=KVSnapshot((LayerKV(x,x*2),),torch.zeros(3,1,3,dtype=torch.long),torch.ones(1,3,dtype=torch.bool))
    copied=snap.inference_copy()
    assert not copied.layers[0].key.requires_grad
    assert copied.layers[0].key.data_ptr()!=x.data_ptr()
    assert copied.position_ids.data_ptr()!=snap.position_ids.data_ptr()
    with torch.no_grad():x.add_(1)
    assert not torch.equal(copied.layers[0].key,x)
    snap.to('cpu').layers[0].value.sum().backward()
    assert torch.equal(x.grad,torch.full_like(x,2))


def test_ready_after_image_before_selection_is_legal_and_future_source_is_not():
    metadata=SourceMetadata(1.,1.7,'ep','nav','model',(0.,0.,0.))
    kw=dict(image_time=1.5,selection_time=1.8,episode_id='ep',navigation_id='nav',model_id='model')
    assert admit_snapshot(metadata,**kw)
    assert not admit_snapshot(replace(metadata,source_time=1.6),**kw)
    assert not admit_snapshot(replace(metadata,ready_time=1.9),**kw)
    for key in ['episode_id','navigation_id','model_id']:
        assert not admit_snapshot(metadata,**(kw|{key:'changed'}))
    with pytest.raises(ValueError):admit_snapshot(metadata,**(kw|{'image_time':float('nan')}))


def test_relative_pose_and_metadata_are_in_current_ego_frame():
    source=(1.,2.,math.pi);current=(1.,1.,math.pi/2)
    dx,dy,yaw=relative_source_pose(source,current)
    assert (dx,dy,yaw)==pytest.approx((1.,0.,math.pi/2))
    meta=SourceMetadata(1.,1.1,'e','n','m',source)
    assert cache_metadata(meta,image_time=2.,current_pose=current).tolist()==pytest.approx([1,1,0,1,0,1],abs=1e-6)
    assert torch.equal(cache_metadata(None,image_time=2.,current_pose=current),torch.zeros(6))
    with pytest.raises(ValueError):cache_metadata(meta,image_time=.9,current_pose=current)


def test_snapshot_rejects_malformed_geometry():
    x=torch.ones(1,1,3,4)
    with pytest.raises(ValueError):KVSnapshot((LayerKV(x,x),),torch.zeros(3,1,2),torch.ones(1,3))
    with pytest.raises(ValueError):LayerKV(x,x[:,:,:2])


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_payload_is_rejected_even_in_a_masked_source_column(bad):
    x=torch.ones(1,1,3,4);x[:,:,0]=bad
    with pytest.raises(ValueError,match='finite'):LayerKV(x,torch.ones_like(x))
    with pytest.raises(ValueError,match='finite'):LayerKV(torch.ones_like(x),x)


def test_integer_payload_is_rejected():
    x=torch.ones(1,1,3,4,dtype=torch.long)
    with pytest.raises(ValueError,match='floating'):LayerKV(x,x)
