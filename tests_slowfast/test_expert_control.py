from types import SimpleNamespace
import pytest
import torch
from slowfast_v11.model import FastWithSlowKV


def api():
    from tools import train_frozen_expert as m
    return m


def test_frozen_vision_reuse_is_exact_and_scoped(language_factory,native_input):
    fast=FastWithSlowKV(language_factory(),lora_rank=2,lora_alpha=4,position_offset=128).eval()
    batch=dict(native=native_input,ego=torch.zeros(1,4),command=torch.tensor([1]))
    with torch.no_grad():
        expected=fast(**batch)
        vision=fast.backbone.get_image_features(native_input['pixel_values'],native_input['image_grid_thw'],return_dict=True).pooler_output
        native=dict(native_input,pixel_values=torch.empty(0))
        with api().cached_vision(fast.backbone,vision,native['image_grid_thw']):
            actual=fast(**dict(batch,native=native))
        assert torch.equal(expected,actual)
        assert torch.equal(expected,fast(**batch))
        with pytest.raises(RuntimeError,match='sentinel'):
            with api().cached_vision(fast.backbone,vision,native['image_grid_thw']):raise RuntimeError('sentinel')
        assert torch.equal(expected,fast(**batch))


def test_snapshot_unpack_preserves_head_order_and_holds_target_positions():
    keys=torch.arange(2*6*20*8,dtype=torch.float32).reshape(2,6,20,8)
    values=-keys
    p=torch.arange(20).view(1,1,20).expand(3,2,20).clone()
    cache=SimpleNamespace(keys=keys,values=values,positions=p,valid=torch.ones(2,20,dtype=torch.bool),index={'a':0,'b':1})
    snap=api().snapshot_for(cache,'a','b',kv_heads=2)
    assert snap.layers[0].key.shape==(1,2,20,4)
    assert torch.equal(snap.layers[0].key[0,:,0].flatten(),keys[1,0,0])
    assert torch.equal(snap.position_ids,p[:,:1])
    assert snap.layers[0].key.untyped_storage().data_ptr()==keys.untyped_storage().data_ptr()
    assert not api().snapshot_for(cache,'a','b',kv_heads=2,masked=True).valid_mask.any()
