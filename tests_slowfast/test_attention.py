from dataclasses import replace
import pytest
import torch

from slowfast_v11.attention import KVBridge, check_compatibility, planning_attention
from slowfast_v11.protocol import LayerKV


def lm_input(width, batch=1, length=16, offset=0):
    x=torch.randn(batch,length,width)
    positions=torch.arange(length).view(1,1,-1).expand(3,batch,-1)+offset
    mask=torch.ones(batch,length,dtype=torch.long)
    return x,positions,mask


def run(lm,x,p,m):
    return lm(inputs_embeds=x,position_ids=p,attention_mask=m,use_cache=False).last_hidden_state


def test_capture_is_normalized_pre_rope_and_preserves_graph(language_factory):
    lm=language_factory(64).language_model.eval()
    bridge=KVBridge(lm,position_offset=128)
    x,p,m=lm_input(64)
    raw=[]
    h=lm.layers[3].self_attn.k_norm.register_forward_hook(lambda mod,args,out:raw.append(out))
    with bridge.capture(p,m) as call:run(lm,x,p,m)
    h.remove()
    assert len(call.snapshot.layers)==6
    assert torch.equal(call.snapshot.layers[0].key,raw[0].transpose(1,2))
    assert call.snapshot.layers[0].key.requires_grad
    assert call.snapshot.layers[0].key.shape==(1,1,16,16)
    assert torch.equal(call.snapshot.position_ids,p)
    with pytest.raises(ValueError,match='positions'):
        with bridge.capture(p+128,m):run(lm,x,p,m)


def test_native_context_isolation_and_disabled_bitwise_identity(language_factory):
    slow=language_factory(64).language_model.eval()
    fast=language_factory().language_model.eval()
    assert check_compatibility(fast,slow)['layers_one_based']==[4,8,12,16,20,24]
    sb=KVBridge(slow,position_offset=128);fb=KVBridge(fast,position_offset=128)
    sx,sp,sm=lm_input(64)
    with sb.capture(sp,sm) as call:run(slow,sx,sp,sm)
    snap=call.snapshot
    changed=replace(snap,layers=tuple(LayerKV(z.key+100,z.value+10) for z in snap.layers))
    x,p,m=lm_input(32,offset=128)
    plain=run(fast,x,p,m)
    for candidate in [snap,changed]:
        with fb.fuse(candidate,p,m,torch.tensor([False])):
            masked=run(fast,x,p,m)
        assert torch.equal(plain,masked)
    with fb.fuse(snap,p,m,torch.tensor([True])):active=run(fast,x,p,m)
    with fb.fuse(changed,p,m,torch.tensor([True])):other=run(fast,x,p,m)
    assert torch.equal(active[:,:-8],plain[:,:-8])
    assert torch.equal(active[:,:-8],other[:,:-8])
    assert (active[:,-8:]-other[:,-8:]).abs().max()>1e-5
    assert torch.equal(plain,run(fast,x,p,m))  # Hooks cannot leak state to the next call.


def test_planning_attention_matches_dense_gqa_reference_and_padding():
    torch.manual_seed(7)
    q=torch.randn(2,4,3,8);k=torch.randn(2,2,7,8);v=torch.randn_like(k)
    sk=torch.randn(2,2,5,8);sv=torch.randn_like(sk)
    fm=torch.tensor([[0,0,1,1,1,1,1],[1,1,1,1,1,1,1]],dtype=torch.bool)
    sm=torch.tensor([[0,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    actual=planning_attention(q,k,v,sk,sv,fm,sm)
    allowed=torch.cat((sm[:,None,:].expand(-1,3,-1),
        fm[:,None,:] & (torch.arange(7)[None,:]<=torch.arange(4,7)[:,None])),dim=-1)
    keys=torch.cat((sk,k),dim=2).repeat_interleave(2,dim=1)
    values=torch.cat((sv,v),dim=2).repeat_interleave(2,dim=1)
    logits=(q@keys.transpose(-1,-2))/8**.5
    reference=logits.masked_fill(~allowed[:,None],float('-inf')).softmax(-1)@values
    assert torch.allclose(actual,reference,atol=2e-6,rtol=2e-6)


def test_mixed_gate_batch_and_exception_cleanup(language_factory):
    lm=language_factory().language_model.eval();bridge=KVBridge(lm,position_offset=128)
    x,p,m=lm_input(32,batch=2)
    m[0,:2]=0
    with bridge.capture(p,m) as call:run(lm,x,p,m)
    p=p+128
    plain=run(lm,x,p,m)
    with bridge.fuse(call.snapshot,p,m,torch.tensor([False,True])):
        actual=run(lm,x,p,m)
    assert torch.equal(plain[0],actual[0])
    assert not torch.equal(plain[1,-8:],actual[1,-8:])
    with pytest.raises(RuntimeError,match='sentinel'):
        with bridge.fuse(call.snapshot,p,m,torch.tensor([True,True])):
            raise RuntimeError('sentinel')
    assert torch.equal(plain,run(lm,x,p,m))


def test_bridge_rejects_native_cache_and_reused_or_mismatched_sessions(language_factory):
    lm=language_factory().language_model.eval();bridge=KVBridge(lm,position_offset=128)
    x,p,m=lm_input(32)
    with pytest.raises(ValueError,match='cache'):
        with bridge.capture(p,m):
            lm(inputs_embeds=x,position_ids=p,attention_mask=m,use_cache=True)
    with pytest.raises(ValueError,match='positions'):
        with bridge.capture(p,m):run(lm,x,p+1,m)
    with pytest.raises(RuntimeError,match='one'):
        with bridge.capture(p,m):
            run(lm,x,p,m);run(lm,x,p,m)
    with bridge.capture(p,m) as call:run(lm,x,p,m)
    assert len(call.snapshot.layers)==6
