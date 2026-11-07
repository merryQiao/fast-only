import pytest
import torch
from slowfast_v11.model import FastWithSlowKV,SlowPrefill
from slowfast_v11.content_diagnostics import attention_probe,source_groups


def api():
    import slowfast_v11.path_diagnostics as module
    return module


def test_empty_path_holds_no_cache_token_and_traces_all_layers(language_factory,native_input):
    fast=FastWithSlowKV(language_factory(),lora_rank=2,lora_alpha=4,position_offset=128).eval()
    slow=SlowPrefill(language_factory(64),lora_rank=2,lora_alpha=4,position_offset=128).eval()
    ego=torch.zeros(1,4);command=torch.tensor([1]);metadata=torch.tensor([[0.,0.,0.,0.,1.,1.]])
    batch=dict(native=native_input,ego=ego,command=command)
    groups=source_groups(native_input,61,2)
    with torch.no_grad():
        snap=slow(**batch).snapshot
        with api().trace_fast(fast,full_context=True) as original:
            plain=fast(**batch)
        with api().force_no_cache(fast),attention_probe(groups,mask_external=True) as fusion:
            with api().trace_fast(fast,full_context=True) as traced:
                empty=fast(**batch,snapshot=snap,metadata=metadata)
        assert len(fusion)==6
        assert torch.allclose(plain,empty,atol=2e-6,rtol=2e-6)
        differences=api().compare_traces(original,traced)
        assert len([k for k in differences if k.startswith('layer_')])==24
        assert 'head_input' in differences and 'post_gate_4' in differences and 'post_projection_24' in differences
        assert all(v['max_abs_difference']<2e-6 for v in differences.values())
        assert torch.equal(fast(**batch),plain)


def test_force_no_cache_is_scoped_and_restores_after_exception(language_factory):
    model=FastWithSlowKV(language_factory(),lora_rank=2,lora_alpha=4).eval()
    value=torch.tensor([[0.,0.,0.,0.,1.,1.]])
    with torch.no_grad():
        before=model.metadata_encoder(value)
        with pytest.raises(RuntimeError,match='sentinel'):
            with api().force_no_cache(model):
                assert torch.equal(model.metadata_encoder(value),model.no_cache[None])
                raise RuntimeError('sentinel')
        assert torch.equal(before,model.metadata_encoder(value))
    with pytest.raises(RuntimeError,match='inference'):
        with api().force_no_cache(model):pass


def test_trace_difference_reports_real_deviation():
    a={'head_input':torch.tensor([1.,2.])};b={'head_input':torch.tensor([2.,2.])}
    d=api().compare_traces(a,b)['head_input']
    assert d['max_abs_difference']==1.
    assert d['rms_difference']==pytest.approx(2**-.5)
