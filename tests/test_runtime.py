import pytest
import torch

from fast_only.common import apply_reproducibility


POLICY={'deterministic_algorithms':True,'warn_only':False,
        'cublas_workspace_config':':4096:8','cudnn_deterministic':True,
        'cudnn_benchmark':False,'allow_tf32':False}


def test_conflicting_workspace_is_rejected(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':16:8')
    with pytest.raises(ValueError,match='CUBLAS_WORKSPACE_CONFIG'):
        apply_reproducibility({'reproducibility':POLICY})


def test_workspace_cannot_be_set_after_cuda_initialization(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    with pytest.raises(RuntimeError,match='before'):
        apply_reproducibility({'reproducibility':POLICY})


def test_explicit_flags_are_applied_strictly_before_cuda(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    old=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
         torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
         torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        result=apply_reproducibility({'reproducibility':POLICY})
        assert result==POLICY
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.use_deterministic_algorithms(old[0],warn_only=old[1])
        torch.backends.cudnn.deterministic=old[2];torch.backends.cudnn.benchmark=old[3]
        torch.backends.cuda.matmul.allow_tf32=old[4];torch.backends.cudnn.allow_tf32=old[5]
