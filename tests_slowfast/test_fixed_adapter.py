"""The portable fixed baseline keeps one constant source and the native head."""
import copy
import hashlib
import subprocess
import sys

import pytest
import torch
from torch import nn


def example():
    from slowfast_v11.reader_fusion import ResidualFusion

    torch.manual_seed(42)
    fusion = ResidualFusion(fast_width=12, reader_width=768, hidden=16)
    with torch.no_grad():
        fusion.residual.weight.normal_(std=.05)
        fusion.residual.bias.normal_(std=.01)
    reference = torch.randn(1, 8, 768)
    current = torch.randn(3, 8, 12)
    head = nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 7), nn.GELU(), nn.Linear(7, 3))
    return fusion, reference, current, head.eval().requires_grad_(False)


def test_constant_adapter_exactly_matches_reference_fusion_and_is_frozen():
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter

    fusion, reference, current, head = example()
    expected = head(fusion(current, reference.expand(len(current), -1, -1)))
    adapter = FixedFeatureAdapter(fusion, reference)
    assert torch.equal(head(adapter(current)), expected)
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert "reference" in dict(adapter.named_buffers())
    assert torch.equal(adapter.reference, reference)
    assert not adapter.reference.requires_grad
    with torch.no_grad():
        reference.add_(1)
        fusion.residual.weight.add_(1)
    assert torch.equal(head(adapter(current)), expected)


@pytest.mark.parametrize("reference", [
    torch.zeros(2, 8, 768), torch.zeros(1, 7, 768), torch.zeros(1, 8, 12),
    torch.full((1, 8, 768), float("nan")), torch.zeros(1, 8, 768, dtype=torch.long),
])
def test_adapter_rejects_nonconstant_or_invalid_reference(reference):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter

    fusion, _, _, _ = example()
    with pytest.raises(ValueError, match="reference"):
        FixedFeatureAdapter(fusion, reference)


@pytest.mark.parametrize("current", [
    torch.zeros(0, 8, 12), torch.zeros(2, 7, 12), torch.zeros(2, 8, 13),
    torch.full((1, 8, 12), float("inf")), torch.zeros(1, 8, 12, dtype=torch.long),
])
def test_adapter_rejects_invalid_current_features(current):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter

    fusion, reference, _, _ = example()
    with pytest.raises(ValueError, match="current"):
        FixedFeatureAdapter(fusion, reference)(current)


def test_fusion_source_width_and_nonfinite_weights_are_rejected():
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter
    from slowfast_v11.reader_fusion import ResidualFusion

    fusion, reference, _, _ = example()
    with pytest.raises(ValueError, match="768"):
        FixedFeatureAdapter(ResidualFusion(12, reader_width=18), reference)
    with torch.no_grad():
        fusion.gate.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        FixedFeatureAdapter(fusion, reference)


def test_reference_does_not_depend_on_target_index_and_no_source_argument_exists():
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter

    fusion, reference, current, _ = example()
    adapter = FixedFeatureAdapter(fusion, reference)
    repeated = current[:1].expand(3, -1, -1).clone()
    result = adapter(repeated)
    assert torch.equal(result[0], result[1]) and torch.equal(result[1], result[2])
    assert torch.equal(adapter.reference, reference)
    with pytest.raises(TypeError):
        adapter(current, reference)


def test_binding_uses_original_head_once_and_removes_hook_after_failure():
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter, bind_fixed_adapter

    fusion, reference, current, head = example()
    adapter = FixedFeatureAdapter(fusion, reference)
    expected = head(adapter(current))
    seen = []
    handle = head.register_forward_hook(lambda module, args, output: seen.append(output))
    before = len(head._forward_pre_hooks)
    with bind_fixed_adapter(head, adapter):
        assert torch.equal(head(current), expected)
        with pytest.raises(ValueError, match="one original head"):
            head(current)
    assert len(seen) == 1
    assert len(head._forward_pre_hooks) == before
    with pytest.raises(RuntimeError, match="abort"):
        with bind_fixed_adapter(head, adapter):
            raise RuntimeError("abort")
    assert len(head._forward_pre_hooks) == before
    handle.remove()


def test_disabled_binding_has_native_head_input_and_no_adapter_access():
    from slowfast_v11.fixed_adapter import bind_fixed_adapter

    _, _, current, head = example()
    expected = head(current)
    before = len(head._forward_pre_hooks)
    with bind_fixed_adapter(head, None, enabled=False):
        assert len(head._forward_pre_hooks) == before
        assert torch.equal(head(current), expected)


def test_bundle_roundtrip_authenticates_file_fast_checkpoint_and_head(tmp_path):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter, load_fixed_adapter, save_fixed_adapter

    fusion, reference, current, head = example()
    adapter = FixedFeatureAdapter(fusion, reference)
    path = tmp_path / "baseline.pt"
    digest = save_fixed_adapter(path, adapter, fast_checkpoint_sha256="a" * 64, head=head)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="a" * 64, head=head)
    assert torch.equal(head(loaded(current)), head(adapter(current)))
    assert torch.equal(loaded.reference, reference)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(ValueError, match="bundle SHA256"):
        load_fixed_adapter(path, expected_sha256="b" * 64, fast_checkpoint_sha256="a" * 64, head=head)
    with pytest.raises(ValueError, match="Fast checkpoint"):
        load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="c" * 64, head=head)
    changed_head = copy.deepcopy(head)
    with torch.no_grad():
        changed_head[1].bias.add_(1)
    with pytest.raises(ValueError, match="head identity"):
        load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="a" * 64, head=changed_head)
    changed_topology = copy.deepcopy(head)
    changed_topology[2] = nn.ReLU()
    with pytest.raises(ValueError, match="head identity"):
        load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="a" * 64, head=changed_topology)
    # Loading/inference needs only PyTorch and this module, not the training Reader module.
    head_path = tmp_path / "head.pt"
    torch.save(head.state_dict(), head_path)
    script = (
        "import sys; import torch; from torch import nn; "
        "from slowfast_v11.fixed_adapter import load_fixed_adapter; "
        "head=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,7),nn.GELU(),nn.Linear(7,3)); "
        "head.load_state_dict(torch.load(sys.argv[3],weights_only=True)); "
        "adapter=load_fixed_adapter(sys.argv[1],expected_sha256=sys.argv[2],fast_checkpoint_sha256='a'*64,head=head); "
        "assert adapter(torch.zeros(1,8,12)).shape==(1,8,12); "
        "assert 'slowfast_v11.reader_fusion' not in sys.modules; "
        "assert 'slowfast_v11.readability' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script, str(path), digest, str(head_path)], check=True)


def test_constructing_and_loading_bundle_do_not_perturb_training_rng(tmp_path):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter, load_fixed_adapter, save_fixed_adapter

    fusion, reference, _, head = example()
    before = torch.get_rng_state().clone()
    adapter = FixedFeatureAdapter(fusion, reference)
    assert torch.equal(torch.get_rng_state(), before)
    path = tmp_path / "baseline.pt"
    digest = save_fixed_adapter(path, adapter, fast_checkpoint_sha256="a" * 64, head=head)
    load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="a" * 64, head=head)
    assert torch.equal(torch.get_rng_state(), before)


def test_fixed_binding_runs_native_fast_without_a_slow_or_reader(language_factory, native_input):
    from fast_navsim.model import NavsimPlanner
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter, bind_fixed_adapter
    from slowfast_v11.reader_fusion import ResidualFusion

    torch.manual_seed(4)
    fast = NavsimPlanner(language_factory(), lora_rank=2, lora_alpha=4, lora_dropout=0.).eval().requires_grad_(False)
    fusion = ResidualFusion(32, hidden=16)
    with torch.no_grad():
        fusion.residual.weight.normal_(std=.05)
    adapter = FixedFeatureAdapter(fusion, torch.randn(1, 8, 768))
    ego, command = torch.ones(1, 4), torch.tensor([1])
    captured = []
    handle = fast.head.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        native = fast(native_input, ego, command)
    handle.remove()
    assert len(captured) == 1
    expected = fast.head(fusion(captured[0], adapter.reference))
    calls = []
    handle = fast.head.register_forward_hook(lambda module, args, output: calls.append(True))
    with torch.no_grad(), bind_fixed_adapter(fast.head, adapter):
        result = fast(native_input, ego, command)
    assert torch.equal(result, expected) and len(calls) == 1
    with torch.no_grad(), bind_fixed_adapter(fast.head, None, enabled=False):
        assert torch.equal(fast(native_input, ego, command), native)
    handle.remove()
    assert len(fast.head._forward_pre_hooks) == 0


@pytest.mark.parametrize("malformation", ["schema", "reference", "weights", "extra"])
def test_authenticated_but_malformed_bundle_is_rejected(tmp_path, malformation):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter, load_fixed_adapter, save_fixed_adapter

    fusion, reference, _, head = example()
    path = tmp_path / "baseline.pt"
    save_fixed_adapter(path, FixedFeatureAdapter(fusion, reference), fast_checkpoint_sha256="a" * 64, head=head)
    bundle = torch.load(path, weights_only=True)
    if malformation == "schema":
        bundle["schema"] = "unknown"
    elif malformation == "reference":
        bundle["reference"] = reference.expand(2, -1, -1).clone()
    elif malformation == "weights":
        bundle["fusion_state"]["gate.bias"][0] = float("nan")
    else:
        bundle["target_indices"] = torch.arange(8)
    torch.save(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_fixed_adapter(path, expected_sha256=digest, fast_checkpoint_sha256="a" * 64, head=head)
