"""Frozen constant-feature inference without a Slow model or Reader import."""

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.nn import functional as F


_SCHEMA = "slowfast-fixed-feature-adapter-v1"
_CONFIG_KEYS = {"fast_width", "reader_width", "hidden", "fast_norm_eps", "reader_norm_eps"}
_BUNDLE_KEYS = {"schema", "fast_checkpoint_sha256", "head_identity_sha256", "fusion_config",
                "fusion_state", "reference"}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_reference(reference):
    if (not isinstance(reference, torch.Tensor) or reference.shape != (1, 8, 768)
            or not reference.is_floating_point() or not bool(torch.isfinite(reference).all())):
        raise ValueError("Expected one finite constant reference[1,8,768]")


def _validate_state(state):
    if (not isinstance(state, Mapping) or not state
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                   or not bool(torch.isfinite(value).all()) for value in state.values())):
        raise ValueError("Fusion weights must be finite float32 tensors")


class _FrozenFusion(nn.Module):
    """The original ResidualFusion arithmetic, with portable frozen weights."""

    def __init__(self, fast_width, hidden, reader_width=768, fast_norm_eps=1e-5, reader_norm_eps=1e-5):
        super().__init__()
        self.fast_width = _positive_integer(fast_width, "fast_width")
        _positive_integer(hidden, "hidden")
        if reader_width != 768:
            raise ValueError("Fixed fusion source width must be 768")
        self.reader_width = 768
        for eps in (fast_norm_eps, reader_norm_eps):
            if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not 0 < eps < float("inf"):
                raise ValueError("LayerNorm epsilon must be positive and finite")
        # All initialized values are replaced by authenticated trained weights.
        # Export/reload must not advance the paired fusion-training RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.fast_norm = nn.LayerNorm(fast_width, eps=fast_norm_eps)
            self.reader_norm = nn.LayerNorm(768, eps=reader_norm_eps)
            self.combine = nn.Linear(fast_width + 768, hidden)
            self.gate = nn.Linear(hidden, 1)
            self.residual = nn.Linear(hidden, fast_width)

    def forward(self, current, source):
        current, source = current.detach().float(), source.detach().float()
        hidden = F.gelu(self.combine(torch.cat((self.fast_norm(current), self.reader_norm(source)), dim=-1)))
        return current + self.gate(hidden).sigmoid() * self.residual(hidden)


class FixedFeatureAdapter(nn.Module):
    """Exported fusion weights plus one target-independent R0[1,8,768].

    The caller supplies only current Fast features and applies its original
    frozen head to the returned features. Historical source-model training cost
    remains part of this baseline; per-window source computation is absent.
    """

    def __init__(self, fusion, reference):
        super().__init__()
        _validate_reference(reference)
        if not isinstance(fusion, nn.Module) or getattr(fusion, "reader_width", None) != 768:
            raise ValueError("Expected ResidualFusion with source width 768")
        state = fusion.state_dict()
        _validate_state(state)
        try:
            self.fusion = _FrozenFusion(fusion.fast_width, fusion.combine.out_features,
                                        fast_norm_eps=fusion.fast_norm.eps, reader_norm_eps=fusion.reader_norm.eps)
        except AttributeError as error:
            raise ValueError("Expected the existing ResidualFusion topology") from error
        first = next(fusion.parameters())
        self.fusion.to(device=first.device)
        try:
            self.fusion.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Invalid ResidualFusion weight shapes or topology") from error
        self.register_buffer("reference", reference.detach().to(first.device).clone())
        self.requires_grad_(False)
        self.eval()

    def forward(self, current):
        if (not isinstance(current, torch.Tensor) or current.ndim != 3 or current.shape[0] == 0
                or current.shape[1:] != (8, self.fusion.fast_width) or not current.is_floating_point()
                or current.device != self.reference.device or not bool(torch.isfinite(current).all())):
            raise ValueError("Expected finite current Fast features[B,8,fast_width] on the adapter device")
        return self.fusion(current, self.reference.expand(len(current), -1, -1))


@contextmanager
def bind_fixed_adapter(head, adapter, *, enabled=True):
    """Apply the adapter immediately before one original Fast head call.

    Disabled mode installs no hook and does not inspect or call the adapter.
    The hook is removed even if native inference raises an exception.
    """
    if not enabled:
        yield
        return
    if not isinstance(adapter, FixedFeatureAdapter):
        raise ValueError("Expected a FixedFeatureAdapter")
    called = False

    def replace(module, args):
        nonlocal called
        if called:
            raise ValueError("Fixed adapter expects one original head invocation")
        called = True
        if not args:
            raise ValueError("Original head requires current Fast features")
        return (adapter(args[0]), *args[1:])

    handle = head.register_forward_pre_hook(replace)
    try:
        yield
    finally:
        handle.remove()


def _sha256_string(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _hash_file(file):
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _head_identity(head):
    """Authenticate topology (including activations) as well as every tensor."""
    if not isinstance(head, nn.Module):
        raise ValueError("Expected the original Fast head module")
    topology = [(name, f"{type(module).__module__}.{type(module).__qualname__}", module.extra_repr())
                for name, module in head.named_modules()]
    digest = hashlib.sha256(json.dumps(topology, separators=(",", ":")).encode())
    for name, value in sorted(head.state_dict().items()):
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("Original head state must contain finite tensors")
        value = value.detach().cpu().contiguous()
        descriptor = [name, str(value.dtype), list(value.shape)]
        digest.update(json.dumps(descriptor, separators=(",", ":")).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_fixed_adapter(path, adapter, *, fast_checkpoint_sha256, head):
    """Atomically export a CPU tensor bundle; return its SHA256 for the manifest.

    The Fast checkpoint digest must come from the authenticated source file.
    The original head remains in Fast and is identified rather than duplicated.
    """
    if not isinstance(adapter, FixedFeatureAdapter):
        raise ValueError("Expected a FixedFeatureAdapter")
    _sha256_string(fast_checkpoint_sha256, "Fast checkpoint SHA256")
    _validate_reference(adapter.reference)
    _validate_state(adapter.fusion.state_dict())
    fusion = adapter.fusion
    bundle = {
        "schema": _SCHEMA,
        "fast_checkpoint_sha256": fast_checkpoint_sha256,
        "head_identity_sha256": _head_identity(head),
        "fusion_config": {"fast_width": fusion.fast_width, "reader_width": 768,
                          "hidden": fusion.combine.out_features, "fast_norm_eps": fusion.fast_norm.eps,
                          "reader_norm_eps": fusion.reader_norm.eps},
        "fusion_state": {name: value.detach().cpu().clone() for name, value in fusion.state_dict().items()},
        "reference": adapter.reference.detach().cpu().clone(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
            temporary = Path(file.name)
            torch.save(bundle, file)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with path.open("rb") as file:
        return _hash_file(file)


def load_fixed_adapter(path, *, expected_sha256, fast_checkpoint_sha256, head, map_location="cpu"):
    """Verify file, frozen Fast identity and head before loading tensor-only data."""
    _sha256_string(expected_sha256, "Expected bundle SHA256")
    _sha256_string(fast_checkpoint_sha256, "Fast checkpoint SHA256")
    with Path(path).open("rb") as file:
        if not hmac.compare_digest(_hash_file(file), expected_sha256):
            raise ValueError("Fixed adapter bundle SHA256 mismatch")
        file.seek(0)
        bundle = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or set(bundle) != _BUNDLE_KEYS or bundle["schema"] != _SCHEMA:
        raise ValueError("Unsupported fixed adapter bundle schema")
    if bundle["fast_checkpoint_sha256"] != fast_checkpoint_sha256:
        raise ValueError("Fixed adapter Fast checkpoint mismatch")
    if bundle["head_identity_sha256"] != _head_identity(head):
        raise ValueError("Fixed adapter original head identity mismatch")
    config = bundle["fusion_config"]
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("Invalid fixed fusion configuration")
    _validate_reference(bundle["reference"])
    _validate_state(bundle["fusion_state"])
    fusion = _FrozenFusion(**config)
    try:
        fusion.load_state_dict(bundle["fusion_state"], strict=True)
    except RuntimeError as error:
        raise ValueError("Invalid fixed fusion weight shapes or topology") from error
    return FixedFeatureAdapter(fusion, bundle["reference"]).to(map_location)
