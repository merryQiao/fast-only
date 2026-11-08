"""Full language scope and master weights, without a model download or CUDA."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class TinyLoRALinear(nn.Module):
    """A real additive adapter with the same A/B module layout as PEFT."""

    def __init__(self, width):
        super().__init__()
        self.base_layer = nn.Linear(width, width)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(width, 2, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(2, width, bias=False)})
        self.observed = []

    def forward(self, x):
        base = self.base_layer(x)
        adapter = self.lora_B['default'](self.lora_A['default'](x.float()))
        self.observed.append((base.dtype, adapter.dtype))
        return (base + adapter).to(base.dtype)


class FakeTinyLanguage(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, width)
        self.layers = nn.ModuleList([TinyLoRALinear(width), TinyLoRALinear(width)])
        self.norm = nn.LayerNorm(width)
        self.gradient_checkpointing = True
        self.observed = []
        self.raise_in_forward = False

    def forward(self, *, inputs_embeds, use_cache=False, **kwargs):
        self.observed.append((inputs_embeds.dtype, torch.is_autocast_enabled('cpu'), use_cache))
        if self.raise_in_forward:
            raise RuntimeError('native failure')
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = hidden + layer(hidden).tanh()
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class TinyFast(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.language_model = FakeTinyLanguage()
        self.backbone.visual = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.backbone.unused = nn.Parameter(torch.ones(1))
        self.ego_mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 8))
        self.command_embedding = nn.Embedding(4, 8)
        self.metadata_encoder = nn.Sequential(nn.Linear(6, 8), nn.GELU(), nn.Linear(8, 8))
        self.no_cache = nn.Parameter(torch.randn(8))
        self.queries = nn.Parameter(torch.randn(8, 8))
        self.head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 3))
        self.backbone.requires_grad_(False)

    def forward(self, *, snapshot=False):
        language = self.backbone.language_model
        tokens = language.embed_tokens(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        ego = self.ego_mlp(torch.ones(1, 4))
        command = self.command_embedding(torch.tensor([1]))
        metadata = self.metadata_encoder(torch.ones(1, 6)) if snapshot else self.no_cache[None]
        embeds = tokens + ego[:, None] + command[:, None] + metadata[:, None] + self.queries
        hidden = language(inputs_embeds=embeds, use_cache=False).last_hidden_state
        return self.head(hidden.float())


@pytest.fixture
def tiny():
    torch.manual_seed(17)
    return TinyFast()


def api():
    from slowfast_v11 import full_training
    return full_training


def test_configure_trains_every_language_base_weight_and_outer_module(tiny):
    f = api()
    original = {n: p.detach().clone() for n, p in tiny.named_parameters()}
    report = f.configure_full_fast(tiny)
    language = tiny.backbone.language_model
    assert all(p.requires_grad and p.dtype == torch.float32 for p in language.parameters())
    assert all(p.requires_grad and p.dtype == torch.float32
               for n, p in tiny.named_parameters() if not n.startswith('backbone.'))
    assert all(not p.requires_grad and p.dtype == torch.bfloat16
               for p in tiny.backbone.visual.parameters())
    assert not tiny.backbone.visual.training
    assert not tiny.backbone.unused.requires_grad
    assert language.gradient_checkpointing is False
    assert report['language']['numel'] == sum(p.numel() for p in language.parameters())
    assert report['language_base']['numel'] > report['language_lora']['numel']
    assert set(report['language']['names']) == {
        n for n, _ in tiny.named_parameters() if n.startswith('backbone.language_model.')}
    assert all(torch.equal(original[n], p) for n, p in tiny.named_parameters()
               if p.requires_grad)


def test_optimizer_has_exact_full_language_and_new_groups(tiny):
    f = api()
    f.configure_full_fast(tiny)
    optimizer = f.make_full_optimizer(tiny)
    groups = {group['name']: group for group in optimizer.param_groups}
    assert set(groups) == {'language', 'new'}
    assert groups['language']['lr'] == 1e-5 and groups['new']['lr'] == 5e-5
    assert all(g['weight_decay'] == .01 for g in groups.values())
    language_ids = {id(p) for p in tiny.backbone.language_model.parameters()}
    assert {id(p) for p in groups['language']['params']} == language_ids
    all_ids = [id(p) for g in groups.values() for p in g['params']]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in tiny.parameters() if p.requires_grad}
    assert not set(all_ids) & {id(p) for p in tiny.backbone.visual.parameters()}


def test_full_state_preserves_base_weights_and_cpu_clone_isolation(tiny):
    f = api()
    f.configure_full_fast(tiny)
    saved = f.full_state(tiny)
    trainable = dict((n, p) for n, p in tiny.named_parameters() if p.requires_grad)
    assert set(saved) == set(trainable)
    assert any('.base_layer.weight' in name for name in saved)
    assert 'backbone.language_model.embed_tokens.weight' in saved
    assert not any('visual' in name or name == 'backbone.unused' for name in saved)
    for name, value in saved.items():
        assert value.device.type == 'cpu' and value.dtype == torch.float32
        assert not value.requires_grad and value.data_ptr() != trainable[name].data_ptr()
    with torch.no_grad():
        for p in trainable.values():
            p.add_(torch.randn_like(p))
    f.restore_full_state(tiny, saved)
    assert all(torch.equal(p.view(torch.uint8), saved[n].view(torch.uint8))
               for n, p in trainable.items())


@pytest.mark.parametrize('damage', ['missing_base', 'extra', 'shape', 'dtype', 'not_tensor'])
def test_full_restore_rejects_invalid_state_before_any_copy(tiny, damage):
    f = api()
    f.configure_full_fast(tiny)
    before = f.full_state(tiny)
    saved = {n: value + 1 for n, value in before.items()}
    name = next(n for n in saved if '.base_layer.weight' in n)
    if damage == 'missing_base':
        saved.pop(name)
    elif damage == 'extra':
        saved['backbone.visual.weight'] = torch.zeros(1)
    elif damage == 'shape':
        saved[name] = saved[name][:-1]
    elif damage == 'dtype':
        saved[name] = saved[name].to(torch.bfloat16)
    else:
        saved[name] = None
    with pytest.raises(ValueError, match='state|checkpoint|Checkpoint'):
        f.restore_full_state(tiny, saved)
    assert all(torch.equal(value, before[n]) for n, value in f.full_state(tiny).items())


def test_precision_uses_bf16_native_inputs_and_fp32_lora_and_head(tiny):
    f = api()
    f.configure_full_fast(tiny)
    observed = {}
    hooks = []
    for name, module in [('ego', tiny.ego_mlp), ('ego_linear', tiny.ego_mlp[0]),
                         ('metadata', tiny.metadata_encoder), ('head', tiny.head),
                         ('head_norm', tiny.head[0]), ('head_linear', tiny.head[-1])]:
        hooks.append(module.register_forward_hook(
            lambda module, args, output, name=name: observed.__setitem__(name, output.dtype)))
    try:
        with f.full_precision(tiny), torch.autocast('cpu', dtype=torch.bfloat16):
            result = tiny(snapshot=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert result.dtype == torch.float32
    assert observed == dict(ego=torch.float32, ego_linear=torch.float32, metadata=torch.float32,
                            head=torch.float32, head_norm=torch.float32, head_linear=torch.float32)
    assert tiny.backbone.language_model.observed[-1] == (torch.bfloat16, True, False)
    assert all(layer.observed[-1] == (torch.bfloat16, torch.float32)
               for layer in tiny.backbone.language_model.layers)
    assert all(p.dtype == torch.float32 for p in tiny.parameters() if p.requires_grad)


def test_precision_context_restores_all_original_forwards_after_exception(tiny):
    f = api()
    f.configure_full_fast(tiny)
    originals = {n: m.forward for n, m in tiny.named_modules()}
    own_forwards = {n for n, m in tiny.named_modules() if 'forward' in m.__dict__}
    tiny.backbone.language_model.raise_in_forward = True
    with pytest.raises(RuntimeError, match='native failure'):
        with f.full_precision(tiny):
            tiny()
    assert all(m.forward == originals[n] for n, m in tiny.named_modules())
    assert {n for n, m in tiny.named_modules() if 'forward' in m.__dict__} == own_forwards


def test_install_once_policy_supports_repeated_calls_and_explicit_close(tiny):
    f = api()
    f.configure_full_fast(tiny)
    original = tiny.backbone.language_model.forward
    policy = f.install_full_precision(tiny)
    try:
        assert tiny().dtype == torch.float32
        assert tiny().dtype == torch.float32
        with pytest.raises(RuntimeError, match='already|active|installed'):
            f.install_full_precision(tiny)
    finally:
        policy.close()
    policy.close()
    assert tiny.backbone.language_model.forward == original


def test_full_backward_updates_dense_base_and_allows_unused_no_cache_modules(tiny):
    f = api()
    f.configure_full_fast(tiny)
    optimizer = f.make_full_optimizer(tiny)
    before = f.full_state(tiny)
    with f.full_precision(tiny):
        loss = (tiny() - torch.ones(1, 8, 3)).abs().mean()
        loss.backward()
    language = tiny.backbone.language_model
    for layer in language.layers:
        gradient = layer.base_layer.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero()
    assert language.embed_tokens.weight.grad.count_nonzero()
    assert tiny.no_cache.grad is not None
    assert all(p.grad is None for p in tiny.metadata_encoder.parameters())
    assert all(p.grad is None for p in tiny.backbone.visual.parameters())
    optimizer.step()
    for name, p in tiny.named_parameters():
        if '.base_layer.weight' in name:
            assert not torch.equal(p, before[name])


def test_precision_rejects_checkpointing_enabled_after_configuration(tiny):
    f = api()
    f.configure_full_fast(tiny)
    tiny.backbone.language_model.gradient_checkpointing = True
    with f.full_precision(tiny), pytest.raises(ValueError, match='checkpoint'):
        tiny()


@pytest.mark.parametrize('kwargs', [dict(language_lr=0), dict(new_lr=float('nan')),
                                   dict(weight_decay=-1)])
def test_optimizer_rejects_invalid_hyperparameters(tiny, kwargs):
    f = api()
    f.configure_full_fast(tiny)
    with pytest.raises(ValueError):
        f.make_full_optimizer(tiny, **kwargs)


def test_optimizer_rejects_adapter_only_scope(tiny):
    with pytest.raises(ValueError, match='configure|scope|language'):
        api().make_full_optimizer(tiny)


def test_native_qwen_precision_audit_and_full_base_update(
        language_factory, native_input, record_property):
    from slowfast_v11.model import FastWithSlowKV
    from slowfast_v11.validation import canonical_state

    f = api()
    torch.manual_seed(123)
    fast = FastWithSlowKV(language_factory().to(torch.bfloat16), lora_rank=2,
                          lora_alpha=4, lora_dropout=0., lora_parameter_dtype='float32',
                          position_offset=128).eval()
    ego, command = torch.ones(1, 4), torch.tensor([1])
    with torch.no_grad():
        # Use resumed nonzero adapters, so preserving the actual A/B weights matters.
        for name, p in fast.named_parameters():
            if '.lora_B.' in name:
                p.normal_(std=.01)
        original = fast(native_input, ego, command).clone()
    f.configure_full_fast(fast)
    saved = f.full_state(fast)
    adapter_only = canonical_state(fast)
    assert set(adapter_only) < set(saved)
    assert len(fast.backbone.language_model.layers) == 24
    optimizer = f.make_full_optimizer(fast)
    with f.full_precision(fast):
        result = fast(native_input, ego, command)
        # Whole-language AMP also affects the torch GDN fallback's explicitly
        # upcast matmuls. Audit this policy change; test exactness within the
        # same policy below rather than claiming cross-policy byte equality.
        precision_delta = (result.detach() - original).abs().max().item()
        record_property('legacy_bf16_max_abs_precision_delta', precision_delta)
        record_property('precision_audit_scope', 'CPU native torch GDN fallback, BF16 AMP vs no AMP')
        assert torch.isfinite(result).all()
        initialized_prediction = result.detach().clone()
        result.abs().mean().backward()
    for index, layer in enumerate(fast.backbone.language_model.layers):
        named = [(name, p) for name, p in layer.named_parameters() if 'lora_' not in name]
        assert all(p.requires_grad and p.dtype == torch.float32 for _, p in named)
        grads = [p.grad for _, p in named if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), index
        assert any(g.count_nonzero() for g in grads), index
    assert fast.backbone.language_model.embed_tokens.weight.grad.count_nonzero()
    assert fast.backbone.language_model.norm.weight.grad is not None
    assert all(p.grad is None for p in fast.backbone.visual.parameters())
    assert all(p.grad is None for p in fast.metadata_encoder.parameters())
    optimizer.step()
    for index, layer in enumerate(fast.backbone.language_model.layers):
        assert any(not torch.equal(p, saved[f'backbone.language_model.layers.{index}.{name}'])
                   for name, p in layer.named_parameters() if 'lora_' not in name), index
    f.restore_full_state(fast, saved)
    assert all(torch.equal(value, saved[name]) for name, value in f.full_state(fast).items())
    with torch.no_grad(), f.full_precision(fast):
        restored_prediction = fast(native_input, ego, command)
    torch.testing.assert_close(restored_prediction, initialized_prediction, rtol=0, atol=0)


def test_native_qwen_full_precision_direct_kv_gradients(language_factory, native_input):
    from slowfast_v11.model import FastWithSlowKV, SlowPrefill

    f = api()
    torch.manual_seed(321)
    kw = dict(lora_rank=2, lora_alpha=4, lora_dropout=0., lora_parameter_dtype='float32',
              position_offset=128)
    fast = FastWithSlowKV(language_factory().to(torch.bfloat16), **kw).eval()
    slow = SlowPrefill(language_factory().to(torch.bfloat16), **kw).eval().requires_grad_(False)
    ego, command = torch.ones(1, 4), torch.tensor([1])
    with torch.no_grad():
        snapshot = slow(native_input, ego, command).snapshot.inference_copy()
    assert all(layer.key.dtype == torch.bfloat16 for layer in snapshot.layers)
    f.configure_full_fast(fast)
    with f.full_precision(fast):
        result = fast(native_input, ego, command, snapshot=snapshot,
                      metadata=torch.tensor([[0., 0., 0., 0., 1., 1.]]))
        assert result.dtype == torch.float32
        result.abs().mean().backward()
    assert all(p.grad is None for p in slow.parameters())
    assert all(p.grad is None for p in fast.backbone.visual.parameters())
    assert fast.no_cache.grad is not None and fast.no_cache.grad.count_nonzero() == 0
    assert any(p.grad is not None and p.grad.count_nonzero() for p in fast.metadata_encoder.parameters())
    assert fast.bridge._call is None
    for layer in fast.backbone.language_model.layers:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
                   for name, p in layer.named_parameters() if 'lora_' not in name)
