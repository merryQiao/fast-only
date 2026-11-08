"""Full Fast language training, FP32 masters, and complete trainable state.

The existing LoRA modules stay in place with their resumed weights. These
helpers unfreeze the dense language matrices as well; the adapter-only
``validation.canonical_state`` is deliberately not used by this experiment.
"""
from collections.abc import Mapping
from functools import wraps
import math

import torch
from torch import nn


def _groups(model):
    language_ids = {id(p) for p in model.backbone.language_model.parameters()}
    language, new, frozen = [], [], []
    for name, parameter in model.named_parameters():
        group = (language if id(parameter) in language_ids else
                 frozen if name.startswith('backbone.') else new)
        group.append((name, parameter))
    return language, new, frozen


def _describe(named):
    return dict(names=[name for name, _ in named], tensors=len(named),
                numel=sum(parameter.numel() for _, parameter in named))


def configure_full_fast(model):
    """Train the complete language and existing outer modules, freezing vision.

    Returns JSON-safe parameter names/counts for auditing real dense scope.
    Configure before creating the optimizer. No adapters are merged, replaced,
    or initialized, and existing native attention hooks remain untouched.
    """
    language = model.backbone.language_model
    model.requires_grad_(False)
    named_language, named_new, frozen = _groups(model)
    for _, parameter in named_language + named_new:
        parameter.requires_grad_(True)
        parameter.data = parameter.data.to(dtype=torch.float32)
    # Clear stale gradients if a caller is resuming after adapter diagnostics.
    for parameter in model.parameters():
        parameter.grad = None
    model.backbone.visual.to(dtype=torch.bfloat16).eval()
    disable = getattr(language, 'gradient_checkpointing_disable', None)
    if disable is not None:
        disable()
    for module in language.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False
    return dict(language=_describe(named_language), new=_describe(named_new),
                language_base=_describe([(n, p) for n, p in named_language if 'lora_' not in n]),
                language_lora=_describe([(n, p) for n, p in named_language if 'lora_' in n]),
                frozen=_describe(frozen),
                visual=_describe([(n, p) for n, p in frozen if n.startswith('backbone.visual.')]),
                gradient_checkpointing=False, master_dtype='float32', visual_dtype='bfloat16')


def make_full_optimizer(model, language_lr=1e-5, new_lr=5e-5, weight_decay=.01):
    """AdamW with complete language scope and the original outer-module group."""
    if (not math.isfinite(language_lr) or language_lr <= 0 or
            not math.isfinite(new_lr) or new_lr <= 0 or
            not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError('Full optimizer learning rates/weight decay must be finite and valid')
    language, new, frozen = _groups(model)
    if (not language or not new or
            any(not p.requires_grad or p.dtype != torch.float32 for _, p in language + new) or
            any(p.requires_grad for _, p in frozen)):
        raise ValueError('Full training scope requires configure_full_fast before the optimizer')
    return torch.optim.AdamW([
        dict(name='language', params=[p for _, p in language], lr=language_lr),
        dict(name='new', params=[p for _, p in new], lr=new_lr),
    ], weight_decay=weight_decay)


def full_state(model):
    """Independent CPU clones of every trainable parameter, including bases."""
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_full_state(model, state):
    """Validate all trainable keys, dtypes and shapes before any in-place copy."""
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not isinstance(state, Mapping) or set(state) != set(expected):
        actual = set(state) if isinstance(state, Mapping) else set()
        raise ValueError(f'Full checkpoint state keys mismatch: '
                         f'missing={sorted(set(expected) - actual)}, '
                         f'extra={sorted(actual - set(expected))}')
    for name, parameter in expected.items():
        tensor = state[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape or
                tensor.dtype != parameter.dtype):
            raise ValueError(f'Full checkpoint state shape/dtype mismatch for {name}')
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name])


def _float_input(value):
    return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value


class _FullPrecision:
    """An install-once policy removable explicitly or by a context boundary."""

    def __init__(self, model):
        self.model = model
        self._originals = []
        self.active = False

    def _replace(self, module, wrapper):
        # Restore absence of an instance override as well as its original value.
        self._originals.append((module, 'forward' in module.__dict__, module.forward))
        module.forward = wrapper

    def install(self):
        if self.active:
            return self
        if getattr(self.model, '_full_precision_policy', None) is not None:
            raise RuntimeError('A full precision policy is already installed on this model')
        language = self.model.backbone.language_model
        device_type = next(language.parameters()).device.type
        if device_type not in ('cpu', 'cuda'):
            raise ValueError('Full precision language autocast requires CPU or CUDA')
        native_forward = language.forward

        @wraps(native_forward)
        def language_forward(*args, **kwargs):
            if any(getattr(m, 'gradient_checkpointing', False) for m in language.modules()):
                raise ValueError('Full direct-KV training forbids gradient checkpointing')
            if isinstance(kwargs.get('inputs_embeds'), torch.Tensor):
                kwargs['inputs_embeds'] = kwargs['inputs_embeds'].to(torch.bfloat16)
            with torch.autocast(device_type, dtype=torch.bfloat16):
                return native_forward(*args, **kwargs)

        def fp32_forward(module):
            original = module.forward

            @wraps(original)
            def forward(*args, **kwargs):
                with torch.autocast(device_type, enabled=False):
                    return original(*(_float_input(x) for x in args),
                                    **{k: _float_input(v) for k, v in kwargs.items()})
            return forward

        try:
            self._replace(language, language_forward)
            for name, module in language.named_modules():
                if isinstance(module, nn.Linear) and any(
                        part in ('lora_A', 'lora_B') for part in name.split('.')):
                    self._replace(module, fp32_forward(module))
            for name, module in self.model.named_children():
                if name != 'backbone':
                    self._replace(module, fp32_forward(module))
        except Exception:
            self.close()
            raise
        self.active = True
        self.model._full_precision_policy = self
        return self

    def close(self):
        """Restore exact forwards; safe to call more than once."""
        for module, had_override, original in reversed(self._originals):
            if had_override:
                module.forward = original
            else:
                del module.forward
        self._originals.clear()
        if getattr(self.model, '_full_precision_policy', None) is self:
            del self.model._full_precision_policy
        self.active = False

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def full_precision(model):
    """Context: BF16 native language, FP32 LoRA A/B, sensors and planning head.

    The embedding cast precedes native mRoPE, matching the BF16 source-KV RoPE
    convention in the bridge. FP32 language master weights retain gradients.
    All replaced forwards are restored at exit, including after exceptions.
    """
    return _FullPrecision(model)


def install_full_precision(model):
    """Install once for a training loop; call ``close()`` or use ``with``."""
    return full_precision(model).install()
