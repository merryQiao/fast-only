"""Preserve the frozen native visual provider's nonpersistent RoPE precision.

Full-language setup casts the vision module to BF16, including floating
buffers. Qwen's native visual frequencies are explicitly FP32 and are absent
from checkpoints. Reconstruct their canonical native values, rather than
widening already rounded BF16 values. No parameter or model architecture changes.
"""
import hashlib

import torch


def _tensor_hash(value):
    host=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((tuple(host.shape),host.dtype)).encode())
    h.update(host.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore_native_visual_rope(visual):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
    rotary=getattr(visual,'rotary_pos_emb',None)
    if not isinstance(rotary,Qwen3_5VisionRotaryEmbedding):
        raise ValueError('Native visual precision requires the verified Qwen3.5 rotary implementation')
    previous=rotary.inv_freq
    canonical=Qwen3_5VisionRotaryEmbedding(rotary.dim,rotary.theta).inv_freq.to(device=previous.device)
    if canonical.dtype!=torch.float32:raise ValueError('Native visual rotary constructor no longer yields FP32')
    changed=previous.dtype!=canonical.dtype or not torch.equal(previous,canonical)
    receipt=dict(buffer='visual.rotary_pos_emb.inv_freq',before_dtype=str(previous.dtype),
        after_dtype=str(canonical.dtype),changed=changed,before_sha256=_tensor_hash(previous),
        after_sha256=_tensor_hash(canonical),dim=rotary.dim,theta=rotary.theta,
        initialization='unchanged installed native CPU constructor, then device-only transfer',
        parameters_modified=False)
    if changed:rotary.inv_freq=canonical
    return receipt


def buffer_digest(model):
    h=hashlib.sha256()
    for name,value in model.named_buffers():
        h.update(name.encode());h.update(_tensor_hash(value).encode())
    return h.hexdigest()
