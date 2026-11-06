"""One paired update with explicit consumer gradients and independent clipping."""
import math

import torch

from fast_navsim.model import pose_loss


def _gradient_stats(gradients):
    present = [g for g in gradients if g is not None]
    return dict(count=len(gradients), with_gradient=len(present),
                finite=all(bool(torch.isfinite(g).all()) for g in present),
                nonzero=any(bool(torch.count_nonzero(g)) for g in present),
                max_abs=max((g.detach().abs().max().item() for g in present), default=0.))


def _probe_fast_loss(loss, snapshot, slow, expected_live):
    # Request LF derivatives before LS backward; .grad after total backward cannot
    # establish the consumer route because every arm also supervises Slow itself.
    kv = [tensor for layer in snapshot.layers for tensor in (layer.key, layer.value)]
    lora = [(name, p) for name, p in slow.named_parameters()
            if '.self_attn.' in name and 'lora_B' in name
            and ('.k_proj.' in name or '.v_proj.' in name) and p.requires_grad]
    if len(kv) != 12 or len(lora) != 12:
        raise AssertionError('Probe requires six Slow KV layers and their twelve K/V LoRA-B parameters')
    gradients = torch.autograd.grad(loss, kv + [p for _, p in lora],
                                    retain_graph=True, allow_unused=True)
    kv_stats = [dict(key=_gradient_stats([gradients[i]]),
                     value=_gradient_stats([gradients[i + 1]])) for i in range(0, 12, 2)]
    lora_stats = [dict(name=name, **_gradient_stats([gradients[12 + i]]))
                  for i, (name, _) in enumerate(lora)]
    if expected_live:
        if not all(g is not None and bool(torch.isfinite(g).all())
                   and bool(torch.count_nonzero(g)) for g in gradients):
            raise AssertionError('Fresh joint Fast loss must reach every Slow K/V and K/V LoRA-B parameter')
    elif any(g is not None for g in gradients):
        raise AssertionError('Control Fast loss unexpectedly reaches Slow')
    return dict(expected_live=expected_live, kv=kv_stats,
                lora_b=_gradient_stats(gradients[12:]), lora_b_parameters=lora_stats)


def joint_step(fast, slow, opt_fast, opt_slow, fast_batch, slow_batch,
               target_fast, target_slow, feedback, *, arm, mode,
               slow_weight=1., clip=1., probe=False):
    """Update both models; only ``joint/fresh`` retains LF -> Slow derivatives.

    Inputs/targets are already on their respective model devices. Feedback is a
    completed fixed-teacher plan, never differentiable. Callers set trainability
    and train/eval mode. Returned reports contain only JSON-compatible values.
    """
    if arm not in ('no_kv', 'detached', 'joint') or mode not in ('none', 'fresh'):
        raise ValueError('Unsupported joint arm or cache mode')
    if not math.isfinite(slow_weight) or slow_weight <= 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError('Slow weight and per-model clipping must be finite and positive')
    opt_fast.zero_grad(set_to_none=True)
    opt_slow.zero_grad(set_to_none=True)
    fixed_feedback = feedback.detach() if feedback is not None else None
    source = slow(slow_batch['native'], slow_batch['ego'], slow_batch['command'],
                  feedback=fixed_feedback)
    loss_slow = pose_loss(source.trajectory, target_slow)
    kwargs = {}
    reads_cache = arm != 'no_kv' and mode == 'fresh'
    if reads_cache:
        device = fast_batch['ego'].device
        snapshot = (source.snapshot.to(device) if arm == 'joint'
                    else source.snapshot.inference_copy(device))
        metadata = torch.tensor([0., 0., 0., 0., 1., 1.], device=device,
                                dtype=torch.float32).expand(len(fast_batch['ego']), -1)
        kwargs = dict(snapshot=snapshot, metadata=metadata)
    prediction = fast(fast_batch['native'], fast_batch['ego'], fast_batch['command'], **kwargs)
    loss_fast = pose_loss(prediction, target_fast)
    total = loss_fast + slow_weight * loss_slow.to(loss_fast.device)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('Nonfinite paired trajectory loss')
    routing = (_probe_fast_loss(loss_fast, source.snapshot, slow, arm == 'joint' and reads_cache)
               if probe else None)
    total.backward()
    # A shared clipping norm would let Fast alter the Slow-only control updates.
    norms = [torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           clip, error_if_nonfinite=True)
             for model in (fast, slow)]
    if routing is not None:
        visual_parameters = [p for model in (fast, slow) for p in model.backbone.visual.parameters()]
        routing['vision_frozen'] = all(not p.requires_grad for p in visual_parameters)
        routing['vision_has_gradient'] = any(p.grad is not None for p in visual_parameters)
        if not routing['vision_frozen'] or routing['vision_has_gradient']:
            raise AssertionError('Vision and merger must remain frozen without gradients')
    opt_fast.step()
    opt_slow.step()
    return dict(arm=arm, mode=mode, loss_fast=loss_fast.item(), loss_slow=loss_slow.item(),
                loss_total=total.item(), fast_grad_norm=float(norms[0]), slow_grad_norm=float(norms[1]),
                probe=routing)
