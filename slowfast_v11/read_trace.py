"""Scoped, read-only direct-KV diagnostics for complete inference prefills.

The production SDPA tensor is returned unchanged. FP32 softmax decompositions
are numerical references, while local counterfactuals use the current SDPA,
sigmoid gate, and actual output-projection forward (including PEFT adapters).
The module-level patch is deliberately non-reentrant and inference-only.
"""
from contextlib import contextmanager
from dataclasses import replace
import math
from types import SimpleNamespace

import torch

from . import attention
from .protocol import KVSnapshot

_ACTIVE = False


def _layer_index(layer_one_based):
    if type(layer_one_based) is not int or layer_one_based-1 not in attention.LAYERS:
        raise ValueError('Expected one of the six full-attention layer numbers')
    return attention.LAYERS.index(layer_one_based-1)


def _same_layout(target, donor):
    if not isinstance(target,KVSnapshot) or not isinstance(donor,KVSnapshot):
        raise ValueError('Expected source KV snapshots with the same layout')
    if len(target.layers)!=len(attention.LAYERS) or len(donor.layers)!=len(attention.LAYERS):
        raise ValueError('Source layout requires exactly six KV layers')
    if (target.position_ids.device!=donor.position_ids.device
            or not torch.equal(target.position_ids,donor.position_ids)
            or not torch.equal(target.valid_mask,donor.valid_mask)
            or any(a.key.shape!=b.key.shape or a.key.dtype!=b.key.dtype or a.key.device!=b.key.device
                   for a,b in zip(target.layers,donor.layers))):
        raise ValueError('Source KV layout differs; positions/mask/shape/dtype/device must match')


def swap_layer(correct, donor, layer_one_based):
    """Replace exactly one raw pre-RoPE KV layer, retaining target metadata/layout."""
    index=_layer_index(layer_one_based)
    _same_layout(correct,donor)
    layers=list(correct.layers);layers[index]=donor.layers[index]
    return replace(correct,layers=tuple(layers))


def _validate_groups(groups, source_length, device):
    if (not groups or any(not isinstance(name,str) or not name for name in groups)
            or any(not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool
                   or mask.shape!=(source_length,) or mask.device!=device for mask in groups.values())):
        raise ValueError('Source group layout must contain local boolean source-column masks')
    if not torch.equal(torch.stack(list(groups.values())).sum(0),
                       torch.ones(source_length,device=device,dtype=torch.long)):
        raise ValueError('Source group layout must partition every source column exactly once')


@torch.no_grad()
def _float32_reference(q,k,v,source_k,source_v,fast_valid,source_valid,groups):
    """One joint softmax, split into external/self and source-group contributions."""
    b,h,qn,d=q.shape;sn=source_k.shape[2];tn=k.shape[2]
    _validate_groups(groups,sn,q.device)
    if b!=1 or qn>tn or h%k.shape[1]:
        raise ValueError('Read probe layout requires one window and valid GQA planning rows')
    repeats=h//k.shape[1]
    causal=torch.arange(tn,device=q.device)[None,:]<=torch.arange(tn-qn,tn,device=q.device)[:,None]
    allowed=torch.cat((source_valid[:,None,:].expand(b,qn,-1),fast_valid[:,None,:]&causal),-1)
    if not allowed.any(-1).all():
        raise ValueError('Read probe layout contains a query without any valid key')
    # The surrounding native language forward may be inside BF16 autocast.
    with torch.autocast(q.device.type,enabled=False):
        keys=torch.cat((source_k,k),2).repeat_interleave(repeats,dim=1).float()
        logits=q.float()@keys.transpose(-1,-2)/math.sqrt(d)
        probabilities=logits.masked_fill(~allowed[:,None],float('-inf')).softmax(-1)
        pe=probabilities[...,:sn];pf=probabilities[...,sn:]
        sv=source_v.repeat_interleave(repeats,dim=1).float()
        external=pe@sv
        own=pf@v.repeat_interleave(repeats,dim=1).float()
        contributions={name:pe[...,mask]@sv[:,:,mask] for name,mask in groups.items()}
    group_stats={}
    for name,mask in groups.items():
        count=int(mask.sum());mass=pe[...,mask].sum(-1)
        group_stats[name]=dict(tokens=count,visible_tokens=int((source_valid[0]&mask).sum()),
            mass=float(mass.mean()),mass_per_query=mass.mean((0,1)).cpu().tolist(),
            mass_per_head=mass.mean((0,2)).cpu().tolist(),
            mass_per_token=float(mass.mean())/count if count else None)
    return dict(external=external,self=own,combined=external+own,group_contributions=contributions,
        external_mass=float(pe.sum(-1).mean()),self_mass=float(pf.sum(-1).mean()),
        external_mass_per_query=pe.sum(-1).mean((0,1)).cpu().tolist(),
        self_mass_per_query=pf.sum(-1).mean((0,1)).cpu().tolist(),
        external_mass_per_head=pe.sum(-1).mean((0,2)).cpu().tolist(),
        self_mass_per_head=pf.sum(-1).mean((0,2)).cpu().tolist(),groups=group_stats)


def _rows(value):
    return value.transpose(1,2).flatten(2)


def _rms(value):
    return float(value.detach().float().square().mean().sqrt())


def _difference(reference, candidate):
    reference=reference.detach().float();candidate=candidate.detach().float()
    delta=candidate-reference;norm=reference.norm();other_norm=candidate.norm()
    denominator=norm*other_norm
    return dict(equal=bool(torch.equal(reference,candidate)),
        equal_fraction=float((reference==candidate).float().mean()),
        max_abs_difference=float(delta.abs().max()),rms_difference=_rms(delta),
        relative_l2=float(delta.norm()/norm.clamp_min(1e-12)),
        cosine=float((reference*candidate).sum()/denominator) if denominator>0 else None,
        rms_reference=_rms(reference),rms_candidate=_rms(candidate))


def _contribution_stages(projection, pre_gate, gate, *, input_dtype=None, bias=None):
    """Affine contribution P(x)-P(0); native hooks on P must not run again."""
    post_gate=pre_gate*gate
    projection_input=post_gate.to(dtype=input_dtype or post_gate.dtype)
    if bias is None:bias=projection.forward(torch.zeros_like(projection_input)).float()
    projected=projection.forward(projection_input).float()
    return dict(pre_gate=pre_gate,post_gate=post_gate,linear_projection=projected-bias)


def _contribution_rms(stages):
    return {f'{name}_rms':_rms(value) for name,value in stages.items()}


def _reference_record(reference, actual, gate, projection, input_dtype, bias):
    contributions={name:_contribution_stages(projection,_rows(reference[name]),gate,
                       input_dtype=input_dtype,bias=bias) for name in ('external','self','combined')}
    groups={}
    for name,stats in reference['groups'].items():
        stages=_contribution_stages(projection,_rows(reference['group_contributions'][name]),gate,
                                    input_dtype=input_dtype,bias=bias)
        groups[name]={**stats,'contribution':_contribution_rms(stages)}
    projected=projection.forward(contributions['combined']['post_gate'].to(input_dtype)).float()
    reconstructed=contributions['external']['linear_projection']+contributions['self']['linear_projection']+bias
    return dict(probability_dtype='float32_reference',contribution_space='joint_softmax_weighted_values',
        projection_input_dtype=str(input_dtype),projection_compute='current_native_forward_precision_query_rows',
        external_mass=reference['external_mass'],self_mass=reference['self_mass'],
        external_mass_per_query=reference['external_mass_per_query'],
        self_mass_per_query=reference['self_mass_per_query'],
        external_mass_per_head=reference['external_mass_per_head'],
        self_mass_per_head=reference['self_mass_per_head'],
        external=_contribution_rms(contributions['external']),self=_contribution_rms(contributions['self']),
        combined=_contribution_rms(contributions['combined']),groups=groups,
        sdpa_reconstruction=_difference(_rows(actual),_rows(reference['combined'])),
        projection_reconstruction=_difference(projected,reconstructed))


@contextmanager
def _patch(wrapped):
    global _ACTIVE
    if torch.is_grad_enabled():raise RuntimeError('Read diagnostics require no_grad inference')
    if _ACTIVE:raise RuntimeError('Read diagnostics forbid nested/concurrent scopes')
    original=attention.planning_attention
    _ACTIVE=True;attention.planning_attention=wrapped
    try:yield
    finally:
        attention.planning_attention=original;_ACTIVE=False


@contextmanager
def read_probe(model, groups, *, alternatives=None):
    """Observe exactly one complete six-layer enabled Fast bridge prefill.

    Call inside ``torch.no_grad()`` with an eval model. ``alternatives`` maps
    ``reference`` and/or ``shuffled`` to raw pre-RoPE snapshots at exactly the
    active source layout. Supplying it adds fixed-current-Q/Fast-KV variants,
    K-only/V-only donor changes, zero-V, and masked-column local comparisons.

    The yielded object has JSON-safe ordered ``records`` and small owned CPU
    FP32 ``tensors`` named ``pre_gate_4``, ``gate_4``, ``post_gate_4``, and
    ``query_projection_4`` (and the other five layer numbers). Query projection
    calls have a different row count from the native full-context projection;
    the caller should compare them explicitly if kernel rounding matters.
    """
    if torch.is_grad_enabled():raise RuntimeError('Read probe requires no_grad inference')
    bridge=model.bridge
    if model.training or any(module.training for module in bridge.language.modules()):
        raise RuntimeError('Read probe requires an eval model and language modules')
    if alternatives is not None and (not isinstance(alternatives,dict)
                                     or set(alternatives)-{'reference','shuffled'}):
        raise ValueError('Alternative source layout must map reference/shuffled to snapshots')
    alternatives=None if alternatives is None else dict(alternatives)
    groups=dict(groups);probe=SimpleNamespace(records=[],tensors={})
    original=attention.planning_attention

    def wrapped(q,k,v,sk,sv,fm,sm,dropout=0.):
        if dropout:raise ValueError('Read probe requires eval/dropout zero')
        index=len(probe.records)
        if index>=len(attention.LAYERS):raise RuntimeError('Read probe requires exactly six attention calls')
        layer=attention.LAYERS[index];call=bridge._call
        if (call is None or call.mode!='fuse' or call.fused_layers!=list(attention.LAYERS[:index])
                or layer not in call.values or q.shape[0]!=1 or not call.enabled.all()):
            raise ValueError('Read probe layout requires one enabled, ordered native bridge prefill')
        _validate_groups(groups,sk.shape[2],q.device)
        _same_layout(call.snapshot,call.snapshot)
        for donor in (alternatives or {}).values():_same_layout(call.snapshot,donor)
        # Return this exact object after diagnostics; never substitute a reference.
        actual=original(q,k,v,sk,sv,fm,sm,dropout)
        q_projection=call.values[layer]['q_proj']
        b,t,_=q_projection.shape;h=q.shape[1];d=q.shape[-1];qn=q.shape[2]
        gate=q_projection.view(b,t,h,d*2).chunk(2,dim=-1)[1][:,-qn:].reshape(b,qn,-1).sigmoid()
        pre_gate=_rows(actual);post_gate=pre_gate*gate
        projection=bridge.language.layers[layer].self_attn.o_proj
        query_projection=projection.forward(post_gate)
        bias=projection.forward(torch.zeros_like(post_gate)).float()
        reference=_float32_reference(q,k,v,sk,sv,fm,sm,groups)
        gate_float=gate.float()
        row=dict(layer_one_based=layer+1,
            actual=dict(pre_gate_rms=_rms(pre_gate),post_gate_rms=_rms(post_gate),
                query_projection_rms=_rms(query_projection),gate_rms=_rms(gate),
                gate_mean=float(gate_float.mean()),gate_min=float(gate.min()),gate_max=float(gate.max()),
                gate_mean_per_query=gate_float.mean((0,2)).cpu().tolist(),
                gate_mean_per_head=gate_float.view(b,qn,h,d).mean((0,1,3)).cpu().tolist(),
                gate_fraction_lt_0_01=float((gate_float<.01).float().mean()),
                gate_fraction_lt_0_1=float((gate_float<.1).float().mean()),
                gate_fraction_gt_0_9=float((gate_float>.9).float().mean())),
            float32_reference=_reference_record(reference,actual,gate,projection,post_gate.dtype,bias),
            local_variants={})
        if alternatives is not None:
            local={'correct':(actual,False)}
            from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
            for name,donor in alternatives.items():
                source=donor.layers[index]
                _,donor_k=apply_rotary_pos_emb(source.key,source.key,*call.source_rope)
                for suffix,used_k,used_v in [('',donor_k,source.value),('_k_only',donor_k,sv),('_v_only',sk,source.value)]:
                    local[name+suffix]=(original(q,k,v,used_k,used_v,fm,sm,0.),False)
            local['zero_v']=(original(q,k,v,sk,torch.zeros_like(sv),fm,sm,0.),False)
            local['masked']=(original(q,k,v,sk,sv,fm,torch.zeros_like(sm),0.),True)
            for name,(value,masked) in local.items():
                before=_rows(value);after=before*gate
                projected=query_projection if name=='correct' else projection.forward(after)
                row['local_variants'][name]=dict(pre_gate=_difference(pre_gate,before),
                    post_gate=_difference(post_gate,after),post_projection=_difference(query_projection,projected),
                    external_columns_masked=masked)
        for name,value in [('pre_gate',pre_gate),('gate',gate),('post_gate',post_gate),('query_projection',query_projection)]:
            probe.tensors[f'{name}_{layer+1}']=value.detach().float().cpu().clone()
        probe.records.append(row)
        return actual

    with _patch(wrapped):
        yield probe
        if tuple(row['layer_one_based']-1 for row in probe.records)!=attention.LAYERS:
            raise RuntimeError('Read probe requires complete six-layer coverage')


@contextmanager
def mask_layer(layer_one_based):
    """Mask external attention columns at one of six ordered native FA calls."""
    selected=_layer_index(layer_one_based);original=attention.planning_attention;seen=[]
    def wrapped(q,k,v,sk,sv,fm,sm,dropout=0.):
        if dropout:raise ValueError('Layer mask requires eval/dropout zero')
        index=len(seen)
        if index>=len(attention.LAYERS):raise RuntimeError('Layer mask requires exactly six attention calls')
        seen.append(attention.LAYERS[index])
        return original(q,k,v,sk,sv,fm,torch.zeros_like(sm) if index==selected else sm,dropout)
    with _patch(wrapped):
        yield
        if tuple(seen)!=attention.LAYERS:raise RuntimeError('Layer mask requires complete six-layer coverage')
