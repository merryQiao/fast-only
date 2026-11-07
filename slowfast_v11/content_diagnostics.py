"""Inference-only KV interventions; no model/training API or persistent hooks."""
from contextlib import contextmanager
from dataclasses import replace
import math

import torch
from .protocol import LayerKV
from . import attention

_ACTIVE = False


def contrast_donors(rows):
    """Cohort-local observable contrast, without labels or model predictions."""
    rows=list(rows);tokens=[r['token'] for r in rows]
    if len(set(tokens))!=len(tokens):raise ValueError('Duplicate donor token')
    output={}
    for target in sorted(rows,key=lambda r:r['token']):
        if target['split']!='val':continue
        candidates=[r for r in rows if r['split']=='val' and r['cohort']==target['cohort'] and r['token']!=target['token']]
        if not candidates:raise ValueError('No cohort-local donor')
        cross=[r for r in candidates if r['route']!=target['route']]
        candidates=cross or candidates
        speed=lambda r:math.hypot(*r['ego'][:2])
        donor=min(candidates,key=lambda r:(-(r['command']!=target['command']),
                                          -abs(speed(r)-speed(target)),r['token']))
        output[target['token']]=donor['token']
    return output


def transplant_snapshot(target,donor,*,zero_values=False):
    """Keep target positions/mask; substitute complete K/V only at equal layout."""
    if (len(target.layers)!=len(donor.layers)
            or not torch.equal(target.position_ids,donor.position_ids)
            or not torch.equal(target.valid_mask,donor.valid_mask)
            or any(a.key.shape!=b.key.shape or a.key.dtype!=b.key.dtype or a.key.device!=b.key.device
                   for a,b in zip(target.layers,donor.layers))):
        raise ValueError('Incompatible source KV layout; no truncation/resizing allowed')
    return replace(target,layers=tuple(LayerKV(z.key,torch.zeros_like(z.value) if zero_values else z.value)
                                       for z in donor.layers))


def source_groups(native,image_token_id,merge_size=2):
    """Partition Slow's native tokens and 18 numeric/query suffix positions."""
    ids=native['input_ids']
    if ids.shape[0]!=1 or not native['attention_mask'].bool().all():
        raise ValueError('Diagnostic requires one unpadded source')
    n=ids.shape[1];length=n+18;device=ids.device
    visual=(ids[0]==image_token_id).nonzero().flatten()
    counts=(native['image_grid_thw'].prod(-1)//merge_size**2).tolist()
    if len(counts)!=4 or sum(counts)!=len(visual):raise ValueError('Expected four complete source images')
    groups={};offset=0
    for index,count in enumerate(counts):
        mask=torch.zeros(length,dtype=torch.bool,device=device);mask[visual[offset:offset+count]]=True
        groups[f'image_{index}']=mask;offset+=count
    groups['native_other']=torch.zeros(length,dtype=torch.bool,device=device)
    groups['native_other'][:n]=ids[0]!=image_token_id
    for name,start,stop in [('ego',n,n+1),('command',n+1,n+2),('feedback',n+2,n+10),('planning_queries',n+10,n+18)]:
        mask=torch.zeros(length,dtype=torch.bool,device=device);mask[start:stop]=True;groups[name]=mask
    if not torch.equal(torch.stack(list(groups.values())).sum(0),torch.ones(length,device=device,dtype=torch.long)):
        raise ValueError('Source token partition incomplete')
    return groups


def _difference(a,b):
    a=a.detach().float();b=b.detach().float();delta=a-b
    norm_a=a.norm();norm_b=b.norm();denom=norm_a*norm_b
    return dict(equal=bool(torch.equal(a,b)),rms_difference=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/norm_a.clamp_min(1e-12)),
                cosine=float((a*b).sum()/denom) if denom>0 else None,
                rms_reference=float(a.square().mean().sqrt()),rms_donor=float(b.square().mean().sqrt()))


@torch.no_grad()
def kv_difference(target,donor,groups):
    transplant_snapshot(target,donor)  # Validate exact positional layout first.
    output=[]
    for index,(a,b) in enumerate(zip(target.layers,donor.layers)):
        row=dict(layer_one_based=attention.LAYERS[index]+1,key=_difference(a.key,b.key),value=_difference(a.value,b.value),groups={})
        for name,mask in groups.items():
            if mask.any():row['groups'][name]=dict(tokens=int(mask.sum()),key=_difference(a.key[:,:,mask],b.key[:,:,mask]),
                                                  value=_difference(a.value[:,:,mask],b.value[:,:,mask]))
        output.append(row)
    return output


@torch.no_grad()
def _attention_stats(q,k,v,sk,sv,fm,sm,groups):
    """Float32 reference probabilities and contributions BEFORE gate/o_proj."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    b,h,qn,d=q.shape;sn=sk.shape[2];tn=k.shape[2]
    if b!=1 or any(m.shape!=(sn,) or m.dtype!=torch.bool for m in groups.values()):
        raise ValueError('Probe source groups/layout differ')
    repeats=h//k.shape[1]
    keys=repeat_kv(torch.cat((sk,k),dim=2),repeats).float()
    causal=torch.arange(tn,device=q.device)[None,:]<=torch.arange(tn-qn,tn,device=q.device)[:,None]
    valid=torch.cat((sm[:,None,:].expand(b,qn,-1),fm[:,None,:]&causal),dim=-1)
    logits=q.float()@keys.transpose(-1,-2)/math.sqrt(d)
    probabilities=logits.masked_fill(~valid[:,None],float('-inf')).softmax(-1)
    pe=probabilities[...,:sn];pf=probabilities[...,sn:]
    external=pe@repeat_kv(sv,repeats).float();own=pf@repeat_kv(v,repeats).float()
    norms=lambda t:float(t.norm(dim=-1).mean())
    result=dict(external_mass=float(pe.sum(-1).mean()),self_mass=float(pf.sum(-1).mean()),
                external_mass_per_query=pe.sum(-1).mean((0,1)).cpu().tolist(),
                external_value_norm=norms(external),self_value_norm=norms(own),combined_value_norm=norms(external+own),
                contribution_space='pre_sigmoid_gate_and_o_proj',probability_dtype='float32_reference',groups={})
    for name,mask in groups.items():
        count=int(mask.sum());mass=float(pe[...,mask].sum(-1).mean())
        result['groups'][name]=dict(tokens=count,visible_tokens=int((sm[0]&mask).sum()),mass=mass,
                                   mass_per_token=mass/count if count else None)
    return result


@contextmanager
def attention_probe(groups,*,mask_external=False):
    """Scoped eval harness; original SDPA output is returned without recomputation.

    Module-level patch is intentionally single-process/non-reentrant. Never use
    during training, concurrent requests, compilation or a service deployment.
    """
    global _ACTIVE
    if torch.is_grad_enabled():raise RuntimeError('Attention probe requires no_grad inference')
    if _ACTIVE:raise RuntimeError('Attention probe forbids nested/concurrent calls')
    _ACTIVE=True;original=attention.planning_attention;records=[]
    def wrapped(q,k,v,sk,sv,fm,sm,dropout=0.):
        if dropout:raise ValueError('Diagnostic probe requires eval/dropout zero')
        used=torch.zeros_like(sm) if mask_external else sm
        actual=original(q,k,v,sk,sv,fm,used,dropout)
        records.append(_attention_stats(q,k,v,sk,sv,fm,used,groups))
        return actual
    attention.planning_attention=wrapped
    try:yield records
    finally:
        attention.planning_attention=original;_ACTIVE=False


def _ade(prediction,target):
    p=torch.tensor(prediction,dtype=torch.float32);t=torch.tensor(target,dtype=torch.float32)
    return float((p[...,:2]-t[...,:2]).norm(dim=-1).mean())


def _summary(rows):
    from fast_navsim.model import trajectory_metrics
    if not rows:return dict(count=0,modes={},slow=None)
    target=torch.tensor([r['target'] for r in rows]);modes={}
    for name in rows[0]['predictions']:
        predictions=torch.tensor([r['predictions'][name] for r in rows])
        metrics=trajectory_metrics(predictions,target)
        for baseline in ['none','masked','correct','shuffled','contrast','reference']:
            if baseline not in rows[0]['predictions']:continue
            other=torch.tensor([r['predictions'][baseline] for r in rows])
            metrics[f'gain_ADE_vs_{baseline}_m']=trajectory_metrics(other,target)['ADE']-metrics['ADE']
            delta=(predictions[...,:2]-target[...,:2]).norm(dim=-1).mean(-1)-(other[...,:2]-target[...,:2]).norm(dim=-1).mean(-1)
            metrics[f'better_count_vs_{baseline}']=int((delta<0).sum())
            metrics[f'median_delta_ADE_vs_{baseline}_m']=float(delta.quantile(.5))
            metrics[f'mean_plan_change_vs_{baseline}_m']=float((predictions[...,:2]-other[...,:2]).norm(dim=-1).mean())
        modes[name]=metrics
    return dict(count=len(rows),tokens=[r['token'] for r in rows],modes=modes,
                slow=trajectory_metrics(torch.tensor([r['slow_prediction'] for r in rows]),target))


def summarize_records(records):
    output={}
    for cohort in sorted({r['cohort'] for r in records}):
        rows=[r for r in records if r['cohort']==cohort]
        better=[];clear=[]
        for r in rows:
            base=_ade(r['predictions']['none'],r['target']);slow=_ade(r['slow_prediction'],r['target'])
            if slow<base:better.append(r)
            if base-slow>=.1 and slow<=.8*base:clear.append(r)
        output[cohort]=dict(all=_summary(rows),slow_better=_summary(better),clear_slow_advantage=_summary(clear),
            by_log={route:_summary([r for r in rows if r['route']==route]) for route in sorted({r['route'] for r in rows})},
            by_speed={name:_summary([r for r in rows if predicate(r['speed_mps'])]) for name,predicate in
                      [('low',lambda s:s<1),('mid',lambda s:1<=s<10),('high',lambda s:s>=10)]})
    return output
