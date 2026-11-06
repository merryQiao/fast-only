"""Fixed-budget adaptation, stable weight identity and paired offline reporting."""
from collections import defaultdict
import hashlib
import random
import torch
from fast_navsim.model import trajectory_metrics, pose_loss

MODES=['none']*3+['fresh']*4+['stale']*3
STAGES={'fast_warmup':80,'slow_inputs':20,'slow_lora':40,'fast_consumer':120}


def validate_config(c):
    if c['schema_version']!='slowfast_v11_validation' or c['stages']!=STAGES:
        raise ValueError('This bounded diagnostic has a fixed 80/20/40/120 budget')
    if c['consumer_modes']!=MODES:
        raise ValueError('Consumer missing/fresh/stale proportions are fixed')
    if c['fast']['device']==c['slow']['device'] and not c['allow_single_gpu']:
        raise ValueError('Single GPU must be explicitly allowed')
    model=dict(lora_rank=16,lora_alpha=32,lora_dropout=0.,lora_parameter_dtype='float32',position_offset=4096)
    processor=dict(policy='original_pad',pad_multiple=32,brightness=0.)
    if c['model']!=model or c['processor']!=processor:
        raise ValueError('Unsupported model/processor policy drift')
    data=c['data']
    if (data['ages_s']!=[0.,.5,1.,1.5] or data['max_train_rows']!=12 or data['max_val_rows']!=4
            or data['max_source_observations']!=64 or data['moving_speed_mps']!=1.):
        raise ValueError('Requires the bounded 2Hz mini cohort and fixed age/motion thresholds')
    expected=dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01,grad_clip=1.,schedule='constant')
    if c['optimizer']!=expected:
        raise ValueError('Diagnostic optimizer is fixed; do not silently use a formal schedule')


def set_stage(model,stage):
    if stage not in STAGES:raise ValueError('Unknown adaptation stage')
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith('backbone.') or ('lora_' in name and stage!='slow_inputs'))
    model.train()
    model.backbone.visual.eval()


def make_optimizer(model,c):
    groups=[]
    for lora,lr in [(True,c['lora_lr']),(False,c['new_lr'])]:
        params=[p for name,p in model.named_parameters() if p.requires_grad and ('lora_' in name)==lora]
        if params:groups.append(dict(params=params,lr=lr))
    return torch.optim.AdamW(groups,weight_decay=c['weight_decay'])


def perform_step(model,optimizer,batch,target,clip,**kwargs):
    optimizer.zero_grad(set_to_none=True)
    output=model(batch['native'],batch['ego'],batch['command'],**kwargs)
    prediction=output.trajectory if hasattr(output,'trajectory') else output
    loss=pose_loss(prediction,target)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=loss.item(),grad_norm=float(norm))


def canonical_state(model):
    """All adapters/new modules even when stage-C freezes Slow.requires_grad."""
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters()
            if not name.startswith('backbone.') or 'lora_' in name}


def restore_weights(model,state):
    names={name for name,_ in model.named_parameters() if not name.startswith('backbone.') or 'lora_' in name}
    if set(state)!=names:raise ValueError('Adapter/new-module state keys do not match')
    model.load_state_dict(state,strict=False)


def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cycle(items,steps,seed):
    if not items:raise ValueError('Cannot sample an empty pool')
    rng=random.Random(seed);result=[]
    while len(result)<steps:
        shuffled=list(items);rng.shuffle(shuffled);result.extend(shuffled)
    return result[:steps]


def sampling_plan(rows,stale_pairs,steps,seed,modes=MODES):
    if not rows or any(r['split']!='train' for r in rows):
        raise ValueError('Optimizer sampling requires nonempty TRAIN-only rows')
    tokens={r['token'] for r in rows}
    if not stale_pairs or any(p['split']!='train' or p['target_token'] not in tokens for p in stale_pairs):
        raise ValueError('Missing/invalid TRAIN stale-pair coverage; no fresh fallback')
    target_cycle=iter(cycle([r['token'] for r in rows],steps,seed))
    stale_cycle=iter(cycle(stale_pairs,steps,seed+1))
    plan=[]
    for i in range(steps):
        mode=modes[i%len(modes)]
        if mode=='stale':item=dict(next(stale_cycle))
        elif mode in ('none','fresh'):
            token=next(target_cycle)
            item=dict(target_token=token,source_token=token,age_s=0.,split='train')
        else:raise ValueError('Unknown cache mode')
        plan.append(item|{'mode':mode})
    return plan


def paired_summary(records):
    groups=defaultdict(list)
    none={};fresh={}
    for row in records:
        age=row['age_s']
        label=f'stale@{age:g}' if row['mode']=='stale' else row['mode']
        groups[row['stage'],row['split'],label].append(row)
        if row['mode']=='none':none[row['stage'],row['split'],row['token']]=row
        if row['mode']=='fresh':fresh[row['stage'],row['split'],row['token']]=row
    def metrics(rows):
        return trajectory_metrics(torch.tensor([r['prediction'] for r in rows]),torch.tensor([r['target'] for r in rows]))
    output={}
    for (stage,split,label),rows in groups.items():
        m=metrics(rows)
        moving=[r for r in rows if r['speed_mps']>=1.]
        m.update(tokens=[r['token'] for r in rows],moving_samples=len(moving),stationary_or_slow_samples=len(rows)-len(moving))
        if moving:m['moving_metrics']=metrics(moving)
        slow=[r for r in rows if r['speed_mps']<1.]
        if slow:m['stationary_or_slow_metrics']=metrics(slow)
        if label!='none':
            baseline=[none[stage,split,r['token']] for r in rows]
            m['paired_no_kv']=metrics(baseline)
            m['delta_ADE_vs_no_kv']=m['ADE']-m['paired_no_kv']['ADE']
            m['mean_plan_xy_change_vs_no_kv_m']=torch.linalg.vector_norm(
                torch.tensor([r['prediction'] for r in rows])[...,:2]-torch.tensor([r['prediction'] for r in baseline])[...,:2],dim=-1).mean().item()
        if label not in ('none','fresh'):
            baseline=[fresh[stage,split,r['token']] for r in rows]
            m['paired_fresh']=metrics(baseline)
            m['delta_ADE_vs_fresh']=m['ADE']-m['paired_fresh']['ADE']
            m['mean_plan_xy_change_vs_fresh_m']=torch.linalg.vector_norm(
                torch.tensor([r['prediction'] for r in rows])[...,:2]-torch.tensor([r['prediction'] for r in baseline])[...,:2],dim=-1).mean().item()
        output.setdefault(stage,{}).setdefault(split,{})[label]=m
    return output
