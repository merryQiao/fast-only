"""Bounded expanded-data protocol, polynomial donors and explicit LR progress."""
from collections import defaultdict, deque
from collections.abc import Mapping
import math
from numbers import Integral, Real
import random

from .joint_protocol import (_fields, _fixed, _path, _MODEL, _PROCESSOR,
                             _REPRODUCIBILITY)


def validate_expanded_config(c):
    fixed = dict(schema_version='slowfast_v11_expanded',train_seed=42,split_seed=42,
        init_policy='pretrained',allow_single_gpu=True,model=_MODEL,processor=_PROCESSOR,
        stages=dict(fast_warmup=512,slow_adapt=512,fast_consumer=800),arms=['no_kv','joint'],
        loss=dict(slow_weight=1.),optimizer=dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01,
            grad_clip=1.,schedule='warmup_cosine',clip_per_model=True,warmup_steps=32,min_lr_ratio=.1),
        missing_fraction=.3,feedback_policy='shared_frozen_fast_warmup',gradient_probe_count=4,
        train_curve_steps=[200,400,800],train_diagnostic_count=32,eval_cache_capacity=2,
        reproducibility=_REPRODUCIBILITY)
    _fields(c,(*fixed,'fast','slow','data','output'),'expanded config')
    for key,value in fixed.items(): _fixed(c[key],value,key)
    for name in ('fast','slow'):
        _fields(c[name],('model_path','device'),name)
        _path(c[name]['model_path'],f'{name}.model_path')
        _fixed(c[name]['device'],'cuda:0',f'{name}.device')
    policy = dict(ages_s=[0.],max_train_rows=256,max_val_rows=68,
                  max_source_observations=324,moving_speed_mps=1.)
    paths = ('train_manifest','val_manifest','stats','sensor_root')
    _fields(c['data'],(*paths,*policy),'data')
    for key in paths: _path(c['data'][key],f'data.{key}')
    for key,value in policy.items(): _fixed(c['data'][key],value,f'data.{key}')
    _path(c['output'],'output')


def _train_rows(rows):
    result = list(rows)
    tokens = set()
    for row in result:
        if (not isinstance(row,Mapping) or row.get('split') != 'train'
                or row.get('cohort','train') != 'train'):
            raise ValueError('Optimizer and diagnostic sampling require TRAIN-only rows')
        token = row.get('token')
        if not isinstance(token,str) or not token or token in tokens:
            raise ValueError('TRAIN tokens must be nonempty and unique')
        tokens.add(token)
    if not result: raise ValueError('TRAIN cohort cannot be empty')
    return sorted(result,key=lambda row:row['token'])


def epoch_plan(rows, steps, seed):
    """Stable token order, without replacement within each uniform shuffle epoch."""
    if isinstance(steps,bool) or not isinstance(steps,int) or steps <= 0:
        raise ValueError('Steps must be a positive integer')
    if isinstance(seed,bool) or not isinstance(seed,int): raise ValueError('Seed must be an integer')
    tokens = [row['token'] for row in _train_rows(rows)]
    rng = random.Random(seed); result = []
    while len(result) < steps:
        epoch = tokens.copy(); rng.shuffle(epoch); result.extend(epoch)
    return result[:steps]


def speed_bin(row):
    ego = row.get('ego')
    if (not isinstance(ego,(tuple,list)) or len(ego) != 4 or any(
            isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in ego)):
        raise ValueError('Expected four finite raw ego values')
    speed = math.hypot(ego[0],ego[1])
    return 'low_speed' if speed < 1. else 'moving' if speed < 10. else 'high_speed'


def expanded_plan(rows, pairs, c):
    """Uniform TRAIN targets and an independent exact 240 absent / 560 fresh mask."""
    validate_expanded_config(c)
    rows = _train_rows(rows); by_token = {row['token']:row for row in rows}
    order = epoch_plan(rows,c['stages']['fast_consumer'],c['train_seed'])
    modes = ['none']*240 + ['fresh']*560
    random.Random(c['train_seed']+1).shuffle(modes)
    return [dict(target_token=token,source_token=token,split='train',
                 stratum=speed_bin(by_token[token]),mode=mode,age_s=0.)
            for token,mode in zip(order,modes)]


def _matching(adjacency):
    """Deterministic augmenting paths; O(V E), no exponential subset state."""
    owner = {}
    def augment(target, visited):
        for donor in adjacency[target]:
            if donor in visited: continue
            visited.add(donor)
            if donor not in owner or augment(owner[donor],visited):
                owner[donor] = target
                return True
        return False
    for target in range(len(adjacency)):
        if not augment(target,set()): return None
    return {target:donor for donor,target in owner.items()}


def scalable_mismatches(rows):
    """Complete split/cohort-local derangements; prefer an all-cross-log match.

    If that is impossible, allow same-log donors at least two seconds away,
    ordering cross-log edges first. An infeasible cohort is reported unavailable;
    it never receives a partial mapping or a donor from another cohort.
    """
    groups = defaultdict(list); tokens = set(); log_groups = {}
    for row in rows:
        if not isinstance(row,Mapping): raise ValueError('Donor rows must be mappings')
        split = row.get('split'); cohort = row.get('cohort','train' if split == 'train' else 'new_val')
        if split not in ('train','val') or cohort not in (('train',) if split == 'train' else ('new_val','legacy_val')):
            raise ValueError('Expected TRAIN/new VAL/legacy VAL cohorts only')
        token,route,time = row.get('token'),row.get('route'),row.get('timestamp_us')
        if (not isinstance(token,str) or not token or token in tokens or not isinstance(route,str)
                or not route or isinstance(time,bool) or not isinstance(time,Integral)):
            raise ValueError('Invalid or duplicate donor observation identity')
        key = (split,cohort)
        if route in log_groups and log_groups[route] != key:
            raise ValueError('Log leakage across splits or cohorts')
        tokens.add(token); log_groups[route] = key; groups[key].append(row)
    mapping = {}; report = {}
    for (split,cohort),selected in sorted(groups.items()):
        selected.sort(key=lambda row:row['token']); cross = []; fallback = []
        for target in selected:
            cross.append([j for j,donor in enumerate(selected) if donor['route'] != target['route']])
            fallback.append([j for j,donor in enumerate(selected) if donor['token'] != target['token']
                and donor['route'] == target['route']
                and abs(donor['timestamp_us']-target['timestamp_us']) >= 2_000_000])
        match = _matching(cross)
        if match is None: match = _matching([a+b for a,b in zip(cross,fallback)])
        entry = dict(available=match is not None,count=len(selected),cross_log_count=0)
        if match is None:
            entry['reason'] = 'no_complete_derangement'
        else:
            mapping.update({selected[i]['token']:selected[j]['token'] for i,j in match.items()})
            entry['cross_log_count'] = sum(selected[i]['route'] != selected[j]['route'] for i,j in match.items())
        report[f'{split}/{cohort}'] = entry
    return mapping,report


def select_train_diagnostics(rows, count=32):
    """Fixed round robin across available speed bins and logs, with unique tokens."""
    rows = _train_rows(rows)
    if isinstance(count,bool) or not isinstance(count,int) or not 0 < count <= len(rows):
        raise ValueError('Insufficient TRAIN rows or invalid diagnostic count')
    buckets = defaultdict(lambda:defaultdict(list))
    for row in rows:
        route = row.get('route')
        if not isinstance(route,str) or not route: raise ValueError('Expected TRAIN log identities')
        buckets[speed_bin(row)][route].append(row['token'])
    queues = {}
    for name,logs in buckets.items():
        queues[name] = deque()
        for route,items in sorted(logs.items()):
            random.Random(42).shuffle(items)
            queues[name].append(deque(items))
    selected = []
    while len(selected) < count:
        for name in ('low_speed','moving','high_speed'):
            if len(selected) == count: break
            queue = queues.get(name)
            if not queue: continue
            bucket = queue.popleft(); selected.append(bucket.popleft())
            if bucket: queue.append(bucket)
    return selected


def lr_factor(update_step, total_steps, warmup_steps=32, min_ratio=.1):
    """Multiplier for a one-based optimizer update; final update reaches min_ratio."""
    if (any(isinstance(v,bool) or not isinstance(v,int) for v in (update_step,total_steps,warmup_steps))
            or not 0 < warmup_steps < total_steps or not 1 <= update_step <= total_steps
            or isinstance(min_ratio,bool) or not isinstance(min_ratio,Real)
            or not math.isfinite(min_ratio) or not 0 < min_ratio < 1):
        raise ValueError('Invalid warmup/cosine schedule bounds')
    if update_step <= warmup_steps: return update_step/warmup_steps
    progress = (update_step-warmup_steps)/(total_steps-warmup_steps)
    return min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress))


class WarmupCosineScheduler:
    """Prepare update-one LR on construction; call step AFTER each optimizer step.

    Construct before loading optimizer state, then restore optimizer and scheduler
    states in that order. State contains independent scalars, not live tensors.
    """
    def __init__(self, optimizer, total_steps, warmup_steps=32, min_ratio=.1):
        lr_factor(1,total_steps,warmup_steps,min_ratio)
        self.optimizer = optimizer; self.total_steps = total_steps
        self.warmup_steps = warmup_steps; self.min_ratio = min_ratio
        self.peak_lrs = [group['lr'] for group in optimizer.param_groups]
        if not self.peak_lrs or any(not math.isfinite(v) or v <= 0 for v in self.peak_lrs):
            raise ValueError('Optimizer peak LRs must be finite and positive')
        self.completed_steps = 0
        self._prepare_next()

    def _prepare_next(self):
        factor = lr_factor(min(self.completed_steps+1,self.total_steps),self.total_steps,
                           self.warmup_steps,self.min_ratio)
        for group,peak in zip(self.optimizer.param_groups,self.peak_lrs): group['lr'] = peak*factor

    def step(self):
        if self.completed_steps >= self.total_steps: raise ValueError('Schedule budget exhausted')
        self.completed_steps += 1
        self._prepare_next()

    def state_dict(self):
        return dict(schema='warmup_cosine_v1',total_steps=self.total_steps,warmup_steps=self.warmup_steps,
                    min_ratio=self.min_ratio,peak_lrs=self.peak_lrs.copy(),completed_steps=self.completed_steps)

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state,dict) or state.keys() != expected.keys():
            raise ValueError('Scheduler state fields differ')
        for key in expected:
            if key != 'completed_steps' and state[key] != expected[key]:
                raise ValueError('Scheduler configuration or peak LR differs')
        completed = state['completed_steps']
        if isinstance(completed,bool) or not isinstance(completed,int) or not 0 <= completed <= self.total_steps:
            raise ValueError('Invalid scheduler progress')
        self.completed_steps = completed
        self._prepare_next()
