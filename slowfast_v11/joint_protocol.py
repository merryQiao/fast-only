"""Pure configuration and TRAIN sampling for the fixed live-KV joint diagnostic."""
from collections.abc import Mapping
import math
from numbers import Real
import random


_MODEL = dict(lora_rank=16,lora_alpha=32,lora_dropout=0.,lora_parameter_dtype='float32',position_offset=4096)
_PROCESSOR = dict(policy='original_pad',pad_multiple=32,brightness=0.)
_OPTIMIZER = dict(lora_lr=1e-4,new_lr=5e-4,weight_decay=.01,grad_clip=1.,schedule='constant',clip_per_model=True)
_REPRODUCIBILITY = dict(deterministic_algorithms=True,warn_only=False,cublas_workspace_config=':4096:8',
                        cudnn_deterministic=True,cudnn_benchmark=False,allow_tf32=False)
_DATA_POLICY = dict(ages_s=[0.,.5,1.,1.5],max_train_rows=12,max_val_rows=4,max_source_observations=64,
                    moving_speed_mps=1.,moving_fraction=.5)
_DATA_PATHS = ('train_manifest','val_manifest','stats','sensor_root')
_TOP_KEYS = ('schema_version','train_seed','fast','slow','allow_single_gpu','model','data','processor',
             'stages','arms','loss','optimizer','missing_fraction','feedback_policy','gradient_probe_count',
             'train_curve_steps','initial_checkpoint','reproducibility','output')


def _fields(value, expected, path):
    if not isinstance(value,Mapping) or set(value) != set(expected):
        raise ValueError(f'{path} requires exactly the declared fields {sorted(expected)}')


def _fixed(value, expected, path):
    if isinstance(expected,dict):
        _fields(value,expected,path)
        for key,item in expected.items():
            _fixed(value[key],item,f'{path}.{key}')
    elif isinstance(expected,list):
        if not isinstance(value,list) or len(value) != len(expected):
            raise ValueError(f'{path} differs from fixed joint policy')
        for i,item in enumerate(expected):
            _fixed(value[i],item,f'{path}[{i}]')
    elif (value != expected or isinstance(value,bool) != isinstance(expected,bool)
            or (isinstance(expected,int) and not isinstance(expected,bool) and not isinstance(value,int))):
        raise ValueError(f'{path} must equal fixed joint policy {expected!r}')


def _path(value, field):
    if not isinstance(value,str) or not value.strip():
        raise ValueError(f'{field} must be a nonempty local path string')


def validate_joint_config(c):
    """Reject unsupported drift; paths may identify different verified local assets."""
    _fields(c,_TOP_KEYS,'joint config')
    fixed = dict(schema_version='slowfast_v11_joint',train_seed=42,allow_single_gpu=True,
        model=_MODEL,processor=_PROCESSOR,stages=dict(slow_adapt=240,fast_consumer=240),
        arms=['no_kv','detached','joint'],loss=dict(slow_weight=1.),optimizer=_OPTIMIZER,
        missing_fraction=.3,feedback_policy='shared_frozen_fast_warmup',gradient_probe_count=4,
        train_curve_steps=[60,120,240],reproducibility=_REPRODUCIBILITY)
    for key,value in fixed.items():
        _fixed(c[key],value,key)
    for name in ('fast','slow'):
        _fields(c[name],('model_path','device'),name)
        _path(c[name]['model_path'],f'{name}.model_path')
        _fixed(c[name]['device'],'cuda:0',f'{name}.device')
    _fields(c['data'],(*_DATA_PATHS,*_DATA_POLICY),'data')
    for name in _DATA_PATHS:
        _path(c['data'][name],f'data.{name}')
    for name,value in _DATA_POLICY.items():
        _fixed(c['data'][name],value,f'data.{name}')
    for name in ('initial_checkpoint','output'):
        _path(c[name],name)


def _train_strata(rows, moving_threshold):
    if (isinstance(moving_threshold,bool) or not isinstance(moving_threshold,Real)
            or not math.isfinite(moving_threshold) or moving_threshold <= 0):
        raise ValueError('Moving threshold must be finite and positive')
    strata = {'moving':[],'low_speed':[]}
    seen = set()
    for row in rows:
        if not isinstance(row,Mapping) or row.get('split') != 'train':
            raise ValueError('Balanced optimizer sampling accepts TRAIN rows only')
        token = row.get('token')
        if not isinstance(token,str) or not token or token in seen:
            raise ValueError('TRAIN tokens must be nonempty and unique')
        seen.add(token)
        ego = row.get('ego')
        if (not isinstance(ego,(list,tuple)) or len(ego) != 4
                or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in ego)):
            raise ValueError('Sampling requires four finite current raw ego features')
        stratum = 'moving' if math.hypot(ego[0],ego[1]) >= moving_threshold else 'low_speed'
        strata[stratum].append(token)
    if any(not values for values in strata.values()):
        raise ValueError('Balanced sampling requires both moving and low-speed TRAIN strata; no fallback')
    return {name:sorted(tokens) for name,tokens in strata.items()}


def _shuffle_cycles(tokens, count, seed):
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        cycle = list(tokens)
        rng.shuffle(cycle)
        result.extend(cycle)
    return iter(result[:count])


def balanced_source_order(rows, steps, seed, moving_threshold=1.):
    """Return 50/50 TRAIN tokens, without replacement within stratum shuffle cycles.

    Uses current raw velocity only. This generic even-step helper also supports
    small CPU fixtures; the joint config fixes each real phase at 240 updates.
    """
    if isinstance(steps,bool) or not isinstance(steps,int) or steps <= 0 or steps % 2:
        raise ValueError('Balanced source steps must be a positive even integer')
    if isinstance(seed,bool) or not isinstance(seed,int):
        raise ValueError('Sampling seed must be an integer')
    strata = _train_strata(rows,moving_threshold)
    count = steps//2
    labels = ['moving']*count + ['low_speed']*count
    random.Random(seed).shuffle(labels)
    cycles = {name:_shuffle_cycles(strata[name],count,seed+i+1)
              for i,name in enumerate(('moving','low_speed'))}
    return [next(cycles[name]) for name in labels]


def balanced_plan(rows, pairs, c):
    """Fixed 240 TRAIN targets, each stratum independently assigned 30% no-KV.

    The legacy harness supplies pairs, but fresh coverage is checked by that
    harness. Historical pairs and labels never influence this fresh-only plan.
    All arms consume this same order; no_kv overrides cache use in its trainer.
    """
    validate_joint_config(c)
    rows = list(rows)
    steps,seed = c['stages']['fast_consumer'],c['train_seed']
    strata = _train_strata(rows,c['data']['moving_speed_mps'])
    order = balanced_source_order(rows,steps,seed,c['data']['moving_speed_mps'])
    names = {token:name for name,tokens in strata.items() for token in tokens}
    count = steps//2
    absent = round(count*c['missing_fraction'])
    flags = {}
    for i,name in enumerate(('moving','low_speed')):
        modes = ['none']*absent + ['fresh']*(count-absent)
        random.Random(seed+3+i).shuffle(modes)
        flags[name] = iter(modes)
    return [dict(target_token=token,source_token=token,split='train',stratum=names[token],
                 mode=next(flags[names[token]]),age_s=0.) for token in order]


def fresh_probe_steps(plan, count=4):
    """One-based upper fresh quantiles: 25%,50%,75%,100% for the fixed count four."""
    if isinstance(count,bool) or not isinstance(count,int) or count <= 0:
        raise ValueError('Gradient probe count must be a positive integer')
    fresh = []
    for step,item in enumerate(plan,1):
        if item.get('split') != 'train' or item.get('mode') not in ('none','fresh'):
            raise ValueError('Gradient probes require the TRAIN absent/fresh plan')
        if item['mode'] == 'fresh':
            fresh.append(step)
    if len(fresh) < count:
        raise ValueError('Insufficient distinct fresh steps for requested gradient probes')
    return [fresh[(len(fresh)*i+count-1)//count-1] for i in range(1,count+1)]
