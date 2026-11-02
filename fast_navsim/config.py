"""Strict, file-independent validation for the approved NAVSIM experiment.

The preflight retains the formal 8000-update schedule and only caps execution.
Dataset evidence is validated by the data layer; reading a config opens no data.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path

import yaml

LOSS = 'navsim_l1_pose_mean'
MODEL_POLICY = {
    'ego_dim': 4, 'num_commands': 4, 'planning_queries': 8, 'output_dim': 3,
    'continuous_target': False, 'freeze_visual_merger': True,
    'lora_parameter_dtype': 'float32',
    'lora_target_suffixes': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'token_order': ['native_temporal_images_and_text', 'ego', 'command', 'planning_queries'],
}
PROCESSOR_POLICY = {
    'policy': 'original_pad', 'pad_multiple': 32, 'pad_fill': [0, 0, 0],
    'history_times_s': [-1.5, -1., -.5, 0.], 'prompt_template': 'navsim_temporal_v1',
}
DATA_POLICY = {
    'frequency_hz': 2, 'camera': 'CAM_F0', 'history_offsets': [-3, -2, -1, 0],
    'future_offsets': list(range(1, 9)), 'coordinate_convention': 'rear_axle_forward_left_se2',
    'navigation': 'route_command_only', 'ego_fields': ['vx', 'vy', 'ax', 'ay'],
    'cohort': 'bounded_mini_preflight',
}
REPRODUCIBILITY_POLICY = {
    'deterministic_algorithms': True, 'warn_only': False,
    'cublas_workspace_config': ':4096:8', 'cudnn_deterministic': True,
    'cudnn_benchmark': False, 'allow_tf32': False,
}
BENCHMARK_MODES = ['model_only', 'streaming_e2e', 'disk_e2e']
TRAIN_FIELDS = (
    'optimizer', 'steps', 'microbatch', 'gradient_accumulation', 'effective_batch',
    'lora_lr', 'new_lr', 'weight_decay', 'warmup_fraction', 'grad_clip',
    'brightness', 'eval_every', 'checkpoint_every',
)
STAGES = ('formal', 'preflight', 'deep_overfit')


def _fields(value, path, required):
    if not isinstance(value, Mapping):
        raise ValueError(f'{path} must be a mapping')
    missing = set(required) - value.keys()
    if missing:
        raise ValueError(f'{path} missing required fields: {sorted(missing)}')
    unknown = value.keys() - set(required)
    if unknown:
        raise ValueError(f'{path} unsupported fields: {sorted(unknown, key=str)}')


def _fixed(value, expected, path):
    if isinstance(expected, Mapping):
        _fields(value, path, expected)
        for key, item in expected.items():
            _fixed(value[key], item, f'{path}.{key}')
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f'{path} must equal supported policy {expected!r}')
        for index, item in enumerate(expected):
            _fixed(value[index], item, f'{path}[{index}]')
        return
    if isinstance(expected, bool):
        valid_type = isinstance(value, bool)
    elif isinstance(expected, int):
        valid_type = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(expected, float):
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid_type = type(value) is type(expected)
    if not valid_type or value != expected:
        raise ValueError(f'{path} must equal supported policy {expected!r}, got {value!r}')


def _integer(value, path, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{path} must be an integer >= {minimum}' + (f' and <= {maximum}' if maximum is not None else ''))


def _number(value, path, minimum=0., maximum=None, positive=False, exclusive_max=False):
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{path} must be finite numeric')
    if value < minimum or (positive and value == minimum) or (maximum is not None and (
            value > maximum or (exclusive_max and value == maximum))):
        raise ValueError(f'{path} is outside the supported numeric range')


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path} must be a nonempty string')


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(f'{path} must be boolean')


def validate_config(config):
    """Validate every consumed field; return ``config`` without modifying it.

    Unknown fields fail closed, including B2D options and obsolete smoke flags.
    The three execution stages retain their approved training budgets.
    """
    _fields(config, 'config', (
        'schema_version', 'train_seed', 'split_seed', 'device', 'bf16', 'cpu_threads',
        'output', 'loss', 'model', 'processor', 'data', 'train', 'loader',
        'benchmark', 'execution', 'reproducibility',
    ))
    _fixed(config['schema_version'], 'navsim_v1', 'schema_version')
    for key in ('train_seed', 'split_seed'):
        _integer(config[key], key, maximum=2**32 - 1)
    _string(config['device'], 'device')
    _boolean(config['bf16'], 'bf16')
    _integer(config['cpu_threads'], 'cpu_threads', minimum=1)
    _string(config['output'], 'output')
    _fixed(config['loss'], LOSS, 'loss')
    _fixed(config['reproducibility'], REPRODUCIBILITY_POLICY, 'reproducibility')

    execution = config['execution']
    _fields(execution, 'execution', ('stage', 'stop_after'))
    stage = execution['stage']
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f'execution.stage must be one of {STAGES!r}')
    stop = execution['stop_after']
    if stage == 'formal':
        _fixed(stop, None, 'execution.stop_after')
    elif stage == 'preflight':
        _integer(stop, 'execution.stop_after', minimum=1, maximum=20)
    else:
        _fixed(stop, 300, 'execution.stop_after')

    model = config['model']
    _fields(model, 'model', (*MODEL_POLICY, 'path', 'lora_rank', 'lora_alpha', 'lora_dropout'))
    for key, expected in MODEL_POLICY.items():
        _fixed(model[key], expected, f'model.{key}')
    _string(model['path'], 'model.path')
    for key in ('lora_rank', 'lora_alpha'):
        _integer(model[key], f'model.{key}', minimum=1)
    _number(model['lora_dropout'], 'model.lora_dropout', maximum=1., exclusive_max=True)
    _fixed(config['processor'], PROCESSOR_POLICY, 'processor')

    data = config['data']
    paths = ('root', 'train_manifest', 'val_manifest', 'test_manifest', 'stats', 'source_audit')
    _fields(data, 'data', (*DATA_POLICY, *paths))
    for key, expected in DATA_POLICY.items():
        _fixed(data[key], expected, f'data.{key}')
    for key in paths:
        _string(data[key], f'data.{key}')
    train_name = 'deep_train' if stage == 'deep_overfit' else 'train'
    for key, name in (('train_manifest', train_name + '.jsonl'), ('val_manifest', 'val.jsonl'),
                      ('test_manifest', 'test.jsonl'), ('stats', 'stats.json')):
        _fixed(data[key], f'outputs/navsim/data/{name}', f'data.{key}')

    train = config['train']
    _fields(train, 'train', TRAIN_FIELDS)
    _fixed(train['optimizer'], 'adamw', 'train.optimizer')
    for key in ('steps', 'microbatch', 'gradient_accumulation', 'effective_batch', 'eval_every', 'checkpoint_every'):
        _integer(train[key], f'train.{key}', minimum=1)
    if train['effective_batch'] != train['microbatch'] * train['gradient_accumulation']:
        raise ValueError('train.effective_batch must equal microbatch * gradient_accumulation')
    for key in ('lora_lr', 'new_lr', 'grad_clip'):
        _number(train[key], f'train.{key}', positive=True)
    _number(train['weight_decay'], 'train.weight_decay')
    _number(train['warmup_fraction'], 'train.warmup_fraction', maximum=1., exclusive_max=True)
    _number(train['brightness'], 'train.brightness', maximum=1.)
    budget = {
        'steps': 8000, 'microbatch': 2, 'gradient_accumulation': 8, 'effective_batch': 16,
        'brightness': .1, 'eval_every': 500, 'checkpoint_every': 500,
    }
    if stage == 'preflight':
        budget.update(eval_every=1, checkpoint_every=1)
    elif stage == 'deep_overfit':
        budget.update(steps=300, microbatch=1, gradient_accumulation=1, effective_batch=1,
                      brightness=0., eval_every=100, checkpoint_every=150)
    for key, expected in budget.items():
        _fixed(train[key], expected, f'train.{key}')

    loader = config['loader']
    _fields(loader, 'loader', ('workers', 'prefetch', 'pin_memory', 'sampler'))
    _integer(loader['workers'], 'loader.workers')
    _integer(loader['prefetch'], 'loader.prefetch', minimum=1)
    _boolean(loader['pin_memory'], 'loader.pin_memory')
    _fixed(loader['sampler'], 'counter_v1', 'loader.sampler')

    benchmark = config['benchmark']
    _fields(benchmark, 'benchmark', ('enabled', 'warmup', 'repeats', 'modes', 'p95_budget_ms'))
    _boolean(benchmark['enabled'], 'benchmark.enabled')
    _fixed(benchmark['enabled'], stage != 'deep_overfit', 'benchmark.enabled')
    _integer(benchmark['warmup'], 'benchmark.warmup', minimum=1)
    _integer(benchmark['repeats'], 'benchmark.repeats', minimum=100)
    _fixed(benchmark['modes'], BENCHMARK_MODES, 'benchmark.modes')
    _number(benchmark['p95_budget_ms'], 'benchmark.p95_budget_ms', positive=True)
    return config


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses silently overwritten configuration keys."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ValueError('config YAML keys must be scalar and hashable') from exc
            if duplicate:
                raise ValueError(f'duplicate config YAML key: {key}')
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def read_config(path):
    """Read a safe, duplicate-free YAML mapping and apply the NAVSIM contract."""
    try:
        with Path(path).open(encoding='utf-8') as handle:
            config = yaml.load(handle, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f'invalid config YAML: {exc}') from exc
    return validate_config(config)
