"""Validate the implemented Fast-only v2 contract without opening input files.

Unversioned checkpoints/YAML retain their original behavior. Version 2 is
strict: unsupported policy values and unknown keys are errors, so a documented
setting cannot silently become an unused comment. Evidence contents and hashes
are checked by the dataset builder/checkpoint paths, not this schema validator.
"""
from __future__ import annotations

import math
import re
from pathlib import PurePath
from collections.abc import Mapping

LOSS = 'l1_xy_sum_point_batch_mean'
REPRODUCIBILITY_POLICY = {
    'deterministic_algorithms': True, 'warn_only': False,
    'cublas_workspace_config': ':4096:8', 'cudnn_deterministic': True,
    'cudnn_benchmark': False, 'allow_tf32': False,
}
MODEL_POLICY = {
    'ego_dim': 3, 'num_commands': 7, 'planning_queries': 6, 'freeze_visual_merger': True,
    'lora_parameter_dtype': 'float32',
    'lora_target_suffixes': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'token_order': ['native_temporal_images_and_text', 'ego', 'command', 'target', 'planning_queries'],
}
PROCESSOR_POLICY = {
    'policy': 'original_pad', 'min_pixels': None, 'max_pixels': None, 'pad_multiple': 32,
    'pad_fill': [0, 0, 0], 'prompt_template': 'temporal_task_v1',
}
DATA_POLICY = {
    'navigation_mode': 'annotation_near_route_target',
    'navigation_origin': 'ego_center_from_gnss_extrinsic', 'gnss_mount_xy_m': [-1.4, 0],
    'target_lookahead_policy': 'as_recorded_official_near_route', 'target_fallback': 'error',
    'timing_policy': 'nominal_simulation_time_10hz', 'nominal_hz': 10,
    'require_official_label_pose': True, 'geographic_group_policy': 'town_holdout',
    'history_offsets': [-3, -2, -1, 0], 'future_offsets': [5, 10, 15, 20, 25, 30],
    'ego_source': 'speed_sensor_backward_differences', 'coordinate_convention': 'carla_world_to_forward_left',
    'command_policy': {'vocabulary': {'UNKNOWN': 0, 'LEFT': 1, 'RIGHT': 2, 'STRAIGHT': 3,
                                      'LANEFOLLOW': 4, 'CHANGELANELEFT': 5, 'CHANGELANERIGHT': 6},
                       'missing_index': 0, 'invalid': 'reject'},
}
MODES = ['model_only', 'streaming_e2e', 'disk_e2e']


def _fields(value, path, required, optional=()):
    if not isinstance(value, Mapping):
        raise ValueError(f'{path} must be a mapping')
    missing = set(required) - value.keys()
    if missing:
        raise ValueError(f'{path} missing required fields: {sorted(missing)}')
    unknown = value.keys() - set(required) - set(optional)
    if unknown:
        raise ValueError(f'{path} unsupported fields: {sorted(unknown)}')


def _fixed(value, expected, path):
    if isinstance(expected, dict):
        _fields(value, path, expected)
        for key, item in expected.items():
            _fixed(value[key], item, f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f'{path} must equal supported policy {expected!r}')
        for index, item in enumerate(expected):
            _fixed(value[index], item, f'{path}[{index}]')
    elif (value != expected or (isinstance(value, bool) != isinstance(expected, bool))
          or (isinstance(expected, int) and not isinstance(expected, bool) and not isinstance(value, int))):
        raise ValueError(f'{path} must equal supported policy {expected!r}, got {value!r}')


def _integer(value, path, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{path} must be an integer >= {minimum}' + (f' and <= {maximum}' if maximum is not None else ''))


def _number(value, path, minimum=0., maximum=None, strict_minimum=False, strict_maximum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{path} must be finite numeric')
    if value < minimum or (strict_minimum and value == minimum) or (maximum is not None and
            (value > maximum or (strict_maximum and value == maximum))):
        raise ValueError(f'{path} is outside the supported numeric range')


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{path} must be a nonempty string')


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(f'{path} must be boolean')


def validate_data_policy(data):
    """Validate the complete v2 data mapping; do not read manifests/evidence."""
    paths = ('root', 'manifest_dir', 'train_manifest', 'val_manifest', 'test_manifest', 'stats', 'source_audit')
    _fields(data, 'data', (*DATA_POLICY, *paths, 'split_seed', 'val_towns', 'test_towns', 'train_stride', 'eval_stride'))
    for key, expected in DATA_POLICY.items():
        _fixed(data[key], expected, f'data.{key}')
    for key in paths:
        _string(data[key], f'data.{key}')
    directory = PurePath(data['manifest_dir'])
    expected_names = {'train_manifest': ('train.jsonl', 'deep_train.jsonl'),
                      'val_manifest': ('val.jsonl',), 'test_manifest': ('test.jsonl',),
                      'stats': ('stats.json',)}
    for key, names in expected_names.items():
        if PurePath(data[key]) not in {directory / name for name in names}:
            raise ValueError(f'data.{key} must match manifest_dir and supported filename(s) {names}')
    manifests = [data[key] for key in ('train_manifest', 'val_manifest', 'test_manifest')]
    if len(set(manifests)) != 3:
        raise ValueError('data train_manifest, val_manifest and test_manifest must be distinct')
    _integer(data['split_seed'], 'data.split_seed', maximum=2**32-1)
    for key in ('train_stride', 'eval_stride'):
        _integer(data[key], f'data.{key}', minimum=1)
    for key in ('val_towns', 'test_towns'):
        towns = data[key]
        if not isinstance(towns, list) or not towns or any(not isinstance(town, str) or
                not re.fullmatch(r'Town\d{2}', town) for town in towns) or len(set(towns)) != len(towns):
            raise ValueError(f'data.{key} requires unique official TownXX names')
    if set(data['val_towns']) & set(data['test_towns']):
        raise ValueError('data.val_towns and data.test_towns must be disjoint')
    return data


def validate_training_config(config):
    """Return the original config after validation; unversioned inputs pass through."""
    if not isinstance(config, Mapping):
        raise ValueError('config must be a mapping')
    if 'schema_version' not in config:
        return config
    _fixed(config['schema_version'], 2, 'schema_version')
    _fields(config, 'config', ('schema_version', 'train_seed', 'smoke', 'device', 'bf16', 'cpu_threads',
                               'output', 'loss', 'model', 'processor', 'data', 'loader', 'train', 'benchmark', 'execution'),
            optional=('smoke_max_windows', 'reproducibility'))
    if 'reproducibility' in config:
        _fixed(config['reproducibility'], REPRODUCIBILITY_POLICY, 'reproducibility')
    _integer(config['train_seed'], 'train_seed', maximum=2**32-1)
    _boolean(config['smoke'], 'smoke'); _boolean(config['bf16'], 'bf16')
    _integer(config['cpu_threads'], 'cpu_threads', minimum=1)
    _string(config['device'], 'device'); _string(config['output'], 'output')
    _fixed(config['loss'], LOSS, 'loss')
    if config['smoke'] and 'smoke_max_windows' not in config:
        raise ValueError('smoke_max_windows is required in smoke mode')
    if 'smoke_max_windows' in config:
        _integer(config['smoke_max_windows'], 'smoke_max_windows', minimum=1)

    model = config['model']
    _fields(model, 'model', (*MODEL_POLICY, 'path', 'lora_rank', 'lora_alpha', 'lora_dropout'))
    for key, expected in MODEL_POLICY.items():
        _fixed(model[key], expected, f'model.{key}')
    _string(model['path'], 'model.path')
    for key in ('lora_rank', 'lora_alpha'):
        _integer(model[key], f'model.{key}', minimum=1)
    _number(model['lora_dropout'], 'model.lora_dropout', maximum=1., strict_maximum=True)

    processor = config['processor']
    _fields(processor, 'processor', PROCESSOR_POLICY)
    for key, expected in PROCESSOR_POLICY.items():
        _fixed(processor[key], expected, f'processor.{key}')
    validate_data_policy(config['data'])

    loader = config['loader']
    _fields(loader, 'loader', ('workers', 'prefetch', 'pin_memory', 'sampler'))
    _integer(loader['workers'], 'loader.workers')
    _integer(loader['prefetch'], 'loader.prefetch', minimum=1)
    _boolean(loader['pin_memory'], 'loader.pin_memory')
    _fixed(loader['sampler'], 'counter_v1', 'loader.sampler')

    train = config['train']
    _fields(train, 'train', ('optimizer', 'steps', 'microbatch', 'gradient_accumulation', 'effective_batch',
                             'lora_lr', 'new_lr', 'weight_decay', 'warmup_fraction', 'grad_clip',
                             'brightness', 'eval_every', 'checkpoint_every'))
    _fixed(train['optimizer'], 'adamw', 'train.optimizer')
    for key in ('steps', 'microbatch', 'gradient_accumulation', 'effective_batch', 'eval_every', 'checkpoint_every'):
        _integer(train[key], f'train.{key}', minimum=1)
    if train['effective_batch'] != train['microbatch']*train['gradient_accumulation']:
        raise ValueError('train.effective_batch must equal microbatch * gradient_accumulation')
    for key in ('lora_lr', 'new_lr', 'grad_clip'):
        _number(train[key], f'train.{key}', strict_minimum=True)
    _number(train['weight_decay'], 'train.weight_decay')
    _number(train['warmup_fraction'], 'train.warmup_fraction', maximum=1., strict_maximum=True)
    _number(train['brightness'], 'train.brightness', maximum=1.)

    execution = config['execution']
    _fields(execution, 'execution', ('stage', 'stop_after'))
    stage, stop = execution['stage'], execution['stop_after']
    if stage not in ('formal', 'preflight', 'deep_overfit'):
        raise ValueError('execution.stage must be formal, preflight or deep_overfit')
    if stage == 'formal':
        if stop is not None:
            raise ValueError('execution.stop_after must be null for formal training')
    else:
        _integer(stop, 'execution.stop_after', minimum=1)
        if stage == 'preflight' and (stop > 20 or stop >= train['steps']):
            raise ValueError('execution preflight stop_after must be <=20 and less than total train.steps')
        if stage == 'deep_overfit' and not (stop <= train['steps'] <= 1000):
            raise ValueError('execution deep_overfit requires stop_after <= train.steps <=1000')

    benchmark = config['benchmark']
    _fields(benchmark, 'benchmark', ('enabled', 'warmup', 'repeats', 'modes'))
    _boolean(benchmark['enabled'], 'benchmark.enabled')
    _integer(benchmark['warmup'], 'benchmark.warmup', minimum=1)
    _integer(benchmark['repeats'], 'benchmark.repeats', minimum=100 if benchmark['enabled'] else 1)
    _fixed(benchmark['modes'], MODES, 'benchmark.modes')
    return config
