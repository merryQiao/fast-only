from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from fast_only.configuration import validate_data_policy, validate_training_config


def canonical():
    return {
        'schema_version': 2, 'train_seed': 42, 'smoke': False, 'device': 'cuda:0',
        'bf16': True, 'cpu_threads': 4, 'output': 'outputs/example',
        'loss': 'l1_xy_sum_point_batch_mean',
        'execution': {'stage': 'formal', 'stop_after': None},
        'reproducibility': {'deterministic_algorithms': True, 'warn_only': False,
                            'cublas_workspace_config': ':4096:8', 'cudnn_deterministic': True,
                            'cudnn_benchmark': False, 'allow_tf32': False},
        'model': {'path': 'models/Qwen3.5-0.8B', 'lora_rank': 16, 'lora_alpha': 32,
                  'lora_dropout': .05, 'ego_dim': 3, 'num_commands': 7, 'planning_queries': 6,
                  'freeze_visual_merger': True, 'lora_parameter_dtype': 'float32',
                  'lora_target_suffixes': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                  'token_order': ['native_temporal_images_and_text', 'ego', 'command', 'target', 'planning_queries']},
        'processor': {'policy': 'original_pad', 'min_pixels': None, 'max_pixels': None,
                      'pad_multiple': 32, 'pad_fill': [0, 0, 0], 'prompt_template': 'temporal_task_v1'},
        'data': {'root': 'missing/root', 'manifest_dir': 'missing',
                 'train_manifest': 'missing/train.jsonl', 'val_manifest': 'missing/val.jsonl',
                 'test_manifest': 'missing/test.jsonl', 'stats': 'missing/stats.json',
                 'navigation_mode': 'annotation_near_route_target',
                 'navigation_origin': 'ego_center_from_gnss_extrinsic', 'gnss_mount_xy_m': [-1.4, 0],
                 'target_lookahead_policy': 'as_recorded_official_near_route', 'target_fallback': 'error',
                 'timing_policy': 'nominal_simulation_time_10hz', 'nominal_hz': 10,
                 'source_audit': 'missing/source_audit.json', 'require_official_label_pose': True,
                 'split_seed': 42, 'geographic_group_policy': 'town_holdout',
                 'val_towns': ['Town12'], 'test_towns': ['Town06'], 'train_stride': 1, 'eval_stride': 5,
                 'history_offsets': [-3, -2, -1, 0], 'future_offsets': [5, 10, 15, 20, 25, 30],
                 'ego_source': 'speed_sensor_backward_differences',
                 'coordinate_convention': 'carla_world_to_forward_left',
                 'command_policy': {'vocabulary': {'UNKNOWN': 0, 'LEFT': 1, 'RIGHT': 2, 'STRAIGHT': 3,
                                                  'LANEFOLLOW': 4, 'CHANGELANELEFT': 5, 'CHANGELANERIGHT': 6},
                                    'missing_index': 0, 'invalid': 'reject'}},
        'loader': {'workers': 2, 'prefetch': 2, 'pin_memory': True, 'sampler': 'counter_v1'},
        'train': {'optimizer': 'adamw', 'steps': 8000, 'microbatch': 2, 'gradient_accumulation': 8,
                  'effective_batch': 16, 'lora_lr': .0001, 'new_lr': .0005, 'weight_decay': .01,
                  'warmup_fraction': .03, 'grad_clip': 1., 'brightness': .1,
                  'eval_every': 500, 'checkpoint_every': 500},
        'benchmark': {'enabled': True, 'warmup': 10, 'repeats': 100,
                      'modes': ['model_only', 'streaming_e2e', 'disk_e2e']}}


def test_schema_two_validates_without_opening_dataset_files_and_is_not_mutated():
    config = canonical(); before = deepcopy(config)
    assert validate_training_config(config) is config
    assert validate_data_policy(config['data']) is config['data']
    assert config == before


def test_legacy_config_passes_through_unchanged():
    legacy = {'seed': 42, 'processor': {'min_pixels': 16384}, 'custom_old_key': object()}
    assert validate_training_config(legacy) is legacy


@pytest.mark.parametrize('path,value', [
    ('schema_version', 3), ('schema_version', 2.0), ('train_seed', True), ('loss', 'mse'),
    ('model.ego_dim', 4), ('model.ego_dim', 3.0), ('model.num_commands', 8), ('model.planning_queries', 5),
    ('model.freeze_visual_merger', False), ('model.lora_parameter_dtype', 'bfloat16'), ('model.lora_target_suffixes', ['q_proj']),
    ('model.token_order', ['ego', 'target']),
    ('processor.policy', 'resize'), ('processor.max_pixels', 147456),
    ('processor.pad_multiple', 16), ('processor.pad_fill', [127, 127, 127]),
    ('processor.prompt_template', 'new_prompt'),
    ('data.navigation_mode', 'zero'), ('data.navigation_origin', 'gnss_sensor'),
    ('data.gnss_mount_xy_m', [0, 0]), ('data.target_lookahead_policy', 'future_gt'),
    ('data.target_fallback', 'zero'), ('data.timing_policy', 'frame_index_fallback'),
    ('data.nominal_hz', 20), ('data.require_official_label_pose', False),
    ('data.geographic_group_policy', 'random_window'), ('data.history_offsets', [-4, -2, -1, 0]),
    ('data.future_offsets', [1, 2, 3, 4, 5, 6]), ('data.ego_source', 'future_acceleration'),
    ('data.coordinate_convention', 'carla_right'), ('data.command_policy.invalid', 'clip'),
    ('data.command_policy.missing_index', -1), ('data.command_policy.vocabulary.LEFT', 2),
    ('loader.sampler', 'global_rng'), ('loader.workers', -1), ('loader.prefetch', 0),
    ('train.optimizer', 'sgd'), ('train.effective_batch', 15), ('train.microbatch', 0),
    ('train.brightness', 1.1), ('train.lora_lr', 0), ('train.grad_clip', 0),
    ('benchmark.repeats', 20), ('benchmark.warmup', 0), ('benchmark.modes', ['disk_e2e']),
    ('data.val_towns', ['Town06']), ('data.train_stride', 0), ('data.source_audit', ''),
])
def test_changed_or_invalid_policy_is_rejected_with_key(path, value):
    config = canonical(); target = config
    parts = path.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(ValueError, match=parts[-1]):
        validate_training_config(config)


@pytest.mark.parametrize('section', ['model', 'processor', 'data', 'loader', 'train', 'benchmark', 'execution', 'reproducibility'])
def test_every_required_field_and_unknown_field_are_checked(section):
    for field in canonical()[section]:
        config = canonical(); del config[section][field]
        with pytest.raises(ValueError, match=field):
            validate_training_config(config)
    config = canonical(); config[section]['unused_typo'] = 1
    with pytest.raises(ValueError, match='unused_typo'):
        validate_training_config(config)


def test_shipped_configs_match_formal_preflight_and_deep_overfit_budgets():
    configs = {}
    for name in ('fast_only_b2d', 'preflight_smoke', 'deep_overfit_v2'):
        configs[name] = yaml.safe_load(Path(f'configs/{name}.yaml').read_text())
        validate_training_config(configs[name])
    formal, preflight, deep = (configs[name] for name in ('fast_only_b2d', 'preflight_smoke', 'deep_overfit_v2'))
    assert formal['train']['steps'] == 8000 and formal['train']['effective_batch'] == 16
    assert preflight['train']['steps'] == 8000 and not preflight['smoke']
    assert formal['execution'] == {'stage': 'formal', 'stop_after': None}
    assert preflight['execution'] == {'stage': 'preflight', 'stop_after': 4}
    assert deep['execution'] == {'stage': 'deep_overfit', 'stop_after': 300}
    # The bounded preflight exercises exactly the beginning of the formal LR
    # schedule, including its 240-update warmup; stopping does not redefine it.
    import math
    assert math.ceil(preflight['train']['steps'] * preflight['train']['warmup_fraction']) == 240
    for key, value in formal['train'].items():
        if key not in {'eval_every', 'checkpoint_every'}:
            assert preflight['train'][key] == value, key
    assert deep['train']['steps'] == 300 and deep['smoke'] and deep['smoke_max_windows'] == 1
    assert deep['train']['microbatch'] == deep['train']['gradient_accumulation'] == 1
    assert deep['train']['brightness'] == 0 and not deep['benchmark']['enabled']
    assert len({config['output'] for config in configs.values()}) == 3
    assert all(config['output'].endswith('_deterministic') for config in configs.values())
    assert all(config['reproducibility'] == formal['reproducibility'] for config in configs.values())
    assert all(config['data']['stats'] == formal['data']['stats'] for config in configs.values())
    assert deep['data']['train_manifest'].endswith('/deep_train.jsonl')


@pytest.mark.parametrize('key', ['train_manifest', 'val_manifest', 'test_manifest', 'stats'])
def test_manifest_paths_match_builder_destinations(key):
    data = canonical()['data']
    data[key] = 'somewhere_else/unused.jsonl'
    with pytest.raises(ValueError, match=key):
        validate_data_policy(data)


@pytest.mark.parametrize('stage,stop,total', [
    ('formal', 4, 8000), ('invalid', None, 8000),
    ('preflight', None, 8000), ('preflight', 0, 8000), ('preflight', 21, 8000),
    ('preflight', 4, 4), ('preflight', True, 8000),
    ('deep_overfit', None, 300), ('deep_overfit', 0, 300),
    ('deep_overfit', 301, 300), ('deep_overfit', 1001, 1001),
])
def test_execution_caps_are_enforced_without_changing_total_schedule(stage, stop, total):
    config = canonical()
    config['execution'] = {'stage': stage, 'stop_after': stop}
    config['train']['steps'] = total
    with pytest.raises(ValueError, match='execution'):
        validate_training_config(config)


@pytest.mark.parametrize('stage,stop,total', [('formal', None, 8000),
    ('preflight', 4, 8000), ('preflight', 20, 8000), ('deep_overfit', 300, 300)])
def test_supported_execution_caps_validate(stage, stop, total):
    config = canonical()
    config['execution'] = {'stage': stage, 'stop_after': stop}
    config['train']['steps'] = total
    assert validate_training_config(config) is config


def test_reproducibility_block_is_optional_for_existing_schema_two_configs():
    config = canonical()
    del config['reproducibility']
    assert validate_training_config(config) is config


@pytest.mark.parametrize('key,value', [
    ('deterministic_algorithms', False), ('warn_only', True),
    ('cublas_workspace_config', ':16:8'), ('cudnn_deterministic', False),
    ('cudnn_benchmark', True), ('allow_tf32', True), ('allow_tf32', 0),
])
def test_reproducibility_values_fail_closed(key, value):
    config = canonical()
    config['reproducibility'][key] = value
    with pytest.raises(ValueError, match=key):
        validate_training_config(config)
