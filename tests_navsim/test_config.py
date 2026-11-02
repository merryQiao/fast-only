"""CPU-only tests for the NAVSIM schema and bounded execution contracts."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def api():
    assert (ROOT / 'fast_navsim/config.py').is_file(), 'NAVSIM config validator is missing'
    return import_module('fast_navsim.config')


def canonical(stage='formal'):
    config = {
        'schema_version': 'navsim_v1', 'train_seed': 42, 'split_seed': 42,
        'device': 'cuda:0', 'bf16': True, 'cpu_threads': 4,
        'output': 'outputs/navsim/formal', 'loss': 'navsim_l1_pose_mean',
        'model': {
            'path': 'models/Qwen3.5-0.8B', 'lora_rank': 16, 'lora_alpha': 32,
            'lora_dropout': .05, 'lora_parameter_dtype': 'float32',
            'ego_dim': 4, 'num_commands': 4, 'planning_queries': 8,
            'output_dim': 3, 'continuous_target': False, 'freeze_visual_merger': True,
            'lora_target_suffixes': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            'token_order': ['native_temporal_images_and_text', 'ego', 'command', 'planning_queries'],
        },
        'processor': {
            'policy': 'original_pad', 'pad_multiple': 32, 'pad_fill': [0, 0, 0],
            'history_times_s': [-1.5, -1., -.5, 0.], 'prompt_template': 'navsim_temporal_v1',
        },
        'data': {
            'root': 'data/navsim_smoke', 'train_manifest': 'outputs/navsim/data/train.jsonl',
            'val_manifest': 'outputs/navsim/data/val.jsonl', 'test_manifest': 'outputs/navsim/data/test.jsonl',
            'stats': 'outputs/navsim/data/stats.json', 'source_audit': 'outputs/navsim/audit/source_audit.json',
            'frequency_hz': 2, 'camera': 'CAM_F0', 'history_offsets': [-3, -2, -1, 0],
            'future_offsets': list(range(1, 9)), 'coordinate_convention': 'rear_axle_forward_left_se2',
            'navigation': 'route_command_only', 'ego_fields': ['vx', 'vy', 'ax', 'ay'],
            'cohort': 'bounded_mini_preflight',
        },
        'train': {
            'optimizer': 'adamw', 'steps': 8000, 'microbatch': 2, 'gradient_accumulation': 8,
            'effective_batch': 16, 'lora_lr': .0001, 'new_lr': .0005, 'weight_decay': .01,
            'warmup_fraction': .03, 'grad_clip': 1., 'brightness': .1,
            'eval_every': 500, 'checkpoint_every': 500,
        },
        'loader': {'workers': 2, 'prefetch': 2, 'pin_memory': True, 'sampler': 'counter_v1'},
        'benchmark': {'enabled': True, 'warmup': 10, 'repeats': 100,
                      'modes': ['model_only', 'streaming_e2e', 'disk_e2e'], 'p95_budget_ms': 500},
        'execution': {'stage': 'formal', 'stop_after': None},
        'reproducibility': {
            'deterministic_algorithms': True, 'warn_only': False,
            'cublas_workspace_config': ':4096:8', 'cudnn_deterministic': True,
            'cudnn_benchmark': False, 'allow_tf32': False,
        },
    }
    if stage == 'preflight':
        config['output'] = 'outputs/navsim/preflight'
        config['execution'] = {'stage': stage, 'stop_after': 4}
        config['train'].update(eval_every=1, checkpoint_every=1)
    elif stage == 'deep_overfit':
        config['output'] = 'outputs/navsim/deep'
        config['execution'] = {'stage': stage, 'stop_after': 300}
        config['train'].update(steps=300, microbatch=1, gradient_accumulation=1,
                               effective_batch=1, brightness=0., eval_every=100, checkpoint_every=150)
        config['benchmark']['enabled'] = False
        config['data']['train_manifest'] = 'outputs/navsim/data/deep_train.jsonl'
    return config


def assign(config, path, value):
    parts = path.split('.')
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


@pytest.mark.parametrize('stage', ['formal', 'preflight', 'deep_overfit'])
def test_validate_returns_input_without_mutation_or_dataset_io(stage):
    config = canonical(stage)
    before = deepcopy(config)
    assert api().validate_config(config) is config
    assert config == before


@pytest.mark.parametrize('section', [None, 'model', 'processor', 'data', 'train', 'loader',
                                     'benchmark', 'execution', 'reproducibility'])
def test_all_required_fields_and_unknown_keys_are_checked(section):
    for field in canonical() if section is None else canonical()[section]:
        config = canonical()
        target = config if section is None else config[section]
        del target[field]
        with pytest.raises(ValueError, match=field):
            api().validate_config(config)
    config = canonical()
    (config if section is None else config[section])['unused_typo'] = 1
    with pytest.raises(ValueError, match='unused_typo'):
        api().validate_config(config)


@pytest.mark.parametrize('path,value', [
    ('schema_version', 1), ('train_seed', True), ('train_seed', -1), ('split_seed', 42.),
    ('bf16', 1), ('cpu_threads', False), ('cpu_threads', 0), ('device', ''), ('output', ''),
    ('loss', 'l1_xy_sum_point_batch_mean'), ('model.ego_dim', 3), ('model.ego_dim', 4.),
    ('model.num_commands', 7), ('model.planning_queries', 6), ('model.output_dim', 2),
    ('model.continuous_target', True), ('model.continuous_target', 0),
    ('model.freeze_visual_merger', False), ('model.lora_parameter_dtype', 'bfloat16'),
    ('model.lora_target_suffixes', ['q_proj']), ('model.token_order', ['ego', 'target']),
    ('model.lora_rank', True), ('model.lora_rank', 0), ('model.lora_dropout', float('nan')),
    ('processor.policy', 'resize'), ('processor.pad_multiple', 16), ('processor.pad_fill', [0, False, 0]),
    ('processor.history_times_s', [-1.5, -1., -.5, False]), ('processor.prompt_template', 'temporal_task_v1'),
    ('data.frequency_hz', 0), ('data.frequency_hz', -2), ('data.frequency_hz', True),
    ('data.frequency_hz', 10), ('data.camera', 'CAM_F1'), ('data.history_offsets', [-4, -2, -1, 0]),
    ('data.future_offsets', [5, 10, 15, 20, 25, 30]), ('data.navigation', 'target_point'),
    ('data.coordinate_convention', 'carla_world_to_forward_left'), ('data.ego_fields', ['speed']),
    ('data.cohort', 'all_data'), ('data.source_audit', ''), ('data.train_manifest', 'train.jsonl'),
    ('data.val_manifest', 'outputs/navsim/data/train.jsonl'), ('loader.workers', -1),
    ('loader.prefetch', 0), ('loader.pin_memory', 1), ('loader.sampler', 'random'),
    ('train.steps', 1), ('train.microbatch', 0), ('train.effective_batch', 15),
    ('train.optimizer', 'sgd'), ('train.lora_lr', 0), ('train.new_lr', float('inf')),
    ('train.weight_decay', -.1), ('train.warmup_fraction', 1.), ('train.grad_clip', False),
    ('train.brightness', -1), ('benchmark.enabled', 1), ('benchmark.warmup', 0),
    ('benchmark.repeats', 20), ('benchmark.p95_budget_ms', 0), ('benchmark.modes', ['disk_e2e']),
    ('reproducibility.deterministic_algorithms', False), ('reproducibility.warn_only', True),
    ('reproducibility.allow_tf32', 0), ('reproducibility.cudnn_benchmark', True),
])
def test_unsupported_and_badly_typed_values_fail_with_field_name(path, value):
    config = canonical()
    assign(config, path, value)
    with pytest.raises(ValueError, match=path.split('.')[-1]):
        api().validate_config(config)


@pytest.mark.parametrize('path', ['model.lora_dropout', 'train.lora_lr', 'train.new_lr',
                                  'train.weight_decay', 'train.warmup_fraction', 'train.grad_clip',
                                  'train.brightness', 'benchmark.p95_budget_ms'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, '0.1'])
def test_all_float_settings_reject_nonfinite_and_nonnumeric_values(path, value):
    config = canonical()
    assign(config, path, value)
    with pytest.raises(ValueError, match=path.split('.')[-1]):
        api().validate_config(config)


@pytest.mark.parametrize('section', ['model', 'processor', 'data', 'train', 'loader',
                                     'benchmark', 'execution', 'reproducibility'])
@pytest.mark.parametrize('value', [None, [], '', 1])
def test_nested_sections_must_be_mappings(section, value):
    config = canonical()
    config[section] = value
    with pytest.raises(ValueError, match=section):
        api().validate_config(config)


@pytest.mark.parametrize('stage,stop', [('formal', 4), ('preflight', None), ('preflight', 0),
    ('preflight', True), ('preflight', 21), ('preflight', 8000), ('deep_overfit', 301),
    ('deep_overfit', None), ('unknown', 4)])
def test_execution_cannot_bypass_stage_budget(stage, stop):
    config = canonical(stage)
    config['execution'] = {'stage': stage, 'stop_after': stop}
    with pytest.raises(ValueError, match='execution'):
        api().validate_config(config)


def test_preflight_allows_cap_up_to_twenty_without_redefining_formal_schedule():
    config = canonical('preflight')
    config['execution']['stop_after'] = 20
    assert api().validate_config(config) is config
    config['train']['steps'] = 20
    with pytest.raises(ValueError, match='steps'):
        api().validate_config(config)


@pytest.mark.parametrize('stage', ['formal', 'preflight', 'deep_overfit'])
def test_experiments_can_use_distinct_nonempty_output_paths(stage):
    config = canonical(stage)
    config['output'] = f'/tmp/navsim_runs/{stage}_rerun_2'
    assert api().validate_config(config) is config


@pytest.mark.parametrize('stage,name', [('formal', 'navsim'), ('preflight', 'navsim_preflight'),
                                      ('deep_overfit', 'navsim_deep')])
def test_shipped_configs_are_explicit_and_match_approved_contract(stage, name):
    path = ROOT / f'configs/{name}.yaml'
    assert path.is_file(), f'missing shipped {name} config'
    assert api().read_config(path) == canonical(stage)


def test_read_config_validates_and_rejects_duplicate_yaml_keys(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(canonical('preflight')))
    assert api().read_config(path) == canonical('preflight')
    path.write_text(path.read_text() + '\ntrain_seed: 99\n')
    with pytest.raises(ValueError, match='duplicate.*train_seed'):
        api().read_config(path)


@pytest.mark.parametrize('content', ['', '[]', 'true', 'schema_version: [', 'a: !unknown value'])
def test_read_config_rejects_empty_nonmapping_and_invalid_yaml(tmp_path, content):
    path = tmp_path / 'invalid.yaml'
    path.write_text(content)
    with pytest.raises(ValueError):
        api().read_config(path)


@pytest.mark.parametrize('path,value', [
    ('processor.history_times_s', [-1.5, 10**1000, -.5, 0.]),
    ('train.lora_lr', 10**1000),
])
def test_oversized_numeric_input_is_reported_as_a_validation_error(path, value):
    config = canonical()
    assign(config, path, value)
    with pytest.raises(ValueError, match=path.split('.')[-1]):
        api().validate_config(config)
