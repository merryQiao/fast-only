"""Explicit NAVSIM V1 launch and content-stable resume configuration."""
import copy
import math
import os
import re
from pathlib import Path

import yaml


def read_config(path):
    config = yaml.safe_load(Path(path).read_text())
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            value = os.path.expanduser(os.path.expandvars(value))
            if re.search(r'\$\{[^}]+\}|\$[A-Za-z_][A-Za-z_0-9]*', value):
                raise ValueError(f'Configuration contains unresolved environment placeholder: {value}')
        return value
    config = expand(config)
    validate_config(config)
    return config


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema') != 'navsim_cluster_v1':
        raise ValueError('Expected navsim_cluster_v1 configuration schema')
    for section in ('execution', 'assets', 'data', 'model', 'train', 'evaluation', 'reproducibility'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Missing configuration section: {section}')
    ec, tc, mc = config['execution'], config['train'], config['model']
    if ec.get('mode') not in ('formal', 'smoke'):
        raise ValueError('execution.mode must be formal or smoke')
    for key in ('expected_world_size', 'expected_local_world_size', 'timeout_s'):
        _positive_int(ec.get(key), f'execution.{key}')
    expected = (16, 8) if ec['mode'] == 'formal' else (1, 1)
    if (ec['expected_world_size'], ec['expected_local_world_size']) != expected:
        raise ValueError(f'{ec["mode"]} requires world_size {expected[0]} and local_world_size {expected[1]}')
    if ec.get('backend') != 'nccl':
        raise ValueError('Native model training requires NCCL; Gloo is reserved for the separate CPU algorithm smoke')
    if tc.get('arms') != ['fast_only', 'slow_fast'] or tc.get('seeds') != [42, 43]:
        raise ValueError('Training protocol requires only fast_only/slow_fast and fixed seeds 42,43')
    for key in ('epochs', 'micro_batch', 'gradient_accumulation', 'evaluate_every_epochs',
                'checkpoint_every_updates', 'keep_full_checkpoints'):
        _positive_int(tc.get(key), f'train.{key}')
    if ec['mode'] == 'formal' and tc['epochs'] != 8:
        raise ValueError('Formal training requires eight full epochs')
    if tc['keep_full_checkpoints'] != 2 or tc['evaluate_every_epochs'] != 1:
        raise ValueError('Protocol requires epoch validation and retention of two full checkpoints')
    for key in ('language_lr', 'new_lr', 'gradient_clip'):
        value = tc.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'train.{key} must be finite and positive')
    if not 0 <= tc.get('weight_decay', -1) or not math.isfinite(tc['weight_decay']):
        raise ValueError('train.weight_decay must be finite and nonnegative')
    for key in ('warmup_fraction', 'minimum_lr_ratio'):
        if not isinstance(tc.get(key), (int, float)) or not 0 <= tc[key] <= 1:
            raise ValueError(f'train.{key} must be in [0,1]')
    if tc.get('source_feedback') != 'frozen_source_teacher':
        raise ValueError('Fresh source training requires frozen_source_teacher feedback')
    if tc.get('loss', 'official_unweighted_pose_l1') != 'official_unweighted_pose_l1':
        raise ValueError('NAVSIM loss is unchanged official unweighted pose L1')
    if mc.get('init') != 'pretrained' or mc.get('gradient_checkpointing') is not False:
        raise ValueError('Use pretrained Fast initialization with gradient checkpointing disabled')
    if mc.get('lora_parameter_dtype') != 'float32':
        raise ValueError('Full Fast training requires FP32 master parameters')
    fixed_model = dict(lora_rank=16, lora_alpha=32, lora_dropout=0., position_offset=4096)
    if any(mc.get(key) != value for key, value in fixed_model.items()):
        raise ValueError('Keep the existing rank16/alpha32/dropout0/offset4096 six-layer Direct-KV interface')
    if config.get('processor') != dict(policy='original_pad', pad_multiple=32, brightness=0.,
                                        prompt_template='navsim_temporal_v1'):
        raise ValueError('Use the unchanged original-image pad32 temporal NAVSIM processor')
    if config['data'].get('normalization') != 'full_train_fast__preserved_source_slow':
        raise ValueError('Fast requires full TRAIN statistics and frozen source preserves its statistics')
    validation_policy = ('exclude_upstream_training_groups' if ec['mode'] == 'formal'
                         else 'inherited_smoke_only')
    configured_policy = config['data'].get('validation_policy',
                                            'inherited_smoke_only' if ec['mode'] == 'smoke' else None)
    if configured_policy != validation_policy:
        raise ValueError(f'data.validation_policy must be {validation_policy} for {ec["mode"]}')
    for key in ('fast_model', 'slow_model', 'source_bundle', 'source_checkpoint_sha256'):
        if not isinstance(config['assets'].get(key), str) or not config['assets'][key]:
            raise ValueError(f'Missing assets.{key}')
    if not re.fullmatch('[0-9a-f]{64}', config['assets']['source_checkpoint_sha256']):
        raise ValueError('Expected pinned source checkpoint SHA256')
    for key in ('bundle', 'sensor_root'):
        if not isinstance(config['data'].get(key), str) or not config['data'][key]:
            raise ValueError(f'Missing data.{key}')
    workers = config['data'].get('workers')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError('data.workers must be a nonnegative integer')
    _positive_int(config['data'].get('prefetch_factor'), 'data.prefetch_factor')
    if config['evaluation'] != dict(primary='val_ADE', final_joint_modes=['correct', 'fixed'],
                                    fixed_source='lexicographically_first_train_token'):
        raise ValueError('Evaluation must select VAL ADE and use a fixed TRAIN source at the joint endpoint')
    return config


def validate_topology(config, *, world_size, local_world_size):
    ec = config['execution']
    actual = (world_size, local_world_size)
    expected = (ec['expected_world_size'], ec['expected_local_world_size'])
    if actual != expected:
        raise ValueError(f'Launch topology {actual} differs from configured topology {expected}; no substitution is permitted')


def semantic_config(config):
    """Resume semantics exclude relocatable roots and I/O scheduling choices.

    Actual data/statistics, asset bytes and source bundle identities are supplied
    separately by PreparedData and PlanningBackend. Paths never stand in for a hash.
    """
    result = copy.deepcopy(config)
    for key in ('fast_model', 'slow_model', 'source_bundle'):
        result['assets'].pop(key, None)
    for key in ('bundle', 'sensor_root', 'workers', 'prefetch_factor'):
        result['data'].pop(key, None)
    result.pop('output', None)
    result['train']['loss'] = 'official_unweighted_pose_l1'
    return result
