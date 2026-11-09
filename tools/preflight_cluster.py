"""Read-only asset/data identity preflight without constructing a native model.

This is not a substitute for a real 16-rank target-environment train/resume.
"""
import argparse
import json
from pathlib import Path
import platform
import time

import torch

from fast_cluster.config import baseline_only, semantic_config, split_history, validate_config
from fast_only.common import fingerprint, write_json


def _asset_manifest(root):
    """Match the native asset inventory without importing model/task constructors."""
    root = Path(root)
    paths = sorted(path for path in root.iterdir() if path.is_file()
                   and path.suffix in ('.json', '.safetensors', '.jinja', '.txt', '.model'))
    if not any(path.suffix == '.safetensors' for path in paths):
        raise ValueError(f'No local base safetensors under {root}')
    return {path.name: fingerprint(path) for path in paths}


def run_preflight(config):
    from fast_cluster.data import PreparedData
    validate_config(config)
    start = time.monotonic()
    data = PreparedData(config['data']['bundle'], config['data']['sensor_root'],
                        allow_smoke=config['execution']['mode'] == 'smoke')
    from fast_cluster.source_contract import validate_source_data
    validate_source_data(config, data)
    split = split_history(config)
    bundle = None
    if split:
        assets = {role: _asset_manifest(config['assets'][role + '_model']) for role in ('fast', 'slow')}
    elif baseline_only(config):
        assets = {'fast': _asset_manifest(config['assets']['fast_model'])}
    else:
        from fast_cluster.models import load_source
        bundle = load_source(config)
        assets = bundle['assets']
    train_windows, val_windows = len(data.dataset('train')), len(data.dataset('val'))
    if split and config['execution']['mode'] == 'smoke' and (train_windows, val_windows) != (8, 4):
        raise ValueError('Split-history Smoke requires exactly 8 TRAIN and 4 VAL windows')
    if (data.stats.get('source_split') != 'train' or data.stats.get('count') != train_windows):
        raise ValueError('Preflight requires statistics computed from this complete TRAIN manifest')
    initialization = ('pretrained_split_history' if split else 'original_pretrained_fast_only'
                      if bundle is None else 'legacy_frozen_source_bundle')
    result = dict(schema='navsim_split_history_preflight_v2' if split else 'navsim_cluster_preflight_v1',
        data=data.identity, semantic_config=semantic_config(config),
        train_windows=train_windows, val_windows=val_windows,
        validation_policy=data.identity.get('provenance', {}).get('validation_policy', 'inherited_smoke_only'),
        full_train_stats=data.stats, source_bundle_sha256=None if bundle is None else bundle['_file_sha256'],
        frozen_source_stats=None if bundle is None else bundle['stats'], source_assets=assets,
        initialization=initialization,
        python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, gpu_available=torch.cuda.is_available(),
        visible_gpus=torch.cuda.device_count(), elapsed_s=time.monotonic()-start,
        distributed_training_executed=False,
        remaining='Target 16-rank native update, validation, checkpoint and resume required')
    if split:
        result.update(source_initialization=dict(config['source']),
                      model_statistics=dict(fast=data.stats, slow=data.stats),
                      declared_source_scope=dict(frozen=['backbone_including_lora', 'head'],
                                                 trainable=['ego_mlp', 'command_embedding', 'feedback_encoder',
                                                            'feedback_type', 'no_feedback', 'queries']),
                      model_initialized=False)
    if torch.cuda.is_available():
        result['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def main(argv=None):
    from fast_cluster.config import read_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = run_preflight(read_config(args.config))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
