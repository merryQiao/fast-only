"""Read-only identity preflight; optional bounded native update on current GPU.

This is not a substitute for a real 16-rank target-environment train/resume.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time

import torch

from fast_cluster.models import load_source
from fast_only.common import write_json


def main():
    from fast_cluster.config import read_config
    from fast_cluster.data import PreparedData
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    start = time.monotonic()
    data = PreparedData(config['data']['bundle'], config['data']['sensor_root'],
                        allow_smoke=config['execution']['mode'] == 'smoke')
    from fast_cluster.source_contract import validate_source_data
    validate_source_data(config, data)
    bundle = load_source(config)
    result = dict(schema='navsim_cluster_preflight_v1', data=data.identity,
        train_windows=len(data.dataset('train')), val_windows=len(data.dataset('val')),
        validation_policy=data.identity.get('provenance', {}).get('validation_policy', 'inherited_smoke_only'),
        full_train_stats=data.stats, source_bundle_sha256=bundle['_file_sha256'],
        frozen_source_stats=bundle['stats'], source_assets=bundle['assets'],
        python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, gpu_available=torch.cuda.is_available(),
        visible_gpus=torch.cuda.device_count(), elapsed_s=time.monotonic()-start,
        distributed_training_executed=False,
        remaining='Target 16-rank native update, validation, checkpoint and resume required')
    if torch.cuda.is_available():
        result['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
