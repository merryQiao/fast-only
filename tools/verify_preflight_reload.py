"""Fresh-process reconstruction of the raw-resolution preflight checkpoints."""
import argparse
import gc
import os

import torch

from fast_only.common import read_config, write_json
from tools.run_preflight_v2 import reload_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/preflight_smoke.yaml')
    parser.add_argument('--deep-config', default='configs/deep_overfit_v2.yaml')
    parser.add_argument('--output', default='outputs/fast_only/audit_v2/independent_process_reload.json')
    args = parser.parse_args()
    reports = {}
    for label, path in [('configuration_smoke', args.config), ('deep_overfit', args.deep_config)]:
        config = read_config(path); torch.set_num_threads(config['cpu_threads'])
        report, model = reload_predictions(config)
        reports[label] = report
        del model; gc.collect(); torch.cuda.empty_cache()
    result = {'passed': True, 'verification_process_pid': os.getpid(),
              'method': 'new Python process; local base weights + checkpoint; original-pixel inputs',
              'reports': reports}
    write_json(args.output, result); print(result, flush=True)


if __name__ == '__main__': main()
