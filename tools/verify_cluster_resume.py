"""Compare actual native training runs and independently reload predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fast_only.common import fingerprint, write_json


def assert_same(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(f'Resume tensor differs: {path}')
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or not np.array_equal(a, b):
            raise AssertionError(f'Resume ndarray differs: {path}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(f'Resume keys differ: {path}')
        for key in a:
            assert_same(a[key], b[key], path+'.'+str(key))
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise AssertionError(f'Resume sequence differs: {path}')
        for index, (left, right) in enumerate(zip(a, b)):
            assert_same(left, right, path+'.'+str(index))
    elif a != b:
        raise AssertionError(f'Resume scalar differs: {path}')


def verify(config_path, uninterrupted, resumed, output, device='cuda:0'):
    from fast_cluster.config import read_config
    from fast_cluster.data import PreparedData
    from fast_cluster.models import build_backend
    from fast_cluster.train import _apply_reproducibility
    config = read_config(config_path)
    _apply_reproducibility(config)
    reference = torch.load(uninterrupted, map_location='cpu', weights_only=False)
    restored = torch.load(resumed, map_location='cpu', weights_only=False)
    fields = ('schema', 'identity', 'world_size', 'progress', 'model', 'fast_buffers',
              'optimizer', 'scheduler', 'rng_by_rank')
    if 'source_tasks' in reference or 'source_tasks' in restored:
        fields += ('source_tasks',)
    for field in fields:
        assert_same(reference[field], restored[field], field)
    identity = reference['identity']
    data = PreparedData(config['data']['bundle'], config['data']['sensor_root'],
                        allow_smoke=config['execution']['mode'] == 'smoke')
    backend = build_backend(config, data.stats, identity['arm'], identity['seed'], device)
    batch = backend.prepare([data.dataset('val')[0]])
    backend.eval()
    predictions = []
    with torch.no_grad():
        for saved in (reference, restored):
            backend.restore_fast_state(saved['model'])
            if hasattr(backend,'restore_source_state'):
                backend.restore_source_state(saved.get('source_tasks'))
            predictions.append(backend(batch).detach().cpu().clone())
    assert_same(*predictions, 'independent_reloaded_prediction')
    backend.check_frozen()
    receipt = dict(schema='navsim_cluster_native_resume_v1', fields_bitwise_equal=list(fields),
        independent_reload_prediction_bitwise_equal=True,
        prediction_max_abs_difference=float((predictions[0]-predictions[1]).abs().max()),
        source='real disk checkpoints from separate training processes',
        world_size=reference['world_size'], device=str(device),
        uninterrupted_sha256=fingerprint(uninterrupted), resumed_sha256=fingerprint(resumed),
        update=reference['progress']['update'], sample_token=data.dataset('val').row(0)['token'],
        prediction=predictions[0].tolist())
    write_json(output, receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--uninterrupted', required=True)
    parser.add_argument('--resumed', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    verify(args.config, args.uninterrupted, args.resumed, args.output, args.device)


if __name__ == '__main__':
    main()
