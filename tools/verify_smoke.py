"""Independently reload a smoke checkpoint, reproduce predictions and benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fast_only.benchmark import measure
from fast_only.common import NativeCollator, make_model, verify_checkpoint_inputs, write_json
from fast_only.data import FastDataset
from fast_only.evaluate import evaluate_dataset, save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/fast_only/smoke')
    parser.add_argument('--max-loss-ratio', type=float, default=.25)
    parser.add_argument('--benchmark-repeats', type=int, default=20)
    args = parser.parse_args()
    run = Path(args.run)
    state = torch.load(run/'last.pt', map_location='cpu', weights_only=False)
    config = state['config']
    if not config['smoke']:
        raise ValueError('This verifier is for train-only smoke experiments')
    verify_checkpoint_inputs(state)
    torch.set_num_threads(config.get('cpu_threads', 4))
    if not all(torch.isfinite(value).all().item() for value in state['model'].values()):
        raise AssertionError('Nonfinite checkpoint parameters')
    model = make_model(config); model.load_trainable_state(state['model'])
    dataset = FastDataset(config['data']['train_manifest'], config['data']['stats'])
    collator = NativeCollator(config['model']['path'], **config['processor'])
    metrics, prediction, gt = evaluate_dataset(model, dataset, collator, config['device'])
    reference = np.load(run/'final/predictions.npz')
    delta = float(np.max(np.abs(prediction.numpy()-reference['prediction'])))
    if delta > 1e-5:
        raise AssertionError(f'Reloaded predictions differ from saved final predictions by {delta}')
    initial = json.loads((run/'initial_train_metrics.json').read_text())
    ratio = metrics['loss_l1_xy']/initial['loss_l1_xy']
    if ratio > args.max_loss_ratio:
        raise AssertionError(f'Overfit loss ratio {ratio:.4f} exceeds {args.max_loss_ratio}')
    metrics.update(cohort='train_smoke', optimizer_step=state['step'])
    write_json(run/'reload/metrics.json', metrics)
    save_predictions(run/'reload', dataset, prediction, gt, cohort='train_smoke')
    benchmark = measure(model, dataset, collator, config['device'], warmup=5, repeats=args.benchmark_repeats)
    write_json(run/'reload/benchmark.json', benchmark)
    p0 = json.loads((run/'p0_forward_backward.json').read_text())
    logs = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
    steps = [row['step'] for row in logs]
    if steps != list(range(1, state['step']+1)):
        raise AssertionError('Training log steps do not match checkpoint')
    report = {'passed': True, 'cohort': 'train_smoke', 'windows': len(dataset),
              'optimizer_steps': state['step'], 'initial': initial, 'reloaded': metrics,
              'loss_ratio_final_over_initial': ratio, 'loss_reduction_percent': (1-ratio)*100,
              'reload_max_abs_prediction_difference_m': delta,
              'finite_checkpoint': True, 'gradient_checks': p0['gradients'],
              'layout': p0['layout'], 'end_to_end_ms': benchmark['end_to_end_ms'],
              'limitations': ['Single route and train-only cohort; no generalization evidence.',
                             'No per-frame timestamps: explicit official10Hz frame-index assumption.',
                             'Unverified near-node generation: navigation target fixed to zero.',
                             'Small smoke image resolution; latency does not establish full-resolution10Hz.']}
    write_json(run/'verification.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, [row['train_loss_l1_xy'] for row in logs], alpha=.4, label='Sample training L1')
    evaluated = [(0, initial['loss_l1_xy'])] + [(row['step'], row['train_smoke_metrics']['loss_l1_xy'])
                                                 for row in logs if 'train_smoke_metrics' in row]
    ax.plot(*zip(*evaluated), 'o-', label=f'All {len(dataset)} training windows, eval mode')
    ax.set_xlabel('Optimizer step'); ax.set_ylabel('Mean |dx|+|dy| (m)')
    ax.set_yscale('log'); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(run/'training_curve.png', dpi=140); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
