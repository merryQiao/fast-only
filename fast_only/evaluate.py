"""Open-loop metrics and visualizations; smoke metrics are explicitly train-only."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .common import NativeCollator, make_model, predict, to_device, trajectory_metrics, verify_checkpoint_inputs, write_json


@torch.no_grad()
def evaluate_dataset(model, dataset, collator, device):
    was_training = model.training
    model.eval()
    predictions = []; targets = []
    for i in range(len(dataset)):
        batch = to_device(collator([dataset[i]]), device)
        predictions.append(predict(model, batch).float().cpu())
        targets.append(batch['trajectory_gt'].float().cpu())
    model.train(was_training)
    if not predictions:
        raise ValueError('Cannot evaluate an empty dataset')
    pred, gt = torch.cat(predictions), torch.cat(targets)
    return trajectory_metrics(pred, gt), pred, gt


def save_predictions(output, dataset, pred, gt, *, cohort, max_plots=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'predictions.npz', prediction=pred.numpy(), trajectory_gt=gt.numpy())
    route_indices = defaultdict(list)
    from .diagnostics import select_visualization_indices
    plot_indices = set(select_visualization_indices(dataset, max_samples=max_plots))
    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]; meta = sample['metadata']
        metric = trajectory_metrics(pred[i:i+1], gt[i:i+1])
        rows.append({'route': meta['route'], 'frame': meta['frame'], 'cohort': cohort,
                     'prediction': pred[i].tolist(), 'trajectory_gt': gt[i].tolist(), **metric})
        route_indices[meta['route']].append(i)
        if i not in plot_indices:
            continue
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])
        for j, image in enumerate(sample['images']):
            ax = fig.add_subplot(gs[0, j]); ax.imshow(image); ax.axis('off')
            ax.set_title(f't{(j-3)*.1:+.1f}s')
        ax = fig.add_subplot(gs[1, :])
        # Horizontal axis points left; vertical axis points forward.
        ax.plot(np.r_[0, gt[i, :, 1]], np.r_[0, gt[i, :, 0]], 'o-', label='GT')
        ax.plot(np.r_[0, pred[i, :, 1]], np.r_[0, pred[i, :, 0]], 'x--', label='Prediction')
        ax.scatter([0], [0], c='black', marker='^', label='Ego')
        ax.invert_xaxis(); ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.3)
        ax.set_xlabel('Ego y: left + (m)'); ax.set_ylabel('Ego x: forward + (m)'); ax.legend()
        fig.suptitle(f'{cohort} | frame {meta["frame"]} | ADE={metric["ADE"]:.3f}m FDE={metric["FDE"]:.3f}m')
        fig.tight_layout(); fig.savefig(output / f'case_{i:03d}.png', dpi=120); plt.close(fig)
    write_json(output / 'predictions.json', rows)
    metric_keys = ['route', 'frame', 'cohort', 'ADE', 'FDE', 'loss_l1_xy'] + [f'L2@{i*.5:g}s' for i in range(1, 7)]
    with (output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    write_json(output / 'by_route.json', {route: trajectory_metrics(pred[indices], gt[indices])
                                         for route, indices in route_indices.items()})
    write_json(output / 'visualization_selection.json', {'method': 'scene_command_round_robin',
                'indices': sorted(plot_indices), 'distinct_cases': len(plot_indices),
                'available_cases': len(dataset)})
    return len(rows)


def main():
    from .data import FastDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cohort', required=True, choices=['train_smoke', 'val', 'test'])
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    verify_checkpoint_inputs(state)
    manifest = args.manifest or config['data']['train_manifest']
    dataset = FastDataset(manifest, config['data']['stats'])
    splits = {row['split'] for row in dataset.rows}
    expected = 'train' if args.cohort == 'train_smoke' else args.cohort
    if splits != {expected}:
        raise ValueError(f'Cohort label {args.cohort} contradicts manifest splits {splits}')
    if args.cohort == 'train_smoke' and not config.get('smoke', False):
        raise ValueError('train_smoke evaluation requires a smoke checkpoint')
    model = make_model(config); model.load_trainable_state(state['model'])
    collator = NativeCollator(config['model']['path'], **config['processor'])
    metrics, pred, gt = evaluate_dataset(model, dataset, collator, config['device'])
    from .diagnostics import constant_velocity_metrics, audit_constant_velocity_anchor
    metrics['cv_reference'] = constant_velocity_metrics(dataset)
    metrics['cv_numeric_anchor'] = audit_constant_velocity_anchor(dataset)
    metrics.update(cohort=args.cohort, optimizer_step=state['step'])
    save_predictions(args.output, dataset, pred, gt, cohort=args.cohort, max_plots=10)
    write_json(Path(args.output) / 'metrics.json', metrics)
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
