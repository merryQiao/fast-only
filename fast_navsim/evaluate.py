"""Open-loop NAVSIM pose metrics and descriptive validation interventions.

These bounded-cohort measurements are not an official NAVSIM/PDM score.
Image and command interventions measure output sensitivity, not semantic skill.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np
import torch

from fast_only.common import predict, to_device, write_json
from .model import trajectory_metrics

HISTORY_TIMES = [-1.5, -1., -.5, 0.]
FUTURE_TIMES = [.5 * i for i in range(1, 9)]


def _limit(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _round_robin(buckets, limit):
    selected = []
    cursor = 0
    while len(selected) < limit:
        for key in sorted(buckets):
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                if len(selected) == limit:
                    break
        cursor += 1
    return selected


def select_visualization_indices(dataset, max_samples=10):
    """Deterministic log/scene/command round robin, without duplicate padding."""
    _limit(max_samples, 'max_samples')
    buckets = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        buckets[(str(row['route']), str(row['scene']), int(row['command']))].append(index)
    return _round_robin(buckets, min(max_samples, len(dataset)))


def _forward(model, sample, collator, device):
    batch = to_device(collator([sample], seed=0), device)
    output = predict(model, batch).detach().float().cpu()
    if output.shape != (1, 8, 3) or not torch.isfinite(output).all():
        raise ValueError('Evaluation requires finite [1,8,3] NAVSIM predictions')
    return output, batch['trajectory_gt'].detach().float().cpu()


@torch.no_grad()
def evaluate_dataset(model, dataset, collator, device):
    """Return official pose-loss/open-loop metrics, CPU predictions and targets."""
    if not len(dataset):
        raise ValueError('Cannot evaluate an empty dataset')
    predictions, targets = [], []
    was_training = model.training
    model.eval()
    try:
        for index in range(len(dataset)):
            prediction, target = _forward(model, dataset[index], collator, device)
            predictions.append(prediction)
            targets.append(target)
    finally:
        model.train(was_training)
    prediction, target = torch.cat(predictions), torch.cat(targets)
    return trajectory_metrics(prediction, target), prediction, target


def constant_velocity_metrics(dataset):
    """Reference x=vx*t,y=vy*t,heading=0 using only raw current velocity.

    Current-frame velocity remains constant over eight half-second intervals.
    Relative orientation remains zero. Acceleration and future metadata are not
    used to generate predictions; future poses are used only as metric targets.
    """
    if not len(dataset):
        raise ValueError('Cannot evaluate an empty dataset')
    velocities = torch.tensor([row['ego'][:2] for row in dataset.rows], dtype=torch.float32)
    if velocities.shape != (len(dataset), 2) or not torch.isfinite(velocities).all():
        raise ValueError('Constant velocity needs finite raw current vx/vy in m/s')
    times = torch.tensor(FUTURE_TIMES)
    prediction = torch.zeros(len(dataset), 8, 3)
    prediction[..., :2] = velocities[:, None, :] * times[None, :, None]
    target = torch.tensor([row['trajectory_gt'] for row in dataset.rows], dtype=torch.float32)
    result = trajectory_metrics(prediction, target)
    result.update(method='current_vx_vy_constant_velocity_reference',
                  heading_assumption='constant orientation: relative heading zero at every horizon',
                  limitation='Descriptive open-loop reference; not an official NAVSIM/PDM score.')
    return result


def _plot_case(path, sample, prediction, target, cohort, metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if len(sample['images']) != 4:
        raise ValueError('Plotting requires four chronological images')
    fig = plt.figure(figsize=(15, 8))
    try:
        grid = fig.add_gridspec(2, 4, height_ratios=[1, 2])
        for index, (image, time) in enumerate(zip(sample['images'], HISTORY_TIMES)):
            ax = fig.add_subplot(grid[0, index])
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f't{time:+.1f}s')
        xy = fig.add_subplot(grid[1, :2])
        for poses, style, label in ((target, 'o-', 'GT'), (prediction, 'x--', 'Prediction')):
            xy.plot(np.r_[0, poses[:, 1]], np.r_[0, poses[:, 0]], style, label=label)
        xy.scatter([0], [0], c='black', marker='^', label='Current rear axle')
        xy.invert_xaxis()
        xy.set_aspect('equal', adjustable='datalim')
        xy.set_xlabel('Ego y: left + (m)')
        xy.set_ylabel('Ego x: forward + (m)')
        xy.grid(alpha=.3)
        xy.legend()
        heading = fig.add_subplot(grid[1, 2:])
        heading.plot(FUTURE_TIMES, target[:, 2], 'o-', label='GT')
        heading.plot(FUTURE_TIMES, prediction[:, 2], 'x--', label='Prediction')
        heading.set_xlabel('Future time (s)')
        heading.set_ylabel('Relative heading (rad)')
        heading.grid(alpha=.3)
        heading.legend()
        meta = sample['metadata']
        fig.suptitle(f'{cohort} | {meta["route"]} | scene {meta["scene"]} | '
                     f'frame {meta["frame"]} | command {meta["command"]} | '
                     f'ADE={metric["ADE"]:.3f} m FDE={metric["FDE"]:.3f} m')
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def save_predictions(output, dataset, pred, gt, cohort, max_plots=10):
    """Save pose arrays, per-case metrics and stratified XY/heading plots."""
    _limit(max_plots, 'max_plots')
    pred = torch.as_tensor(pred).detach().float().cpu()
    gt = torch.as_tensor(gt).detach().float().cpu()
    trajectory_metrics(pred, gt)
    if len(pred) != len(dataset) or len(dataset.rows) != len(dataset):
        raise ValueError('Prediction count must match dataset rows')
    if not isinstance(cohort, str) or not cohort.strip():
        raise ValueError('cohort must be a nonempty label')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'predictions.npz', prediction=pred.numpy(), trajectory_gt=gt.numpy(),
             future_times_s=np.array(FUTURE_TIMES), history_times_s=np.array(HISTORY_TIMES),
             token=np.array([str(row['token']) for row in dataset.rows]),
             command=np.array([row['command'] for row in dataset.rows], dtype=np.int64))
    selected = select_visualization_indices(dataset, max_samples=max_plots)
    plot_indices = set(selected)
    rows, by_route = [], defaultdict(list)
    for index, meta in enumerate(dataset.rows):
        metric = trajectory_metrics(pred[index:index+1], gt[index:index+1])
        rows.append({key: meta[key] for key in ('route', 'scene', 'frame', 'token', 'command')})
        rows[-1].update(cohort=cohort, prediction=pred[index].tolist(),
                        trajectory_gt=gt[index].tolist(), **metric)
        by_route[str(meta['route'])].append(index)
        if index in plot_indices:
            _plot_case(output / f'case_{index:03d}.png', dataset[index], pred[index].numpy(),
                       gt[index].numpy(), cohort, metric)
    write_json(output / 'predictions.json', rows)
    fields = ['route', 'scene', 'frame', 'token', 'command', 'cohort', 'samples',
              'ADE', 'FDE', 'loss_l1_pose', 'loss_l1_xy', 'heading_mae_rad',
              'heading_final_mae_rad'] + [f'L2@{time:g}s' for time in FUTURE_TIMES]
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'by_route.json', {route: trajectory_metrics(pred[indices], gt[indices])
                                         for route, indices in sorted(by_route.items())})
    write_json(output / 'visualization_selection.json', {
        'method': 'log_scene_command_round_robin', 'indices': selected,
        'distinct_cases': len(selected), 'available_cases': len(dataset),
        'history_times_s': HISTORY_TIMES, 'future_times_s': FUTURE_TIMES,
        'xy_units': 'm', 'heading_units': 'rad',
    })
    return len(rows)


def _cycle_donors(metadata):
    groups = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row['route'])].append(index)
    grouped = [index for route in sorted(groups, key=lambda key: (-len(groups[key]), key))
               for index in groups[route]]
    count = len(metadata)
    order = [None] * count
    slots = list(range(0, count, 2)) + list(range(1, count, 2))
    for slot, index in zip(slots, grouped):
        order[slot] = index
    return {index: order[(position+1) % count] for position, index in enumerate(order)}


def _changes(original, changed):
    delta = changed - original
    return {'mean_point_change_m': torch.linalg.vector_norm(delta[..., :2], dim=-1).mean().item(),
            'mean_heading_change_rad': torch.atan2(delta[..., 2].sin(), delta[..., 2].cos()).abs().mean().item()}


def _summarize(pairs, metadata, method):
    log_counts = dict(sorted(Counter(str(row['route']) for row in metadata).items()))
    report = dict(status='measured' if pairs else 'insufficient', samples=len(pairs),
                  selected_samples=len(metadata), logs=len(log_counts), log_counts=log_counts,
                  method=method, pairs=pairs, mean_point_change_m=None, mean_heading_change_rad=None,
                  ground_truth_used_for_prediction=False)
    if pairs:
        for key in ('mean_point_change_m', 'mean_heading_change_rad'):
            report[key] = sum(pair[key] for pair in pairs)/len(pairs)
    return report


@torch.no_grad()
def channel_sensitivity(model, dataset, collator, device, max_samples=200):
    """VAL-only all-four-image cycle and synthetic (command+1)%4 interventions.

    Keep ego/command fixed during image swaps; keep images/ego fixed during
    command intervention. Fixed augmentation seeds isolate the changed channel.
    Interventions measure output change, with no semantic-performance claim.
    """
    _limit(max_samples, 'max_samples', minimum=1)
    if not len(dataset) or any(row.get('split') != 'val' for row in dataset.rows):
        raise ValueError('Channel sensitivity requires a nonempty val-only dataset')
    buckets = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        buckets[str(row['route'])].append(index)
    indices = _round_robin(buckets, min(max_samples, len(dataset)))
    metadata = [dataset.rows[index] for index in indices]
    donors = _cycle_donors(metadata) if len(indices) > 1 else {}
    image_pairs, command_pairs = [], []
    was_training = model.training
    model.eval()
    try:
        for position, index in enumerate(indices):
            sample = dataset[index]
            original, _ = _forward(model, sample, collator, device)
            meta = metadata[position]
            identity = {'index': index, 'log': str(meta['route']), 'scene': meta['scene'],
                        'frame': meta['frame'], 'token': meta['token']}
            if donors:
                donor_position = donors[position]
                donor_index = indices[donor_position]
                swapped = dict(sample, images=dataset[donor_index]['images'])
                changed, _ = _forward(model, swapped, collator, device)
                donor_log = str(metadata[donor_position]['route'])
                image_pairs.append(dict(identity, donor_index=donor_index, donor_log=donor_log,
                                        cross_log=identity['log'] != donor_log, **_changes(original, changed)))
            command = int(sample['command'])
            if command not in range(4):
                raise ValueError('Command intervention requires four-way commands 0..3')
            intervened = dict(sample, command=(sample['command'] + 1) % 4)
            changed, _ = _forward(model, intervened, collator, device)
            command_pairs.append(dict(identity, original_command=command,
                                      intervened_command=(command + 1) % 4, **_changes(original, changed)))
    finally:
        model.train(was_training)
    image = _summarize(image_pairs, metadata, 'fixed_cycle_derangement_all_four_images_only')
    image.update(cross_log_pairs=sum(pair['cross_log'] for pair in image_pairs),
                 same_log_pairs=sum(not pair['cross_log'] for pair in image_pairs))
    command = _summarize(command_pairs, metadata, 'command_plus_one_modulo_four')
    return {'cohort': 'val', 'image_swap': image, 'command_intervention': command,
            'limitation': 'Descriptive channel sensitivity only; no semantic performance or official NAVSIM/PDM score claim.'}
