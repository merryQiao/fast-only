"""Independent motion, visual sensitivity and representative-case diagnostics.

Future motion audit evidence is used only for selecting a numeric audit subset.
It is never passed to the planner or used to generate predictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import torch

from .common import predict, to_device, trajectory_metrics


def _metadata(dataset, index):
    return dataset.rows[index] if hasattr(dataset, 'rows') else dataset[index]['metadata']


def constant_velocity_predictions(dataset):
    """Predict x=v*t, y=0 at .5..3 seconds using raw sensor speed in m/s."""
    if not len(dataset):
        raise ValueError('Cannot evaluate an empty dataset')
    speeds = torch.tensor([_metadata(dataset, i)['ego'][0] for i in range(len(dataset))],
                          dtype=torch.float32)
    if not torch.isfinite(speeds).all():
        raise ValueError('Constant velocity needs finite raw speed in m/s')
    times = torch.arange(1, 7, dtype=torch.float32) * .5
    x = speeds[:, None] * times
    return torch.stack((x, torch.zeros_like(x)), dim=-1)


def evaluate_constant_velocity(dataset):
    """Return the ordinary ADE/FDE/six-horizon L2 contract, prediction and GT."""
    prediction = constant_velocity_predictions(dataset)
    targets = []
    for index in range(len(dataset)):
        metadata = _metadata(dataset, index)
        target = metadata.get('trajectory_gt')
        if target is None:
            target = dataset[index]['trajectory_gt']
        targets.append(torch.as_tensor(target, dtype=torch.float32))
    target = torch.stack(targets)
    return trajectory_metrics(prediction, target), prediction, target


def constant_velocity_metrics(dataset):
    return evaluate_constant_velocity(dataset)[0]


def audit_constant_velocity_anchor(dataset, *, min_speed_mps=1., max_delta_speed_mps=.1,
                                   max_delta_yaw_rad=.01, max_point_error_m=.02,
                                   min_samples=1):
    """Audit geometry on independently established sustained straight/steady motion.

    Membership uses raw sensor evidence across current..current+30, never GT
    displacement or the eventual CV error. Missing evidence is insufficient.
    The strict centimetre gate applies only to this subset; CV remains a
    descriptive reference for arbitrary turning/accelerating cohorts.
    """
    if min_samples < 1:
        raise ValueError('min_samples must be positive')
    criteria = dict(min_speed_mps_exclusive=min_speed_mps, max_delta_speed_mps=max_delta_speed_mps,
                    max_delta_yaw_rad=max_delta_yaw_rad, max_point_error_m=max_point_error_m,
                    required_motion_frames=31, horizons_s=[i*.5 for i in range(1, 7)])
    selected = []; excluded = Counter()
    for index in range(len(dataset)):
        evidence = _metadata(dataset, index).get('audit_motion')
        if not isinstance(evidence, dict) or evidence.get('num_frames') != 31:
            excluded['missing_motion_evidence'] += 1
            continue
        try:
            speed, delta_v, delta_yaw = [float(evidence[key]) for key in
                                        ('speed_min_mps', 'max_delta_speed_mps', 'max_delta_yaw_rad')]
            if not all(math.isfinite(value) for value in (speed, delta_v, delta_yaw)) or min(delta_v, delta_yaw) < 0:
                raise ValueError('invalid evidence')
        except (KeyError, TypeError, ValueError):
            excluded['invalid_motion_evidence'] += 1
            continue
        if speed <= min_speed_mps or delta_v > max_delta_speed_mps or delta_yaw > max_delta_yaw_rad:
            excluded['not_sustained_straight_steady'] += 1
            continue
        selected.append(index)
    report = dict(status='insufficient', selected_samples=len(selected), total_samples=len(dataset),
                  selected_indices=selected, criteria=criteria, excluded=dict(excluded),
                  max_point_error_m=None, metrics=None,
                  selection_source='sustained_raw_sensor_speed_compass_audit_only',
                  limitation='CV is a geometry anchor only for independently verified straight steady motion.')
    if len(selected) < min_samples:
        return report
    _, prediction, target = evaluate_constant_velocity(dataset)
    prediction, target = prediction[selected], target[selected]
    error = torch.linalg.vector_norm(prediction-target, dim=-1)
    maximum = error.max().item()
    report.update(status='passed' if maximum <= max_point_error_m else 'failed',
                  max_point_error_m=maximum, metrics=trajectory_metrics(prediction, target))
    return report


def select_visualization_indices(dataset, max_samples=10):
    """Round-robin deterministic scene/command buckets without duplicate cases."""
    if max_samples < 0:
        raise ValueError('max_samples must be nonnegative')
    buckets = defaultdict(list)
    for index in range(len(dataset)):
        metadata = _metadata(dataset, index)
        buckets[(str(metadata.get('scene', 'unknown')), int(metadata.get('command', 0)))].append(index)
    selected = []; cursor = 0
    while len(selected) < min(max_samples, len(dataset)):
        for key in sorted(buckets):
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                if len(selected) == min(max_samples, len(dataset)):
                    break
        cursor += 1
    return selected


def _image_swap_cycle(metadata):
    """One fixed cycle, arranging route groups to avoid same-route neighbours."""
    groups = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row.get('route', 'unknown'))].append(index)
    grouped = [index for route in sorted(groups, key=lambda key: (-len(groups[key]), key))
               for index in groups[route]]
    n = len(grouped)
    order = [None] * n
    # Standard alternating placement maximizes separation of the largest group.
    slots = list(range(0, n, 2)) + list(range(1, n, 2))
    for slot, index in zip(slots, grouped, strict=True):
        order[slot] = index
    donor = {}
    for position, index in enumerate(order):
        donor[index] = order[(position+1) % n]
    return donor


@torch.no_grad()
def evaluate_image_swap(model, dataset, collator, device, max_samples=200):
    """Swap all four images through a fixed cycle, preserving current features.

    Reports output change in metres (mean point Euclidean distance), with actual
    split and route labels. This measures sensitivity, not prediction accuracy.
    """
    if max_samples < 1:
        raise ValueError('max_samples must be positive')
    # Cover available routes before taking more windows from any one route.
    # A long first route must not hide cross-route donors behind the sample cap.
    route_indices = defaultdict(list)
    for index in range(len(dataset)):
        route_indices[str(_metadata(dataset, index).get('route', 'unknown'))].append(index)
    indices = []; cursor = 0
    limit = min(max_samples, len(dataset))
    while len(indices) < limit:
        for route in sorted(route_indices):
            if cursor < len(route_indices[route]):
                indices.append(route_indices[route][cursor])
                if len(indices) == limit:
                    break
        cursor += 1
    metadata = [_metadata(dataset, index) for index in indices]
    splits = sorted({str(row.get('split', 'unknown')) for row in metadata})
    routes = {str(row.get('route', 'unknown')) for row in metadata}
    report = dict(status='insufficient', samples=len(indices),
                  cohort=splits[0] if len(splits) == 1 else ('mixed' if splits else 'empty'),
                  splits=splits, different_route_available=len(routes) > 1,
                  cross_route_pairs=0, mean_point_change_m=None, per_route={}, pairs=[],
                  method='fixed_cycle_derangement_all_four_images_only',
                  metric='mean Euclidean distance between original and swapped six-point outputs (m)',
                  ground_truth_used_for_prediction=False)
    if len(indices) < 2:
        return report
    donor = _image_swap_cycle(metadata)
    per_route = defaultdict(list); pairs = []
    was_training = model.training
    model.eval()
    try:
        for position, index in enumerate(indices):
            donor_position = donor[position]
            donor_index = indices[donor_position]
            current = dataset[index]
            swapped = dict(current, images=dataset[donor_index]['images'])
            # Same augmentation seed isolates image content from augmentation RNG.
            original_batch = to_device(collator([current], seed=0), device)
            swapped_batch = to_device(collator([swapped], seed=0), device)
            original = predict(model, original_batch).detach().float().cpu()
            changed = predict(model, swapped_batch).detach().float().cpu()
            if original.shape != (1, 6, 2) or changed.shape != original.shape or not torch.isfinite(original).all() or not torch.isfinite(changed).all():
                raise ValueError('Image swap requires finite [1,6,2] planner predictions')
            delta = torch.linalg.vector_norm(original-changed, dim=-1).mean().item()
            route = str(metadata[position].get('route', 'unknown'))
            donor_route = str(metadata[donor_position].get('route', 'unknown'))
            per_route[route].append(delta)
            pairs.append(dict(index=index, donor_index=donor_index, route=route,
                              frame=metadata[position].get('frame'), donor_route=donor_route,
                              donor_frame=metadata[donor_position].get('frame'),
                              split=metadata[position].get('split', 'unknown'),
                              mean_point_change_m=delta, different_route=route != donor_route))
    finally:
        model.train(was_training)
    report.update(status='measured', pairs=pairs,
                  cross_route_pairs=sum(pair['different_route'] for pair in pairs),
                  mean_point_change_m=sum(pair['mean_point_change_m'] for pair in pairs)/len(pairs),
                  per_route={route: dict(samples=len(values), mean_point_change_m=sum(values)/len(values))
                             for route, values in sorted(per_route.items())})
    return report
