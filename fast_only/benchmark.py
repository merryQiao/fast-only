"""Synchronized full-planner model, decoded-stream and disk latency distributions."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .common import NativeCollator, make_model, predict, read_config, to_device, verify_checkpoint_inputs, write_json

MODES = ('model_only', 'streaming_e2e', 'disk_e2e')


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _distribution(samples):
    return {**{f'p{q}': float(np.percentile(samples, q)) for q in (50, 95, 99)},
            'mean': float(np.mean(samples)), 'samples': samples, 'count': len(samples)}


def _input_audit(sample, batch, model, collator):
    native = batch['native']
    grid = native.get('image_grid_thw')
    grid_list = grid.detach().cpu().tolist() if grid is not None else None
    visual = getattr(getattr(model, 'backbone', None), 'visual', None)
    merge = getattr(visual, 'spatial_merge_size', None)
    processor = getattr(getattr(collator, 'processor', None), 'image_processor', None)
    patch_size = getattr(processor, 'patch_size', None)
    result = {'raw_image_sizes_wh': [list(image.size) for image in sample.get('images', [])],
              'image_grid_thw': grid_list,
              'image_tokens_per_frame': ((grid.prod(-1) // merge**2).detach().cpu().tolist()
                                         if grid is not None and merge else None),
              'pixel_values_shape': list(native['pixel_values'].shape) if 'pixel_values' in native else None,
              'input_ids_shape': list(native['input_ids'].shape) if 'input_ids' in native else None,
              'processor_patch_size': patch_size,
              'processed_sizes_wh': ([[g[2]*patch_size, g[1]*patch_size] for g in grid_list]
                                     if grid_list is not None and isinstance(patch_size, int) else None)}
    if 'image_audit' in batch:
        result['collator_image_audit'] = batch['image_audit']
    return result


def _component_profile(model, batch, device):
    """Extra forward with boundary synchronization; never an E2E estimate."""
    timings = {}; starts = {}; hooks = []
    backbone = getattr(model, 'backbone', None)
    modules = [('vision', getattr(backbone, 'visual', None)),
               ('backbone', getattr(backbone, 'language_model', None)),
               ('head', getattr(model, 'head', None))]
    for name, module in modules:
        if module is None:
            continue
        def before(_module, _args, name=name):
            synchronize(device)
            starts[name] = time.perf_counter()
        def after(_module, _args, _output, name=name):
            synchronize(device)
            timings[name] = (time.perf_counter()-starts[name])*1000
        hooks.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        predict(model, batch)
        synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
    return timings


@torch.no_grad()
def measure(model, dataset, collator, device, *, warmup=10, repeats=100,
            modes=None, profile_components=True):
    """Measure batch-one full forwards, including native vision on every call.

    Model-only reuses one already-preprocessed device input (never features).
    Streaming decodes each raw PIL observation before starting its timer, then
    includes processor, H2D and full model. Disk starts before ``dataset[i]`` and
    includes JPEG read/decode too. Streaming and disk cycle the same dataset
    indices; only the model-only mode uses a fixed canonical input. No image
    feature/KV cache is used. OS cache is warm; this is not cold-storage latency.
    """
    if warmup < 1 or repeats < 1:
        raise ValueError('Benchmark requires positive warmup and repeat counts')
    modes = list(MODES) if modes is None else list(modes)
    if not modes or any(mode not in MODES for mode in modes) or len(modes) != len(set(modes)):
        raise ValueError(f'Benchmark modes must be a nonempty unique subset of {MODES}')
    if not len(dataset):
        raise ValueError('Benchmark requires a nonempty dataset')
    was_training = model.training
    model.eval()
    is_cuda = torch.device(device).type == 'cuda'
    result = {'warmup': warmup, 'repeats': repeats, 'batch_size': 1, 'modes': modes,
              'quantile_warning': ('Fewer than 100 repeats: tail quantiles are exploratory.'
                                   if repeats < 100 else None),
              'memory_by_mode': {}, 'preprocessing_by_mode': {},
              'notes': 'All modes run native vision, language backbone and trajectory head on every forward. '
                       'No feature or KV cache. Model-only uses one preprocessed device input; '
                       'streaming starts after raw-image decode and includes native processor + H2D; '
                       'disk includes dataset read/decode + native processor + H2D. '
                       'Warm OS file cache; GPU synchronized at timed boundaries. '
                       'Component times come from one extra instrumented pass, not summed quantiles.'}
    batch = None
    try:
        for mode in modes:
            elapsed = []; prep = []; transfer = []
            if mode == 'model_only':
                sample = dataset[0]
                cpu_batch = collator([sample])
                batch = to_device(cpu_batch, device)
                synchronize(device)
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            for i in range(warmup + repeats):
                if mode == 'streaming_e2e':
                    sample = dataset[i % len(dataset)]  # deliberately outside the timer
                synchronize(device)
                t0 = time.perf_counter()
                if mode == 'disk_e2e':
                    sample = dataset[i % len(dataset)]
                if mode != 'model_only':
                    cpu_batch = collator([sample])
                    t1 = time.perf_counter()
                    batch = to_device(cpu_batch, device)
                    synchronize(device)
                    t2 = time.perf_counter()
                else:
                    t1 = t2 = t0
                predict(model, batch)
                synchronize(device)
                t3 = time.perf_counter()
                if i >= warmup:
                    elapsed.append((t3-t0)*1000)
                    prep.append((t1-t0)*1000)
                    transfer.append((t2-t1)*1000)
                if i == 0:
                    result.setdefault('input_audit', _input_audit(sample, cpu_batch, model, collator))
            result[mode + '_ms'] = _distribution(elapsed)
            result['preprocessing_by_mode'][mode] = {
                'read_preprocess_ms': _distribution(prep), 'host_to_device_ms': _distribution(transfer)}
            result['memory_by_mode'][mode] = {
                'peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20 if is_cuda else None,
                'peak_reserved_mib': torch.cuda.max_memory_reserved(device)/2**20 if is_cuda else None}
        result['component_ms_single_profile_pass'] = (_component_profile(model, batch, device)
                                                       if profile_components else {})
        result['layout'] = getattr(model, 'last_layout', {})
        for key in ('peak_allocated_mib', 'peak_reserved_mib'):
            result[key] = max((mem[key] for mem in result['memory_by_mode'].values()
                               if mem[key] is not None), default=None)
        if 'disk_e2e_ms' in result:
            result['end_to_end_ms'] = result['disk_e2e_ms']  # old report compatibility
            result['within_100ms_p95'] = result['disk_e2e_ms']['p95'] <= 100
            disk_prep = result['preprocessing_by_mode']['disk_e2e']
            result['read_preprocess_ms_p50'] = disk_prep['read_preprocess_ms']['p50']
            result['host_to_device_ms_p50'] = disk_prep['host_to_device_ms']['p50']
        return result
    finally:
        model.train(was_training)


def main():
    from .data import FastDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, help='Measured forwards per mode (default: config or 100)')
    parser.add_argument('--warmup', type=int, help='Warmup forwards per mode (default: config or 10)')
    parser.add_argument('--modes', '--mode', nargs='+', choices=MODES)
    parser.add_argument('--profile-components', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else None
    config = state['config'] if state else read_config(args.config)
    if state:
        verify_checkpoint_inputs(state)
    torch.set_num_threads(config.get('cpu_threads', 4))
    model = make_model(config)
    if state:
        model.load_trainable_state(state['model'])
    dataset = FastDataset(config['data']['train_manifest'], config['data']['stats'])
    collator = NativeCollator(config['model']['path'], **config['processor'])
    options = {key: value for key, value in config.get('benchmark', {}).items()
               if key in {'warmup', 'repeats', 'modes', 'profile_components'}}
    options.update({key: getattr(args, key) for key in ('warmup', 'repeats', 'modes', 'profile_components')
                    if getattr(args, key) is not None})
    result = measure(model, dataset, collator, config['device'], **options)
    result.update(checkpoint=args.checkpoint, processor_policy=config['processor'],
                  checkpoint_inputs_verified=state is not None)
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
