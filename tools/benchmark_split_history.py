"""Bounded Fast-only and synchronous fresh-KV runtime timing, without labels.

Compare each trained Fast-only checkpoint with its cache enabled. The sync
reference uses the SAME cooperative checkpoint as async, and reuses one prepared
Fast vision result across its detached no-KV candidate and final KV forward.
"""
import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import threading
import time

import torch

from fast_cluster.async_runtime import (NavigationEpoch, SnapshotCandidate, SnapshotStore,
    _audit_backend, _close_backend, _frame_cache_stats, load_fragment)
from fast_cluster.config import read_config
from slowfast_v11.protocol import cache_metadata


def _cache_stats(backend):
    return _frame_cache_stats(backend)


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _trajectory(value):
    value = value.detach().to('cpu', non_blocking=False).clone()
    if value.shape != (1, 8, 3) or not torch.isfinite(value).all():
        raise ValueError('Benchmark requires finite completed CPU trajectories [1,8,3]')
    return value


def _snapshot_bytes(snapshot):
    tensors = [t for layer in snapshot.layers for t in (layer.key, layer.value)]
    tensors.extend((snapshot.position_ids, snapshot.valid_mask))
    return sum(t.numel()*t.element_size() for t in tensors)


def run_benchmark(config, checkpoint, fragment, output, *, mode='fast_only', ticks=12,
        fast_device='cuda:0', slow_device='cuda:1', allow_colocated_smoke=False,
        wall_timeout_s=180., backend_module='fast_cluster.models'):
    if mode not in ('fast_only', 'sync_fresh'):
        raise ValueError('Benchmark mode must be fast_only or sync_fresh')
    rows = load_fragment(fragment)
    if type(ticks) is not int or not 1 <= ticks <= len(rows):
        raise ValueError('Bounded benchmark ticks must fit the continuous fragment')
    if not math.isfinite(wall_timeout_s) or wall_timeout_s <= 0:
        raise ValueError('Benchmark wall timeout must be finite and positive')
    if mode == 'sync_fresh' and fast_device == slow_device and not allow_colocated_smoke:
        raise ValueError('Same-device sync benchmark requires explicit allow_colocated_smoke')
    config = copy.deepcopy(config)
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    config['runtime_checkpoint'] = checkpoint
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    module = importlib.import_module(backend_module)
    fast = slow = None
    stop = threading.Event()
    condition = threading.Condition()
    shared = dict(latest=None, done=False)
    plans, releases = [], []
    source_thread = None
    failure = None
    report = dict(schema='navsim_split_history_runtime_benchmark_v1', mode=mode,
        runtime_checkpoint=checkpoint, fast_device=fast_device,
        slow_device=slow_device if mode == 'sync_fresh' else None,
        colocated_smoke=mode == 'sync_fresh' and fast_device == slow_device,
        multi_gpu_validated=False, quality_evaluated=False,
        scope='bounded runtime timing only; no quality, closed-loop, or target-cluster performance claim',
        source_policy='recorded real 2 Hz wall-clock arrivals; choose latest available observation, log all drops',
        four_frame_control='frozen RGB and per-frame visual feature cache enabled; full language history still processed',
        sync_policy='same observation detached no-KV Fast candidate, Slow prefill, complete CPU transfer, final Fast; one prepared Fast vision result',
        timing_scope='decode/Processor/vision plus language/head CPU submission; sync includes candidate, Slow input/preprocessing/prefill, D2H and Fast-device transfer',
        requested_plans=ticks, plans=plans, source_releases=releases)
    startup = time.monotonic()
    try:
        fast = module.make_runtime_fast(config, checkpoint, fast_device)
        report.update(fast_interface=fast.interface_info, fast_model_id=fast.model_id,
                      fast_cache_before=_cache_stats(fast), fast_frozen_before=_audit_backend(fast))
        frames = fast.interface_info.get('image_frames')
        if frames == 4 and (report['fast_cache_before'] is None or not report['fast_cache_before']['enabled']):
            raise ValueError('The four-frame benchmark requires its frozen visual cache enabled')
        if mode == 'sync_fresh':
            slow = module.make_runtime_slow(config, slow_device)
            report.update(slow_model_id=slow.model_id, slow_frozen_before=_audit_backend(slow))
        t0 = time.monotonic()+.05
        report.update(startup_s=time.monotonic()-startup, playback_start=t0)
        epochs = NavigationEpoch()

        def source_loop():
            for index, raw in enumerate(rows):
                due = t0+(raw['timestamp_us']-rows[0]['timestamp_us'])/1e6
                if stop.wait(max(0., due-time.monotonic())):
                    break
                obs = dict(raw, observation_time=due, source_index=index,
                    navversion=epochs.update(raw['episode_id'], raw['command']))
                released = time.monotonic()
                releases.append(dict(index=index, token=raw['token'], timestamp_us=raw['timestamp_us'],
                                     observation_time=due, release_time=released))
                with condition:
                    shared['latest'] = obs
                    condition.notify_all()
            with condition:
                shared['done'] = True
                condition.notify_all()

        source_thread = threading.Thread(target=source_loop, name='benchmark-source', daemon=True)
        source_thread.start()
        store = SnapshotStore(fast.device)
        last_index = -1
        while len(plans) < ticks:
            if time.monotonic()-t0 > wall_timeout_s:
                raise TimeoutError('Split-history benchmark exceeded wall timeout')
            with condition:
                obs = shared['latest']
                if obs is None or obs['source_index'] == last_index:
                    if shared['done']:
                        break
                    condition.wait(timeout=.02)
                    continue
            cycle_start = time.monotonic()
            prepared = fast.prepare(obs)
            prepared_at = time.monotonic()
            candidate_wall = slow_wall = d2h_wall = transfer_wall = 0.
            candidate_timings = None
            slow_timings = None
            snapshot = metadata = None
            source_time = ready_time = selection_time = None
            payload_bytes = 0
            if mode == 'sync_fresh':
                candidate_start = time.monotonic()
                with torch.inference_mode():
                    candidate, candidate_timings = fast.predict(prepared, None, None)
                candidate = _trajectory(candidate)
                candidate_wall = time.monotonic()-candidate_start
                slow_start = time.monotonic()
                with torch.inference_mode():
                    captured = slow.prefill(obs, candidate)
                slow_timings = getattr(slow, 'last_timings', None)
                _sync(getattr(slow, 'device', slow_device))
                slow_wall = time.monotonic()-slow_start
                d2h_start = time.monotonic()
                owned = captured.inference_copy('cpu')
                d2h_wall = time.monotonic()-d2h_start
                payload_bytes = _snapshot_bytes(owned)
                candidate_snapshot = SnapshotCandidate(len(plans)+1, owned,
                    obs['observation_time'], obs['episode_id'], obs['navversion'], slow.model_id,
                    tuple(obs['ego_pose']), obs['token'], time.monotonic())
                published = store.publish(candidate_snapshot)
                selection_time = time.monotonic()
                lease = store.select(image_time=obs['observation_time'], selection_time=selection_time,
                    episode_id=obs['episode_id'], navversion=obs['navversion'], model_id=slow.model_id)
                if lease is None:
                    raise RuntimeError('Completed synchronous source was not causally admitted')
                snapshot = lease.snapshot
                metadata = cache_metadata(snapshot.metadata, image_time=obs['observation_time'],
                                          current_pose=obs['ego_pose'], device=fast.device)[None]
                source_time, ready_time = snapshot.metadata.source_time, snapshot.metadata.ready_time
                transfer_wall = published.transfer_s
            final_start = time.monotonic()
            with torch.inference_mode():
                trajectory, final_timings = fast.predict(prepared, snapshot, metadata)
            trajectory = _trajectory(trajectory)
            submit = time.monotonic()
            plans.append(dict(token=obs['token'], source_index=obs['source_index'],
                timestamp_us=obs['timestamp_us'], image_time=obs['observation_time'],
                submit_time=submit, source_to_submit_s=submit-obs['observation_time'],
                cycle_wall_s=submit-cycle_start, preparation_wall_s=prepared_at-cycle_start,
                candidate_wall_s=candidate_wall, slow_prefill_wall_s=slow_wall,
                slow_timings=slow_timings,
                snapshot_d2h_wall_s=d2h_wall, snapshot_fast_transfer_wall_s=transfer_wall,
                snapshot_bytes=payload_bytes, snapshot_source_time=source_time,
                snapshot_ready_time=ready_time, selection_time=selection_time,
                final_predict_wall_s=submit-final_start,
                dropped_source_windows=max(0, obs['source_index']-last_index-1),
                preparation_timings=prepared.get('timings', {}), candidate_timings=candidate_timings,
                final_timings=final_timings, fast_cache=_cache_stats(fast),
                trajectory=trajectory.tolist()))
            last_index = obs['source_index']
    except BaseException as exc:
        failure = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        if source_thread is not None:
            source_thread.join(timeout=2.)
        for name, backend in [('fast', fast), ('slow', slow)]:
            if backend is None:
                continue
            try:
                after = _audit_backend(backend)
                report[name+'_frozen_after'] = after
                report[name+'_frozen_unchanged'] = after == report[name+'_frozen_before']
                if not report[name+'_frozen_unchanged']:
                    raise ValueError(f'Inference {name} parameters/buffers changed')
            except BaseException as exc:
                if failure is None:
                    failure = exc
                    report['error'] = f'{type(exc).__name__}: {exc}'
            finally:
                _close_backend(backend)
        report.update(planning_count=len(plans), source_count=len(releases),
            source_frequency_hz=(len(releases)-1)/(releases[-1]['release_time']-releases[0]['release_time']) if len(releases)>1 else 0.,
            planning_frequency_hz=(len(plans)-1)/(plans[-1]['submit_time']-plans[0]['submit_time']) if len(plans)>1 else 0.,
            fast_cache_after=_cache_stats(fast), dropped_source_windows=sum(p['dropped_source_windows'] for p in plans),
            source_thread_stopped=source_thread is None or not source_thread.is_alive(),
            stop_reason='requested_plan_limit' if len(plans)==ticks else 'error' if failure else 'source_exhausted',
            completed_without_error=failure is None)
        if plans:
            for key in ('source_to_submit_s', 'cycle_wall_s', 'preparation_wall_s', 'slow_prefill_wall_s'):
                report['mean_'+key] = sum(p[key] for p in plans)/len(plans)
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if failure is not None:
        raise failure
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/navsim_v1_split_history_smoke.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fragment', default='outputs/slowfast_cluster/async_fragment/fragment.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['fast_only', 'sync_fresh'], required=True)
    parser.add_argument('--fast-device', default='cuda:0')
    parser.add_argument('--slow-device', default='cuda:1')
    parser.add_argument('--allow-colocated-smoke', action='store_true')
    parser.add_argument('--ticks', type=int, choices=range(1, 25), default=12)
    parser.add_argument('--wall-timeout-s', type=float, default=180.)
    return parser


def main():
    args = build_parser().parse_args()
    result = run_benchmark(read_config(args.config), args.checkpoint, args.fragment, args.output,
        mode=args.mode, ticks=args.ticks, fast_device=args.fast_device, slow_device=args.slow_device,
        allow_colocated_smoke=args.allow_colocated_smoke, wall_timeout_s=args.wall_timeout_s)
    print(json.dumps({key: result[key] for key in ['mode', 'planning_count', 'source_frequency_hz',
        'planning_frequency_hz', 'fast_cache_after', 'quality_evaluated', 'multi_gpu_validated']}))


if __name__ == '__main__':
    main()
