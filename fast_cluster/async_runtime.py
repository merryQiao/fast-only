"""Bounded live inference around unchanged Fast/Slow models.

All publication clocks use time.monotonic. Recorded sensor times are mapped to
that clock only after both models are ready. CUDA tensors never cross processes.
"""
from dataclasses import dataclass, replace
import copy
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import tempfile
import threading
import time
import traceback

import torch

from slowfast_v11.protocol import KVSnapshot, SourceMetadata, admit_snapshot, cache_metadata


class WorkerFailure(RuntimeError):
    pass


class NavigationEpoch:
    def __init__(self):
        self.previous = None
        self.epoch = -1

    def update(self, episode, command):
        identity = (episode, command)
        if identity != self.previous:
            self.epoch += 1
            self.previous = identity
        return self.epoch


class LatestValue:
    """A bounded latest-value mailbox, with explicit replacement accounting."""
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None
        self.replaced = 0

    def put(self, value):
        with self.lock:
            if (self.value is not None and hasattr(value, 'version')
                    and self.value.version >= value.version):
                return
            if self.value is not None:
                self.replaced += 1
            self.value = value

    def take(self):
        with self.lock:
            value, self.value = self.value, None
            return value


class FixedRefresh:
    def __init__(self, start, period):
        if not math.isfinite(period) or period <= 0:
            raise ValueError('Refresh period must be finite and positive')
        self.start, self.period = start, period
        self.last_tick = -1
        self.busy = False

    def poll(self, now):
        tick = math.floor((now - self.start) / self.period)
        if tick < 0 or tick <= self.last_tick:
            return None
        event = dict(tick=tick, scheduled_time=self.start + tick * self.period,
                     skipped_overdue_ticks=max(0, tick - self.last_tick - 1),
                     kind='busy_skip' if self.busy else 'dispatch')
        self.last_tick = tick
        if not self.busy:
            self.busy = True
        return event

    def complete(self):
        self.busy = False


@dataclass(frozen=True)
class SnapshotCandidate:
    version: int
    snapshot: KVSnapshot
    source_time: float
    episode_id: str
    navversion: int
    model_id: str
    source_pose: tuple
    source_token: str = ''
    slow_completed_time: float = 0.


@dataclass(frozen=True)
class PublishedSnapshot:
    version: int
    snapshot: KVSnapshot
    source_token: str
    transfer_s: float


class SnapshotStore:
    """Only the Fast owner publishes and selects. A selected lease never changes."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.current = None

    def publish(self, candidate, now=time.monotonic):
        if len(candidate.snapshot.layers) != 6:
            raise ValueError('Publication requires all six Slow KV layers')
        if self.current is not None and candidate.version <= self.current.version:
            return None
        begin = now()
        # Owned blocking CPU->device transfer, including positions/mask. No generic
        # nonblocking D2H is used at either side of the process boundary.
        local = candidate.snapshot.inference_copy(self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        ready = now()
        metadata = SourceMetadata(candidate.source_time, ready, candidate.episode_id,
                                  str(candidate.navversion), candidate.model_id,
                                  tuple(candidate.source_pose))
        local = replace(local, metadata=metadata)
        published = PublishedSnapshot(candidate.version, local, candidate.source_token, ready - begin)
        self.current = published  # All fields are complete before this assignment.
        return published

    def select(self, *, image_time, selection_time, episode_id, navversion, model_id):
        local = self.current
        if local is None or not admit_snapshot(local.snapshot.metadata,
                image_time=image_time, selection_time=selection_time, episode_id=episode_id,
                navigation_id=str(navversion), model_id=model_id):
            return None
        return local


def validate_observation(observation):
    """Reject annotations rather than silently passing them to a model backend."""
    allowed = {'images', 'image_paths', 'ego', 'command', 'ego_pose', 'timestamp_us',
               'token', 'episode_id', 'navversion', 'frame_index'}
    if set(observation) - allowed:
        raise ValueError(f'Runtime observations are input-only; unexpected fields: {sorted(set(observation)-allowed)}')
    required = {'ego', 'command', 'ego_pose', 'timestamp_us', 'token', 'episode_id'}
    if not required <= set(observation):
        raise ValueError(f'Missing raw observation fields: {sorted(required-set(observation))}')
    images = observation.get('image_paths', observation.get('images'))
    if images is None or len(images) != 4:
        raise ValueError('Four chronological source images are required')
    if len(observation['ego']) != 4 or len(observation['ego_pose']) != 3:
        raise ValueError('Expected raw ego4 and SE(2) ego_pose')
    if not all(math.isfinite(float(v)) for v in [*observation['ego'], *observation['ego_pose']]):
        raise ValueError('Observation ego/pose must be finite')
    if type(observation['command']) is not int or observation['command'] not in range(4):
        raise ValueError('Official command must be integer 0..3')
    if type(observation['timestamp_us']) is not int:
        raise ValueError('Recorded timestamps must be integer microseconds')
    if not all(isinstance(observation[x], str) and observation[x] for x in ['token', 'episode_id']):
        raise ValueError('Observation identity must be nonempty')
    return dict(observation)


def load_fragment(fragment):
    if isinstance(fragment, (str, Path)):
        path = Path(fragment).resolve()
        content = json.loads(path.read_text())
        rows = content['observations']
        for row in rows:
            relative = row.get('image_paths', [])
            for name in relative:
                candidate = Path(name)
                if candidate.is_absolute() or '..' in candidate.parts:
                    raise ValueError('Fragment images must use safe relative paths')
            row['image_paths'] = [str(path.parent / name) for name in relative]
    else:
        rows = fragment
    rows = [validate_observation(row) for row in rows]
    if len(rows) < 2:
        raise ValueError('Continuous fragment needs at least two observations')
    if any(abs(b['timestamp_us'] - a['timestamp_us'] - 500_000) > 1000 for a, b in zip(rows, rows[1:])):
        raise ValueError('Continuous source requires recorded 2 Hz timestamp spacing')
    return rows


def align_feedback(completed, source):
    if completed is None or any(completed[key] != source[key] for key in ['episode_id', 'navversion']):
        return None, None
    age = source['observation_time'] - completed['observation_time']
    if age < 0:
        return None, None
    trajectory = completed['trajectory'].detach().cpu().clone().float()
    if trajectory.shape != (1, 8, 3) or not torch.isfinite(trajectory).all():
        raise ValueError('Completed Fast feedback requires finite CPU [1,8,3]')
    old_x, old_y, old_yaw = completed['ego_pose']
    new_x, new_y, new_yaw = source['ego_pose']
    c, s = math.cos(old_yaw), math.sin(old_yaw)
    x, y = trajectory[..., 0].clone(), trajectory[..., 1].clone()
    gx, gy = c*x-s*y+old_x-new_x, s*x+c*y+old_y-new_y
    c, s = math.cos(new_yaw), math.sin(new_yaw)
    trajectory[..., 0], trajectory[..., 1] = c*gx+s*gy, -s*gx+c*gy
    yaw = trajectory[..., 2] + old_yaw - new_yaw
    trajectory[..., 2] = torch.atan2(yaw.sin(), yaw.cos())
    target_times = [completed['observation_time'] + .5*i for i in range(1, 9)]
    provenance = {key: completed[key] for key in ['token', 'observation_time', 'completed_time',
                                                'episode_id', 'navversion', 'ego_pose']}
    provenance.update(age_s=age, absolute_target_times=target_times,
                      points_already_due=sum(t <= source['observation_time'] for t in target_times),
                      transform='source ego SE2 only; no resampling or action matching',
                      producer='latest completed main Fast plan; may be KV-enabled; matched source observation')
    return trajectory, provenance


def select_slow_source(latest, completed):
    """The existing fixed feedback time encoder requires plan/source identity.

    Keep the source observation that produced the latest compatible completed
    plan. A newer observation cannot relabel the old eight plan times as its own
    .5..4 second horizon. Missing feedback uses the latest available raw input.
    """
    if (completed is None or any(completed[key] != latest[key]
            for key in ['episode_id', 'navversion'])
            or completed['observation_time'] > latest['observation_time']):
        return latest, None, None
    matched = completed['input_observation']
    if (matched['token'] != completed['token']
            or matched['observation_time'] != completed['observation_time']
            or tuple(matched['ego_pose']) != tuple(completed['ego_pose'])):
        raise ValueError('Completed feedback must retain its exact source observation')
    source = dict(matched)
    feedback, provenance = align_feedback(completed, source)
    provenance.update(feedback_source_matches_observation=True,
        source_policy='latest compatible completed Fast plan and its exact observation snapshot')
    return source, feedback, provenance


def _audit_backend(backend):
    audit = getattr(backend, 'audit_frozen', None)
    return audit() if callable(audit) else {'available': False}


def _frame_cache_stats(backend):
    cache = getattr(backend, 'frame_cache', getattr(backend, 'cache', None))
    return cache.stats() if cache is not None and callable(getattr(cache, 'stats', None)) else None


def _close_backend(backend):
    close = getattr(backend, 'close', None)
    if callable(close):
        close()
        return
    models = [getattr(backend, 'model', None), getattr(backend, 'fast', None),
              getattr(getattr(backend, 'provider', None), 'slow', None)]
    for model in models:
        bridge = getattr(model, 'bridge', None)
        if bridge is not None:
            bridge.close()


def _receive_candidate(message):
    """Map a complete owned CPU file. Never queue the mapped tensor itself."""
    snapshot = torch.load(message['snapshot_path'], map_location='cpu',
                          mmap=True, weights_only=False)
    if not isinstance(snapshot, KVSnapshot) or len(snapshot.layers) != 6:
        raise ValueError('Mapped snapshot must contain six native KV layers')
    if any(layer.key.device.type != 'cpu' for layer in snapshot.layers):
        raise ValueError('Snapshot transport permits owned CPU tensors only')
    return SnapshotCandidate(snapshot=snapshot, **message['candidate_fields'])


def _slow_worker(config, device, backend_module, requests, results, acknowledgments, transport):
    """Spawn entry point: this process alone owns the Slow model and GPU context."""
    try:
        torch.set_num_threads(int(config.get('runtime_cpu_threads', 2)))
        backend = importlib.import_module(backend_module).make_runtime_slow(config, device)
        results.put(dict(kind='ready', pid=mp.current_process().pid, model_id=backend.model_id,
                         interface_info=getattr(backend, 'interface_info', {}),
                         frozen_audit=_audit_backend(backend), frame_cache=_frame_cache_stats(backend)))
        while True:
            request = requests.get()
            if request is None:
                break
            start = time.monotonic()
            with torch.inference_mode():
                captured = backend.prefill(request['observation'], request['feedback'])
                if len(captured.layers) != 6:
                    raise ValueError('Slow prefill must capture all six layers')
                owned = captured.inference_copy('cpu')
            complete = time.monotonic()
            obs = request['observation']
            provenance = request['feedback_provenance']
            if provenance is not None:
                provenance = dict(provenance,
                    age_at_slow_completion_s=complete-provenance['observation_time'],
                    points_due_at_slow_completion=sum(t <= complete for t in provenance['absolute_target_times']))
            fields = dict(version=request['version'], source_time=obs['observation_time'],
                episode_id=obs['episode_id'], navversion=obs['navversion'], model_id=backend.model_id,
                source_pose=tuple(obs['ego_pose']), source_token=obs['token'], slow_completed_time=complete)
            path = Path(transport)/f'snapshot_{request["version"]:08d}.pt'
            # Native payloads exceed this host's 64 MiB /dev/shm. The queue carries
            # ONLY this descriptor; torch's ForkingPickler must never see the KV.
            torch.save(owned, path)
            payload_bytes = sum(t.numel()*t.element_size() for layer in owned.layers
                                for t in (layer.key, layer.value))
            payload_bytes += sum(t.numel()*t.element_size() for t in (owned.position_ids, owned.valid_mask))
            results.put(dict(kind='snapshot', snapshot_path=str(path), candidate_fields=fields,
                             cpu_snapshot_bytes=payload_bytes, transport_write_complete=time.monotonic(), slow_start=start,
                             slow_completed=complete, slow_wall_s=complete-start,
                             slow_timings=getattr(backend, 'last_timings', None),
                             source_age_at_slow_start_s=start-obs['observation_time'],
                             feedback_provenance=provenance))
            if acknowledgments.get() != request['version']:
                raise RuntimeError('Slow snapshot receipt acknowledgment mismatch')
            # Linux mappings retain their storage after unlink, including while
            # a selected candidate awaits its blocking CPU->Fast-device transfer.
            path.unlink()
        results.put(dict(kind='stopped', frozen_audit=_audit_backend(backend), frame_cache=_frame_cache_stats(backend)))
    except BaseException:
        results.put(dict(kind='error', traceback=traceback.format_exc()))
    finally:
        _close_backend(locals().get('backend'))


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_async(config, fragment, checkpoint, output, *, fast_device='cuda:0', slow_device='cuda:1',
              allow_colocated_smoke=False, max_plans=16, refresh_period_s=1., wall_timeout_s=180.,
              startup_timeout_s=300., backend_module='fast_cluster.models', transport_root=None):
    """Run a bounded wall-clock replay, returning and writing all lifecycle evidence.

    Fast is owned by the caller process. A source thread and controller thread keep
    arrivals/absolute refresh ticks independent of Fast's model forward. Slow is a
    real spawned process. Snapshot transfer is blocking and honestly included in
    observed Fast submission latency, but never waits for an incomplete prefill.
    """
    config = copy.deepcopy(config)
    if checkpoint is not None:
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        config['runtime_checkpoint'] = checkpoint
    split_history = config.get('schema') == 'navsim_split_history_v2'
    rows = load_fragment(fragment)
    if type(max_plans) is not int or not 1 <= max_plans <= len(rows):
        raise ValueError('max_plans must fit the bounded source fragment')
    if fast_device == slow_device and not allow_colocated_smoke:
        raise ValueError('Shared device requires explicit allow_colocated_smoke')
    if any(not math.isfinite(value) or value <= 0
           for value in (wall_timeout_s, startup_timeout_s, refresh_period_s)):
        raise ValueError('Timeouts and refresh period must be finite and positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    transport_parent = Path(transport_root or os.environ.get('TMPDIR') or output)
    transport_parent.mkdir(parents=True, exist_ok=True)
    transport = Path(tempfile.mkdtemp(prefix='async-kv-', dir=transport_parent)).resolve()
    context = mp.get_context('spawn')
    requests, results = context.Queue(maxsize=1), context.Queue(maxsize=2)
    acknowledgments = context.Queue(maxsize=1)
    worker = context.Process(target=_slow_worker,
        args=(config, slow_device, backend_module, requests, results, acknowledgments, str(transport)), name='slow-prefill')
    stop = threading.Event()
    condition = threading.Condition()
    shared = dict(latest=None, feedback=None, error=None, source_done=False, busy=False)
    pending = LatestValue()
    refreshes, slow_events, releases, plans, publications = [], [], [], [], []
    threads = []
    fast = None
    startup = time.monotonic()
    worker.start()
    report = dict(schema='cluster_async_inference_v1', fast_device=fast_device,
                  slow_device=slow_device, colocated_smoke=fast_device == slow_device,
                  shared_device_functional_smoke=fast_device == slow_device,
                  multi_gpu_validated=False, refresh_period_s=refresh_period_s,
                  feedback_policy='latest compatible completed main Fast plan (may be KV-enabled) with its exact source observation',
                  feedback_training_difference=('runtime uses latest completed main Fast plan (may be KV-enabled); training uses detached same-observation no-KV Fast candidate'
                      if split_history else 'runtime feedback producer differs from frozen training teacher'),
                  runtime_checkpoint=checkpoint,
                  source_clock='recorded microsecond deltas anchored to common time.monotonic after ready barrier',
                  timing_scope='decode/Processor/vision prepare + language/head predict + snapshot transfer + submit',
                  transport_mode='file_backed_cpu_mmap', transport_directory=str(transport),
                  transport_scope='same host only; use node-local NVMe for the target runtime; no inter-node RPC',
                  plans=plans, publications=publications, refreshes=refreshes,
                  slow_events=slow_events, source_releases=releases)
    failure = None
    clean = False
    try:
        fast = importlib.import_module(backend_module).make_runtime_fast(config, checkpoint, fast_device)
        deadline = startup + startup_timeout_s
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('Slow model initialization exceeded startup timeout')
            try:
                ready = results.get(timeout=.1)
            except queue.Empty:
                if not worker.is_alive():
                    raise WorkerFailure(f'Slow worker exited before ready (exit={worker.exitcode})')
                continue
            if ready['kind'] == 'error':
                raise WorkerFailure(ready['traceback'])
            if ready['kind'] != 'ready':
                raise WorkerFailure('Expected Slow startup-ready message')
            break
        report.update(slow_worker_pid=ready['pid'], fast_model_id=fast.model_id,
                      slow_model_id=ready['model_id'], fast_interface=fast.interface_info,
                      slow_interface=ready['interface_info'],
                      fast_frozen_before=_audit_backend(fast), slow_frozen_before=ready['frozen_audit'],
                      fast_cache_before=_frame_cache_stats(fast), slow_cache_before=ready['frame_cache'],
                      startup_s=time.monotonic()-startup, requested_plans=max_plans)
        t0 = time.monotonic() + .05
        report['playback_start'] = t0
        epochs = NavigationEpoch()

        def source_loop():
            for index, raw in enumerate(rows):
                due = t0 + (raw['timestamp_us']-rows[0]['timestamp_us'])/1e6
                if stop.wait(max(0., due-time.monotonic())):
                    break
                obs = dict(raw, observation_time=due, release_time=time.monotonic(),
                           navversion=epochs.update(raw['episode_id'], raw['command']), source_index=index)
                releases.append(dict(index=index, token=obs['token'], observation_time=due,
                                     release_time=obs['release_time'], timestamp_us=obs['timestamp_us']))
                with condition:
                    shared['latest'] = obs
                    condition.notify_all()
            with condition:
                shared['source_done'] = True
                condition.notify_all()

        policy = FixedRefresh(t0, refresh_period_s)

        def controller_loop():
            version = 0
            try:
                while not stop.is_set():
                    try:
                        message = results.get(timeout=.005)
                    except queue.Empty:
                        message = None
                    if message is not None:
                        if message['kind'] == 'error':
                            raise WorkerFailure(message['traceback'])
                        if message['kind'] != 'snapshot':
                            raise WorkerFailure('Unexpected Slow lifecycle message')
                        try:
                            candidate = _receive_candidate(message)
                        finally:
                            acknowledgments.put(message['candidate_fields']['version'], timeout=.2)
                        pending.put(candidate)
                        slow_events.append({k: v for k, v in message.items()
                                            if k not in ('snapshot_path', 'candidate_fields')} |
                            dict(version=candidate.version, source_time=candidate.source_time,
                                 source_token=candidate.source_token))
                        policy.complete()
                        with condition:
                            shared['busy'] = False
                    if not worker.is_alive():
                        raise WorkerFailure(f'Slow worker exited during playback (exit={worker.exitcode})')
                    event = policy.poll(time.monotonic())
                    if event is None:
                        continue
                    event['checked_time'] = time.monotonic()
                    if event['kind'] == 'dispatch':
                        with condition:
                            obs = shared['latest']
                            completed = shared['feedback']
                        if obs is None:
                            event['kind'] = 'no_observation'
                            policy.complete()
                        else:
                            version += 1
                            obs, feedback, provenance = select_slow_source(obs, completed)
                            if provenance is not None:
                                if split_history:
                                    provenance['producer'] = 'latest completed main Fast plan (may be KV-enabled); training uses detached same-observation no-KV Fast candidate; matched source observation'
                                provenance.update(age_at_dispatch_s=event['checked_time']-provenance['observation_time'],
                                    points_due_at_dispatch=sum(t <= event['checked_time']
                                                              for t in provenance['absolute_target_times']))
                            event.update(version=version, source_token=obs['token'],
                                         source_time=obs['observation_time'])
                            requests.put(dict(version=version, observation=obs, feedback=feedback,
                                              feedback_provenance=provenance), timeout=.2)
                            with condition:
                                shared['busy'] = True
                    refreshes.append(event)
            except BaseException as exc:
                with condition:
                    shared['error'] = exc
                    condition.notify_all()

        threads = [threading.Thread(target=source_loop, name='source-replay', daemon=True),
                   threading.Thread(target=controller_loop, name='slow-controller', daemon=True)]
        for thread in threads:
            thread.start()
        store = SnapshotStore(fast.device)
        last_index = -1
        while len(plans) < max_plans:
            if time.monotonic()-t0 > wall_timeout_s:
                raise TimeoutError('Async replay exceeded wall timeout')
            with condition:
                if shared['error'] is not None:
                    raise shared['error']
                obs = shared['latest']
                if obs is None or obs['source_index'] == last_index:
                    if shared['source_done']:
                        break
                    condition.wait(timeout=.02)
                    continue
            prepare_start = time.monotonic()
            prepared = fast.prepare(obs)
            prepared_at = time.monotonic()
            # Freeze this observation's four frames through preparation and LLM.
            # A Slow result completing during vision preparation may be selected.
            candidate = pending.take()
            if candidate is not None:
                # Old episodes/navigation cannot reappear when a late job finishes.
                if candidate.source_time > obs['observation_time']:
                    pending.put(candidate)  # Preserve it for a later eligible image.
                elif (candidate.episode_id == obs['episode_id'] and candidate.navversion == obs['navversion']
                        and candidate.model_id == ready['model_id']):
                    published = store.publish(candidate)
                    if published:
                        publications.append(dict(version=published.version, source_token=published.source_token,
                            source_time=published.snapshot.metadata.source_time,
                            ready_time=published.snapshot.metadata.ready_time, transfer_s=published.transfer_s))
            selected_at = time.monotonic()
            lease = store.select(image_time=obs['observation_time'], selection_time=selected_at,
                                 episode_id=obs['episode_id'], navversion=obs['navversion'],
                                 model_id=ready['model_id'])
            metadata = None if lease is None else cache_metadata(lease.snapshot.metadata,
                image_time=obs['observation_time'], current_pose=obs['ego_pose'], device=fast.device)[None]
            forward_start = time.monotonic()
            with torch.inference_mode():
                trajectory, timings = fast.predict(prepared, None if lease is None else lease.snapshot, metadata)
            completed_at = time.monotonic()
            trajectory = trajectory.detach().cpu().clone()
            if trajectory.shape != (1, 8, 3) or not torch.isfinite(trajectory).all():
                raise ValueError('Fast must return a finite completed CPU trajectory [1,8,3]')
            with condition:
                was_busy = shared['busy']
                shared['feedback'] = dict(trajectory=trajectory, observation_time=obs['observation_time'],
                    completed_time=completed_at, token=obs['token'], episode_id=obs['episode_id'],
                    navversion=obs['navversion'], ego_pose=obs['ego_pose'], input_observation=dict(obs))
            submit = time.monotonic()
            row = dict(token=obs['token'], source_index=obs['source_index'], timestamp_us=obs['timestamp_us'],
                episode_id=obs['episode_id'], navversion=obs['navversion'],
                image_time=obs['observation_time'], source_release_time=obs['release_time'],
                selection_time=selected_at, forward_start=forward_start, completed_time=completed_at,
                submit_time=submit, source_to_submit_s=submit-obs['observation_time'],
                preparation_wall_s=prepared_at-prepare_start, predict_wall_s=completed_at-forward_start,
                dropped_source_windows=max(0, obs['source_index']-last_index-1),
                snapshot_version=None if lease is None else lease.version,
                snapshot_source_time=None if lease is None else lease.snapshot.metadata.source_time,
                snapshot_ready_time=None if lease is None else lease.snapshot.metadata.ready_time,
                snapshot_source_token=None if lease is None else lease.source_token,
                slow_inflight_at_completion=was_busy, trajectory=trajectory.tolist(),
                timings={**prepared.get('timings', {}), **timings})
            plans.append(row)
            last_index = obs['source_index']
    except BaseException as exc:
        failure = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2.)
        # The publisher remains alive until every received shared snapshot is owned
        # locally. Wait for any outstanding Slow job, consume its CPU transport,
        # then request an orderly exit. Shutdown has a finite wall-clock bound.
        try:
            requests.put(None, timeout=.5)
        except queue.Full:
            pass
        shutdown_deadline = time.monotonic() + min(15., max(2., wall_timeout_s))
        while worker.is_alive() and time.monotonic() < shutdown_deadline:
            try:
                message = results.get(timeout=.05)
                if message['kind'] == 'snapshot':
                    try:
                        _receive_candidate(message)
                    finally:
                        acknowledgments.put(message['candidate_fields']['version'], timeout=.2)
                if message['kind'] == 'stopped':
                    report['slow_frozen_after'] = message['frozen_audit']
                    report['slow_cache_after'] = message['frame_cache']
                if message['kind'] == 'error' and failure is None:
                    failure = WorkerFailure(message['traceback'])
                    report['error'] = str(failure)
            except queue.Empty:
                pass
            worker.join(timeout=.01)
        if worker.is_alive():
            worker.terminate(); worker.join(timeout=2.)
        else:
            clean = worker.exitcode == 0 and not any(thread.is_alive() for thread in threads)
        # A small final status message can already be flushed when the process
        # exits; no shared tensor remains unacknowledged at this point.
        while not worker.is_alive():
            try:
                message = results.get_nowait()
            except queue.Empty:
                break
            if message['kind'] == 'stopped':
                report['slow_frozen_after'] = message['frozen_audit']
                report['slow_cache_after'] = message['frame_cache']
            elif message['kind'] == 'error' and failure is None:
                failure = WorkerFailure(message['traceback'])
                report['error'] = str(failure)
        if fast is not None:
            report['fast_cache_after'] = _frame_cache_stats(fast)
            try:
                report['fast_frozen_after'] = _audit_backend(fast)
                if 'fast_frozen_before' in report:
                    report['fast_frozen_unchanged'] = report['fast_frozen_before'] == report['fast_frozen_after']
                if 'slow_frozen_after' in report:
                    report['slow_frozen_unchanged'] = report['slow_frozen_before'] == report['slow_frozen_after']
                if report.get('fast_frozen_unchanged') is False or report.get('slow_frozen_unchanged') is False:
                    raise ValueError('Inference parameters or buffers changed')
            except BaseException as exc:
                if failure is None:
                    failure = exc
                    report['error'] = f'{type(exc).__name__}: {exc}'
            finally:
                _close_backend(fast)
        requests.cancel_join_thread(); requests.close()
        results.cancel_join_thread(); results.close()
        acknowledgments.cancel_join_thread(); acknowledgments.close()
        shutil.rmtree(transport)
        count = len(releases)
        report.update(planning_count=len(plans), source_count=count,
            source_frequency_hz=(count-1)/(releases[-1]['release_time']-releases[0]['release_time']) if count > 1 else 0.,
            planning_frequency_hz=(len(plans)-1)/(plans[-1]['submit_time']-plans[0]['submit_time']) if len(plans)>1 else 0.,
            consumed_versions=sorted({p['snapshot_version'] for p in plans if p['snapshot_version'] is not None}),
            fast_completed_while_slow_busy=sum(p['slow_inflight_at_completion'] for p in plans),
            dropped_source_windows=sum(p['dropped_source_windows'] for p in plans),
            coalesced_cpu_snapshots=pending.replaced, busy_refresh_skips=sum(x['kind']=='busy_skip' for x in refreshes),
            maximum_cpu_snapshot_bytes=max((x['cpu_snapshot_bytes'] for x in slow_events), default=0),
            transport_clean=not transport.exists(),
            shutdown_clean=clean, slow_worker_exitcode=worker.exitcode,
            completed_without_error=failure is None)
        _json_write(output/'result.json', report)
    if failure is not None:
        raise failure
    return report
