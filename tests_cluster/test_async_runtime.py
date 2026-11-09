"""Runtime causality and a real spawned CPU Slow worker; no model downloads."""
import math
import io
import json
import os
import tarfile
import time

import pytest
import torch

from slowfast_v11.protocol import KVSnapshot, LayerKV


def snapshot(value=1.):
    x = torch.full((1, 2, 3, 4), value)
    return KVSnapshot(tuple(LayerKV(x.clone(), x.clone()) for _ in range(6)),
                      torch.zeros(3, 1, 3, dtype=torch.long),
                      torch.ones(1, 3, dtype=torch.bool))


def observation(index=0, command=1):
    return dict(image_paths=['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
                ego=[1., 0., 0., 0.], command=command, ego_pose=[0., 0., 0.],
                timestamp_us=1_000_000 + index * 500_000, token=f't{index}',
                episode_id='episode', navversion=0)


def test_navigation_epoch_changes_when_command_returns():
    from fast_cluster.async_runtime import NavigationEpoch
    epochs = NavigationEpoch()
    assert [epochs.update('ep', c) for c in [1, 1, 2, 1]] == [0, 0, 1, 2]
    assert epochs.update('other', 1) == 3


def test_fixed_refresh_skips_busy_and_overdue_ticks_without_drift():
    from fast_cluster.async_runtime import FixedRefresh
    clock = FixedRefresh(10., 1.)
    assert clock.poll(10.)['scheduled_time'] == 10.
    assert clock.poll(10.9) is None
    assert clock.poll(11.)['kind'] == 'busy_skip'
    clock.complete()
    event = clock.poll(14.2)
    assert event['scheduled_time'] == 14.
    assert event['skipped_overdue_ticks'] == 2
    assert event['kind'] == 'dispatch'


def test_snapshot_publication_owns_storage_and_requires_all_layers():
    from fast_cluster.async_runtime import SnapshotStore, SnapshotCandidate
    original = snapshot()
    store = SnapshotStore('cpu')
    candidate = SnapshotCandidate(1, original, 1., 'ep', 0, 'model', (0., 0., 0.))
    published = store.publish(candidate, now=lambda: 2.)
    original.layers[0].key.add_(5)
    assert published.snapshot.layers[0].key.flatten()[0] == 1
    with pytest.raises(ValueError, match='six'):
        store.publish(SnapshotCandidate(2, KVSnapshot(original.layers[:5], original.position_ids,
                      original.valid_mask), 1., 'ep', 0, 'model', (0., 0., 0.)))


def test_admission_requires_source_ready_and_identity_and_fixes_one_version():
    from fast_cluster.async_runtime import SnapshotStore, SnapshotCandidate
    store = SnapshotStore('cpu')
    args = dict(image_time=2., selection_time=3., episode_id='ep', navversion=0,
                model_id='model')
    store.publish(SnapshotCandidate(1, snapshot(), 1., 'ep', 0, 'model', (0., 0., 0.)),
                  now=lambda: 2.5)
    lease = store.select(**args)
    assert lease.version == 1
    assert store.select(**(args | {'selection_time': 2.4})) is None
    assert store.select(**(args | {'image_time': .9})) is None
    assert store.select(**(args | {'navversion': 2})) is None
    assert store.select(**(args | {'episode_id': 'other'})) is None
    assert store.select(**(args | {'model_id': 'other'})) is None
    store.publish(SnapshotCandidate(2, snapshot(2.), 2., 'ep', 0, 'model', (0., 0., 0.)),
                  now=lambda: 2.8)
    assert store.select(**args).version == 2
    assert lease.version == 1 and lease.snapshot.layers[0].key.flatten()[0] == 1


def test_latest_value_coalesces_backlog():
    from fast_cluster.async_runtime import LatestValue
    box = LatestValue()
    box.put('old'); box.put('new')
    assert box.take() == 'new'
    assert box.take() is None
    assert box.replaced == 1


def test_input_rejects_ground_truth_and_future_fields():
    from fast_cluster.async_runtime import validate_observation
    for field in ['trajectory_gt', 'future_timestamps_us', 'future_poses']:
        with pytest.raises(ValueError, match='input-only'):
            validate_observation(observation() | {field: []})
    assert validate_observation(observation())['timestamp_us'] == 1_000_000


def test_feedback_transforms_frame_and_preserves_original_target_times():
    from fast_cluster.async_runtime import align_feedback
    plan = torch.tensor([[[1., 0., 0.]] * 8])
    completed = dict(trajectory=plan, observation_time=1., completed_time=1.2,
                     token='old', episode_id='ep', navversion=0, ego_pose=(1., 0., 0.))
    source = observation() | dict(observation_time=2., episode_id='ep', ego_pose=(0., 0., math.pi / 2))
    aligned, provenance = align_feedback(completed, source)
    assert aligned[0, 0].tolist() == pytest.approx([0., -2., -math.pi / 2], abs=1e-6)
    assert provenance['absolute_target_times'] == [1.5 + .5 * i for i in range(8)]
    assert provenance['age_s'] == 1.
    assert provenance['points_already_due'] == 2
    assert align_feedback(completed | {'navversion': 1}, source) == (None, None)


class CPUFast:
    device = 'cpu'
    model_id = 'fast-cpu'
    interface_info = {'layers': 6, 'backend': 'CPU contract test'}

    def prepare(self, obs):
        return {'observation': obs, 'timings': {'decode_s': 0., 'processor_s': 0., 'vision_s': 0.,
                                              'prepare_complete_monotonic': time.monotonic()}}

    def predict(self, prepared, snap, metadata):
        time.sleep(.015)
        assert snap is None or len(snap.layers) == 6
        return torch.zeros(1, 8, 3), {'llm_head_s': .015, 'model_total_s': .015}


class CPUSlow:
    model_id = 'slow-cpu'

    def __init__(self, config):
        self.config = config

    def prefill(self, obs, feedback):
        if self.config.get('expected_runtime_checkpoint'):
            assert self.config['runtime_checkpoint'] == self.config['expected_runtime_checkpoint']
        time.sleep(self.config.get('slow_delay_s', .12))
        if self.config.get('fail_worker'):
            raise RuntimeError('intentional child failure')
        if self.config.get('native_shape'):
            x = torch.ones(1, 2, 8255, 256, dtype=torch.bfloat16)
            return KVSnapshot(tuple(LayerKV(x.clone(), x.clone()) for _ in range(6)),
                              torch.zeros(3, 1, 8255, dtype=torch.long),
                              torch.ones(1, 8255, dtype=torch.bool))
        return snapshot(float(obs['timestamp_us']))


def make_runtime_fast(config, checkpoint, device):
    return CPUFast()


def make_runtime_slow(config, device):
    return CPUSlow(config)


def test_real_spawned_cpu_worker_replays_wall_clock_and_publishes(tmp_path):
    from fast_cluster.async_runtime import run_async
    result = run_async({}, [observation(i) for i in range(8)], None, tmp_path / 'run',
                       fast_device='cpu', slow_device='cpu', allow_colocated_smoke=True,
                       max_plans=5, refresh_period_s=.2, wall_timeout_s=12,
                       backend_module='tests_cluster.test_async_runtime')
    assert result['slow_worker_pid'] != os.getpid()
    assert result['planning_count'] == 5
    assert result['source_frequency_hz'] == pytest.approx(2., rel=.05)
    assert result['multi_gpu_validated'] is False
    assert result['consumed_versions']
    assert result['fast_completed_while_slow_busy'] > 0
    assert result['shutdown_clean'] is True
    assert all(row['selection_time'] >= row['timings']['prepare_complete_monotonic']
               for row in result['plans'])
    assert all(row['snapshot_source_time'] <= row['image_time'] and
               row['snapshot_ready_time'] <= row['selection_time']
               for row in result['plans'] if row['snapshot_version'] is not None)


def test_spawned_worker_error_is_propagated_and_stopped(tmp_path):
    from fast_cluster.async_runtime import run_async, WorkerFailure
    with pytest.raises(WorkerFailure, match='intentional child failure'):
        run_async({'fail_worker': True}, [observation(i) for i in range(8)], None,
                  tmp_path / 'failed', fast_device='cpu', slow_device='cpu',
                  allow_colocated_smoke=True, max_plans=6, refresh_period_s=.2,
                  wall_timeout_s=12, backend_module='tests_cluster.test_async_runtime')


def test_local_extract_preserves_original_bytes_and_rejects_escape(tmp_path):
    from PIL import Image
    from tools.prepare_async_fragment import extract_selected
    image = io.BytesIO()
    Image.new('RGB', (1920, 1080), color=(10, 20, 30)).save(image, format='JPEG')
    data = image.getvalue()
    prefix = tmp_path / 'prefix.gz'
    with tarfile.open(prefix, 'w:gz') as archive:
        member = tarfile.TarInfo('original/a.jpg'); member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    out = tmp_path / 'images'; out.mkdir()
    receipt = extract_selected(prefix, out, {'original/a.jpg': 'safe/a.jpg'})
    assert (out / 'safe/a.jpg').read_bytes() == data
    assert receipt['safe/a.jpg']['size_wh'] == [1920, 1080]
    with pytest.raises(ValueError, match='safe relative'):
        extract_selected(prefix, out, {'original/a.jpg': '../escape.jpg'})


def test_fragment_load_rejects_absolute_or_parent_paths(tmp_path):
    from fast_cluster.async_runtime import load_fragment
    for name in ['../bad.jpg', '/absolute.jpg']:
        path = tmp_path / 'fragment.json'
        path.write_text(json.dumps({'observations': [observation(i) | {'image_paths': [name]*4}
                                                   for i in range(2)]}))
        with pytest.raises(ValueError, match='safe relative'):
            load_fragment(path)


def test_cli_defaults_to_distinct_gpus_and_explicit_colocation():
    from tools.run_cluster_async import build_parser
    args = build_parser().parse_args(['--checkpoint', 'weights.pt', '--output', 'run'])
    assert (args.fast_device, args.slow_device) == ('cuda:0', 'cuda:1')
    assert args.refresh_period_s == 1.
    assert args.allow_colocated_smoke is False
    assert args.config == 'configs/navsim_v1_split_history_smoke.yaml'


def test_native_size_cpu_snapshot_uses_descriptor_transport_and_cleans_files(tmp_path):
    from fast_cluster.async_runtime import run_async
    before_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        result = run_async({'native_shape': True}, [observation(i) for i in range(6)], None,
            tmp_path / 'native', fast_device='cpu', slow_device='cpu', allow_colocated_smoke=True,
            max_plans=3, refresh_period_s=.2, wall_timeout_s=15,
            backend_module='tests_cluster.test_async_runtime', transport_root=tmp_path)
    finally:
        torch.set_num_threads(before_threads)
    assert result['transport_mode'] == 'file_backed_cpu_mmap'
    assert result['maximum_cpu_snapshot_bytes'] > 100_000_000
    assert result['consumed_versions']
    assert result['transport_clean'] is True
    assert not list(tmp_path.rglob('snapshot_*.pt'))


def test_nonfinite_runtime_timeout_is_rejected_before_startup(tmp_path):
    from fast_cluster.async_runtime import run_async
    target = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='finite and positive'):
        run_async({}, [observation(i) for i in range(3)], None, target,
                  fast_device='cpu', slow_device='cpu', allow_colocated_smoke=True,
                  max_plans=2, wall_timeout_s=float('nan'),
                  backend_module='tests_cluster.test_async_runtime')
    assert not target.exists()


def test_slow_feedback_uses_its_matching_observation_when_latest_is_newer():
    from fast_cluster.async_runtime import select_slow_source
    old = observation(1) | dict(observation_time=1.5)
    latest = observation(3) | dict(observation_time=2.5)
    completed = dict(trajectory=torch.zeros(1, 8, 3), observation_time=1.5,
                     completed_time=1.7, token=old['token'], episode_id='episode', navversion=0,
                     ego_pose=old['ego_pose'], input_observation=old)
    source, feedback, provenance = select_slow_source(latest, completed)
    assert source['token'] == old['token']
    assert source['observation_time'] == completed['observation_time']
    assert torch.equal(feedback, completed['trajectory'])
    assert provenance['absolute_target_times'] == [2. + .5*i for i in range(8)]
    assert provenance['feedback_source_matches_observation'] is True
    source, feedback, provenance = select_slow_source(latest | {'navversion': 2}, completed)
    assert source['token'] == latest['token'] and feedback is None and provenance is None


def test_runtime_injects_same_absolute_checkpoint_for_slow_worker(tmp_path):
    from fast_cluster.async_runtime import run_async
    checkpoint = tmp_path/'same_checkpoint.pt'
    config = {'expected_runtime_checkpoint': str(checkpoint.resolve())}
    result = run_async(config, [observation(i) for i in range(4)], checkpoint,
        tmp_path/'same_source', fast_device='cpu', slow_device='cpu', allow_colocated_smoke=True,
        max_plans=2, refresh_period_s=.2, wall_timeout_s=12,
        backend_module='tests_cluster.test_async_runtime')
    assert result['runtime_checkpoint'] == str(checkpoint.resolve())
    assert 'runtime_checkpoint' not in config
