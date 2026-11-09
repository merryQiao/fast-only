"""Frozen per-frame RGB/ViT cache contracts, using a native-shaped CPU provider."""
from types import SimpleNamespace
import os

from PIL import Image
import pytest
import torch


class NativeVision:
    def __init__(self):
        self.visual = SimpleNamespace(spatial_merge_size=2)
        self.calls = []

    def get_image_features(self, pixel_values, image_grid_thw, return_dict=True):
        assert image_grid_thw.shape == (1, 3)
        self.calls.append((len(pixel_values), image_grid_thw.tolist()))
        return SimpleNamespace(pooler_output=(pixel_values.reshape(-1, 4, pixel_values.shape[-1]).sum(1),))


def native(frames):
    return dict(pixel_values=torch.cat([torch.full((4, 3), float(index)) for index in frames]),
                image_grid_thw=torch.tensor([[1, 2, 2]]*len(frames), dtype=torch.long))


def test_per_image_native_calls_and_cache_off_math_match_cached_overlap():
    from fast_cluster.vision_cache import FrozenFrameCache, encode_frames
    provider = NativeVision()
    cache = FrozenFrameCache('frozen-provider', capacity=8)
    first = encode_frames(provider, native(range(4)), cache=cache, keys=list('abcd'))
    second = encode_frames(provider, native(range(1, 5)), cache=cache, keys=list('bcde'))
    reference = NativeVision()
    uncached = encode_frames(reference, native(range(1, 5)))
    assert len(provider.calls) == 5 and len(reference.calls) == 4
    assert all(call[0] == 4 for call in provider.calls + reference.calls)
    assert all(torch.equal(a, b) for a, b in zip(second, uncached))
    assert first[0].shape == (1, 3)
    assert cache.stats()['feature_hits'] == 3
    assert cache.stats()['feature_misses'] == 5


def test_image_cache_decodes_once_and_owns_rgb_copies(tmp_path, monkeypatch):
    from fast_cluster.vision_cache import FrozenFrameCache
    path = tmp_path/'frame.jpg'
    Image.new('RGB', (16, 12), (30, 60, 90)).save(path)
    original = Image.open
    opened = []

    def open_image(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(Image, 'open', open_image)
    cache = FrozenFrameCache('provider')
    image, key = cache.image(path)
    expected = image.getpixel((0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    again, other_key = cache.image(path)
    assert len(opened) == 1 and key == other_key
    assert again.mode == 'RGB' and again.getpixel((0, 0)) == expected
    assert cache.stats()['image_hits'] == 1
    assert cache.stats()['rgb_bytes'] == 16*12*3
    previous_mtime = path.stat().st_mtime_ns
    Image.new('RGB', (16, 12), (90, 60, 30)).save(path)
    # Some shared filesystems expose coarser mutation timestamps than the two
    # immediate writes in this test; make the stat identity change explicit.
    os.utime(path, ns=(previous_mtime+2_000_000_000, previous_mtime+2_000_000_000))
    changed, changed_key = cache.image(path)
    assert changed_key != key and len(opened) == 2
    assert changed.getpixel((0, 0)) != expected


def test_feature_ownership_capacity_and_provider_invalidation():
    from fast_cluster.vision_cache import FrozenFrameCache
    cache = FrozenFrameCache('old', capacity=2)
    producer = torch.ones(2, 3, requires_grad=True)
    first = cache.feature('a', lambda: producer)
    first.add_(3)
    producer.data.add_(5)
    hit = cache.feature('a', lambda: pytest.fail('cache hit recomputed'))
    assert torch.equal(hit, torch.ones(2, 3)) and not hit.requires_grad
    cache.feature('b', lambda: torch.zeros(2, 3))
    cache.feature('c', lambda: torch.zeros(2, 3))
    assert cache.stats()['entries'] == 2
    assert cache.stats()['evictions'] == 1
    assert cache.stats()['feature_bytes'] == 2*2*3*4
    cache.clear(provider_id='new')
    assert cache.stats()['entries'] == 0
    assert cache.stats()['provider_id'] == 'new'
    assert torch.equal(cache.feature('a', lambda: torch.full((2, 3), 9.)), torch.full((2, 3), 9.))


def test_disabled_cache_still_uses_per_image_execution_without_reuse():
    from fast_cluster.vision_cache import FrozenFrameCache, encode_frames
    cache = FrozenFrameCache('provider', enabled=False)
    provider = NativeVision()
    one = encode_frames(provider, native(range(4)), cache=cache, keys=list('abcd'))
    two = encode_frames(provider, native(range(4)), cache=cache, keys=list('abcd'))
    assert len(provider.calls) == 8
    assert all(torch.equal(a, b) for a, b in zip(one, two))
    assert cache.stats()['entries'] == 0 and cache.stats()['feature_hits'] == 0


@pytest.mark.parametrize('mutation', ['pixel_count', 'grid_shape', 'grid_dtype', 'grid_zero', 'keys', 'labels'])
def test_native_layout_and_input_only_contract_are_validated(mutation):
    from fast_cluster.vision_cache import FrozenFrameCache, encode_frames
    packed = native(range(4)); keys = list('abcd')
    if mutation == 'pixel_count': packed['pixel_values'] = packed['pixel_values'][:-1]
    if mutation == 'grid_shape': packed['image_grid_thw'] = torch.ones(4, 2, dtype=torch.long)
    if mutation == 'grid_dtype': packed['image_grid_thw'] = packed['image_grid_thw'].float()
    if mutation == 'grid_zero': packed['image_grid_thw'][0, 1] = 0
    if mutation == 'keys': keys = keys[:-1]
    if mutation == 'labels': packed['trajectory_gt'] = torch.zeros(8, 3)
    with pytest.raises(ValueError):
        encode_frames(NativeVision(), packed, cache=FrozenFrameCache('provider'), keys=keys)


def test_merged_native_output_token_shape_is_validated():
    from fast_cluster.vision_cache import encode_frames
    provider = NativeVision()
    provider.get_image_features = lambda *args, **kwargs: SimpleNamespace(pooler_output=(torch.zeros(2, 3),))
    with pytest.raises(ValueError, match='token'):
        encode_frames(provider, native([0]))


class BenchmarkFast:
    device = 'cpu'
    model_id = 'CPU benchmark provider'
    interface_info = {'image_frames': 4, 'backend': 'CPU benchmark contract'}

    def __init__(self, config):
        from fast_cluster.vision_cache import FrozenFrameCache
        self.frame_cache = FrozenFrameCache('CPU benchmark provider', enabled=config.get('cache_enabled', True))
        self.vision = NativeVision()

    def prepare(self, observation):
        from fast_cluster.vision_cache import encode_frames
        index = (observation['timestamp_us']-1_000_000)//500_000
        features = encode_frames(self.vision, native(range(index, index+4)),
                                 self.frame_cache, list(range(index, index+4)))
        return dict(observation=observation, vision=features, timings={'vision_s': 0.})

    def predict(self, prepared, snapshot, metadata):
        observation = prepared['observation']
        assert len(prepared['vision']) == 4
        if snapshot is not None:
            assert snapshot.metadata.source_time == observation['observation_time']
        plan = torch.full((1, 8, 3), float(observation['timestamp_us']))
        return plan, {'llm_head_s': 0., 'model_total_s': 0.}


class BenchmarkSlow:
    model_id = 'CPU benchmark provider'

    def prefill(self, observation, feedback):
        from tests_cluster.test_async_runtime import snapshot
        assert feedback.device.type == 'cpu' and not feedback.requires_grad
        assert (feedback == float(observation['timestamp_us'])).all()
        return snapshot()


def make_runtime_fast(config, checkpoint, device):
    return BenchmarkFast(config)


def make_runtime_slow(config, device):
    assert config['runtime_checkpoint'] == config['expected_runtime_checkpoint']
    return BenchmarkSlow()


def test_sync_benchmark_matches_observation_and_reuses_cached_fast_vision(tmp_path):
    from tools.benchmark_split_history import run_benchmark
    from tests_cluster.test_async_runtime import observation
    checkpoint = tmp_path/'same.pt'
    result = run_benchmark({'expected_runtime_checkpoint': str(checkpoint.resolve())}, checkpoint,
        [observation(i) for i in range(4)], tmp_path/'sync', mode='sync_fresh',
        fast_device='cpu', slow_device='cpu', allow_colocated_smoke=True,
        ticks=2, wall_timeout_s=5, backend_module='tests_cluster.test_vision_cache')
    assert result['planning_count'] == 2
    assert result['source_frequency_hz'] == pytest.approx(2., rel=.1)
    assert result['fast_cache_after']['feature_hits'] == 3
    assert result['fast_cache_after']['feature_misses'] == 5
    assert result['quality_evaluated'] is False
    assert all(row['snapshot_source_time'] == row['image_time'] for row in result['plans'])
    assert all(row['snapshot_ready_time'] <= row['selection_time'] for row in result['plans'])


def test_four_frame_baseline_benchmark_requires_enabled_cache(tmp_path):
    from tools.benchmark_split_history import run_benchmark
    from tests_cluster.test_async_runtime import observation
    with pytest.raises(ValueError, match='four-frame.*cache'):
        run_benchmark({'cache_enabled': False}, tmp_path/'same.pt',
            [observation(i) for i in range(4)], tmp_path/'baseline', mode='fast_only',
            fast_device='cpu', slow_device='cpu', ticks=2,
            backend_module='tests_cluster.test_vision_cache')
