from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from fast_only.benchmark import measure


class CountingDataset:
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 2

    def __getitem__(self, index):
        self.reads += 1
        return {'images': [Image.new('RGB', (128, 72)) for _ in range(4)], 'index': index}


class CountingCollator:
    def __init__(self):
        self.calls = 0

    def __call__(self, samples):
        self.calls += 1
        return {'native': {'pixel_values': torch.ones(4, 1),
                           'image_grid_thw': torch.tensor([[1, 4, 8]] * 4),
                           'input_ids': torch.ones(1, 6, dtype=torch.long)},
                'ego': torch.ones(1, 3), 'command': torch.ones(1, dtype=torch.long),
                'nav': torch.ones(1, 2)}


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.visual = torch.nn.Identity()
        self.backbone.visual.spatial_merge_size = 2
        self.backbone.language_model = torch.nn.Identity()
        self.head = torch.nn.Identity()
        self.last_layout = {'image_tokens_per_frame': [8] * 4}
        self.calls = 0

    def forward(self, native, ego, command, nav):
        self.calls += 1
        return self.head(self.backbone.language_model(self.backbone.visual(native['pixel_values'])))


@pytest.mark.parametrize('mode,reads,collations', [('model_only', 1, 1),
                                                  ('streaming_e2e', 5, 5),
                                                  ('disk_e2e', 5, 5)])
def test_modes_measure_full_model_with_distinct_preprocessing_boundaries(mode, reads, collations):
    dataset = CountingDataset(); collator = CountingCollator(); model = CountingModel()
    result = measure(model, dataset, collator, 'cpu', warmup=2, repeats=3,
                     modes=[mode], profile_components=False)
    timing = result[mode + '_ms']
    assert len(timing['samples']) == 3
    assert set(('p50', 'p95', 'p99', 'mean', 'samples')) <= timing.keys()
    assert dataset.reads == reads and collator.calls == collations
    assert model.calls == 5
    assert model.training  # benchmark restores original mode
    assert result['input_audit']['raw_image_sizes_wh'] == [[128, 72]] * 4
    assert result['input_audit']['image_grid_thw'] == [[1, 4, 8]] * 4
    assert result['input_audit']['image_tokens_per_frame'] == [8] * 4
    if mode == 'disk_e2e':
        assert result['end_to_end_ms'] == timing


def test_all_modes_default_and_separate_component_profile():
    dataset = CountingDataset(); collator = CountingCollator(); model = CountingModel().eval()
    result = measure(model, dataset, collator, 'cpu', warmup=1, repeats=2)
    assert all(name + '_ms' in result for name in ['model_only', 'streaming_e2e', 'disk_e2e'])
    assert model.calls == 10  # nine timed/warmup full passes plus one separate profile
    assert set(result['component_ms_single_profile_pass']) == {'vision', 'backbone', 'head'}
    assert not model.training
    assert result['repeats'] == 2
    assert result['quantile_warning']


def test_streaming_excludes_disk_time_but_disk_mode_includes_it(monkeypatch):
    # A logical clock proves exact boundaries without wall-clock tolerances.
    import fast_only.benchmark as benchmark
    clock = SimpleNamespace(value=0.)
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock.value)
    class TimedDataset(CountingDataset):
        def __getitem__(self, index):
            clock.value += .010
            return super().__getitem__(index)
    class TimedCollator(CountingCollator):
        def __call__(self, samples):
            clock.value += .020
            return super().__call__(samples)
    class TimedModel(CountingModel):
        def forward(self, *args):
            clock.value += .030
            return super().forward(*args)
    result = measure(TimedModel(), TimedDataset(), TimedCollator(), 'cpu',
                     warmup=1, repeats=2, profile_components=False)
    assert result['model_only_ms']['mean'] == pytest.approx(30)
    assert result['streaming_e2e_ms']['mean'] == pytest.approx(50)
    assert result['disk_e2e_ms']['mean'] == pytest.approx(60)


def test_model_mode_restored_after_exception():
    model = CountingModel()
    class BrokenCollator:
        def __call__(self, samples):
            raise RuntimeError('processor failed')
    with pytest.raises(RuntimeError, match='processor failed'):
        measure(model, CountingDataset(), BrokenCollator(), 'cpu', warmup=1, repeats=1)
    assert model.training


@pytest.mark.parametrize('kwargs', [dict(warmup=0), dict(repeats=0), dict(modes=['wrong']), dict(modes=[])])
def test_invalid_benchmark_settings_fail(kwargs):
    with pytest.raises(ValueError):
        measure(CountingModel(), CountingDataset(), CountingCollator(), 'cpu', **kwargs)
