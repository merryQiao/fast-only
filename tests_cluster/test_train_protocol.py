"""Configuration, resume identity, and evaluation protocol tests."""
import copy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_config_rejects_formal_small_topology_and_unresolved_assets(monkeypatch):
    from fast_cluster.config import read_config, validate_config, validate_topology

    monkeypatch.delenv('FAST_BASE_MODEL', raising=False)
    with pytest.raises(ValueError, match='unresolved'):
        read_config(ROOT / 'configs/navsim_v1_h200_2x8.yaml')
    config = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    formal = copy.deepcopy(config)
    formal['execution']['mode'] = 'formal'
    with pytest.raises(ValueError, match='16.*8'):
        validate_config(formal)
    with pytest.raises(ValueError, match='topology'):
        validate_topology(config, world_size=2, local_world_size=2)


def test_resume_identity_allows_relocation_but_rejects_training_change():
    from fast_cluster.config import read_config, semantic_config
    from fast_cluster.distributed import verify_identity

    config = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    moved = copy.deepcopy(config)
    moved['assets']['fast_model'] = '/new/fast'
    moved['assets']['slow_model'] = '/new/slow'
    moved['assets']['source_bundle'] = '/new/source.pt'
    moved['data']['bundle'] = '/new/manifest'
    moved['data']['sensor_root'] = '/new/sensors'
    moved['data']['workers'] = 9
    moved['data']['prefetch_factor'] = 8
    assert semantic_config(moved) == semantic_config(config)
    identity = {'semantic_config': semantic_config(config), 'model': {'sha': 'same'},
                'data': {'sha': 'same'}, 'world_size': 1}
    verify_identity(identity, {**identity, 'semantic_config': semantic_config(moved)})
    moved['train']['epochs'] = 2
    with pytest.raises(ValueError, match='identity'):
        verify_identity(identity, {**identity, 'semantic_config': semantic_config(moved)})
    with pytest.raises(ValueError, match='world'):
        verify_identity(identity, {**identity, 'world_size': 2})


def test_validation_policy_is_mode_specific_and_formal_is_explicit():
    from fast_cluster.config import read_config, validate_config

    smoke = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    smoke['data']['validation_policy'] = 'exclude_upstream_training_groups'
    with pytest.raises(ValueError, match='validation_policy'):
        validate_config(smoke)
    smoke['data']['validation_policy'] = 'inherited_smoke_only'
    validate_config(smoke)
    formal = copy.deepcopy(smoke)
    formal['execution'].update(mode='formal', expected_world_size=16, expected_local_world_size=8)
    formal['train']['epochs'] = 8
    with pytest.raises(ValueError, match='validation_policy'):
        validate_config(formal)
    formal['data']['validation_policy'] = 'exclude_upstream_training_groups'
    validate_config(formal)
    formal['data'].pop('validation_policy')
    with pytest.raises(ValueError, match='validation_policy'):
        validate_config(formal)


def test_per_sample_loss_retains_official_unweighted_pose_l1():
    from fast_cluster.train import per_sample_pose_loss

    prediction = torch.arange(48.).reshape(2, 8, 3)
    target = torch.zeros_like(prediction)
    assert torch.equal(per_sample_pose_loss(prediction, target), prediction.mean((1, 2)))
    with pytest.raises(ValueError, match='8,3'):
        per_sample_pose_loss(prediction[:, :7], target[:, :7])


def test_eval_uses_unwrapped_backend_and_rank_sharding_without_padding():
    from fast_cluster.train import evaluate

    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []
        def prepare(self, samples):
            self.calls.extend(samples)
            return {'trajectory_gt': torch.zeros(len(samples), 8, 3), 'samples': samples}
        def forward(self, batch):
            return torch.ones(len(batch['samples']), 8, 3)
    backend = Backend()
    metrics = evaluate(backend, list(range(5)), rank=1, world_size=2,
                       micro_batch=2, device=torch.device('cpu'), reduce=False)
    assert backend.calls == [1, 3]
    assert metrics['samples'] == 2
    assert metrics['ADE'] == pytest.approx(2 ** .5)
    assert metrics['FDE'] == pytest.approx(2 ** .5)


def test_streamed_eval_horizon_and_wrapped_heading_metrics_match_existing_navsim():
    from fast_cluster.train import evaluate
    from fast_navsim.model import trajectory_metrics

    prediction = torch.arange(48.).reshape(2, 8, 3) / 10
    target = torch.zeros_like(prediction)
    prediction[..., 2] = 3.1
    target[..., 2] = -3.1
    prediction[1, -1, 2], target[1, -1, 2] = -3.12, 3.11
    class Backend(torch.nn.Module):
        def prepare(self, samples):
            return {'indices': samples, 'target': target[samples]}
        def forward(self, batch):
            return prediction[batch['indices']]
    actual = evaluate(Backend(), [0, 1], rank=0, world_size=1,
                      micro_batch=1, device='cpu', reduce=False)
    expected = trajectory_metrics(prediction, target)
    for metric in ('ADE', 'FDE', 'loss_l1_pose', 'heading_mae_rad', 'heading_final_mae_rad',
                   *[f'L2@{(i + 1) * .5:g}s' for i in range(8)]):
        assert actual[metric] == pytest.approx(expected[metric], rel=1e-6, abs=1e-7), metric
    assert actual['samples'] == expected['samples'] == 2
    assert actual['heading_mae_rad'] < .1


def test_training_update_weights_partial_group_and_synchronizes_only_final_micro():
    from contextlib import contextmanager
    from fast_cluster.distributed import AccumulationGroup
    from fast_cluster.train import perform_update

    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.no_sync_calls = 0
        def prepare(self, samples):
            return {'x': torch.tensor([s[0] for s in samples]),
                    'valid': torch.tensor([s[1] for s in samples]),
                    'target': torch.ones(len(samples), 8, 3) * 100}
        def forward(self, batch):
            return (self.weight * batch['x']).reshape(-1, 1, 1).expand(-1, 8, 3)
        def check_frozen(self):
            return True
        @contextmanager
        def no_sync(self):
            self.no_sync_calls += 1
            yield
    backend = Backend()
    optimizer = torch.optim.SGD(backend.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    group = AccumulationGroup(0, (((0, True), (1, True)), ((2, True), (0, False))), 3)
    samples = iter([[(1., True), (2., True)], [(3., True), (1000., False)]])
    row = perform_update(backend, backend, optimizer, scheduler, samples, group,
                         world_size=1, gradient_clip=100., device='cpu')
    assert backend.weight.item() == pytest.approx(2.)
    assert backend.no_sync_calls == 1
    assert row['loss_l1_pose'] == pytest.approx(100.)


def test_formal_explicit_smoke_stop_is_rejected_before_initialization():
    from fast_cluster.config import read_config
    from fast_cluster.train import run_training

    config = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    config['execution'].update(mode='formal', expected_world_size=16, expected_local_world_size=8)
    config['train']['epochs'] = 8
    config['data']['validation_policy'] = 'exclude_upstream_training_groups'
    with pytest.raises(ValueError, match='only in Smoke'):
        run_training(config, arm='fast_only', seed=42, output='unused', stop_after_updates=1)


def test_cpu_tiny_trainer_epoch_resume_and_frozen_identity(tmp_path, monkeypatch):
    """Exercise actual trainer/checkpoint control flow without claiming native model coverage."""
    import sys
    from types import SimpleNamespace
    from fast_cluster.config import read_config
    from fast_cluster import train as trainer

    class Dataset:
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, index):
            index, valid = index if isinstance(index, tuple) else (index, True)
            return dict(index=index, valid=valid)
        def row(self, index):
            return dict(token=f'{index:04d}')
    class Data:
        identity = {'tiny_data': 'five_unique_train_two_val'}
        stats = {}
        def __init__(self, *args, **kwargs):
            pass
        def dataset(self, split):
            return Dataset(5 if split == 'train' else 2)
    class Backend(torch.nn.Module):
        model_identity = {'tiny_model': True}
        mutate_frozen = False
        def __init__(self):
            super().__init__()
            self.fast = torch.nn.Linear(1, 24)
            self.audit_calls = 0
        def prepare(self, samples):
            return dict(x=torch.tensor([[s['index'] + 1.] for s in samples]),
                        target=torch.zeros(len(samples), 8, 3),
                        valid=torch.tensor([s['valid'] for s in samples]))
        def forward(self, batch):
            return self.fast(batch['x']).reshape(-1, 8, 3)
        def check_frozen(self):
            return True
        def audit_frozen(self):
            self.audit_calls += 1
            return {'frozen_identity': self.audit_calls if self.mutate_frozen else 'constant'}
        def diagnostics(self):
            return {'validation_scope': 'CPU tiny control flow only'}
        def fast_state(self):
            return {key: value.clone() for key, value in self.fast.state_dict().items()}
        def restore_fast_state(self, state):
            self.fast.load_state_dict(state)
        def set_eval_source(self, mode):
            pass
    source_guard = dict(calls=0, reject=False)
    def validate_fake_source(config, data):
        source_guard['calls'] += 1
        if source_guard['reject']:
            raise ValueError('untrusted fake source provenance')
    def build(*args):
        assert source_guard['calls'] > 0, 'source/data provenance must be checked before building models'
        torch.random.default_generator.manual_seed(42)
        return Backend()
    monkeypatch.setitem(sys.modules, 'fast_cluster.data', SimpleNamespace(PreparedData=Data))
    monkeypatch.setitem(sys.modules, 'fast_cluster.models', SimpleNamespace(build_backend=build))
    monkeypatch.setitem(sys.modules, 'fast_cluster.source_contract',
                        SimpleNamespace(validate_source_data=validate_fake_source))
    monkeypatch.setitem(sys.modules, 'tools.train_direct_full', SimpleNamespace(gradient_audit=lambda model: {'tiny': True}))
    import slowfast_v11.full_training as full_training
    monkeypatch.setattr(full_training, 'make_full_optimizer',
                        lambda model, **kwargs: torch.optim.AdamW(model.parameters(), lr=1e-3))
    monkeypatch.setattr(trainer, 'initialize', lambda config: SimpleNamespace(
        device=torch.device('cpu'), rank=0, world_size=1, local_rank=0))
    for name in ('reset_peak_memory_stats', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)
    config = read_config(ROOT / 'configs/navsim_v1_smoke.yaml')
    interrupted, control = tmp_path / 'interrupted', tmp_path / 'control'
    trainer.run_training(config, arm='fast_only', seed=42, output=interrupted, stop_after_updates=1)
    first = torch.load(interrupted / 'latest.pt', weights_only=False)
    assert first['progress']['sampler']['cursor'] == 2
    resumed = trainer.run_training(config, arm='fast_only', seed=42, output=interrupted,
                                    resume=interrupted / 'latest.pt')
    trainer.run_training(config, arm='fast_only', seed=42, output=control)
    actual = torch.load(interrupted / 'latest.pt', weights_only=False)
    expected = torch.load(control / 'latest.pt', weights_only=False)
    assert resumed['completed'] and resumed['windows_seen'] == 5
    assert actual['progress'] == expected['progress']
    assert all(torch.equal(actual['model'][key], value) for key, value in expected['model'].items())
    formal = copy.deepcopy(config)
    formal['execution'].update(mode='formal', expected_world_size=16, expected_local_world_size=8)
    formal['train']['epochs'] = 8
    formal['data']['validation_policy'] = 'exclude_upstream_training_groups'
    preflight = trainer.run_training(formal, arm='fast_only', seed=42,
                                     output=tmp_path / 'preflight', preflight_updates=1)
    assert not preflight['completed'] and preflight['stage'] == 'preflight'
    assert preflight['preflight_partial_val']['scope'] == 'preflight_partial_val'
    assert preflight['preflight_partial_val']['samples'] == 2
    assert preflight['best_val_ADE'] is None
    assert not (tmp_path / 'preflight' / 'best_weights.pt').exists()
    boundary = trainer.run_training(formal, arm='fast_only', seed=42,
                                    output=tmp_path / 'preflight_boundary', preflight_updates=3)
    assert not boundary['completed'] and boundary['equivalent_epochs'] == 1.
    assert boundary['best_val_ADE'] is None
    assert boundary['preflight_partial_val']['used_for_model_selection'] is False
    assert not (tmp_path / 'preflight_boundary' / 'best_weights.pt').exists()
    import json
    boundary_rows = [json.loads(line) for line in
                     (tmp_path / 'preflight_boundary' / 'train.jsonl').read_text().splitlines()]
    assert not any('val' in row for row in boundary_rows)
    boundary_state = torch.load(tmp_path / 'preflight_boundary' / 'latest.pt', weights_only=False)
    assert boundary_state['progress']['sampler']['cursor'] == 5
    assert boundary_state['progress']['best_val_ADE'] is None
    Backend.mutate_frozen = True
    with pytest.raises(ValueError, match='Frozen.*changed'):
        trainer.run_training(config, arm='fast_only', seed=42, output=tmp_path / 'mutated', stop_after_updates=1)
    source_guard['reject'] = True
    with pytest.raises(ValueError, match='untrusted fake source provenance'):
        trainer.run_training(formal, arm='fast_only', seed=42,
                             output=tmp_path / 'untrusted_source', preflight_updates=1)
