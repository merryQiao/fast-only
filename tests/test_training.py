import random
import json

import numpy as np
import pytest
import torch

from fast_only.train import save_checkpoint, restore_checkpoint, build_scheduler, training_stop_step, training_accounting
from fast_only.common import experiment_fingerprints, verify_checkpoint_inputs


class TinyModel(torch.nn.Linear):
    def trainable_state(self):
        return self.state_dict()

    def load_trainable_state(self, state):
        self.load_state_dict(state)


def test_checkpoint_restores_optimizer_scheduler_and_all_rng(tmp_path):
    model = TinyModel(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = build_scheduler(optimizer, 20, .1)
    model(torch.ones(1, 2)).sum().backward(); optimizer.step(); scheduler.step()
    path = tmp_path / 'checkpoint.pt'
    config = {'seed': 42}
    save_checkpoint(path, model, optimizer, scheduler, config, 1, 3., {'manifest': 'abc'})
    expected = (random.random(), np.random.rand(), torch.rand(3))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad(): model.weight.add_(2)
    state = restore_checkpoint(path, model, optimizer, scheduler)
    assert state['step'] == 1 and state['config'] == config
    assert state['data_fingerprints']['manifest'] == 'abc'
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert all(torch.equal(model.state_dict()[k], v) for k, v in original.items())
    assert scheduler.last_epoch == 1 and optimizer.state_dict()['state']


def test_scheduler_warmup_and_cosine_end():
    model = TinyModel(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.)
    scheduler = build_scheduler(optimizer, 10, .2)
    rates = [optimizer.param_groups[0]['lr']]
    for _ in range(10):
        optimizer.step(); scheduler.step(); rates.append(optimizer.param_groups[0]['lr'])
    assert rates[0] < rates[1]
    assert rates[1] == 1.
    assert rates[-1] == 0.


def test_resume_rejects_earlier_stop_step():
    with pytest.raises(ValueError, match='restored'):
        training_stop_step(200, 100, 50)
    assert training_stop_step(200, 100, 150) == 150


def test_reloading_detects_changed_base_weights_and_embeds_stats(tmp_path):
    model_path = tmp_path / 'base'; model_path.mkdir()
    (model_path/'config.json').write_text('{}')
    (model_path/'weights.safetensors').write_bytes(b'original base weights')
    manifest = tmp_path/'train.jsonl'; manifest.write_text('{}\n')
    stats = tmp_path/'stats.json'; stats.write_text('{"ego_mean":[1,2,3]}')
    cfg = {'model': {'path': str(model_path)},
           'data': {'train_manifest': str(manifest), 'stats': str(stats)}}
    fingerprints = experiment_fingerprints(cfg)
    model = TinyModel(2, 2); optimizer = torch.optim.AdamW(model.parameters())
    scheduler = build_scheduler(optimizer, 10, .03)
    path = tmp_path/'state.pt'
    save_checkpoint(path, model, optimizer, scheduler, cfg, 1, 1., fingerprints)
    state = torch.load(path, weights_only=False)
    assert state['normalization_stats'] == json.loads(stats.read_text())
    verify_checkpoint_inputs(state)
    (model_path/'weights.safetensors').write_bytes(b'different base weights')
    with pytest.raises(ValueError, match='fingerprint'):
        verify_checkpoint_inputs(state)


def test_window_accounting_counts_accumulated_microbatches():
    assert training_accounting(4, 2, 8, 100) == {'windows_seen': 64, 'epoch_equivalent': .64}
    assert training_accounting(200, 1, 1, 8) == {'windows_seen': 200, 'epoch_equivalent': 25.}


def test_checkpoint_preserves_consumed_prefetch_cursor(tmp_path):
    model = TinyModel(2, 2); optimizer = torch.optim.AdamW(model.parameters())
    scheduler = build_scheduler(optimizer, 10, .03)
    sampler = {'sampler_version': 'counter_v1', 'seed': 42, 'next_batch': 32,
               'microbatch': 2, 'dataset_size': 100}
    path = tmp_path/'cursor.pt'
    save_checkpoint(path, model, optimizer, scheduler, {'seed': 42}, 4, 1., {},
                    sampler_state=sampler, windows_seen=64)
    restored = restore_checkpoint(path, model, optimizer, scheduler)
    assert restored['sampler_state'] == sampler
    assert restored['windows_seen'] == 64
