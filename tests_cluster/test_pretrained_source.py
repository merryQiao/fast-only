"""New source modules learn through a frozen backbone, without a Mini teacher."""
from types import SimpleNamespace

import pytest
import torch

from fast_cluster.pretrained_source import PretrainedSource


class TinySlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.language_model = torch.nn.Linear(8, 8)
        self.backbone.visual = torch.nn.Linear(8, 8)
        self.ego_mlp = torch.nn.Linear(4, 8)
        self.command_embedding = torch.nn.Embedding(4, 8)
        self.queries = torch.nn.Parameter(torch.randn(8, 8))
        self.feedback_encoder = torch.nn.Linear(3, 8)
        self.feedback_type = torch.nn.Parameter(torch.randn(8))
        self.no_feedback = torch.nn.Parameter(torch.randn(8))
        self.head = torch.nn.Linear(8, 3)
        self.register_buffer('feedback_times', torch.arange(1, 9)*.5)

    def forward(self, native, ego, command, feedback=None):
        z = self.ego_mlp(ego)+self.command_embedding(command)+self.queries.mean(0)
        z = z + (self.no_feedback if feedback is None else
                 self.feedback_encoder(feedback.mean(1))+self.feedback_type)
        return SimpleNamespace(snapshot=self.backbone.language_model(z))


def test_new_source_inputs_learn_through_frozen_language_and_feedback_is_detached():
    source = PretrainedSource(TinySlow(), {'ego_mean': [0.]*4, 'ego_std': [1.]*4})
    feedback = torch.randn(2, 8, 3, requires_grad=True)
    snapshot = source({}, torch.ones(2, 4), torch.tensor([1, 2]), feedback)
    snapshot.square().mean().backward()
    assert feedback.grad is None
    assert all(not p.requires_grad and p.grad is None for p in source.slow.backbone.parameters())
    assert all(not p.requires_grad and p.grad is None for p in source.slow.head.parameters())
    report = source.gradient_report()
    assert report['finite'] and report['nonzero'] and report['tensors_with_gradient'] > 0
    assert source.slow.ego_mlp.weight.grad.abs().sum() > 0
    assert source.slow.queries.grad.abs().sum() > 0
    source.check_frozen()


def test_new_source_state_is_complete_and_independent_of_live_tensors():
    source = PretrainedSource(TinySlow(), {})
    saved = source.task_state()
    with torch.no_grad():
        source.slow.queries.add_(1)
    assert not torch.equal(saved['queries'], source.slow.queries)
    source.restore_task_state(saved)
    assert torch.equal(saved['queries'], source.slow.queries)
    with pytest.raises(ValueError):
        source.restore_task_state({})
    source.freeze_for_inference()
    assert all(not p.requires_grad for p in source.parameters())
    source.restore_task_state(saved)
    assert source.task_state().keys() == saved.keys()
