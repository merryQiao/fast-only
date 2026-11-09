"""Train only new source conditions through an unchanged pretrained backbone."""
import torch
from torch import nn


class PretrainedSource(nn.Module):
    TASK_NAMES = ('queries', 'feedback_type', 'no_feedback')
    TASK_PREFIXES = ('ego_mlp.', 'command_embedding.', 'feedback_encoder.')

    def __init__(self, slow, stats):
        super().__init__()
        self.slow, self.stats = slow, stats
        slow.requires_grad_(False)
        self.task_names = tuple(name for name, _ in slow.named_parameters()
            if name in self.TASK_NAMES or name.startswith(self.TASK_PREFIXES))
        if not self.task_names:
            raise ValueError('No declared new Slow input/query modules were found')
        parameters = dict(slow.named_parameters())
        for name in self.task_names:
            parameters[name].data = parameters[name].data.float()
            parameters[name].requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained vision/language do not acquire training-time state.
        self.slow.backbone.eval()
        self.slow.head.eval()
        return self

    def forward(self, native, normalized_ego, command, feedback=None):
        # no_grad here would sever task encoder/Query learning through the
        # frozen language model. Only the already completed plan is detached.
        feedback = None if feedback is None else feedback.detach()
        return self.slow(native, normalized_ego, command, feedback=feedback).snapshot

    def check_frozen(self):
        for name, parameter in self.slow.named_parameters():
            if name not in self.task_names and (parameter.requires_grad or parameter.grad is not None):
                raise ValueError(f'Frozen Slow backbone/unused head changed gradient scope: {name}')
        return True

    def task_state(self):
        parameters = dict(self.slow.named_parameters())
        return {name: parameters[name].detach().cpu().clone() for name in self.task_names}

    def restore_task_state(self, state):
        parameters = dict(self.slow.named_parameters())
        if not isinstance(state, dict) or set(state) != set(self.task_names):
            raise ValueError('Missing or incompatible trained Slow input/query task state')
        for name in self.task_names:
            value, parameter = state[name], parameters[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f'Slow task state shape/dtype differs: {name}')
        with torch.no_grad():
            for name in self.task_names:
                parameters[name].copy_(state[name])

    def freeze_for_inference(self):
        self.eval().requires_grad_(False)
        return self

    def gradient_report(self):
        parameters = dict(self.slow.named_parameters())
        gradients = [parameters[name].grad for name in self.task_names if parameters[name].grad is not None]
        return dict(tensors_with_gradient=len(gradients),
                    finite=all(bool(torch.isfinite(g).all()) for g in gradients),
                    nonzero=any(bool(torch.count_nonzero(g)) for g in gradients),
                    backbone_frozen=self.check_frozen(),
                    total_trainable_parameters=sum(parameters[name].numel() for name in self.task_names))
