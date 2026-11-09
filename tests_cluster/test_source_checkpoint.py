import pytest
import torch

from fast_cluster.distributed import save_checkpoint, load_checkpoint, save_best_weights


class Backend:
    def __init__(self):
        self.fast = torch.nn.Linear(1, 1)
        self.source = torch.nn.Linear(1, 1)
    def fast_state(self):
        return {k: v.detach().clone() for k,v in self.fast.state_dict().items()}
    def restore_fast_state(self, state):
        self.fast.load_state_dict(state)
    def source_state(self):
        return {k: v.detach().clone() for k,v in self.source.state_dict().items()}
    def restore_source_state(self, state):
        if state is None:
            raise ValueError('Missing trained source task state')
        self.source.load_state_dict(state)


def test_complete_and_best_checkpoints_keep_the_trained_source(tmp_path):
    backend = Backend()
    optimizer = torch.optim.AdamW([*backend.fast.parameters(), *backend.source.parameters()], lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    (backend.fast(torch.ones(1,1))+backend.source(torch.ones(1,1))).square().sum().backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    source = backend.source_state()
    identity = {'world_size': 1, 'protocol': 'pretrained_source_inputs'}
    path = tmp_path/'update_000000001.pt'
    save_checkpoint(path, backend, optimizer, scheduler, identity=identity,
                    progress={'update': 1}, device='cpu')
    save_best_weights(tmp_path/'best.pt', backend, identity=identity, metrics={'ADE': 1.})
    for file in (path, tmp_path/'best.pt'):
        state = torch.load(file, weights_only=False)
        assert all(torch.equal(state['source_tasks'][k], v) for k,v in source.items())
    with torch.no_grad():
        backend.source.weight.add_(4)
    load_checkpoint(path, backend, optimizer, scheduler, identity=identity, device='cpu')
    assert all(torch.equal(backend.source_state()[k], v) for k,v in source.items())
