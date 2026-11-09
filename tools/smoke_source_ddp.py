"""Exercise source dropout across no_sync boundaries with actual Gloo ranks."""
from datetime import timedelta
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from fast_cluster.distributed import EpochPlan, valid_scaled_loss


class ConditionalPlanner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fast = torch.nn.Linear(1, 1)
        self.source_input = torch.nn.Linear(1, 1)
        self.source_body = torch.nn.Linear(1, 1).requires_grad_(False)

    def forward(self, x, source_on):
        with torch.no_grad():
            candidate = self.fast(x).detach()
        output = self.fast(x)
        if source_on:
            output = output + self.source_body(self.source_input(x+candidate))
        return output


def _enabled(update, micro, rank):
    return update == 0 and ((micro == 0 and rank == 0) or (micro == 1 and rank == 1))


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.random.default_generator.manual_seed(7)
        model = ConditionalPlanner()
        reference = ConditionalPlanner()
        reference.load_state_dict(model.state_dict())
        wrapped = DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False)
        params = [p for p in model.parameters() if p.requires_grad]
        ref_params = [p for p in reference.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=.01, weight_decay=.02)
        ref_optimizer = torch.optim.SGD(ref_params, lr=.01, weight_decay=.02)
        plans = [EpochPlan(13, 2, 2, r, 2, 42, 0) for r in range(2)]
        groups = [list(p) for p in plans]
        for update, group in enumerate(groups[rank]):
            optimizer.zero_grad(set_to_none=True)
            for micro, rows in enumerate(group.microbatches):
                x = torch.tensor([[i/10.+.1] for i,_ in rows])
                valid = torch.tensor([v for _,v in rows])
                context = wrapped.no_sync() if micro+1 < len(group.microbatches) else __import__('contextlib').nullcontext()
                with context:
                    loss = (wrapped(x,_enabled(update,micro,rank)).flatten()-1.).square()
                    valid_scaled_loss(loss,valid,world_size=2,global_valid_count=group.valid_count).backward()
            optimizer.step()
            ref_optimizer.zero_grad(set_to_none=True)
            losses = []
            for r in range(2):
                for micro, rows in enumerate(groups[r][update].microbatches):
                    x = torch.tensor([[i/10.+.1] for i,_ in rows])
                    valid = torch.tensor([v for _,v in rows])
                    losses.append(((reference(x,_enabled(update,micro,r)).flatten()-1.).square()*valid).sum())
            (sum(losses)/group.valid_count).backward()
            ref_optimizer.step()
            for actual, expected in zip(params,ref_params):
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
                copy = actual.detach().clone()
                dist.broadcast(copy,0)
                assert torch.equal(actual,copy)
            assert all(p.grad is None for p in model.source_body.parameters())
        if rank == 0:
            with open(output,'w') as stream:
                json.dump(dict(world_size=2,backend='gloo',updates=2,valid_samples=13,
                    source_on_off_no_sync_matches_global_reference=True,
                    all_source_off_update_matches_reference=True,
                    frozen_body_has_no_gradients=True,parameters_synchronized=True),stream)
    finally:
        dist.destroy_process_group()



def run_smoke(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/'result.json'
    mp.spawn(_worker,args=(str(output/'rendezvous'),str(receipt)),nprocs=2,join=True)
    return json.loads(receipt.read_text())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(run_smoke(args.output),indent=2))
