"""Score historical Fast/fixed-adapter references on the newly held-out logs."""
import argparse
import json
from pathlib import Path

import torch

from fast_only.common import apply_reproducibility,fingerprint,write_json
from fast_navsim.model import trajectory_metrics
from slowfast_v11.fixed_adapter import load_fixed_adapter
from slowfast_v11.validation import canonical_state,state_digest
from tools.cache_direct_full import DirectCache
from tools.direct_full_common import read_config,source_info,load_fast,direct_forward


@torch.no_grad()
def run(config,cache_root,output):
    c=read_config(config);out=Path(output)
    if out.exists():raise FileExistsError('Use a new reference evaluation destination')
    source,_,_=source_info(c);apply_reproducibility(source['config']);torch.set_num_threads(4)
    cache=DirectCache(cache_root,config);fast,_,_,_=load_fast(c);fast.eval().requires_grad_(False)
    before=state_digest(canonical_state(fast));parent=Path(c['adapter_run']);result=json.loads((parent/'result.json').read_text())
    adapters={}
    for seed in [42,43,44]:
        item=result['fixed_bundles'][str(seed)]
        adapters[seed]=load_fixed_adapter(item['path'],expected_sha256=item['sha256'],
            fast_checkpoint_sha256=c['fast_checkpoint_sha256'],head=fast.head,map_location='cuda:0')
    records=[]
    for n,row in enumerate(cache.val,1):
        packet=cache.packet(row['token']);captured=[]
        hook=fast.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        try:baseline=direct_forward(fast,packet,'full_fast_only')
        finally:hook.remove()
        if len(captured)!=1 or not torch.equal(baseline.float().cpu(),packet['original_prediction'].cpu()):raise ValueError('Historical Fast native cache prediction differs')
        predictions={'original_fast':baseline}
        predictions.update({f'historical_fixed_seed_{seed}':fast.head(adapter(captured[0])) for seed,adapter in adapters.items()})
        for name,pred in predictions.items():
            records.append(dict(token=row['token'],route=row['route'],method=name,prediction=pred[0].float().cpu().tolist(),target=row['trajectory_gt']))
        if n%64==0:print(f'historical fixed-adapter reference {n}/256',flush=True)
    if state_digest(canonical_state(fast))!=before:raise ValueError('Historical frozen Fast changed')
    metrics={name:trajectory_metrics(torch.tensor([r['prediction'] for r in records if r['method']==name]),
        torch.tensor([r['target'] for r in records if r['method']==name])) for name in predictions}
    write_json(out,dict(records=records,metrics=metrics,cache_manifest_sha256=cache.fingerprint,
        adapter_source_result_sha256=fingerprint(parent/'result.json'),bundle_identities=result['fixed_bundles'],
        caveat='historical TRAIN256 adapters; previous training/data budget differs; score reference only',
        new_slow_or_reader_calls=0,source_code_sha256=fingerprint(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_direct_full.yaml')
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.config,a.cache,a.output)


if __name__=='__main__':main()
