"""Paired source interventions on the fixed correct-KV training endpoint."""
import argparse
import json
from pathlib import Path

import torch

from fast_only.common import fingerprint,write_json
from fast_navsim.model import trajectory_metrics
from slowfast_v11.full_training import full_precision,restore_full_state
from slowfast_v11.validation import state_digest
from tools.direct_full_common import direct_forward,equal_source_layout
from tools.train_direct_full import setup,load_result_checkpoint,prediction_record,FIXED_SOURCE


@torch.no_grad()
def run(config,cache_root,init,run_root,output):
    root=Path(run_root);out=Path(output)
    if out.exists():raise FileExistsError('Use a new diagnostic destination')
    model,cache,c,identity,_,_=setup(config,cache_root,init)
    result=json.loads((root/'result.json').read_text());verified=json.loads((root/'verification.json').read_text())
    if result['arm']!='full_correct_kv' or result['identity']!=identity or not verified['passed']:raise ValueError('Correct-KV fixed endpoint not verified')
    final=load_result_checkpoint(root,'final.pt',result,'checkpoint_sha256')
    if state_digest(final['model'])!=result['model_digest']:raise ValueError('Complete model digest differs')
    restore_full_state(model,final['model']);del final
    expected={r['token']:torch.tensor(r['prediction']) for r in json.loads((root/'final_predictions.json').read_text())}
    if fingerprint(root/'final_predictions.json')!=verified['predictions_sha256']:raise ValueError('Verified endpoint predictions changed')
    groups={route:[r for r in cache.val if r['route']==route] for route in sorted({r['route'] for r in cache.val})}
    routes=list(groups)
    if len(routes)!=8 or any(len(v)!=32 for v in groups.values()):raise ValueError('Expected fixed8x32 validation groups')
    donors={row['token']:groups[routes[(i+4)%8]][j]['token'] for i,route in enumerate(routes) for j,row in enumerate(groups[route])}
    model.eval();model.backbone.visual.eval();records=[];exact=0
    with full_precision(model):
        for n,row in enumerate(cache.val,1):
            token=row['token'];packet=cache.packet(token);correct=cache.source(token)
            for mode in ['correct','off','reference','shuffled']:
                source_token=token if mode=='correct' else None if mode=='off' else FIXED_SOURCE if mode=='reference' else donors[token]
                source=None if source_token is None else correct if source_token==token else cache.source(source_token)
                if source is not None:equal_source_layout(correct,source)
                prediction=direct_forward(model,packet,'full_fast_only' if source is None else 'full_correct_kv',source)
                record=prediction_record(row,prediction,'full_correct_kv','val',source_token);record['mode']=mode;records.append(record)
                if mode=='correct':
                    if not torch.equal(prediction[0].float().cpu(),expected[token]):raise ValueError('Correct intervention fails exact independent endpoint reproduction')
                    exact+=1
                del source,prediction
            del packet,correct
            if n%64==0:print(f'correct-checkpoint source interventions {n}/256',flush=True)
    metrics={mode:trajectory_metrics(torch.tensor([r['prediction'] for r in records if r['mode']==mode]),
        torch.tensor([r['target'] for r in records if r['mode']==mode])) for mode in ['correct','off','reference','shuffled']}
    write_json(out,dict(passed=True,metrics=metrics,records=records,donors=donors,correct_endpoint_exact_count=exact,
        checkpoint_sha256=result['checkpoint_sha256'],cache_sha256=cache.fingerprint,source_code_sha256=fingerprint(__file__),
        caveat='same-checkpoint diagnostics; off also changes the no-cache metadata/native path; primary is separately trained matched-budget arms'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_direct_full.yaml');p.add_argument('--cache',default='outputs/slowfast_direct_full/cache_01')
    p.add_argument('--init',default='outputs/slowfast_direct_full/p0_01/shared_init.pt');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.cache,a.init,a.run,a.output)


if __name__=='__main__':main()
