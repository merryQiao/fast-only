"""Independent full-holdout reload and six actual final optimizer updates."""
import argparse
import json
from pathlib import Path

import torch

from fast_only.common import apply_reproducibility,fingerprint,write_json
from fast_navsim.model import pose_loss
from slowfast_v11.expanded_protocol import epoch_plan,WarmupCosineScheduler
from slowfast_v11.reader_fusion import ResidualFusion
from tools.diagnose_slowfast_readability import state_cpu
from tools.train_reader_fusion import make_head,evaluate,cached_forward
from tools.train_slowfast_joint import compare_trees,capture_rng,restore_rng


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh verification destination')
    result=json.loads((root/'result.json').read_text());p=result['protocol'];c=p['config'];features=result['features']
    for path,digest in [(root/'frozen_features.pt',features['feature_bank_sha256']),(root/'rows.json',features['rows_sha256']),
                        (Path(c['parent_run'])/'frozen_features.pt',c['old_feature_bank_sha256']),(Path(c['parent_run'])/'rows.json',c['old_rows_sha256'])]:
        if fingerprint(path)!=digest:raise ValueError('Authenticated current/training features differ')
    original=json.loads((Path(p['parent_protocol']['config']['source_run'])/'result.json').read_text())
    apply_reproducibility(original['config']);torch.set_num_threads(4)
    bank=torch.load(root/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    training_bank=torch.load(Path(c['parent_run'])/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    rows=json.loads((root/'rows.json').read_text());held=[r for r in rows if r['cohort']=='holdout']
    train_rows=[r for r in rows if r['cohort']=='train'];by_token={r['token']:r for r in train_rows}
    head=make_head(bank['head'],'cuda:0');comparisons={}
    for seed in c['train_seeds']:
        comparisons[str(seed)]={};order=epoch_plan(train_rows,4096,seed)
        for arm in c['arms']:
            dest=root/f'seed_{seed}'/arm;r=result['training']['seeds'][str(seed)][arm]
            for name,key in [('before_final.pt','before_final_sha256'),('final.pt','checkpoint_sha256')]:
                if fingerprint(dest/name)!=r[key]:raise ValueError('Fusion checkpoint identity differs')
            before=torch.load(dest/'before_final.pt',map_location='cpu',weights_only=False);after=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
            for checkpoint,step in [(before,511),(after,512)]:
                if (checkpoint['next_index']!=step or checkpoint['order']!=order or checkpoint['protocol']!=p
                        or checkpoint['arm']!=arm or checkpoint['config']!={**c,'seed':seed}):raise ValueError('Replay protocol/sampler differs')
            fusion=ResidualFusion(fast_width=bank['fast'].shape[-1],**c['fusion']).cuda();fusion.load_state_dict(after['fusion'])
            observed=evaluate(fusion,head,bank,held)
            path=dest/'holdout_predictions.json'
            if fingerprint(path)!=result['holdout'][str(seed)][arm]['predictions_sha256']:raise ValueError('Stored holdout prediction identity differs')
            predictions=compare_trees(json.loads(path.read_text()),observed)
            fusion.load_state_dict(before['fusion']);fusion.train()
            optimizer=torch.optim.AdamW(fusion.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
            scheduler=WarmupCosineScheduler(optimizer,512,c['warmup_steps'],c['min_lr_ratio'])
            optimizer.load_state_dict(before['optimizer']);scheduler.load_state_dict(before['scheduler'])
            restore_rng(before['rng']);optimizer.zero_grad(set_to_none=True);tokens=order[-8:]
            target=torch.tensor([by_token[t]['trajectory_gt'] for t in tokens],device='cuda:0')
            loss=pose_loss(cached_forward(fusion,head,training_bank,tokens,arm),target);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(fusion.parameters(),c['grad_clip'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();fusion.eval()
            with torch.no_grad():last=cached_forward(fusion,head,training_bank,tokens,arm).cpu()
            actual=dict(fusion=state_cpu(fusion),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=capture_rng(),reference_prediction=last)
            replay={k:compare_trees(after[k],value) for k,value in actual.items()}
            comparisons[str(seed)][arm]=dict(conditions=len(observed),full_holdout_predictions=predictions,
                actual_step_replay=replay,last_step_loss=float(loss.detach()),last_step_grad_norm=float(norm),before=511,after=512)
            print(f'verified seed{seed}/{arm} reload={predictions["passed"]} replay={all(x["passed"] for x in replay.values())}',flush=True)
    passed=all(x['full_holdout_predictions']['passed'] and all(v['passed'] for v in x['actual_step_replay'].values()) for pair in comparisons.values() for x in pair.values())
    write_json(output,dict(passed=passed,comparisons=comparisons,scope='same_H800_runtime_all256new_windows_four_conditions_and_six_actual_single_updates',new_network_bytes=0))
    if not passed:raise ValueError('Independent paired study verification failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.run,a.output)


if __name__=='__main__':main()
