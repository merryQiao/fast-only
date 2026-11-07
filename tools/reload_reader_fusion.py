"""Independent full-VAL reload and actual final-update replay for frozen fusion."""
import argparse
import json
from pathlib import Path

import torch

from fast_only.common import apply_reproducibility,fingerprint,write_json
from fast_navsim.model import pose_loss
from slowfast_v11.expanded_protocol import WarmupCosineScheduler,epoch_plan
from slowfast_v11.reader_fusion import ResidualFusion
from tools.diagnose_slowfast_readability import state_cpu
from tools.train_reader_fusion import cached_forward,evaluate,make_head
from tools.train_slowfast_joint import restore_rng,capture_rng,compare_trees


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh verification output file')
    result=json.loads((root/'result.json').read_text());protocol=result['protocol'];c=protocol['config']
    for name,key in [('frozen_features.pt','frozen_features_sha256'),('rows.json','rows_sha256')]:
        if fingerprint(root/name)!=protocol[key]:raise ValueError('Frozen bank/row SHA differs')
    for path,digest in [(Path(c['fast_run'])/'final.pt',c['fast_checkpoint_sha256']),
                        (Path(c['feature_run'])/'kv_final.pt',c['reader_checkpoint_sha256']),
                        (Path(c['feature_run'])/'frozen_cache/manifest.json',protocol['feature_manifest_sha256'])]:
        if fingerprint(path)!=digest:raise ValueError('Source checkpoint/manifest identity differs')
    parent=json.loads((Path(c['source_run'])/'result.json').read_text())
    for field in ['train_manifest','val_manifest','stats']:
        if fingerprint(parent['config']['data'][field])!=protocol['source_data'][field]:raise ValueError('Source data identity differs')
    apply_reproducibility(parent['config']);torch.set_num_threads(4)
    bank=torch.load(root/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    rows=json.loads((root/'rows.json').read_text());train={r['token']:r for r in rows if r['cohort']=='train'}
    expected_order=epoch_plan(list(train.values()),c['steps']*c['batch_size'],c['seed'])
    head=make_head(bank['head'],'cuda:0');comparisons={}
    for arm in c['arms']:
        sub=root/arm;r=result['arms'][arm]
        for name,key in [('final.pt','checkpoint_sha256'),('before_final.pt','before_final_sha256')]:
            if fingerprint(sub/name)!=r[key]:raise ValueError('Fusion checkpoint SHA differs')
        after=torch.load(sub/'final.pt',map_location='cpu',weights_only=False)
        before=torch.load(sub/'before_final.pt',map_location='cpu',weights_only=False)
        for saved,step in [(before,c['steps']-1),(after,c['steps'])]:
            if (saved['arm']!=arm or saved['config']!=c or saved['order']!=expected_order or saved['next_index']!=step
                    or saved['protocol']!=protocol):raise ValueError('Fusion resume protocol/order differs')
        fusion=ResidualFusion(fast_width=bank['fast'].shape[-1],**c['fusion']).cuda()
        fusion.load_state_dict(after['fusion'])
        actual=evaluate(fusion,head,bank,[r for r in rows if r['cohort']!='train'])
        expected=json.loads((sub/'predictions.json').read_text())
        predictions=compare_trees(expected,actual)
        fusion.load_state_dict(before['fusion']);fusion.train()
        optimizer=torch.optim.AdamW(fusion.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        scheduler=WarmupCosineScheduler(optimizer,c['steps'],c['warmup_steps'],c['min_lr_ratio'])
        optimizer.load_state_dict(before['optimizer']);scheduler.load_state_dict(before['scheduler'])
        restore_rng(before['rng']);optimizer.zero_grad(set_to_none=True)
        tokens=expected_order[-c['batch_size']:]
        loss=pose_loss(cached_forward(fusion,head,bank,tokens,arm),torch.tensor([train[t]['trajectory_gt'] for t in tokens],device='cuda:0'))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(fusion.parameters(),c['grad_clip'],error_if_nonfinite=True)
        optimizer.step();scheduler.step();fusion.eval()
        with torch.no_grad():last=cached_forward(fusion,head,bank,tokens,arm).cpu()
        replay={k:compare_trees(after[k],v) for k,v in dict(fusion=state_cpu(fusion),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),rng=capture_rng(),reference_prediction=last).items()}
        comparisons[arm]=dict(full_VAL_predictions=predictions,conditions=len(actual),actual_step_replay=replay,
            last_step_loss=float(loss),last_step_grad_norm=float(norm),replayed_before=c['steps']-1,replayed_after=c['steps'])
    passed=all(v['full_VAL_predictions']['passed'] and all(x['passed'] for x in v['actual_step_replay'].values()) for v in comparisons.values())
    evidence=dict(passed=passed,comparisons=comparisons,authenticated_frozen_features=True,
        scope='same_H800_runtime_all68_VAL_four_modes_and_actual_single_update_per_arm',new_download_bytes=0)
    write_json(output,evidence);print(json.dumps(evidence),flush=True)
    if not passed:raise ValueError('Fusion reload/replay mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.run,a.output)


if __name__=='__main__':main()
