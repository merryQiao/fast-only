"""Paired fusion-only seeds; new logs are excluded from all optimizer updates."""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from fast_only.common import apply_reproducibility,fingerprint,write_json
from slowfast_v11.expanded_protocol import select_train_diagnostics
from slowfast_v11.reader_fusion import ResidualFusion
from slowfast_v11.validation import state_digest
from tools.train_reader_fusion import train_arm,make_head,evaluate,metrics
from tools.train_slowfast_joint import compare_trees


def authenticated_parent(c):
    parent=Path(c['parent_run']);p=json.loads((parent/'protocol.json').read_text())
    for name,key,expected in [('frozen_features.pt','frozen_features_sha256',c['old_feature_bank_sha256']),
                              ('rows.json','rows_sha256',c['old_rows_sha256'])]:
        if p[key]!=expected or fingerprint(parent/name)!=expected:raise ValueError('Authenticated previous feature data differs')
    if p['fast_checkpoint']!=c['fast_checkpoint_sha256'] or p['reader_checkpoint']!=c['reader_checkpoint_sha256'] or p['source_checkpoint']!=c['source_checkpoint_sha256']:
        raise ValueError('Frozen source versions differ')
    return p


def config(path):
    c=yaml.safe_load(Path(path).read_text())
    if (c['schema']!='adapter_paired_study_v1' or c['train_seeds']!=[42,43,44] or c['arms']!=['correct','reference']
            or c['steps']!=512 or c['batch_size']!=8 or c['train_size']!=256 or c['holdout_logs']!=8
            or c['holdout_windows_per_log']!=32 or c['analysis']['bootstrap_replicates']!=20000
            or c['analysis']['practical_mean_gain_m']!=.02 or not c['analysis']['require_positive_CI_lower_bound']):
        raise ValueError('Predeclared adapter study protocol differs')
    fixed=dict(warmup_steps=32,min_lr_ratio=.1,learning_rate=5e-4,weight_decay=.01,grad_clip=1.,
               curve_steps=[128,256,384,512],fusion=dict(reader_width=768,hidden=256,gate_bias=-2.))
    if any(c[k]!=v for k,v in fixed.items()):raise ValueError('Preregistered optimizer/fusion recipe differs')
    return c


def train(config_path,root):
    c=config(config_path);root=Path(root)
    if root.exists():raise FileExistsError('Use a fresh training root')
    p=authenticated_parent(c);parent=Path(c['parent_run']);old_result=json.loads((parent/'result.json').read_text())
    original=json.loads((Path(p['config']['source_run'])/'result.json').read_text())
    apply_reproducibility(original['config']);torch.set_num_threads(4);root.mkdir(parents=True)
    bank=torch.load(parent/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    rows=json.loads((parent/'rows.json').read_text());head=make_head(bank['head'],'cuda:0')
    train_rows=[r for r in rows if r['cohort']=='train'];by_token={r['token']:r for r in rows}
    diagnostics=[by_token[t] for t in select_train_diagnostics(train_rows,32)]
    protocol=dict(schema=c['schema'],config=c,config_sha256=fingerprint(config_path),parent_protocol=p,
        training_bank_sha256=c['old_feature_bank_sha256'],training_rows_sha256=c['old_rows_sha256'],
        holdout_never_in_optimizer=True,source_code_sha256={f:fingerprint(f) for f in ['tools/train_adapter_study.py','tools/train_reader_fusion.py','slowfast_v11/reader_fusion.py']})
    protocol=json.loads(json.dumps(protocol));write_json(root/'protocol.json',protocol)
    results={};identity={}
    for seed in c['train_seeds']:
        dest=root/f'seed_{seed}';dest.mkdir();results[str(seed)]={}
        for arm in c['arms']:
            cc=copy.deepcopy(c);cc['seed']=seed
            model,result=train_arm(cc,arm,bank,head,rows,diagnostics,dest/arm,protocol)
            results[str(seed)][arm]=result
            if seed==42:
                old=torch.load(parent/arm/'final.pt',map_location='cpu',weights_only=False)
                current=torch.load(dest/arm/'final.pt',map_location='cpu',weights_only=False)
                weights=compare_trees(old['fusion'],current['fusion'])
                predictions=compare_trees(json.loads((parent/arm/'predictions.json').read_text()),json.loads((dest/arm/'predictions.json').read_text()))
                if not weights['passed'] or not predictions['passed']:raise ValueError('Seed42 no longer reproduces prior fusion experiment')
                identity[arm]=dict(weights=weights,old_VAL_predictions=predictions,
                                   parent_checkpoint_sha256=old_result['arms'][arm]['checkpoint_sha256'])
        pair=results[str(seed)]
        if pair['correct']['initial_digest']!=pair['reference']['initial_digest'] or pair['correct']['order']!=pair['reference']['order']:
            raise ValueError('Paired initialization/order differs')
    write_json(root/'training.json',dict(completed=True,protocol=protocol,seeds=results,seed42_prior_exact=identity,
        new_backbone_updates=0,new_reader_updates=0,holdout_used_for_selection=False,download_bytes=0))
    print('All six fixed-budget fusion trainings completed; new holdout not accessed',flush=True)


def evaluate_study(root):
    from slowfast_v11.fixed_adapter import FixedFeatureAdapter,save_fixed_adapter
    root=Path(root);trained=json.loads((root/'training.json').read_text());p=trained['protocol'];c=p['config']
    features=json.loads((root/'features.json').read_text())
    if fingerprint(root/'frozen_features.pt')!=features['feature_bank_sha256'] or fingerprint(root/'rows.json')!=features['rows_sha256']:
        raise ValueError('New frozen feature identity differs')
    original=json.loads((Path(p['parent_protocol']['config']['source_run'])/'result.json').read_text())
    apply_reproducibility(original['config']);torch.set_num_threads(4)
    bank=torch.load(root/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    rows=json.loads((root/'rows.json').read_text());held=[r for r in rows if r['cohort']=='holdout'];head=make_head(bank['head'],'cuda:0')
    if len(held)!=256 or len({r['route'] for r in held})!=8 or any(r['split']!='val' for r in held):raise ValueError('Fixed holdout coverage differs')
    evaluation={};bundles={}
    for seed in c['train_seeds']:
        evaluation[str(seed)]={}
        for arm in c['arms']:
            dest=root/f'seed_{seed}'/arm;r=trained['seeds'][str(seed)][arm]
            if fingerprint(dest/'final.pt')!=r['checkpoint_sha256']:raise ValueError('Trained fusion checkpoint differs')
            saved=torch.load(dest/'final.pt',map_location='cpu',weights_only=False)
            fusion=ResidualFusion(fast_width=bank['fast'].shape[-1],**c['fusion']).cuda();fusion.load_state_dict(saved['fusion'])
            records=evaluate(fusion,head,bank,held)
            path=dest/'holdout_predictions.json'
            if path.exists():raise FileExistsError('Holdout already evaluated')
            write_json(path,records)
            evaluation[str(seed)][arm]=dict(summary={mode:metrics([r for r in records if r['mode']==mode]) for mode in ['correct','off','reference','shuffled']},
                predictions_sha256=fingerprint(path),conditions=len(records))
            if arm=='reference':
                adapter=FixedFeatureAdapter(fusion,bank['reader']['reference'][:1])
                path=root/f'seed_{seed}'/'fixed_adapter.pt'
                digest=save_fixed_adapter(path,adapter,fast_checkpoint_sha256=c['fast_checkpoint_sha256'],head=head)
                bundles[str(seed)]=dict(path=str(path),sha256=digest,source_training_checkpoint=r['checkpoint_sha256'])
    write_json(root/'result.json',dict(completed=True,protocol=p,features=features,training=trained,
        holdout=evaluation,fixed_bundles=bundles,new_holdout_used_for_selection=False,new_backbone_updates=0,new_Reader_updates=0,
        new_network_response_body_bytes=features['new_network_response_body_bytes']))
    print('Frozen three-seed holdout evaluation completed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_adapter_study.yaml');p.add_argument('--run',required=True)
    p.add_argument('--phase',choices=['train','evaluate'],required=True);a=p.parse_args()
    if a.phase=='train':train(a.config,a.run)
    else:evaluate_study(a.run)


if __name__=='__main__':main()
