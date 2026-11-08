"""Collect fixed-endpoint, recording-paired direct-KV full-language evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from fast_only.common import fingerprint,write_json
from fast_navsim.evaluate import constant_velocity_metrics
from fast_navsim.model import trajectory_metrics
from slowfast_v11.adapter_data import recording_group
from slowfast_v11.adapter_statistics import paired_gain_matrix,paired_adapter_statistics
from slowfast_v11.expanded_protocol import lr_factor
from tools.direct_full_common import read_config
from tools.train_direct_full import FIXED_SOURCE,build_draw_order,order_digest

MODES=('correct','off','reference','shuffled')
FULL_STATE_KEYS=529
REPLAY_CHECKS={'model','optimizer','scheduler','rng','sampler','reference_predictions'}
METRIC_TOLERANCE=1e-6


def require(condition,message):
    if not condition:raise ValueError(message)


def metrics(records):
    return trajectory_metrics(torch.tensor([r['prediction'] for r in records],dtype=torch.float32),
        torch.tensor([r['target'] for r in records],dtype=torch.float32))


def checked_ades(records):
    """Infer effects from poses, rejecting drift in any saved window metric."""
    metrics(records)  # Validate complete finite eight-pose arrays before reduction.
    prediction=torch.tensor([r['prediction'] for r in records],dtype=torch.float32)
    target=torch.tensor([r['target'] for r in records],dtype=torch.float32)
    delta=prediction-target
    distance=torch.linalg.vector_norm(delta[...,:2],dim=-1)
    values=dict(ADE=distance.mean(-1),FDE=distance[:,-1],loss_l1_pose=delta.abs().mean((-2,-1)))
    for index,record in enumerate(records):
        for name,value in values.items():
            if name in record:
                saved=record[name]
                require(isinstance(saved,(int,float)) and not isinstance(saved,bool) and
                    np.isfinite(saved) and abs(saved-float(value[index]))<=METRIC_TOLERANCE,
                    f'Saved window {name} metric differs from recomputed poses')
    return values['ADE'].tolist()


def validate_metrics(records,saved,label):
    actual=metrics(records)
    require(isinstance(saved,dict) and set(saved)==set(actual),f'{label} metric fields differ')
    require(all(isinstance(saved[k],(int,float)) and not isinstance(saved[k],bool) and
        np.isfinite(saved[k]) and abs(saved[k]-value)<=METRIC_TOLERANCE for k,value in actual.items()),
        f'{label} saved/recomputed metric disagreement')
    return actual


def validate_records(records,val,*,arm=None,mode=None,sources=None,require_split=True):
    expected={r['token']:r for r in val}
    lookup={r['token']:r for r in records}
    require(len(lookup)==len(records)==len(expected) and set(lookup)==set(expected),
        'Endpoint/diagnostic condition coverage differs')
    for token,record in lookup.items():
        row=expected[token]
        require(record['target']==row['trajectory_gt'],'Endpoint target differs from authenticated VAL')
        require(record['route']==row['route'],'Endpoint route/recording differs from authenticated VAL')
        require((not require_split or record.get('split')=='val') and
            ('split' not in record or record['split']=='val'),'Endpoint split differs from VAL')
        if arm is not None:
            require(record.get('arm')==arm and record.get('command')==row['command'],
                'Endpoint arm/command metadata differs')
        if mode is not None:require(record.get('mode')==mode,'Endpoint diagnostic mode differs')
        if sources is not None:
            require('source_token' in record and record['source_token']==sources[token],
                'Endpoint/diagnostic source token differs')
    ades=checked_ades(records)
    return [dict(record,ADE=ade) for record,ade in zip(records,ades)]


def exact_check(check,label):
    require(isinstance(check,dict) and check.get('passed') is True and
        check.get('max_abs_difference')==0 and check.get('differing_fields')==0 and
        check.get('first_differences')==[],f'{label} exact replay/reload proof missing or inexact')


def validate_proofs(result,verification,replay):
    require(verification.get('full_state_keys')==FULL_STATE_KEYS and
        result['scope']['language']['tensors']+result['scope']['new']['tensors']==FULL_STATE_KEYS,
        'Full state reload does not cover all 529 trainable language/outer tensors')
    require(verification.get('samples')==256 and verification.get('arm')==result['arm'] and
        verification.get('model_digest')==result['model_digest'] and
        verification.get('val_used_for_selection') is False,'Full endpoint reload metadata differs')
    exact_check(verification.get('reference_prediction_check'),'Reference prediction reload')
    checks=replay.get('checks')
    require(replay.get('arm')==result['arm'] and replay.get('replayed_updates')==1 and
        replay.get('micro_batches')==4 and isinstance(checks,dict) and set(checks)==REPLAY_CHECKS,
        'Actual final four-microbatch replay proof is incomplete')
    for name,check in checks.items():exact_check(check,f'{name} replay')
    require(checks['model'].get('tensor_count')==FULL_STATE_KEYS and
        checks['optimizer'].get('tensor_count')==3*result['optimizer_state_parameters'] and
        checks['rng'].get('tensor_count',0)>=1,'Full model/optimizer/RNG replay tensor evidence differs')


def validate_training(log,protocol,result,train,c,arm):
    order=build_draw_order(train,c['optimizer_updates'],c['gradient_accumulation'],c['train_seed'])
    digest=order_digest(order)
    require(result.get('optimizer_updates')==1024 and len(log)==1024 and
        protocol.get('arm')==arm and protocol.get('config')==c and
        protocol.get('identity')==result['identity'] and protocol.get('val_used_for_selection') is False and
        protocol.get('fixed_source_token')==FIXED_SOURCE and protocol.get('draw_order')==order and
        protocol.get('draw_order_sha256')==result['draw_order_sha256']==digest,
        'Training protocol/update budget/order differs from authenticated TRAIN')
    for update,row in enumerate(log,1):
        cursor=update*4
        require(row.get('update')==update and row.get('arm')==arm and row.get('draw_index')==cursor and
            row.get('windows_seen')==cursor and row.get('accumulation_index')==0 and
            row.get('equivalent_train_epochs')==cursor/512 and row.get('tokens')==order[cursor-4:cursor],
            'Training update/cursor/accumulation or TRAIN-only order differs')
        micros=row.get('micro_losses')
        require(isinstance(micros,list) and len(micros)==4 and
            np.isfinite(micros).all() and all(v>=0 for v in micros) and
            np.isfinite(row['loss']) and abs(row['loss']-float(np.mean(micros)))<=METRIC_TOLERANCE and
            np.isfinite(row['grad_norm']) and row['grad_norm']>=0,'Invalid accumulated training loss/gradient trace')
        factor=lr_factor(update,1024,c['warmup_updates'],c['minimum_lr_ratio'])
        expected=dict(language=c['language_lr']*factor,new=c['new_lr']*factor)
        require(set(row.get('learning_rates',{}))==set(expected) and all(
            np.isclose(row['learning_rates'][key],value,rtol=1e-10,atol=1e-15) for key,value in expected.items()),
            'Training learning-rate schedule differs')


def validate_diagnostics(diagnostics,correct,val,checkpoint_sha,cache_sha):
    require(diagnostics.get('passed') is True and diagnostics.get('correct_endpoint_exact_count')==256 and
        diagnostics.get('cache_sha256')==cache_sha and diagnostics.get('checkpoint_sha256')==checkpoint_sha,
        'Same-checkpoint diagnostic checkpoint/cache proof differs')
    tokens={r['token'] for r in val};by_token={r['token']:r for r in val};donors=diagnostics.get('donors',{})
    require(set(donors)==tokens and set(donors.values())==tokens and all(
        recording_group(by_token[token]['route'])!=recording_group(by_token[donor]['route'])
        for token,donor in donors.items()),'Diagnostic shuffled source mapping is not a cross-recording permutation')
    records=diagnostics.get('records',[])
    require(len(records)==1024 and {r.get('mode') for r in records}==set(MODES),
        'Diagnostic mode/condition coverage differs')
    recomputed={}
    correct_lookup={r['token']:r for r in correct}
    for mode in MODES:
        sources={token:token if mode=='correct' else None if mode=='off' else
            FIXED_SOURCE if mode=='reference' else donors[token] for token in tokens}
        selected=validate_records([r for r in records if r['mode']==mode],val,
            arm='full_correct_kv',mode=mode,sources=sources)
        if mode=='correct':
            require(all(r['prediction']==correct_lookup[r['token']]['prediction'] for r in selected),
                'Diagnostic correct predictions differ from independently verified endpoint')
        recomputed[mode]=validate_metrics(selected,diagnostics['metrics'][mode],f'Diagnostic {mode}')
    require(set(diagnostics['metrics'])==set(MODES),'Diagnostic metric modes differ')
    return recomputed


def paired(correct,reference,c):
    a={r['token']:r for r in correct};b={r['token']:r for r in reference}
    if len(a)!=len(correct) or len(b)!=len(reference) or len(a)!=256 or len(b)!=256 or set(a)!=set(b):raise ValueError('Incomplete/duplicate paired endpoint')
    correct_ade=dict(zip((r['token'] for r in correct),checked_ades(correct)))
    reference_ade=dict(zip((r['token'] for r in reference),checked_ades(reference)))
    rows=[]
    for token,r in a.items():
        other=b[token]
        if r['route']!=other['route'] or r['target']!=other['target']:raise ValueError('Paired target/recording differs')
        rows.append(dict(seed=42,log=recording_group(r['route']),token=token,
            ade_correct=correct_ade[token],ade_reference=reference_ade[token]))
    logs=sorted({r['log'] for r in rows})
    if len(logs)!=8:raise ValueError('Expected8 independent validation recording groups')
    gains=paired_gain_matrix(rows,seeds=[42],logs=logs)
    result=paired_adapter_statistics(gains,seeds=[42],logs=logs,replicates=c['bootstrap_replicates'],
        analysis_seed=c['analysis_seed'],threshold_m=c['practical_mean_ade_gain_m'])
    result['seed_scope']='one full-language training seed; recording-group bootstrap only; no training-seed uncertainty'
    result['paired_window_median_gain_m']=float(np.median([r['ade_reference']-r['ade_correct'] for r in rows]))
    return result


def run(config,run_root,cache_root,reference_path,diagnostic_path,output):
    c=read_config(config);root=Path(run_root);cache_root=Path(cache_root);out=Path(output)
    if out.exists():raise FileExistsError('Use a new immutable report destination')
    cache=json.loads((cache_root/'manifest.json').read_text());cache_sha=fingerprint(cache_root/'manifest.json')
    rows=cache['rows'];val=[r for r in rows if r['split']=='val'];train=[r for r in rows if r['split']=='train'];val_tokens={r['token'] for r in val}
    if len(val)!=256 or len(val_tokens)!=256 or len(train)!=512 or not cache['completed']:raise ValueError('Fixed cache incomplete')
    train_groups={recording_group(r['route']) for r in train};val_groups=Counter(recording_group(r['route']) for r in val)
    require(len(train_groups)==12 and len(val_groups)==8 and set(val_groups.values())=={32} and
        not train_groups&set(val_groups) and min(r['token'] for r in train)==FIXED_SOURCE,
        'TRAIN/VAL recording-group boundary or fixed source differs')
    arms={};predictions={};identities=[]
    for arm in c['arms']:
        directory=root/arm;result=json.loads((directory/'result.json').read_text())
        verification=json.loads((directory/'verification.json').read_text());replay=json.loads((directory/'replay.json').read_text())
        records=json.loads((directory/'final_predictions.json').read_text())
        log=[json.loads(line) for line in (directory/'train.jsonl').read_text().splitlines()]
        protocol=json.loads((directory/'protocol.json').read_text())
        if (not result['completed'] or not verification['passed'] or not replay['passed'] or result['arm']!=arm
                or verification['checkpoint_sha256']!=result['checkpoint_sha256'] or replay['checkpoint_sha256']!=result['checkpoint_sha256']
                or fingerprint(directory/'final_predictions.json')!=verification['predictions_sha256']
                or result['identity']['cache_sha256']!=cache_sha or len(log)!=1024 or log[-1]['windows_seen']!=4096
                or result['val_used_for_selection'] or result['windows_seen']!=4096 or result['equivalent_train_epochs']!=8
                or {r['token'] for r in records}!=val_tokens or len(records)!=256):raise ValueError('Incomplete/unmatched training, replay or endpoint proof')
        validate_proofs(result,verification,replay)
        validate_training(log,protocol,result,train,c,arm)
        for name,key in [('final.pt','checkpoint_sha256'),('before_final.pt','before_final_sha256')]:
            require(fingerprint(directory/name)==result[key],'Training checkpoint artifact SHA256 differs')
        sources={token:None if arm=='full_fast_only' else token if arm=='full_correct_kv' else FIXED_SOURCE for token in val_tokens}
        records=validate_records(records,val,arm=arm,mode='trained',sources=sources)
        arm_metrics=validate_metrics(records,verification['metrics'],'Verified endpoint')
        identities.append((result['identity'],result['draw_order_sha256']))
        predictions[arm]=records
        arms[arm]=dict(result=result,verification=verification,replay=replay,metrics=arm_metrics,
            training_log_sha256=fingerprint(directory/'train.jsonl'),protocol_sha256=fingerprint(directory/'protocol.json'))
    if any(item!=identities[0] for item in identities):raise ValueError('Three arms do not share init/data/order/code')
    ref=json.loads(Path(reference_path).read_text())
    if ref['cache_manifest_sha256']!=cache_sha:raise ValueError('Historical reference evaluated a different cohort')
    reference_metrics={}
    methods={'original_fast','historical_fixed_seed_42','historical_fixed_seed_43','historical_fixed_seed_44'}
    require(len(ref['records'])==1024 and {r['method'] for r in ref['records']}==methods,
        'Historical reference condition coverage differs')
    for method in sorted(methods):
        records=validate_records([r for r in ref['records'] if r['method']==method],val,require_split=False)
        reference_metrics[method]=validate_metrics(records,ref['metrics'][method],f'Historical reference {method}')
    diagnostics=json.loads(Path(diagnostic_path).read_text())
    diagnostic_metrics=validate_diagnostics(diagnostics,predictions['full_correct_kv'],val,
        arms['full_correct_kv']['result']['checkpoint_sha256'],cache_sha)
    if fingerprint(cache_root/'slow_predictions.json')!=cache['slow_predictions_sha256']:raise ValueError('Frozen Slow predictions changed')
    targets={r['token']:r for r in val}
    slow=[dict(r,target=targets[r['token']]['trajectory_gt']) for r in json.loads((cache_root/'slow_predictions.json').read_text()) if r['token'] in targets]
    if len(slow)!=256 or {r['token'] for r in slow}!=val_tokens:raise ValueError('Frozen Slow endpoint incomplete')
    slow=validate_records(slow,val)
    primary=paired(predictions['full_correct_kv'],predictions['full_fixed_kv'],c)
    fast=paired(predictions['full_correct_kv'],predictions['full_fast_only'],c)
    traffic=json.loads((Path(c['data_dir'])/'network_budget.json').read_text())
    if traffic['phase']!='complete' or traffic['used']>c['new_response_body_cap_bytes']:raise ValueError('Traffic audit incomplete/overcap')
    result=dict(schema='direct_full_report_v1',completed=True,config=c,arms=arms,
        primary_correct_vs_separately_trained_fixed=primary,correct_vs_fully_adapted_fast_only=fast,
        historical_reference_metrics=reference_metrics,historical_fixed_mean_ade_over_three_seeds=float(np.mean([
            reference_metrics[f'historical_fixed_seed_{seed}']['ADE'] for seed in [42,43,44]])),
        frozen_slow_metrics=metrics(slow),
        same_correct_checkpoint_metrics=diagnostic_metrics,data_traffic=traffic,
        cache_manifest_sha256=cache_sha,reference_file_sha256=fingerprint(reference_path),diagnostics_file_sha256=fingerprint(diagnostic_path),
        source_code_sha256=fingerprint(__file__),
        limitations=['TRAIN512/12groups,NAVSIMmini; not full navtrain','VAL256/8fresh groups,one training seed',
            'offline synchronous freshKV; no stale-cache,async,closed-loop claim','historical fixed-adapter has different prior data/training budget'])
    class Dataset:
        def __init__(self,rows):self.rows=rows
        def __len__(self):return len(self.rows)
    result['constant_velocity_metrics']=constant_velocity_metrics(Dataset(val))
    write_json(out,result)
    print(json.dumps(dict(metrics={arm:item['metrics']['ADE'] for arm,item in arms.items()},primary_gain_m=primary['mean_gain_m'],ci95_m=primary['ci95_m']),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_direct_full.yaml');p.add_argument('--run',required=True)
    p.add_argument('--cache',default='outputs/slowfast_direct_full/cache_01');p.add_argument('--reference',required=True);p.add_argument('--diagnostics',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.run,a.cache,a.reference,a.diagnostics,a.output)


if __name__=='__main__':main()
