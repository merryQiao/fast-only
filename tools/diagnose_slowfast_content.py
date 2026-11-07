"""Fixed-checkpoint offline content/softmax diagnostics. No training/download."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch

from fast_only.common import fingerprint,write_json
from slowfast_v11.attention import LAYERS
from slowfast_v11.content_diagnostics import (contrast_donors,source_groups,transplant_snapshot,
    kv_difference,attention_probe,summarize_records)
from slowfast_v11.validation import canonical_state,state_digest
from tools.reload_slowfast_expanded import checked_checkpoint
from tools.train_slowfast_expanded import ExpandedExperiment,SnapshotLRU

MODES=('none','correct','shuffled','contrast','masked','zero_v','reference')


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Diagnostic requires a new output directory')
    original=json.loads((root/'result.json').read_text())
    output.mkdir(parents=True);started=time.monotonic()
    experiment=ExpandedExperiment(copy.deepcopy(original['config']));experiment.output=output
    saved=checked_checkpoint(root/'joint_final.pt',experiment,original,'joint',800,
                             original['arms']['joint']['checkpoint_sha256'])
    references={(r['token'],r['mode']):r['prediction'] for r in saved['reference_predictions']}
    del saved
    fast,slow=experiment.fast.eval(),experiment.slow.eval()
    before={name:state_digest(canonical_state(getattr(experiment,name))) for name in ['fast','slow']}
    parameter_versions={name:tuple((n,p._version) for n,p in getattr(experiment,name).named_parameters())
                        for name in ['fast','slow']}
    rows=experiment.cohorts['new_val']+experiment.cohorts['legacy_val']
    donors=contrast_donors(experiment.rows)
    reference_token=min(r['token'] for r in experiment.train)
    plan=dict(schema='kv_content_diagnostic_v1',source_checkpoint_sha256=original['arms']['joint']['checkpoint_sha256'],
        checkpoint_arm='joint',checkpoint_step=800,reference_token=reference_token,
        reference_rule='lexicographically_first_TRAIN_token_no_GT_or_error_selection',modes=list(MODES),
        selected_tokens=[r['token'] for r in rows],cohort_counts={'new_val':64,'legacy_val':4},
        contrast_rule='same_cohort;prefer_cross_log;different_command_first;max_abs_current_speed_delta;token_tiebreak',
        donors={r['token']:dict(shuffled=experiment.mismatches[r['token']],contrast=donors[r['token']]) for r in rows},
        contrast_candidates={r['token']:[dict(token=d['token'],cross_log=d['route']!=r['route'],
            different_command=d['command']!=r['command'],
            abs_speed_difference_mps=abs(math.hypot(*r['ego'][:2])-math.hypot(*d['ego'][:2])))
            for d in sorted(rows,key=lambda item:item['token']) if d['cohort']==r['cohort'] and d['token']!=r['token']]
            for r in rows},
        fixed_conditions=['current_observation','metadata','source_positions','source_valid_layout','model_weights'],
        clear_slow_advantage=dict(absolute_ADE_gain_m=.1,relative_ADE_gain=.2),
        masked_invariance_token=rows[0]['token'],data_identities=experiment.data_identities,
        model_assets=experiment.assets,metadata_fixed_across_cache_modes=True,
        scope='previously_observed_offline_VAL;not_closed_loop;not_test_or_model_selection',
        new_training_updates=0,new_download_bytes=0,source_code_sha256={str(p):fingerprint(p) for p in
            [Path('slowfast_v11/content_diagnostics.py'),Path('tools/diagnose_slowfast_content.py')]})
    write_json(output/'protocol.json',plan)
    cache=SnapshotLRU(2)
    @torch.no_grad()
    def source(token):
        def build():
            batch=experiment.batch(token,'slow')
            prediction=slow(**batch,feedback=experiment.feedback[token].to(experiment.c['slow']['device']))
            groups=source_groups(batch['native'],slow.backbone.config.image_token_id,
                                 slow.backbone.visual.spatial_merge_size)
            return (prediction.snapshot.inference_copy(experiment.c['fast']['device']),
                    prediction.trajectory[0].float().cpu().tolist(),groups)
        return cache.get(token,build)
    def versions(snapshot):
        return (snapshot.position_ids._version,snapshot.valid_mask._version,
                *(t._version for z in snapshot.layers for t in (z.key,z.value)))
    records=[];max_reference_diff=0.;verified_references=0;invariance=None
    with torch.no_grad():
        reference,_,reference_groups=source(reference_token)
        reference_versions=versions(reference)
        for index,row in enumerate(rows,1):
            token=row['token'];correct,slow_prediction,groups=source(token)
            original_versions=versions(correct)
            metadata=experiment.metadata(experiment.fresh[token]);metadata_copy=metadata.clone()
            batch=experiment.batch(token,'fast')
            entry=dict(token=token,cohort=row['cohort'],route=row['route'],command=row['command'],
                       speed_mps=math.hypot(*row['ego'][:2]),target=row['trajectory_gt'],slow_prediction=slow_prediction,
                       metadata=metadata.cpu().tolist(),predictions={},attention={},donors={},kv_differences={})
            for mode in MODES:
                if mode=='none':
                    result=fast(**batch)
                    stats=[]
                else:
                    donor_token=(experiment.mismatches[token] if mode=='shuffled' else
                                 donors[token] if mode=='contrast' else reference_token if mode=='reference' else token)
                    donor,_,donor_groups=((reference,None,reference_groups) if mode=='reference' else
                                          (correct,None,groups) if donor_token==token else source(donor_token))
                    if set(groups)!=set(donor_groups) or any(not torch.equal(groups[k],donor_groups[k]) for k in groups):
                        raise ValueError('Donor token-type layout differs')
                    marker=versions(donor)
                    candidate=transplant_snapshot(correct,donor,zero_values=mode=='zero_v')
                    if mode in ('shuffled','contrast','reference'):
                        entry['kv_differences'][mode]=kv_difference(correct,donor,groups)
                    with attention_probe(groups,mask_external=mode=='masked') as stats:
                        result=fast(**batch,snapshot=candidate,metadata=metadata)
                    if len(stats)!=6:raise ValueError('Diagnostic did not traverse all six fusion layers')
                    if versions(donor)!=marker:raise ValueError('Diagnostic mutated donor KV')
                    for layer,statistics in zip(LAYERS,stats):statistics['layer_one_based']=layer+1
                    d=experiment.by_token[donor_token]
                    entry['donors'][mode]=dict(token=donor_token,cohort=d['cohort'],route=d['route'],
                        command=d['command'],speed_mps=math.hypot(*d['ego'][:2]),cross_log=d['route']!=row['route'],
                        different_command=d['command']!=row['command'],actual_metadata_intentionally_not_used=donor_token!=token)
                prediction=result[0].float().cpu()
                entry['predictions'][mode]=prediction.tolist();entry['attention'][mode]=stats
                oldmode={'none':'none','correct':'fresh','shuffled':'shuffled'}.get(mode)
                if oldmode:
                    reference_prediction=torch.tensor(references[token,oldmode],dtype=torch.float32)
                    difference=float((prediction-reference_prediction).abs().max())
                    max_reference_diff=max(max_reference_diff,difference);verified_references+=1
                    if difference!=0.:raise ValueError(f'Saved {oldmode} prediction changed for {token}: {difference}')
                if not torch.equal(metadata,metadata_copy):raise ValueError('Fixed metadata changed')
            if index==1:
                wrong,_,_=source(donors[token])
                with attention_probe(groups,mask_external=True):
                    masked_wrong=fast(**batch,snapshot=transplant_snapshot(correct,wrong),metadata=metadata)[0].float().cpu()
                delta=float((masked_wrong-torch.tensor(entry['predictions']['masked'])).abs().max())
                invariance=dict(token=token,max_abs_difference=delta,passed=delta==0.)
                if not invariance['passed']:raise ValueError('Masked output changed with external content')
            if versions(correct)!=original_versions or versions(reference)!=reference_versions:
                raise ValueError('Diagnostic mutated source KV')
            records.append(entry)
            with (output/'records.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
            experiment.log(f'content {index}/{len(rows)} {row["cohort"]} token={token}')
    after={name:state_digest(canonical_state(getattr(experiment,name))) for name in ['fast','slow']}
    if before!=after:raise ValueError('Diagnostic mutated model parameters')
    if parameter_versions!={name:tuple((n,p._version) for n,p in getattr(experiment,name).named_parameters())
                            for name in ['fast','slow']}:
        raise ValueError('Diagnostic mutated a frozen or trainable model parameter')
    summary=summarize_records(records)
    result=dict(schema='kv_content_diagnostic_result_v1',completed=True,summary=summary,
        verified_reference_predictions=verified_references,max_reference_difference=max_reference_diff,
        masked_content_invariance=invariance,weights_unchanged=before==after,
        all_parameter_versions_unchanged=True,
        weights_before=before,weights_after=after,source_tensors_unchanged=True,
        checkpoint_sha256=plan['source_checkpoint_sha256'],elapsed_s=time.monotonic()-started,
        gpu=torch.cuda.get_device_name(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        new_training_updates=0,new_download_bytes=0,test_used=False,val_used_for_selection=False)
    write_json(output/'result.json',result)
    experiment.log('Fixed content diagnostic complete; no training updates')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/slowfast_v11_expanded/run_01')
    parser.add_argument('--output',default='outputs/slowfast_kv_content/run_01')
    args=parser.parse_args();run(args.run,args.output)


if __name__=='__main__':main()
