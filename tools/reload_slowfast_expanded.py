"""Independent expanded-cohort prediction reload and scheduled joint-step replay."""
import argparse
import copy
import json
from pathlib import Path

import torch

from fast_only.common import fingerprint,write_json
from slowfast_v11.validation import canonical_state,restore_weights,state_digest,make_optimizer
from tools.train_slowfast_joint import capture_rng,restore_rng,compare_trees
from tools.reload_slowfast_validation import compare_predictions
from tools.train_slowfast_expanded import ExpandedExperiment


def checked_checkpoint(path,experiment,result,arm,step,sha):
    if fingerprint(path)!=sha:raise ValueError('Expanded checkpoint SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('schema')!='slowfast_expanded_v1' or saved.get('arm')!=arm or saved.get('step')!=step
            or saved['config']!=experiment.c or saved['sampler']!=dict(next_index=step,plan=experiment.plan)
            or saved['diagnostic_tokens']!=experiment.diagnostic_tokens or saved['probe_steps']!=experiment.probe_steps
            or saved['shared_initialization']!=result['shared_initialization']):
        raise ValueError('Expanded protocol/state boundary mismatch')
    if saved['assets']!=experiment.assets or saved['data_identities']!=experiment.data_identities:
        raise ValueError('Model or data identities differ')
    teacher=state_digest(saved['feedback_producer'])
    if (teacher!=result['feedback_producer_digest'] or teacher!=saved['feedback_producer_digest']
            or teacher!=saved['shared_initialization']['fast_digest']):
        raise ValueError('Shared pretrained-warmup feedback producer differs')
    if set(saved['feedback'])!=set(experiment.sources):raise ValueError('Feedback source coverage differs')
    for feedback in saved['feedback'].values():
        if feedback.shape!=(1,8,3) or not bool(torch.isfinite(feedback).all()):
            raise ValueError('Invalid saved numeric feedback')
    if experiment.feedback and not compare_trees(experiment.feedback,saved['feedback'])['passed']:
        raise ValueError('Feedback differs across arms/replay checkpoints')
    experiment.feedback=saved['feedback'];experiment.feedback_producer=saved['feedback_producer']
    experiment.feedback_digest=teacher;experiment.shared_initialization=saved['shared_initialization']
    restore_weights(experiment.fast,saved['fast']);restore_weights(experiment.slow,saved['slow'])
    for name in ['fast','slow']:
        if state_digest(canonical_state(getattr(experiment,name)))!=state_digest(saved[name]):
            raise ValueError('Restored adapter/module state differs')
    return saved


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Reload requires a new output directory')
    result=json.loads((root/'result.json').read_text())
    if result.get('schema')!='slowfast_expanded_result_v1':raise ValueError('Unsupported expanded result')
    output.mkdir(parents=True)
    experiment=ExpandedExperiment(copy.deepcopy(result['config']));experiment.output=output
    comparisons={}
    rows=experiment.cohorts['new_val']+experiment.cohorts['legacy_val']
    for arm in experiment.c['arms']:
        saved=checked_checkpoint(root/f'{arm}_final.pt',experiment,result,arm,800,result['arms'][arm]['checkpoint_sha256'])
        experiment.evaluate_rows(f'{arm}_final',rows)
        actual=[r for r in experiment.records if r['stage']==f'{arm}_final']
        # Tokens are unique across cohorts; legacy comparator requires split/stage
        # normalization. Cohorts remain separate in references/evidence/results.
        normalize=lambda records:[dict(r,split='val',stage='after_consumer') for r in records]
        comparisons[arm]={}
        for cohort in ['new_val','legacy_val']:
            references=[r for r in saved['reference_predictions'] if r['split']==cohort]
            observed=[r for r in actual if r['split']==cohort]
            comparisons[arm][cohort]=compare_predictions(normalize(references),normalize(observed))
        experiment.log(f'Independent {arm} reload: {comparisons[arm]}')
        del saved
    before=checked_checkpoint(root/'replay_before.pt',experiment,result,'joint',experiment.replay_step-1,
                              result['replay_checkpoints']['before']['checkpoint_sha256'])
    experiment.trainable()
    experiment.opt_fast=make_optimizer(experiment.fast,experiment.c['optimizer'])
    experiment.opt_slow=make_optimizer(experiment.slow,experiment.c['optimizer'])
    # Construct with original peak LR before restoring optimizer's current LR.
    experiment.schedule_fast=experiment.scheduler(experiment.opt_fast,800)
    experiment.schedule_slow=experiment.scheduler(experiment.opt_slow,800)
    for name in ['fast','slow']:
        getattr(experiment,'opt_'+name).load_state_dict(before['optimizer_'+name])
        getattr(experiment,'schedule_'+name).load_state_dict(before['scheduler_'+name])
    token=experiment.plan[experiment.replay_step-1]['target_token']
    experiment.batch(token,'fast');experiment.batch(token,'slow')
    restore_rng(before['rng'])
    update=experiment.paired_update('joint',experiment.replay_step-1)
    prediction=experiment.replay_prediction(token)
    actual=dict(fast=canonical_state(experiment.fast),slow=canonical_state(experiment.slow),
        optimizer_fast=experiment.opt_fast.state_dict(),optimizer_slow=experiment.opt_slow.state_dict(),
        scheduler_fast=experiment.schedule_fast.state_dict(),scheduler_slow=experiment.schedule_slow.state_dict(),
        rng=capture_rng(),replay_prediction=prediction)
    after=checked_checkpoint(root/'replay_after.pt',experiment,result,'joint',experiment.replay_step,
                             result['replay_checkpoints']['after']['checkpoint_sha256'])
    replay={key:compare_trees(after[key],value) for key,value in actual.items()}
    passed=all(v['passed'] for arm in comparisons.values() for v in arm.values()) and all(v['passed'] for v in replay.values())
    evidence=dict(schema='slowfast_expanded_reload_v1',passed=passed,comparisons=comparisons,replay=replay,
        replayed_step=experiment.replay_step,replayed_token=token,replayed_update=update,
        assets_verified=True,data_identities_verified=True,teacher_verified=True,
        checkpoint_sha256_verified=True,training_updates=1,test_used=False,
        scope='same-GPU/runtime exact reload and one actual scheduled step; not cross-GPU or full DDP resume')
    write_json(output/'verification.json',evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();evidence=run(args.run,args.output)
    print(json.dumps(dict(passed=evidence['passed'],comparisons=evidence['comparisons'],replay=evidence['replay'])))
    return 0 if evidence['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
