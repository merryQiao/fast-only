"""Fresh-process three-arm VAL reload and actual interrupted joint-step replay."""
import argparse
import copy
import json
from pathlib import Path

import torch

from fast_only.common import fingerprint, write_json
from slowfast_v11.validation import canonical_state, restore_weights, state_digest, make_optimizer
from tools.train_slowfast_joint import JointExperiment, restore_rng, capture_rng, compare_trees
from tools.reload_slowfast_validation import compare_predictions


def _checked_checkpoint(path, experiment, *, arm, step, sha=None):
    if sha is not None and fingerprint(path) != sha:
        raise ValueError('Joint checkpoint SHA differs from result')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved.get('schema')!='slowfast_joint_v1' or saved.get('arm')!=arm or saved.get('step')!=step
            or saved['config']!=experiment.c or saved['sampler']!=dict(next_index=step,plan=experiment.plan)):
        raise ValueError('Checkpoint protocol, branch or next sampler position differs')
    if (saved['assets']!=experiment.assets or saved['data_identities']!=experiment.data_identities
            or saved['parent_checkpoint_sha256']!=experiment.parent_sha
            or saved['feedback_producer_digest']!=experiment.feedback_digest
            or state_digest(saved['feedback_producer'])!=experiment.feedback_digest
            or not compare_trees(saved['feedback'],experiment.feedback)['passed']):
        raise ValueError('Checkpoint assets, data, parent or teacher feedback differs')
    restore_weights(experiment.fast,saved['fast']); restore_weights(experiment.slow,saved['slow'])
    if (state_digest(canonical_state(experiment.fast))!=state_digest(saved['fast'])
            or state_digest(canonical_state(experiment.slow))!=state_digest(saved['slow'])):
        raise ValueError('Restored state identity differs')
    return saved


def run(run_directory, output):
    root=Path(run_directory); output=Path(output)
    if output.exists(): raise FileExistsError('Reload output must be a new directory')
    result=json.loads((root/'result.json').read_text())
    if result.get('schema')!='slowfast_joint_result_v1': raise ValueError('Unsupported joint run')
    output.mkdir(parents=True)
    experiment=JointExperiment(copy.deepcopy(result['config']))
    experiment.output=output
    experiment.load_parent()
    comparisons={}
    for arm in experiment.c['arms']:
        saved=_checked_checkpoint(root/f'{arm}_final.pt',experiment,arm=arm,step=240,
                                  sha=result['arms'][arm]['checkpoint_sha256'])
        experiment.evaluate_selected(f'{arm}_final',['val'],include_stale=True)
        actual=[r for r in experiment.records if r['stage']==f'{arm}_final']
        # Reuse the strict shape/condition-set comparator; stage normalization is
        # only an adapter to its legacy label, not a change to the stored records.
        normalize=lambda rows:[dict(r,stage='after_consumer') for r in rows]
        comparisons[arm]=compare_predictions(normalize(saved['reference_predictions']),normalize(actual))
        experiment.log(f'Independent {arm} VAL reload: {comparisons[arm]}')
        del saved
    before=_checked_checkpoint(root/'replay_before.pt',experiment,arm='joint',step=experiment.replay_step-1,
                               sha=result['replay_checkpoints']['before']['checkpoint_sha256'])
    experiment.trainable()
    experiment.opt_fast=make_optimizer(experiment.fast,experiment.c['optimizer'])
    experiment.opt_slow=make_optimizer(experiment.slow,experiment.c['optimizer'])
    experiment.opt_fast.load_state_dict(before['optimizer_fast'])
    experiment.opt_slow.load_state_dict(before['optimizer_slow'])
    token=experiment.plan[experiment.replay_step-1]['target_token']
    # Finish model construction and image preparation before restoring RNG.
    experiment.batch(token,'fast'); experiment.batch(token,'slow')
    restore_rng(before['rng'])
    update=experiment.paired_update('joint',experiment.replay_step-1)
    prediction=experiment.replay_prediction(token)
    actual=dict(fast=canonical_state(experiment.fast),slow=canonical_state(experiment.slow),
                optimizer_fast=experiment.opt_fast.state_dict(),optimizer_slow=experiment.opt_slow.state_dict(),
                rng=capture_rng(),replay_prediction=prediction)
    after=_checked_checkpoint(root/'replay_after.pt',experiment,arm='joint',step=experiment.replay_step,
                              sha=result['replay_checkpoints']['after']['checkpoint_sha256'])
    replay={name:compare_trees(after[name],value) for name,value in actual.items()}
    passed=all(c['passed'] for c in comparisons.values()) and all(c['passed'] for c in replay.values())
    evidence=dict(schema='slowfast_joint_reload_v1',passed=passed,
        scope='independent process on same GPU/runtime; exact prediction and actual next-step replay',
        comparisons=comparisons,replay=replay,replayed_step=experiment.replay_step,
        replayed_token=token,replayed_update=update,parent_checkpoint_sha256=experiment.parent_sha,
        assets_verified=True,data_identities_verified=True,teacher_feedback_verified=True,
        replay_checkpoint_sha256_verified=True,
        test_used=False,training_updates=1,cross_gpu_bitwise_claim=False)
    write_json(output/'verification.json',evidence)
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); evidence=run(args.run,args.output)
    print(json.dumps(dict(passed=evidence['passed'],comparisons=evidence['comparisons'],replay=evidence['replay'])))
    return 0 if evidence['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
