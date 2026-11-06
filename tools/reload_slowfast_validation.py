"""Independently reload a completed bounded KV diagnostic and verify held-out predictions.

This command restores saved adapter/module weights and completed numeric feedback.
It never trains, regenerates feedback, or overwrites original run files.
"""
import argparse
import copy
import json
import math
from numbers import Real
from pathlib import Path


_MODES = ('none', 'fresh', 'shuffled', 'stale')


def _condition_key(record):
    if record.get('split') != 'val' or record.get('stage') != 'after_consumer':
        raise ValueError('Prediction records must be VAL after_consumer only')
    token, mode, source = record.get('token'), record.get('mode'), record.get('source_token')
    age = record.get('age_s')
    if (not isinstance(token,str) or not token or mode not in _MODES
            or isinstance(age,bool) or not isinstance(age,Real) or not math.isfinite(age) or age < 0):
        raise ValueError('Invalid prediction condition identity')
    if source is not None and (not isinstance(source,str) or not source):
        raise ValueError('Invalid source token')
    if ((mode == 'none' and (source is not None or age != 0))
            or (mode == 'fresh' and (source != token or age != 0))
            or (mode == 'shuffled' and (source is None or source == token or age != 0))
            or (mode == 'stale' and (source is None or age <= 0))):
        raise ValueError('Cache mode, source and age disagree')
    return token, mode, float(age), source


def _index_predictions(records):
    indexed = {}
    for record in records:
        key = _condition_key(record)
        if key in indexed:
            raise ValueError('Duplicate prediction condition')
        prediction = record.get('prediction')
        if (not isinstance(prediction,(list,tuple)) or len(prediction) != 8
                or any(not isinstance(row,(list,tuple)) or len(row) != 3 for row in prediction)):
            raise ValueError('Every trajectory prediction must have shape [8,3]')
        flat = [value for row in prediction for value in row]
        if any(isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) for value in flat):
            raise ValueError('Trajectory predictions must contain finite numeric values')
        indexed[key] = flat
    if not indexed:
        raise ValueError('Held-out prediction comparison requires nonempty records')
    return indexed


def compare_predictions(reference_records, actual_records, *, atol=0.):
    """Compare complete condition sets with an explicit absolute-only tolerance.

    Malformed data and missing/duplicate conditions raise ValueError. Numerical
    differences return passed=False, permitting a failed verification artifact.
    """
    if isinstance(atol,bool) or not isinstance(atol,Real) or not math.isfinite(atol) or atol < 0:
        raise ValueError('Absolute tolerance must be a finite nonnegative number')
    reference, actual = _index_predictions(reference_records), _index_predictions(actual_records)
    if reference.keys() != actual.keys():
        raise ValueError('Held-out prediction condition coverage or identity differs')
    by_mode = {}
    for key, expected in reference.items():
        difference = max(abs(float(a)-float(b)) for a,b in zip(expected,actual[key]))
        item = by_mode.setdefault(key[1],dict(count=0,max_abs_difference=0.,passed=True))
        item['count'] += 1
        item['max_abs_difference'] = max(item['max_abs_difference'],difference)
        item['passed'] = item['passed'] and difference <= atol
    maximum = max(item['max_abs_difference'] for item in by_mode.values())
    return dict(passed=maximum <= atol,count=len(reference),atol=float(atol),
                max_abs_difference=maximum,by_mode=by_mode)


def _evaluation_specs(experiment):
    """Rebuild the full held-out condition list independently of checkpoint records."""
    specs = []
    for row in experiment.rows:
        if row['split'] != 'val':
            continue
        token = row['token']
        fresh = experiment.fresh[token]
        specs.extend([(token,'none',fresh,None),(token,'fresh',fresh,token)])
        if token in experiment.mismatches:
            specs.append((token,'shuffled',fresh,experiment.mismatches[token]))
        specs.extend((token,'stale',pair,pair.source_token) for pair in experiment.pairs
                     if pair.split == 'val' and pair.target_token == token and pair.age_s > 0)
    return specs


def reload_and_compare(checkpoint_path, output_path, *, atol=0.):
    checkpoint_path, output_path = Path(checkpoint_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError('Reload evidence output must be a new JSON file')
    # Validate the tolerance before heavyweight imports/model construction.
    if isinstance(atol,bool) or not isinstance(atol,Real) or not math.isfinite(atol) or atol < 0:
        raise ValueError('Absolute tolerance must be a finite nonnegative number')

    import torch
    from fast_only.common import fingerprint
    from slowfast_v11.validation import canonical_state, restore_weights, state_digest
    from tools.validate_slowfast_v11 import Experiment

    checkpoint_sha = fingerprint(checkpoint_path)
    result_path = checkpoint_path.parent/'result.json'
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if result is not None and checkpoint_sha != result.get('checkpoint_sha256'):
        raise ValueError('Checkpoint SHA256 differs from original result.json')
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if checkpoint.get('schema') != 'slowfast_validation_v1' or checkpoint.get('stage') != 'fast_consumer':
        raise ValueError('Requires the final bounded validation checkpoint')
    if checkpoint.get('step') != checkpoint['config']['stages']['fast_consumer']:
        raise ValueError('Checkpoint does not contain the completed fixed consumer budget')
    references = checkpoint['reference_predictions']
    reference_index = _index_predictions(references)
    fast_digest, slow_digest = state_digest(checkpoint['fast']), state_digest(checkpoint['slow'])
    teacher_digest = state_digest(checkpoint['feedback_producer'])
    if teacher_digest != checkpoint['feedback_producer_digest']:
        raise ValueError('Completed-feedback teacher state identity differs')
    if result is not None:
        for name, digest in [('final_fast_digest',fast_digest),('final_slow_digest',slow_digest),
                             ('feedback_producer_digest',teacher_digest)]:
            if result.get(name) != digest:
                raise ValueError(f'Checkpoint state identity differs from original {name}')

    experiment = Experiment(copy.deepcopy(checkpoint['config']))
    # Experiment.__init__ invokes verify_assets against the pinned local model
    # files; compare those verified identities to the saved run as well.
    if experiment.assets != checkpoint['assets']:
        raise ValueError('Restored local model/processor identities differ from checkpoint')
    if experiment.data_identities != checkpoint.get('data_identities'):
        raise ValueError('Data identities differ from checkpoint manifests, statistics, metadata or images')
    if result is not None and result.get('data_identities') != checkpoint['data_identities']:
        raise ValueError('Data identities differ between checkpoint and original result.json')
    restore_weights(experiment.fast,checkpoint['fast'])
    restore_weights(experiment.slow,checkpoint['slow'])
    if (state_digest(canonical_state(experiment.fast)) != fast_digest
            or state_digest(canonical_state(experiment.slow)) != slow_digest):
        raise ValueError('Restored canonical state differs from saved checkpoint')
    experiment.fast.eval()
    experiment.slow.eval()
    experiment.feedback = checkpoint['feedback']
    specs = _evaluation_specs(experiment)
    expected_keys = {(token,mode,float(pair.age_s),source) for token,mode,pair,source in specs}
    if set(reference_index) != expected_keys or len(specs) != len(expected_keys):
        raise ValueError('Saved references do not cover every rebuilt held-out condition')
    needed_sources = {source for _,_,_,source in specs if source is not None}
    target_tokens = {token for token,_,_,_ in specs}
    for token in needed_sources:
        feedback = experiment.feedback.get(token)
        if not isinstance(feedback,torch.Tensor) or feedback.shape != (1,8,3) or not torch.isfinite(feedback).all():
            raise ValueError('Missing or invalid saved completed numeric source feedback')
    all_sources = experiment.sources
    experiment.sources = {token:all_sources[token] for token in needed_sources}
    experiment.export_cache()
    if experiment.slow_digest != slow_digest:
        raise ValueError('Frozen Slow identity changed while exporting source snapshots')
    # Fast target batches also use sources to locate their current observations.
    experiment.sources = {token:all_sources[token] for token in needed_sources | target_tokens}
    actual = []
    with torch.no_grad():
        for token,mode,pair,source in specs:
            batch = experiment.batch(token,'fast')
            kwargs = {} if source is None else dict(
                snapshot=experiment.cache[source].to(experiment.c['fast']['device']),
                metadata=experiment.metadata(pair))
            prediction = experiment.fast(batch['native'],batch['ego'],batch['command'],**kwargs)
            actual.append(dict(stage='after_consumer',split='val',token=token,mode=mode,
                source_token=source,age_s=pair.age_s,prediction=prediction.float().cpu()[0].tolist()))
    comparison = compare_predictions(references,actual,atol=atol)
    evidence = dict(schema='slowfast_validation_reload_v1',
        scope='independent-process final VAL prediction reload; no training or continuation claim',
        checkpoint=checkpoint_path.name,checkpoint_sha256=checkpoint_sha,
        checkpoint_sha256_verified_against_result=result is not None,
        asset_identities_verified=True,assets=experiment.assets,
        data_identities_verified=True,data_identities=experiment.data_identities,
        state_identities_verified=True,final_fast_digest=fast_digest,final_slow_digest=slow_digest,
        feedback_producer_digest=teacher_digest,
        feedback_digest=state_digest({token:experiment.feedback[token] for token in sorted(needed_sources)}),
        source_cache_count=len(needed_sources),held_out_tokens=sorted(target_tokens),
        devices={name:experiment.c[name]['device'] for name in ('fast','slow')},
        reference_stage='after_consumer',reference_split='val',comparison=comparison,
        optimizer_restored=False,training_updates=0,test_used=False,
        predictions=actual)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x') as stream:
        json.dump(evidence,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--atol',type=float,default=0.,help='Explicit absolute tolerance; default requires exact predictions')
    args = parser.parse_args()
    evidence = reload_and_compare(args.checkpoint,args.output,atol=args.atol)
    print(json.dumps(evidence['comparison'],sort_keys=True))
    return 0 if evidence['comparison']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
