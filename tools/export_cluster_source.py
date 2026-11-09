"""Export authenticated frozen Slow + causal feedback teacher task state only."""
import argparse
import json
import os
from pathlib import Path

import torch

from fast_only.common import fingerprint, write_json
from fast_cluster.models import asset_manifest, validate_source_bundle
from slowfast_v11.validation import state_digest


def export_source(source_run, destination, expected_sha):
    root, destination = Path(source_run), Path(destination)
    result = json.loads((root/'result.json').read_text())
    checkpoint = root/'joint_final.pt'
    actual = fingerprint(checkpoint)
    if actual != expected_sha or result['arms']['joint']['checkpoint_sha256'] != actual:
        raise ValueError('Original frozen source checkpoint SHA differs')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['schema'] != 'slowfast_expanded_v1' or saved['arm'] != 'joint':
        raise ValueError('Expected original joint Slow source checkpoint')
    if saved['data_identities'] != result['data_identities'] or saved['assets'] != result['assets']:
        raise ValueError('Original source report differs from embedded data/base identity')
    stats_path = Path(result['config']['data']['stats'])
    if fingerprint(stats_path) != result['data_identities']['stats']:
        raise ValueError('Original source TRAIN statistics differ')
    assets = {}
    for role in ('fast', 'slow'):
        base = Path(result['config'][role]['model_path'])
        assets[role] = asset_manifest(base)
        if any(assets[role].get(k) != v for k, v in result['assets'][role].items()):
            raise ValueError(f'Original {role} base assets differ')
    bundle = dict(schema='navsim_frozen_source_v1', source_checkpoint_sha256=actual,
        slow=saved['slow'], feedback_teacher=saved['feedback_producer'],
        slow_digest=state_digest(saved['slow']),
        feedback_teacher_digest=state_digest(saved['feedback_producer']),
        stats=json.loads(stats_path.read_text()), model_kwargs=result['config']['model'],
        processor=result['config']['processor'], assets=assets,
        upstream_data_identity=result['data_identities'],
        scope='old Mini-adapted frozen task states; not a newly V1-trained Slow',
        training_feedback='original frozen Fast teacher; no future labels',
        runtime_feedback='latest completed Fast plan, separately exercised by functional async Smoke')
    if bundle['feedback_teacher_digest'] != saved['feedback_producer_digest']:
        raise ValueError('Original frozen feedback teacher digest differs')
    validate_source_bundle(bundle, actual)
    from fast_cluster.source_contract import validate_bundle_provenance
    validate_bundle_provenance(bundle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {k: v for k, v in bundle.items() if k not in ('slow', 'feedback_teacher')}
    receipt.update(bundle_sha256=fingerprint(destination), bytes=destination.stat().st_size)
    write_json(destination.with_suffix('.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    receipt = export_source(args.source_run, args.output, args.expected_sha256)
    print(json.dumps({k: receipt[k] for k in ('bundle_sha256', 'bytes', 'slow_digest', 'feedback_teacher_digest')}))


if __name__ == '__main__':
    main()
