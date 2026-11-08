"""Frozen six-layer source-read localization. No training or network access."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from fast_only.common import fingerprint, write_json
from fast_navsim.model import trajectory_metrics
from slowfast_v11.adapter_data import recording_group
from slowfast_v11.attention import LAYERS
from slowfast_v11.content_diagnostics import kv_difference, source_groups
from slowfast_v11.full_training import full_precision, restore_full_state
from slowfast_v11.path_diagnostics import trace_fast
from slowfast_v11.validation import state_digest
from tools.direct_full_common import equal_source_layout, snapshot
from tools.train_direct_full import setup, load_result_checkpoint, FIXED_SOURCE
from tools.train_frozen_expert import cached_vision

MODES = ('correct', 'reference', 'shuffled')


def select_interventions(rows, per_group=8):
    """Equal recordings, round robin command/speed buckets; never read labels."""
    if (type(per_group) is not int or per_group < 1 or not rows or
            len({r['token'] for r in rows}) != len(rows) or
            any(r['split'] != 'val' for r in rows)):
        raise ValueError('Expected unique VAL rows and a positive per-group count')
    groups = defaultdict(list)
    for row in rows:
        groups[recording_group(row['route'])].append(row)
    chosen = []
    for group in sorted(groups):
        buckets = defaultdict(list)
        for row in sorted(groups[group], key=lambda r:r['token']):
            speed = math.hypot(*row['ego'][:2])
            buckets[(row['command'], 0 if speed < 1 else 1 if speed < 10 else 2)].append(row)
        if len(groups[group]) < per_group:
            raise ValueError('Insufficient windows; refuse silent cohort shortening')
        selected = []
        while len(selected) < per_group:
            for key in sorted(buckets):
                if buckets[key]:
                    selected.append(buckets[key].pop(0))
                    if len(selected) == per_group:
                        break
        chosen.extend(selected)
    return chosen


def tensor_comparison(reference, candidate):
    a = reference.detach().float()
    b = candidate.detach().float()
    if a.shape != b.shape or not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Expected matching nonempty finite trace tensors')
    d = a - b
    denom = a.norm() * b.norm()
    return dict(rms_difference=float(d.square().mean().sqrt()),
                reference_rms=float(a.square().mean().sqrt()),
                candidate_rms=float(b.square().mean().sqrt()),
                relative_l2=float(d.norm() / a.norm().clamp_min(1e-12)),
                max_abs_difference=float(d.abs().max()),
                equal_fraction=float((a == b).float().mean()),
                cosine=float((a * b).sum() / denom) if denom > 0 else None)


def grouped_distribution(rows, getter):
    if not rows:
        return dict(count=0)
    values = np.asarray([getter(row) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic aggregate')
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[recording_group(row['route'])].append(value)
    means = {key:float(np.mean(v)) for key,v in sorted(groups.items())}
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                minimum=float(values.min()), maximum=float(values.max()),
                zero_fraction=float(np.mean(values == 0)), recordings=len(means),
                equal_recording_mean=float(np.mean(list(means.values()))), by_recording=means)


def weights_digest(model):
    """Hash every frozen and formerly trainable parameter, not adapter state only."""
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        host = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(host.shape), host.dtype)).encode())
        h.update(host.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def versions(model):
    return tuple((name, parameter._version) for name, parameter in model.named_parameters())


def source_versions(snap):
    return tuple(t._version for t in [snap.position_ids, snap.valid_mask,
        *(t for layer in snap.layers for t in (layer.key, layer.value))])


@contextmanager
def extra_trace(model):
    records = {}
    handles = []
    def save(name, value):
        records[name] = value[:, -8:].detach().float().cpu().clone()
    norm = model.backbone.language_model.norm
    handles.append(norm.register_forward_pre_hook(lambda m,args:save('final_norm_input', args[0])))
    handles.append(norm.register_forward_hook(lambda m,args,out:save('final_norm_output', out)))
    for index, module in enumerate(model.head):
        handles.append(module.register_forward_hook(
            lambda m,args,out,n=index:save(f'head_stage_{n}', out)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def context_trace(model, reference=None):
    """Keep one baseline's Context on device and compare all rows in 24 blocks."""
    stored = {} if reference is None else reference
    checks = {}
    handles = []
    for index, layer in enumerate(model.backbone.language_model.layers, 1):
        def hook(module,args,out,n=index):
            value = out[0] if isinstance(out, (tuple,list)) else out
            value = value[:, :-8]
            if reference is None:
                stored[n] = value.detach().clone()
                checks[n] = True
            else:
                checks[n] = torch.equal(value, stored[n])
                if not checks[n]:
                    raise ValueError(f'External source changed forbidden Context rows at layer {n}')
        handles.append(layer.register_forward_hook(hook))
    try:
        yield stored, checks
    finally:
        for handle in handles:
            handle.remove()


def forward_snapshot(model, packet, source):
    with cached_vision(model.backbone, packet['vision'], packet['batch']['native']['image_grid_thw']):
        return model(**packet['batch'], snapshot=source,
            metadata=torch.tensor([[0., 0., 0., 0., 1., 1.]], device=packet['batch']['ego'].device))


def flatten_numeric(value, prefix=''):
    out = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            out.update(flatten_numeric(item, name))
        elif type(item) in (int, float) and math.isfinite(item):
            out[name] = float(item)
    return out


def summarize(records, interventions):
    summary = {}
    for cohort, selected in [('all', records), ('moving_ge_1mps', [r for r in records if r['speed_mps'] >= 1]),
                              ('high_ge_10mps', [r for r in records if r['speed_mps'] >= 10])]:
        if not selected:
            continue
        part = dict(count=len(selected), metrics={}, natural_differences={}, probes={})
        target = torch.tensor([r['target'] for r in selected])
        for mode in MODES:
            part['metrics'][mode] = trajectory_metrics(torch.tensor([r['predictions'][mode] for r in selected]), target)
            part['probes'][mode] = {}
            for index, layer in enumerate(LAYERS):
                flat = [dict(route=r['route'], values=flatten_numeric(r['probes'][mode][index])) for r in selected]
                fields = sorted(set.intersection(*(set(r['values']) for r in flat)))
                part['probes'][mode][str(layer+1)] = {
                    field:grouped_distribution(flat, lambda r,k=field:r['values'][k]) for field in fields}
        for mode in MODES[1:]:
            part['natural_differences'][mode] = {
                name:{field:grouped_distribution(selected,
                    lambda r,n=name,k=field:r['natural_differences'][mode][n][k])
                    for field in ['rms_difference', 'reference_rms', 'relative_l2', 'equal_fraction']}
                for name in selected[0]['natural_differences'][mode]}
        summary[cohort] = part
    grouped = defaultdict(list)
    for row in interventions:
        grouped[row['mode'],row['layer_one_based']].append(row)
    summary['single_layer_interventions'] = {
        f'{mode}_{layer}':{field:grouped_distribution(rows,lambda r,k=field:r[k])
            for field in ['ADE_gain_vs_correct_m', 'mean_plan_xy_change_m', 'max_plan_xy_change_m']}
        for (mode,layer),rows in sorted(grouped.items())}
    return summary


@torch.no_grad()
def run(config, cache_root, init, run_root, output):
    from slowfast_v11.read_trace import read_probe, swap_layer, mask_layer
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserve prior evidence; choose a fresh output directory')
    model, cache, c, identity, _, _ = setup(config, cache_root, init)
    root = Path(run_root)
    result = json.loads((root/'result.json').read_text())
    verified = json.loads((root/'verification.json').read_text())
    diagnostic_path = root.parent/'correct_checkpoint_diagnostics.json'
    old = json.loads(diagnostic_path.read_text())
    if (result['arm'] != 'full_correct_kv' or result['identity'] != identity or not verified['passed'] or
            not old['passed'] or old['checkpoint_sha256'] != result['checkpoint_sha256'] or
            old['cache_sha256'] != cache.fingerprint or
            fingerprint(root/'final_predictions.json') != verified['predictions_sha256']):
        raise ValueError('Verified correct-trained endpoint or diagnostic identities differ')
    saved = load_result_checkpoint(root, 'final.pt', result, 'checkpoint_sha256')
    if state_digest(saved['model']) != result['model_digest']:
        raise ValueError('Full language endpoint digest differs')
    restore_full_state(model, saved['model'])
    del saved
    gc.collect()
    model.eval().requires_grad_(False)
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Inference model has stale gradients')
    expected = {(r['token'], r['mode']):torch.tensor(r['prediction']) for r in old['records'] if r['mode'] in MODES}
    if len(expected) != 3*256:
        raise ValueError('Prior source intervention coverage differs')
    donors = old['donors']
    chosen = select_interventions(cache.val)
    if len(cache.val) != 256 or len(chosen) != 64:
        raise ValueError('Expected fixed VAL256 and metadata-selected intervention64')
    chosen_tokens = {r['token'] for r in chosen}
    before = weights_digest(model)
    parameter_versions = versions(model)
    out.mkdir(parents=True)
    protocol = dict(schema='frozen_direct_kv_read_trace_v1', checkpoint_sha256=result['checkpoint_sha256'],
        cache_sha256=cache.fingerprint, prior_diagnostic_sha256=fingerprint(diagnostic_path),
        input_config_sha256=fingerprint(config), model_all_parameters_sha256=before,
        metadata=[0.,0.,0.,0.,1.,1.], modes=list(MODES), layers_one_based=[i+1 for i in LAYERS],
        selected_tokens=[r['token'] for r in cache.val], intervention_tokens=[r['token'] for r in chosen],
        intervention_selection='8 per recording; command/speed (<1,1..10,>=10 m/s) bucket round robin, token order',
        reference_token=FIXED_SOURCE, donors=donors, total_single_layer_interventions=64*6*3,
        new_training_updates=0, new_response_body_bytes=0,
        scope='previously observed VAL, offline diagnostics only; not new-log confirmation or closed-loop',
        local_readout='fixed current Q/FastKV/gate, source keys rotated at unchanged source positions',
        projection='actual PEFT forward invoked directly on query rows, bypass parent instrumentation hooks; native full-M difference audited',
        source_code_sha256={p:fingerprint(p) for p in ['tools/trace_direct_kv.py', 'slowfast_v11/read_trace.py',
            'slowfast_v11/attention.py', 'slowfast_v11/content_diagnostics.py',
            'slowfast_v11/path_diagnostics.py', 'slowfast_v11/full_training.py']})
    write_json(out/'protocol.json', protocol)
    records = []
    interventions = []
    exact = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with full_precision(model):
        reference_packet = cache.source(FIXED_SOURCE)
        reference = snapshot(reference_packet)
        for index,row in enumerate(cache.val,1):
            token = row['token']
            packet = cache.packet(token)
            source_packet = cache.source(token)
            donor_packet = cache.source(donors[token])
            equal_source_layout(source_packet, reference_packet)
            equal_source_layout(source_packet, donor_packet)
            sources = dict(correct=snapshot(source_packet), reference=reference, shuffled=snapshot(donor_packet))
            source_marks = {key:source_versions(value) for key,value in sources.items()}
            groups = source_groups(packet['batch']['native'], model.backbone.config.image_token_id,
                model.backbone.visual.spatial_merge_size)
            entry = dict(token=token, route=row['route'], command=row['command'],
                speed_mps=math.hypot(*row['ego'][:2]), target=row['trajectory_gt'],
                source_tokens=dict(correct=token, reference=FIXED_SOURCE, shuffled=donors[token]),
                source_differences={name:kv_difference(sources['correct'],sources[name],groups) for name in MODES[1:]},
                predictions={}, probes={}, natural_differences={}, context_checks={}, projection_shape_audit={})
            baseline_trace = None
            baseline_context = None
            for mode in MODES:
                alternatives = {k:sources[k] for k in MODES[1:]} if mode == 'correct' else None
                with read_probe(model, groups, alternatives=alternatives) as probe:
                    with trace_fast(model) as traced, extra_trace(model) as extra:
                        with context_trace(model, baseline_context) as (context,checks):
                            prediction = forward_snapshot(model, packet, sources[mode])
                if len(probe.records) != 6 or len(checks) != 24 or not all(checks.values()):
                    raise ValueError('Incomplete six-layer read or Context invariance coverage')
                actual = prediction[0].float().cpu()
                if not torch.equal(actual, expected[token,mode]):
                    raise ValueError(f'Passive instrumentation changes prior prediction for {token}/{mode}')
                exact += 1
                traced.update(extra)
                for layer in LAYERS:
                    number = layer+1
                    if not torch.equal(probe.tensors[f'post_gate_{number}'], traced[f'post_gate_{number}']):
                        raise ValueError('Probe gate reconstruction differs from actual bridge input')
                    traced[f'pre_gate_{number}'] = probe.tensors[f'pre_gate_{number}']
                entry['projection_shape_audit'][mode] = {str(layer+1):tensor_comparison(
                    traced[f'post_projection_{layer+1}'], probe.tensors[f'query_projection_{layer+1}']) for layer in LAYERS}
                entry['predictions'][mode] = actual.tolist()
                entry['probes'][mode] = probe.records
                entry['context_checks'][mode] = dict(checked_layers=24, all_rows_exact=True)
                if mode == 'correct':
                    baseline_trace = traced
                    baseline_context = context
                else:
                    if traced.keys() != baseline_trace.keys():
                        raise ValueError('Passive trace stage coverage differs')
                    entry['natural_differences'][mode] = {name:tensor_comparison(baseline_trace[name],traced[name]) for name in traced}
                del probe, prediction, actual
            del baseline_context, context, baseline_trace, traced, extra
            if token in chosen_tokens:
                target = torch.tensor(row['trajectory_gt'])
                base = torch.tensor(entry['predictions']['correct'])
                base_ade = float((base[:,:2]-target[:,:2]).norm(dim=-1).mean())
                for mode in ['reference','shuffled','masked']:
                    for layer in LAYERS:
                        source = sources['correct'] if mode == 'masked' else swap_layer(sources['correct'],sources[mode],layer+1)
                        scope = mask_layer(layer+1) if mode == 'masked' else nullcontext()
                        with scope:
                            prediction = forward_snapshot(model, packet, source)[0].float().cpu()
                        delta = (prediction[:,:2]-base[:,:2]).norm(dim=-1)
                        item = dict(token=token, route=row['route'], mode=mode, layer_one_based=layer+1,
                            source_token=None if mode == 'masked' else entry['source_tokens'][mode],
                            prediction=prediction.tolist(), target=row['trajectory_gt'], baseline_prediction=base.tolist(),
                            ADE_gain_vs_correct_m=base_ade-float((prediction[:,:2]-target[:,:2]).norm(dim=-1).mean()),
                            mean_plan_xy_change_m=float(delta.mean()), max_plan_xy_change_m=float(delta.max()))
                        interventions.append(item)
                        with (out/'interventions.jsonl').open('a') as stream:
                            stream.write(json.dumps(item,allow_nan=False)+'\n')
                # Any scope patch leakage must fail immediately on an uninstrumented real pass.
                plain = forward_snapshot(model,packet,sources['correct'])[0].float().cpu()
                if not torch.equal(plain,expected[token,'correct']):
                    raise ValueError('Single-layer intervention leaked into subsequent baseline')
            if source_marks != {key:source_versions(value) for key,value in sources.items()}:
                raise ValueError('Read localization mutated an immutable source tensor')
            if versions(model) != parameter_versions:
                raise ValueError('Read localization mutated a model parameter')
            records.append(entry)
            with (out/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(entry,allow_nan=False)+'\n')
            if torch.cuda.max_memory_allocated() > 76*2**30:
                raise MemoryError('Read-only diagnostic exceeds 76GiB memory cap')
            del packet, source_packet, donor_packet, sources, groups, entry
            if index == 1 or index % 8 == 0:
                print(f'read trace {index}/256, exact={exact}, single-layer={len(interventions)}/1152, elapsed={time.monotonic()-started:.1f}s',flush=True)
    after = weights_digest(model)
    if before != after or versions(model) != parameter_versions or exact != 768 or len(interventions) != 1152:
        raise ValueError('Frozen-model or complete coverage acceptance check failed')
    summary = summarize(records,interventions)
    write_json(out/'summary.json',summary)
    verification = dict(schema='frozen_direct_kv_read_trace_result_v1', completed=True,
        new_training_updates=0, new_response_body_bytes=0, exact_prior_predictions=exact,
        parameter_versions_unchanged=True, all_parameter_sha256_before=before, all_parameter_sha256_after=after,
        source_tensors_unchanged=True, context_24layers_all_rows_exact_natural_passes=768,
        context_checks_exclude_single_layer_interventions=True,
        single_layer_interventions=len(interventions), scope_cleanup_baseline_rechecks=64,
        protocol_sha256=fingerprint(out/'protocol.json'), records_sha256=fingerprint(out/'records.jsonl'),
        interventions_sha256=fingerprint(out/'interventions.jsonl'), summary_sha256=fingerprint(out/'summary.json'),
        elapsed_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(out/'result.json', verification)
    print(json.dumps(verification,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/slowfast_direct_full.yaml')
    p.add_argument('--cache',default='outputs/slowfast_direct_full/cache_01')
    p.add_argument('--init',default='outputs/slowfast_direct_full/p0_01/shared_init.pt')
    p.add_argument('--run',default='outputs/slowfast_direct_full/run_01/full_correct_kv')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    run(a.config,a.cache,a.init,a.run,a.output)


if __name__ == '__main__':
    main()
