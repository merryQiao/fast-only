"""Fixed-budget full Fast language training from authenticated frozen vision/KV.

Run one arm per process. --verify reloads final.pt for the fresh VAL endpoint;
--replay reloads before_final.pt and repeats the actual last accumulated update.
No language embedding, language hidden state or training graph is cached.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from fast_only.common import apply_reproducibility, fingerprint, seed_all, write_json
from fast_navsim.model import pose_loss, trajectory_metrics
from slowfast_v11.adapter_data import recording_group
from slowfast_v11.expanded_protocol import WarmupCosineScheduler, select_train_diagnostics
from slowfast_v11.full_training import (
    configure_full_fast, full_precision, full_state, make_full_optimizer,
    restore_full_state,
)
from slowfast_v11.validation import state_digest
from tools.direct_full_common import read_config, load_fast, direct_forward
from tools.train_slowfast_joint import capture_rng, restore_rng, compare_trees


FIXED_SOURCE = '006be99cbfcf5751'
ARMS = ('full_fast_only', 'full_correct_kv', 'full_fixed_kv')


def cpu_tree(value):
    """Detach checkpoint tensors from every live model/optimizer storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def order_digest(order):
    return hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest()


def build_draw_order(rows, updates=1024, accumulation=4, seed=42):
    """Shuffle the authenticated TRAIN list independently of ambient RNG."""
    tokens = [r['token'] for r in rows]
    if (not tokens or len(set(tokens)) != len(tokens) or
            any(r.get('split') != 'train' for r in rows)):
        raise ValueError('Expected a nonempty, unique TRAIN cohort only')
    if (type(updates) is not int or updates <= 0 or
            type(accumulation) is not int or accumulation <= 0):
        raise ValueError('Invalid train update/accumulation budget')
    rng = random.Random(seed)
    order = []
    while len(order) < updates*accumulation:
        epoch = tokens.copy()
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:updates*accumulation]


def sampler_state(order, draw_index, accumulation=4):
    if (type(accumulation) is not int or accumulation <= 0 or
            type(draw_index) is not int or not 0 <= draw_index <= len(order) or
            draw_index % accumulation or len(order) % accumulation):
        raise ValueError('Sampler draw cursor must be an accumulation boundary')
    return dict(schema='direct_full_sampler_v1', order_sha256=order_digest(order),
                draw_index=draw_index, total_draws=len(order),
                accumulation=accumulation, accumulation_index=0,
                completed_updates=draw_index//accumulation)


def validate_sampler(saved, order, accumulation):
    if not isinstance(saved, dict):
        raise ValueError('Invalid sampler state')
    expected = sampler_state(order, saved.get('draw_index'), accumulation)
    if saved != expected:
        raise ValueError('Saved sampler cursor/order differs')
    return expected['draw_index']


def accumulated_step(model, optimizer, scheduler, tokens, loss_fn, clip=1., probe=None):
    """Backpropagate mean micro losses, clip once, and advance AdamW once."""
    if not tokens:
        raise ValueError('An update requires microbatch tokens')
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        for token in tokens:
            loss = loss_fn(token)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite or nonscalar full-language micro loss')
            losses.append(float(loss.detach()))
            (loss/len(tokens)).backward()
            del loss
        probes = probe() if probe is not None else None
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip,
            error_if_nonfinite=True)
        rates = {g.get('name', str(i)): float(g['lr'])
                 for i, g in enumerate(optimizer.param_groups)}
        optimizer.step()
        scheduler.step()
        result = dict(loss=sum(losses)/len(losses), micro_losses=losses,
                      grad_norm=float(norm), learning_rates=rates)
        if probes is not None:
            result['gradient_probes'] = probes
        return result
    finally:
        optimizer.zero_grad(set_to_none=True)


def checkpoint_payload(model, optimizer, scheduler, order, draw_index, config, identity, arm,
                       reference_predictions=None):
    sampler = sampler_state(order, draw_index, config['gradient_accumulation'])
    if scheduler.completed_steps != sampler['completed_updates']:
        raise ValueError('Scheduler and sampler update cursors differ')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Checkpoint must be saved with accumulation index zero and cleared gradients')
    state = full_state(model)
    return dict(schema='direct_full_checkpoint_v1', arm=arm, config=copy.deepcopy(config),
                identity=copy.deepcopy(identity), model=state, model_digest=state_digest(state),
                optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                rng=cpu_tree(capture_rng()), sampler=sampler,
                reference_predictions=cpu_tree(reference_predictions or []))


def restore_checkpoint(model, optimizer, scheduler, saved, order, config, identity, arm):
    if (saved.get('schema') != 'direct_full_checkpoint_v1' or saved.get('arm') != arm or
            saved.get('config') != config or saved.get('identity') != identity):
        raise ValueError('Full checkpoint config/arm/source identity differs')
    cursor = validate_sampler(saved.get('sampler'), order, config['gradient_accumulation'])
    if saved.get('model_digest') != state_digest(saved['model']):
        raise ValueError('Full checkpoint model digest differs')
    if saved['scheduler'].get('completed_steps') != cursor//config['gradient_accumulation']:
        raise ValueError('Full checkpoint schedule/sampler cursor differs')
    # Check all state fields that can be checked without mutating the live model.
    candidate = copy.copy(scheduler)
    candidate.optimizer = copy.copy(optimizer)
    candidate.optimizer.param_groups = [dict(g) for g in optimizer.param_groups]
    candidate.load_state_dict(saved['scheduler'])
    live_groups = optimizer.state_dict()['param_groups']
    saved_groups = saved['optimizer']['param_groups']
    if (len(live_groups) != len(saved_groups) or any(
            len(a['params']) != len(b['params']) or a.get('name') != b.get('name')
            for a, b in zip(live_groups, saved_groups))):
        raise ValueError('Full checkpoint optimizer parameter groups differ')
    restore_full_state(model, saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    optimizer.zero_grad(set_to_none=True)
    restore_rng(saved['rng'])
    return cursor


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(checkpoint_sha256=fingerprint(path), model_digest=payload['model_digest'],
                completed_updates=payload['sampler']['completed_updates'])


def gradient_audit(model):
    """Reject a missing/nonfinite dense gradient in any of the 24 native blocks."""
    if any(p.requires_grad or p.grad is not None for p in model.backbone.visual.parameters()):
        raise ValueError('Frozen visual/merger acquired gradients or trainable parameters')
    blocks = model.backbone.language_model.layers
    if len(blocks) != 24:
        raise ValueError('Full Fast training requires all 24 native language blocks')
    report = []
    for index, block in enumerate(blocks, 1):
        named = [(n, p) for n, p in block.named_parameters()
                 if 'lora_' not in n and p.grad is not None]
        item = dict(block=index, tensors_with_grad=len(named),
                    finite=bool(named) and all(bool(torch.isfinite(p.grad).all()) for _, p in named),
                    nonzero=any(bool(p.grad.count_nonzero()) for _, p in named))
        if not item['finite'] or not item['nonzero']:
            raise ValueError(f'Dense base gradient missing/nonfinite/zero in block {index}')
        report.append(item)
    special = {}
    for name, parameter in model.backbone.language_model.named_parameters():
        if name in ('embed_tokens.weight', 'norm.weight'):
            special[name] = dict(finite=parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                                 nonzero=parameter.grad is not None and bool(parameter.grad.count_nonzero()))
            if not all(special[name].values()):
                raise ValueError(f'Full language gradient missing for {name}')
    return dict(blocks=report, embedding_and_final_norm=special, visual_frozen=True)


def source_for(cache, token, arm, device='cuda:0'):
    if arm not in ARMS:
        raise ValueError('Unknown full-language arm')
    return None if arm == 'full_fast_only' else cache.source(
        token if arm == 'full_correct_kv' else FIXED_SOURCE, device=device)


def prediction_record(row, prediction, arm, split, source_token=None):
    prediction = prediction.detach().float().cpu()
    target = torch.tensor([row['trajectory_gt']], dtype=torch.float32)
    # This validates both complete eight-pose tensors and their finite values.
    metrics = trajectory_metrics(prediction, target)
    return dict(arm=arm, split=split, mode='trained', token=row['token'],
                source_token=source_token, route=row['route'], command=row['command'],
                prediction=prediction[0].tolist(), target=row['trajectory_gt'],
                ADE=metrics['ADE'], FDE=metrics['FDE'],
                loss_l1_pose=metrics['loss_l1_pose'])


@torch.no_grad()
def evaluate_rows(model, cache, rows, arm, split, device='cuda:0'):
    """Evaluate fixed rows without advancing the training RNG or sampler."""
    if not rows or split not in ('train', 'val') or any(r['split'] != split for r in rows):
        raise ValueError('Evaluation rows must match the declared train/val split')
    rng = capture_rng()
    was_training = model.training
    model.eval()
    records = []
    try:
        for row in rows:
            token = row['token']
            packet = cache.packet(token, device=device)
            source = source_for(cache, token, arm, device)
            prediction = direct_forward(model, packet, arm, source)
            source_token = None if arm == 'full_fast_only' else (
                token if arm == 'full_correct_kv' else FIXED_SOURCE)
            records.append(prediction_record(row, prediction, arm, split, source_token))
            del packet, source, prediction
        predictions = torch.tensor([r['prediction'] for r in records], dtype=torch.float32)
        targets = torch.tensor([r['target'] for r in records], dtype=torch.float32)
        return trajectory_metrics(predictions, targets), records
    finally:
        model.train(was_training)
        model.backbone.visual.eval()
        restore_rng(rng)


def memory_stats(device='cuda:0'):
    if torch.device(device).type != 'cuda':
        return dict(allocated_gib=0., reserved_gib=0., peak_allocated_gib=0., peak_reserved_gib=0.)
    return dict(allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                reserved_gib=torch.cuda.memory_reserved(device)/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)


def sampled_bases(model):
    sampled = []
    for index, block in enumerate(model.backbone.language_model.layers, 1):
        name, parameter = next((n, p) for n, p in block.named_parameters()
                               if 'lora_' not in n and p.ndim == 2)
        sampled.append((index, name, parameter, parameter.detach().cpu().clone()))
    return sampled


def base_deltas(sampled):
    records = []
    for index, name, parameter, initial in sampled:
        delta = float((parameter.detach().cpu()-initial).abs().max())
        if not 0 < delta < float('inf'):
            raise ValueError(f'Dense base tensor did not update finitely in block {index}')
        records.append(dict(block=index, name=name, max_abs_change=delta))
    return records


def make_schedule_optimizer(model, config):
    optimizer = make_full_optimizer(model, language_lr=config['language_lr'],
                                    new_lr=config['new_lr'], weight_decay=config['weight_decay'])
    scheduler = WarmupCosineScheduler(optimizer, config['optimizer_updates'],
                                      config['warmup_updates'], config['minimum_lr_ratio'])
    return optimizer, scheduler


def model_loss(model, cache, by_token, arm, token, device='cuda:0'):
    packet = cache.packet(token, device=device)
    source = source_for(cache, token, arm, device)
    target = torch.tensor([by_token[token]['trajectory_gt']], dtype=torch.float32, device=device)
    # Returning the loss keeps only this microbatch graph alive until backward.
    return pose_loss(direct_forward(model, packet, arm, source), target)


def select_diagnostic_rows(rows, count=64):
    tokens = select_train_diagnostics(rows, count)
    by_token = {r['token']: r for r in rows}
    selected = [by_token[t] for t in tokens]
    groups = {recording_group(r['route']) for r in rows}
    if count >= len(groups) and {recording_group(r['route']) for r in selected} != groups:
        raise ValueError('Fixed TRAIN diagnostics must cover every available recording group')
    return selected


def train_loop(model, cache, config, identity, order, arm, output, scope, device='cuda:0'):
    """Run the complete declared draw budget without consulting VAL outputs."""
    output = Path(output)
    if any((output/name).exists() for name in ['train.jsonl', 'final.pt', 'before_final.pt', 'result.json']):
        raise FileExistsError('Training outputs already exist; each arm requires a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    by_token = {r['token']: r for r in cache.train}
    if set(order) != set(by_token) or len(order) != config['optimizer_updates']*config['gradient_accumulation']:
        raise ValueError('Optimizer order must cover the authenticated TRAIN budget exactly')
    diagnostic_rows = select_diagnostic_rows(cache.train, config['train_diagnostic_windows'])
    if len(diagnostic_rows) != config['train_diagnostic_windows']:
        raise ValueError('Fixed TRAIN diagnostic cohort is incomplete')
    optimizer, scheduler = make_schedule_optimizer(model, config)
    sampled = sampled_bases(model)
    started = time.monotonic()
    model.train()
    model.backbone.visual.eval()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    protocol = dict(schema='direct_full_training_protocol_v1', arm=arm, config=config,
                    identity=identity, scope=scope, draw_order=order,
                    draw_order_sha256=order_digest(order),
                    train_diagnostic_tokens=[r['token'] for r in diagnostic_rows],
                    train_diagnostic_selector='fixed TRAIN speed-bin/log round robin',
                    train_diagnostic_recording_groups=sorted({recording_group(r['route']) for r in diagnostic_rows}),
                    fixed_source_token=FIXED_SOURCE, val_used_for_selection=False,
                    metadata=[0., 0., 0., 0., 1., 1.],
                    sampler='seed42 repeated independent epoch shuffles of prepared TRAIN512')
    write_json(output/'protocol.json', protocol)
    diagnostics = []
    before_identity = None
    final_tokens = order[-config['gradient_accumulation']:]
    for update in range(1, config['optimizer_updates']+1):
        cursor = (update-1)*config['gradient_accumulation']
        if update == config['optimizer_updates']:
            payload = checkpoint_payload(model, optimizer, scheduler, order, cursor,
                                         config, identity, arm)
            before_identity = atomic_save(payload, output/'before_final.pt')
            del payload
            gc.collect()
        tokens = order[cursor:cursor+config['gradient_accumulation']]
        step_started = time.monotonic()
        inspect = update == 1 or update in config['train_curve_updates']
        result = accumulated_step(model, optimizer, scheduler, tokens,
            lambda token: model_loss(model, cache, by_token, arm, token, device),
            clip=config['gradient_clip'], probe=(lambda: gradient_audit(model)) if inspect else None)
        if inspect:
            result['base_weight_deltas'] = base_deltas(sampled)
        memory = memory_stats(device)
        if memory['peak_allocated_gib'] > config['max_allocated_gib']:
            raise MemoryError('Actual full-language accumulated training exceeds declared GPU memory cap')
        if any(p.requires_grad or p.grad is not None for p in model.backbone.visual.parameters()):
            raise ValueError('Entire frozen visual/merger contract changed')
        draw_index = cursor+len(tokens)
        row = dict(update=update, arm=arm, tokens=tokens, draw_index=draw_index,
                   accumulation_index=0, windows_seen=draw_index,
                   equivalent_train_epochs=draw_index/len(cache.train),
                   update_elapsed_s=time.monotonic()-step_started,
                   elapsed_s=time.monotonic()-started, **memory, **result)
        with (output/'train.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
        if update == 1 or update % 16 == 0 or update == config['optimizer_updates']:
            print(f'{arm} update {update}/{config["optimizer_updates"]}: '
                  f'loss={result["loss"]:.6f}, grad={result["grad_norm"]:.5f}, '
                  f'peak={memory["peak_allocated_gib"]:.2f}GiB, '
                  f'elapsed={time.monotonic()-started:.1f}s', flush=True)
        if update in config['train_curve_updates']:
            metrics, records = evaluate_rows(model, cache, diagnostic_rows, arm, 'train', device)
            diagnostics.append(dict(update=update, windows_seen=draw_index, metrics=metrics,
                                    tokens=[r['token'] for r in records]))
            write_json(output/'train_diagnostics.json', diagnostics)
    _, references = evaluate_rows(model, cache, [by_token[t] for t in final_tokens], arm, 'train', device)
    payload = checkpoint_payload(model, optimizer, scheduler, order, len(order), config, identity, arm,
                                 reference_predictions=references)
    final_identity = atomic_save(payload, output/'final.pt')
    result = dict(schema='direct_full_training_result_v1', completed=True, arm=arm,
                  optimizer_updates=config['optimizer_updates'], windows_seen=len(order),
                  equivalent_train_epochs=len(order)/len(cache.train),
                  checkpoint_sha256=final_identity['checkpoint_sha256'],
                  model_digest=final_identity['model_digest'],
                  before_final_sha256=before_identity['checkpoint_sha256'],
                  before_final_model_digest=before_identity['model_digest'],
                  identity=identity, scope=scope, train_diagnostics=diagnostics,
                  draw_order_sha256=order_digest(order),
                  trainable_numel=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  optimizer_state_parameters=len(optimizer.state),
                  val_used_for_selection=False, final_checkpoint_selected_by='fixed update budget',
                  elapsed_s=time.monotonic()-started, **memory_stats(device))
    write_json(output/'result.json', result)
    del payload, sampled
    return result


def setup(config_path, cache_path, init_path):
    """Authenticate completed P0, original assets and prepared data before training."""
    from tools.cache_direct_full import DirectCache
    config = read_config(config_path)
    strict = dict(train_recording_groups=12, val_recording_groups=8,
                  train_curve_updates=[256, 512, 768, 1024], train_diagnostic_windows=64,
                  max_allocated_gib=76)
    if any(config.get(k) != v for k, v in strict.items()):
        raise ValueError('Predeclared full-language diagnostics/cohort/memory policy differs')
    source = json.loads((Path(config['source_run'])/'result.json').read_text())
    apply_reproducibility(source['config'])
    seed_all(config['train_seed'])
    torch.set_num_threads(4)
    cache = DirectCache(root=cache_path, config=config_path)
    required = ['source_frozen', 'visual_only_cache', 'source_layout_all_exact',
                'vision_native_prediction_all_exact', 'old_raw_KV_reproduced_exact']
    if any(cache.manifest.get(key) is not True for key in required):
        raise ValueError('Frozen full-vision/native source cache gates did not pass')
    if (cache.manifest.get('fixed_source') != FIXED_SOURCE or
            min(r['token'] for r in cache.train) != FIXED_SOURCE):
        raise ValueError('Predeclared fixed TRAIN source identity differs')
    init_path = Path(init_path)
    p0 = json.loads((init_path.parent/'result.json').read_text())
    init_sha = fingerprint(init_path)
    if (p0.get('passed') is not True or p0.get('p0_updates_discarded') is not True or
            p0.get('full_init_restored') is not True or p0.get('shared_init_sha256') != init_sha or
            p0.get('source_checkpoint') != config['source_checkpoint_sha256'] or
            p0.get('fast_checkpoint') != config['fast_checkpoint_sha256']):
        raise ValueError('Completed disposable P0/shared initialization provenance differs')
    for name in ['tools/direct_full_common.py', 'slowfast_v11/full_training.py']:
        if fingerprint(name) != p0['source_code_sha256'][name]:
            raise ValueError('Proven full-language precision/forward implementation changed after P0')
    initial = torch.load(init_path, map_location='cpu', mmap=True, weights_only=False)
    if (initial.get('schema') != 'direct_full_shared_init_v1' or initial.get('config') != config or
            initial.get('source_fast') != config['fast_checkpoint_sha256'] or
            initial.get('digest') != p0['initial_digest'] or
            state_digest(initial['state']) != p0['initial_digest']):
        raise ValueError('Shared initialization is not the exact pre-update P0 state')
    model, _, _, _ = load_fast(config)
    scope = configure_full_fast(model)
    restore_full_state(model, initial['state'])
    actual = full_state(model)
    if state_digest(actual) != p0['initial_digest']:
        raise ValueError('Dense full-language shared initialization restore differs')
    del initial, actual
    gc.collect()
    identity = dict(cache_sha256=cache.fingerprint, init_sha256=init_sha,
                    initial_digest=p0['initial_digest'],
                    source_checkpoint_sha256=config['source_checkpoint_sha256'],
                    fast_checkpoint_sha256=config['fast_checkpoint_sha256'],
                    data=cache.manifest['data'],
                    source_code_sha256={p: fingerprint(p) for p in [
                        'tools/train_direct_full.py', 'tools/direct_full_common.py',
                        'tools/cache_direct_full.py', 'slowfast_v11/full_training.py',
                        'slowfast_v11/expanded_protocol.py']})
    order = build_draw_order(cache.train, config['optimizer_updates'],
                             config['gradient_accumulation'], config['train_seed'])
    # Reinitializing/loading modules may consume RNG. Every arm begins from the
    # exact same seed immediately before its optimizer/training loop.
    seed_all(config['train_seed'])
    return model, cache, config, identity, order, scope


def load_result_checkpoint(output, filename, result, key):
    path = Path(output)/filename
    if fingerprint(path) != result[key]:
        raise ValueError('Saved full checkpoint SHA256 differs from completed training result')
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def verify_or_replay(model, cache, config, identity, order, arm, output, replay=False, device='cuda:0'):
    output = Path(output)
    result = json.loads((output/'result.json').read_text())
    if (not result.get('completed') or result['arm'] != arm or result['identity'] != identity or
            result['optimizer_updates'] != config['optimizer_updates']):
        raise ValueError('Completed training result identity/budget differs')
    final = load_result_checkpoint(output, 'final.pt', result, 'checkpoint_sha256')
    optimizer, scheduler = make_schedule_optimizer(model, config)
    by_token = {r['token']: r for r in cache.train}
    if replay:
        before = load_result_checkpoint(output, 'before_final.pt', result, 'before_final_sha256')
        cursor = restore_checkpoint(model, optimizer, scheduler, before, order, config, identity, arm)
        if cursor != len(order)-config['gradient_accumulation']:
            raise ValueError('Replay checkpoint is not immediately before the actual final update')
        model.train()
        model.backbone.visual.eval()
        update = accumulated_step(model, optimizer, scheduler, order[cursor:],
            lambda token: model_loss(model, cache, by_token, arm, token, device),
            clip=config['gradient_clip'], probe=lambda: gradient_audit(model))
        _, predictions = evaluate_rows(model, cache, [by_token[t] for t in order[cursor:]], arm, 'train', device)
        actual = checkpoint_payload(model, optimizer, scheduler, order, len(order), config, identity, arm,
                                    reference_predictions=predictions)
        checks = {key: compare_trees(final[key], actual[key]) for key in [
            'model', 'optimizer', 'scheduler', 'rng', 'sampler', 'reference_predictions']}
        report = dict(schema='direct_full_replay_v1', arm=arm,
                      passed=all(v['passed'] for v in checks.values()), checks=checks,
                      replayed_updates=1, micro_batches=config['gradient_accumulation'],
                      final_update=update, checkpoint_sha256=result['checkpoint_sha256'])
        write_json(output/'replay.json', report)
        if not report['passed']:
            raise ValueError('Fresh-process final four-microbatch accumulated-update replay differs')
        return report
    cursor = restore_checkpoint(model, optimizer, scheduler, final, order, config, identity, arm)
    if cursor != len(order):
        raise ValueError('Final reload did not exhaust the declared training draw budget')
    _, references = evaluate_rows(model, cache, [by_token[r['token']] for r in final['reference_predictions']], arm, 'train', device)
    reference_check = compare_trees(final['reference_predictions'], references)
    if not reference_check['passed']:
        raise ValueError('Fresh final checkpoint reload predictions differ')
    metrics, records = evaluate_rows(model, cache, cache.val, arm, 'val', device)
    write_json(output/'final_predictions.json', records)
    report = dict(schema='direct_full_endpoint_v1', arm=arm, passed=True, metrics=metrics,
                  samples=len(records), checkpoint_sha256=result['checkpoint_sha256'],
                  model_digest=final['model_digest'], reference_prediction_check=reference_check,
                  full_state_keys=len(final['model']),
                  predictions_sha256=fingerprint(output/'final_predictions.json'),
                  final_checkpoint_selected_by='fixed update budget', val_used_for_selection=False)
    write_json(output/'verification.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/slowfast_direct_full.yaml')
    parser.add_argument('--cache', default='outputs/slowfast_direct_full/cache_01')
    parser.add_argument('--init', default='outputs/slowfast_direct_full/p0_01/shared_init.pt')
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--output', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--replay', action='store_true')
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not args.verify and not args.replay and output.exists():
        raise FileExistsError('Use a fresh directory for each full-language training arm')
    model, cache, config, identity, order, scope = setup(args.config, args.cache, args.init)
    with full_precision(model):
        if args.verify or args.replay:
            result = verify_or_replay(model, cache, config, identity, order, args.arm, output, args.replay)
        else:
            result = train_loop(model, cache, config, identity, order, args.arm, output, scope)
    print(json.dumps(dict(arm=args.arm, completed=True,
                          mode='replay' if args.replay else 'verify' if args.verify else 'train',
                          result=str(output/('replay.json' if args.replay else
                              'verification.json' if args.verify else 'result.json')))), flush=True)
    return result


if __name__ == '__main__':
    main()
