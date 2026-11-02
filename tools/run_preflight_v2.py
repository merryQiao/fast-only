"""Validate the intended training policies on a bounded real original-RGB cohort.

This command never launches the formal 8000-step run and never evaluates TEST.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fast_only.common import (NativeCollator, make_model, predict, read_config,
                              to_device, write_json, verify_checkpoint_inputs)
from fast_only.data import FastDataset
from fast_only.diagnostics import (audit_constant_velocity_anchor, constant_velocity_metrics,
                                   evaluate_image_swap)
from fast_only.evaluate import evaluate_dataset, save_predictions
from fast_only.loading import batch_stream
from fast_only.model import planning_loss
from fast_only.train import build_scheduler, restore_checkpoint, run_training


def _release(context):
    context.clear(); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


@torch.no_grad()
def navigation_sensitivity(model, dataset, collator, device):
    """Change only normalized target input; retain images/ego/current command."""
    model.eval(); changes = []
    for index in range(len(dataset)):
        current = to_device(collator([dataset[index]], seed=0), device)
        donor = dataset[(index+1) % len(dataset)]
        changed = dict(current, nav=donor['nav'].unsqueeze(0).to(device))
        a, b = predict(model, current), predict(model, changed)
        changes.append(torch.linalg.vector_norm(a-b, dim=-1).mean().item())
    return {'cohort': sorted({row['split'] for row in dataset.rows}), 'samples': len(changes),
            'method': 'cycle_targets_only_keep_images_ego_command',
            'mean_point_change_m': float(np.mean(changes)), 'per_sample_m': changes,
            'interpretation': 'Functional target-channel sensitivity, not an accuracy improvement claim.'}


def verify_real_resume(context, config, *, raise_on_mismatch=True, capture_state=False):
    """Replay actual optimizer update 2->3 and compare uninterrupted step 3."""
    output = Path(config['output']); tc = config['train']
    model, optimizer, scheduler = (context[key] for key in ['model', 'optimizer', 'scheduler'])
    before = torch.load(output/'step_000002.pt', map_location='cpu', weights_only=False)
    expected = torch.load(output/'step_000003.pt', map_location='cpu', weights_only=False)
    verify_checkpoint_inputs(before)
    collator = NativeCollator(config['model']['path'], **config['processor'], brightness=tc['brightness'])
    state = restore_checkpoint(output/'step_000002.pt', model, optimizer, scheduler)
    starting = model.trainable_state()
    starting_difference = max((starting[key]-before['model'][key]).abs().max().item() for key in starting)
    learning_rates = [group['lr'] for group in optimizer.param_groups]
    rng_start = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    model.train(); optimizer.zero_grad(set_to_none=True)
    indices = []; losses = []; input_hashes = []
    with batch_stream(context['dataset'], collator, tc['microbatch'], config['train_seed'],
                      state['sampler_state']['next_batch'], tc['gradient_accumulation'],
                      **{key: config['loader'][key] for key in ['workers','prefetch','pin_memory']}) as stream:
        for prepared in stream:
            indices.append(list(prepared.indices))
            digest = hashlib.sha256()
            tensors = dict(prepared.batch['native'], **{key: prepared.batch[key]
                          for key in ['ego','nav','command','trajectory_gt']})
            for key, value in sorted(tensors.items()):
                if isinstance(value, torch.Tensor):
                    array = value.detach().contiguous().cpu().numpy()
                    digest.update(key.encode()); digest.update(str(array.shape).encode())
                    digest.update(memoryview(array).cast('B'))
            input_hashes.append(digest.hexdigest())
            batch = to_device(prepared.batch, config['device'])
            loss = planning_loss(predict(model, batch), batch['trajectory_gt'])
            (loss/tc['gradient_accumulation']).backward(); losses.append(loss.item())
        resumed_cursor = stream.state_dict()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    rng_after_backward = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, tc['grad_clip'], error_if_nonfinite=True)
    optimizer.step(); scheduler.step()
    actual = model.trainable_state()
    difference = max((actual[key].float()-expected['model'][key].float()).abs().max().item()
                     for key in actual)
    differences = sorted([{'name': key,
        'max_abs_difference':(actual[key].float()-expected['model'][key].float()).abs().max().item(),
        'expected_update_max_abs':(expected['model'][key].float()-before['model'][key].float()).abs().max().item()}
        for key in actual],key=lambda item:item['max_abs_difference'],reverse=True)
    moments = []
    live = optimizer.state_dict()
    for identifier, value in live['state'].items():
        reference = expected['optimizer']['state'][identifier]
        moments.append({'id':identifier,
            **{key:(value[key].detach().cpu()-reference[key]).abs().max().item()
               for key in ['exp_avg','exp_avg_sq']},
            'step_actual':float(value['step']), 'step_expected':float(reference['step'])})
    log = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    reference_loss = next(row['train_loss_l1_xy'] for row in log if row['step']==3)
    # Preserve strict numerical gates after enabling deterministic CUDA kernels.
    # Earlier nondeterministic failures remain in the historical v2 output.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        check_indices = sorted({0, len(context['dataset'])//2, len(context['dataset'])-1})
        checks = [to_device(context['collator']([context['dataset'][index]],seed=0),config['device'])
                  for index in check_indices]
        actual_predictions = torch.cat([predict(model,batch).float().cpu() for batch in checks])
        model.load_trainable_state(expected['model'])
        expected_predictions = torch.cat([predict(model,batch).float().cpu() for batch in checks])
        prediction_difference = (actual_predictions-expected_predictions).abs().max().item()
    parameter_tolerance, prediction_tolerance = 1e-6, 1e-5
    report = {'passed': difference <= parameter_tolerance and prediction_difference <= prediction_tolerance
                        and starting_difference == 0 and resumed_cursor == expected['sampler_state']
                        and abs(float(np.mean(losses))-reference_loss) <= 1e-6,
              'restored_step': 2, 'replayed_to_step': 3,
              'max_abs_trainable_parameter_difference': difference,
              'resumed_mean_loss': float(np.mean(losses)), 'uninterrupted_mean_loss': reference_loss,
              'starting_parameter_max_difference':starting_difference,
              'resumed_grad_norm':float(grad_norm),
              'uninterrupted_grad_norm':next(row['grad_norm'] for row in log if row['step']==3),
              'learning_rates_used':learning_rates,'top_parameter_differences':differences[:20],
              'cuda_rng_hash_start':rng_start, 'cuda_rng_hash_after_backward':rng_after_backward,
              'microbatch_input_hashes':input_hashes, 'microbatch_losses':losses,
              'max_prediction_difference_m':prediction_difference,
              'prediction_check_train_indices':check_indices,
              'prediction_tolerance_m':prediction_tolerance,
              'strict_1e6_parameter_check_passed':difference <= 1e-6,
              'bitwise_parameter_match':difference == 0,
              'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled(),
              'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
              'reproducibility': 'Strict deterministic CUDA policy; actual restored optimizer update compared with uninterrupted step3. Historical failed controls are preserved in outputs/fast_only/preflight_v2.',
              'top_moment_differences':sorted(moments,key=lambda item:item['exp_avg'],reverse=True)[:20],
              'sampler_state': resumed_cursor, 'sample_indices': indices,
              'tolerance': parameter_tolerance,
              'method': 'actual CUDA update with dropout/prefetch after checkpoint restore, versus uninterrupted saved step3'}
    restore_checkpoint(output/'last.pt', model, optimizer, scheduler)
    model.train(was_training)
    optimizer.zero_grad(set_to_none=True)
    write_json(output/'resume_verification.json', report)
    if not report['passed'] and raise_on_mismatch:
        raise AssertionError(f'Actual interrupted-update replay mismatch: {report}')
    return (report, actual) if capture_state else report


def context_from_checkpoint(config):
    state = torch.load(Path(config['output'])/'last.pt',map_location='cpu',weights_only=False)
    verify_checkpoint_inputs(state)
    model = make_model(config)
    tc=config['train']
    groups=[{'params':[p for name,p in model.named_parameters() if p.requires_grad and 'lora_' in name], 'lr':tc['lora_lr']},
            {'params':[p for name,p in model.named_parameters() if p.requires_grad and 'lora_' not in name], 'lr':tc['new_lr']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=tc['weight_decay'])
    scheduler=build_scheduler(optimizer,tc['steps'],tc['warmup_fraction'])
    restore_checkpoint(Path(config['output'])/'last.pt',model,optimizer,scheduler)
    return {'model':model,'optimizer':optimizer,'scheduler':scheduler,
            'dataset':FastDataset(config['data']['train_manifest'],config['data']['stats']),
            'collator':NativeCollator(config['model']['path'],**config['processor'])}


def reload_predictions(config):
    output = Path(config['output'])
    state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
    verify_checkpoint_inputs(state)
    model = make_model(config); model.load_trainable_state(state['model'])
    dataset = FastDataset(config['data']['train_manifest'], config['data']['stats'])
    collator = NativeCollator(config['model']['path'], **config['processor'])
    metrics, pred, gt = evaluate_dataset(model, dataset, collator, config['device'])
    original = np.load(output/'final/predictions.npz')['prediction']
    delta = float(np.abs(pred.numpy()-original).max())
    if delta > 1e-5: raise AssertionError(f'Reloaded prediction mismatch {delta}')
    save_predictions(output/'reload', dataset, pred, gt, cohort='train_smoke' if config['smoke'] else 'train')
    write_json(output/'reload/metrics.json', metrics)
    return {'max_abs_prediction_difference_m': delta, 'metrics': metrics, 'step': state['step']}, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/preflight_smoke.yaml')
    parser.add_argument('--deep-config', default='configs/deep_overfit_v2.yaml')
    parser.add_argument('--reuse-preflight', action='store_true',help='Verify and reuse completed preflight checkpoint; never retrain/overwrite it')
    parser.add_argument('--reuse-deep', action='store_true',help='Verify completed deep diagnostic; never retrain/overwrite it')
    args = parser.parse_args()
    config, deep = read_config(args.config), read_config(args.deep_config)
    for current,reuse in ((config,args.reuse_preflight),(deep,args.reuse_deep)):
        if Path(current['output'],'last.pt').exists() and not reuse:
            raise FileExistsError('Preflight output exists; use a new configuration/output directory')
    torch.set_num_threads(config['cpu_threads'])
    cap = config['execution']['stop_after']
    if config['execution']['stage'] != 'preflight' or cap is None or cap > 20 or deep['train']['steps'] > 1000:
        raise ValueError('Preflight refuses formal/large training budgets')
    for field in ['model','processor']:
        if config[field] != deep[field]: raise ValueError(f'Deep diagnostic changed {field}')
    for field in ['navigation_mode','navigation_origin','gnss_mount_xy_m','timing_policy','stats']:
        if config['data'][field] != deep['data'][field]:
            raise ValueError(f'Deep diagnostic changed input policy {field}')
    train = FastDataset(config['data']['train_manifest'], config['data']['stats'])
    val = FastDataset(config['data']['val_manifest'], config['data']['stats'])
    deep_path = Path(deep['data']['train_manifest'])
    # Choose one fixed TRAIN example before any outcome is observed.
    deep_path.write_text(json.dumps(train.rows[0], allow_nan=False)+'\n')
    audit_dir = Path('outputs/fast_only/audit_v2'); audit_dir.mkdir(parents=True, exist_ok=True)
    audit = {'status': 'data_policies_verified_runtime_pending', 'formal_training_started': False,
             'training_config': args.config, 'deep_config': args.deep_config,
             'data': json.loads(Path(config['data']['manifest_dir'],'build_report.json').read_text()),
             'source': json.loads(Path(config['data']['source_audit']).read_text()),
             'test_usage': 'schema_geometry_time audit only; no TEST predictions/metrics',
             'cv_train': constant_velocity_metrics(train), 'cv_val': constant_velocity_metrics(val),
             'cv_anchor_train': audit_constant_velocity_anchor(train),
             'cv_anchor_val': audit_constant_velocity_anchor(val)}
    write_json(audit_dir/'configuration_audit.json', audit)
    print({'phase':'final_configuration_smoke','train':len(train),'val':len(val),
           'original_pixels':True,'training_steps':cap,
           'unchanged_formal_schedule_steps':config['train']['steps']}, flush=True)
    context = context_from_checkpoint(config) if args.reuse_preflight else run_training(config)
    resume = verify_real_resume(context, config)
    model = context['model']; collator = context['collator']
    val_metrics, pred, gt = evaluate_dataset(model, val, collator, config['device'])
    save_predictions(Path(config['output'])/'val', val, pred, gt, cohort='val')
    write_json(Path(config['output'])/'val/metrics.json', val_metrics)
    swap = evaluate_image_swap(model, val, collator, config['device'], max_samples=200)
    target_swap = navigation_sensitivity(model, val, collator, config['device'])
    write_json(Path(config['output'])/'image_swap.json', swap)
    write_json(Path(config['output'])/'navigation_sensitivity.json', target_swap)
    if swap['mean_point_change_m'] is None or swap['mean_point_change_m'] <= 1e-6:
        raise AssertionError('Visual channel sensitivity not established')
    if target_swap['mean_point_change_m'] <= 1e-6:
        raise AssertionError('Navigation target sensitivity not established')
    _release(context); del model, collator; gc.collect(); torch.cuda.empty_cache()
    reload_report, reloaded_model = reload_predictions(config)
    del reloaded_model; gc.collect(); torch.cuda.empty_cache()
    print({'phase':'single_sample_deep_overfit','steps':deep['train']['steps']}, flush=True)
    if args.reuse_deep:
        state=torch.load(Path(deep['output'])/'last.pt',map_location='cpu',weights_only=False)
        verify_checkpoint_inputs(state)
        if state['config'] != deep:
            raise ValueError('Reused deep checkpoint has a different configuration')
        final=json.loads(Path(deep['output'],'final_train_metrics.json').read_text())
    else:
        context = run_training(deep)
        final = context['final_train']
    initial = json.loads(Path(deep['output'],'initial_train_metrics.json').read_text())
    if not args.reuse_deep:
        _release(context)
    deep_reload, deep_model = reload_predictions(deep)
    del deep_model; gc.collect(); torch.cuda.empty_cache()
    loss = final['loss_l1_xy']
    deep_result = {'passed': loss <= .05, 'threshold_l1_xy_m': .05,
                   'initial':initial,'final':final,'reload':deep_reload,
                   'loss_reduction_percent':(1-loss/initial['loss_l1_xy'])*100,
                   'protocol_differences': 'One TRAIN row, microbatch1/accumulation1/brightness0; same original-image/navigation/model/loss/normalizer policies'}
    write_json(Path(deep['output'])/'deep_overfit_verification.json', deep_result)
    summary = {'passed': deep_result['passed'], 'formal_training_started':False,
               'configuration_smoke': {'optimizer_steps':cap,
                    'schedule_budget_steps':config['train']['steps'],
                    'effective_batch':config['train']['effective_batch'], 'val':val_metrics,
                    'reload':reload_report,'resume':resume},
               'deep_overfit':deep_result,'image_swap':swap,'navigation_sensitivity':target_swap,
               'benchmark':json.loads(Path(config['output'],'p0_benchmark.json').read_text()),
               'limitations': ['Nominal simulation time, not independently measured per-sensor timestamps.',
                              'Private expert lookahead thresholds unpublished; current official near waypoint used as recorded.',
                              'Only9 VAL windows available for swap, not200.',
                              'CV numeric-anchor eligibility is reported independently; insufficient is not passed.',
                              'No formal8000-step training or TEST open-loop evaluation.']}
    write_json(audit_dir/'preflight_summary.json', summary)
    print(json.dumps(summary,indent=2), flush=True)
    if not deep_result['passed']:
        raise AssertionError(f'Deep-overfit L1 {loss} exceeds .05m')


if __name__ == '__main__': main()
