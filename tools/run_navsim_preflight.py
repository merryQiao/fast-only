"""Run the bounded real NAVSIM official-configuration acceptance experiment."""
import argparse
import copy
import gc
import json
from pathlib import Path
import numpy as np
import torch
from fast_only.common import predict,to_device,write_json
from fast_only.loading import batch_stream
from fast_navsim.collate import NavsimCollator
from fast_navsim.config import read_config
from fast_navsim.data import NavsimDataset,select_deep_row
from fast_navsim.evaluate import evaluate_dataset,save_predictions,channel_sensitivity,constant_velocity_metrics
from fast_navsim.runtime import make_model,restore_checkpoint,verify_checkpoint
from fast_navsim.train import run_training,perform_update,optimizer_context


def release(context):
    context.clear();gc.collect();torch.cuda.empty_cache()


def compare_formal_config(formal,preflight):
    def difference(a,b,prefix=''):
        changes=[]
        for key in sorted(a.keys()|b.keys()):
            path=f'{prefix}.{key}' if prefix else key
            if isinstance(a.get(key),dict) and isinstance(b.get(key),dict):changes+=difference(a[key],b[key],path)
            elif a.get(key)!=b.get(key):changes.append({'field':path,'formal':a.get(key),'preflight':b.get(key)})
        return changes
    changes=difference(formal,preflight)
    allowed={'output','execution.stage','execution.stop_after','train.eval_every','train.checkpoint_every'}
    if any(row['field'] not in allowed for row in changes):raise ValueError(f'Formal/preflight policy drift: {changes}')
    return {'passed':True,'differences':changes,'formal_started':False}


def verify_real_resume(context,config):
    output=Path(config['output']);tc=config['train']
    before=torch.load(output/'step_000002.pt',map_location='cpu',weights_only=False)
    expected=torch.load(output/'step_000003.pt',map_location='cpu',weights_only=False)
    verify_checkpoint(before)
    model,opt,scheduler=(context[k] for k in ['model','optimizer','scheduler'])
    collator=NavsimCollator(config['model']['path'],**config['processor'],brightness=tc['brightness'])
    restore_checkpoint(output/'step_000002.pt',model,opt,scheduler)
    with batch_stream(context['dataset'],collator,tc['microbatch'],config['train_seed'],
            before['sampler_state']['next_batch'],tc['gradient_accumulation'],
            **{k:config['loader'][k] for k in ['workers','prefetch','pin_memory']}) as stream:
        replay=perform_update(model,opt,scheduler,stream,config)
    actual=model.trainable_state()
    parameter_difference=max((actual[k]-expected['model'][k]).abs().max().item() for k in actual)
    moments=opt.state_dict()['state'];moment_difference=0.
    for key,value in moments.items():
        for field in ['exp_avg','exp_avg_sq','step']:
            moment_difference=max(moment_difference,(value[field].detach().cpu()-expected['optimizer']['state'][key][field]).abs().max().item())
    log=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    reference=next(row for row in log if row['step']==3)
    scheduler_equal=scheduler.state_dict()==expected['scheduler']
    lr_equal=[g['lr'] for g in opt.param_groups]==[g['lr'] for g in expected['optimizer']['param_groups']]
    model.eval()
    with torch.no_grad():
        indices=sorted({0,len(context['dataset'])//2,len(context['dataset'])-1})
        batches=[to_device(context['collator']([context['dataset'][i]]),config['device']) for i in indices]
        a=torch.cat([predict(model,b).float().cpu() for b in batches])
        model.load_trainable_state(expected['model'])
        b=torch.cat([predict(model,b).float().cpu() for b in batches])
        pose_difference=(a-b).abs().max().item()
    result={'passed':parameter_difference==0 and moment_difference==0 and pose_difference==0 and scheduler_equal and lr_equal
                    and replay['sampler_state']==expected['sampler_state']
                    and replay['input_hashes']==reference['input_hashes']
                    and replay['train_loss_l1_pose']==reference['train_loss_l1_pose'],
            'restored_step':2,'replayed_step':3,'max_parameter_difference':parameter_difference,
            'max_optimizer_moment_difference':moment_difference,'max_prediction_difference':pose_difference,
            'replayed_loss':replay['train_loss_l1_pose'],'uninterrupted_loss':reference['train_loss_l1_pose'],
            'replayed_grad_norm':replay['grad_norm'],'uninterrupted_grad_norm':reference['grad_norm'],
            'input_hashes_equal':replay['input_hashes']==reference['input_hashes'],
            'scheduler_state_equal':scheduler_equal,'learning_rates_equal':lr_equal,
            'sampler_state':replay['sampler_state'],'prediction_check_indices':indices,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
            'method':'actual dropout/prefetch optimizer update2->3 against uninterrupted checkpoint; exact gates'}
    restore_checkpoint(output/'last.pt',model,opt,scheduler);model.train();opt.zero_grad(set_to_none=True)
    write_json(output/'resume_verification.json',result)
    if not result['passed']:raise AssertionError(f'Actual NAVSIM resume mismatch: {result}')
    return result


def reload_predictions(config):
    output=Path(config['output']);state=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    verify_checkpoint(state)
    if state['config']!=config:raise ValueError('Reload YAML differs from checkpoint config')
    model=make_model(config);model.load_trainable_state(state['model'])
    dataset=NavsimDataset(config['data']['train_manifest'],config['data']['stats'])
    collator=NavsimCollator(config['model']['path'],**config['processor'])
    metrics,pred,gt=evaluate_dataset(model,dataset,collator,config['device'])
    original=np.load(output/'final/predictions.npz')['prediction']
    difference=float(np.max(np.abs(pred.numpy()-original)))
    if difference!=0:raise AssertionError(f'Reload pose mismatch: {difference}')
    result={'passed':True,'step':state['step'],'samples':len(dataset),
            'max_pose_difference':difference,'metrics':metrics}
    write_json(output/'reload_verification.json',result)
    return result,model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/navsim_preflight.yaml')
    parser.add_argument('--deep-config',default='configs/navsim_deep.yaml')
    parser.add_argument('--formal-config',default='configs/navsim.yaml')
    args=parser.parse_args()
    config,deep,formal=(read_config(p) for p in [args.config,args.deep_config,args.formal_config])
    torch.set_num_threads(config['cpu_threads'])
    if config['execution']['stage']!='preflight' or config['execution']['stop_after']!=4:
        raise ValueError('Canonical NAVSIM preflight must execute exactly4 updates')
    for key in ['model','processor','loader','reproducibility','loss']:
        if config[key]!=deep[key]:raise ValueError(f'Deep input/model policy drift: {key}')
    for key in ['root','stats','navigation','coordinate_convention','frequency_hz','source_audit']:
        if config['data'][key]!=deep['data'][key]:raise ValueError(f'Deep data policy drift: {key}')
    for c in [config,deep]:
        output=Path(c['output'])
        if output.exists() and any(output.iterdir()):raise FileExistsError('Use new output directory for a new preflight')
    audit=Path('outputs/navsim/audit');audit.mkdir(parents=True,exist_ok=True)
    comparison=compare_formal_config(formal,config)
    train=NavsimDataset(config['data']['train_manifest'],config['data']['stats'])
    val=NavsimDataset(config['data']['val_manifest'],config['data']['stats'])
    deep_rows=[json.loads(line) for line in Path(deep['data']['train_manifest']).read_text().splitlines() if line]
    if deep_rows!=[select_deep_row(train.rows)]:
        raise ValueError('Deep window must be the first TRAIN row with current speed>=1m/s, chosen before outcomes')
    write_json(audit/'configuration_audit.json',{'formal_preflight_comparison':comparison,
        'cv_train':constant_velocity_metrics(train),'cv_val':constant_velocity_metrics(val),
        'train_windows':len(train),'val_windows':len(val),
        'protocol':'official NAVSIM 2Hz input/eight4s SE2 poses; bounded mini smoke, not navtest/EPDMS',
        'deep_acceptance':{'pose_l1_max':.05,'xy_ade_m_max':.05,'heading_mae_rad_max':.02,'loss_reduction_min_percent':95.},
        'formal_started':False,'test_usage':'none; no TEST optimization or metrics'})
    context=run_training(config)
    resume=verify_real_resume(context,config)
    metrics,pred,gt=evaluate_dataset(context['model'],context['val'],context['collator'],config['device'])
    save_predictions(Path(config['output'])/'val',context['val'],pred,gt,cohort='val')
    write_json(Path(config['output'])/'val/metrics.json',metrics)
    channels=channel_sensitivity(context['model'],context['val'],context['collator'],config['device'])
    write_json(Path(config['output'])/'channel_sensitivity.json',channels)
    for key in ['image_swap','command_intervention']:
        if channels[key]['mean_point_change_m']<=1e-6:raise AssertionError(f'NAVSIM {key} channel unresponsive')
    release(context)
    reloaded,model=reload_predictions(config);del model;gc.collect();torch.cuda.empty_cache()
    context=run_training(deep)
    initial=json.loads(Path(deep['output'],'initial_train_metrics.json').read_text())
    final=context['final_train'];release(context)
    deep_reload,model=reload_predictions(deep);del model;gc.collect();torch.cuda.empty_cache()
    reduction=(1-final['loss_l1_pose']/initial['loss_l1_pose'])*100
    deep_report={'passed':final['loss_l1_pose']<=.05 and final['ADE']<=.05
                 and final['heading_mae_rad']<=.02 and reduction>=95,
                 'initial':initial,'final':final,'loss_reduction_percent':reduction,'reload':deep_reload,
                 'protocol_differences':'single TRAIN row, microbatch1/accumulation1/brightness0, separate300-step diagnostic schedule; same full-TRAIN stats/native images/ego/command/model/pose loss'}
    write_json(Path(deep['output'])/'deep_overfit_verification.json',deep_report)
    benchmark=json.loads(Path(config['output'],'p0_benchmark.json').read_text())
    result={'passed':deep_report['passed'],'correctness_passed':deep_report['passed'],
            'latency_passed':benchmark['within_budget_p95'],
            'configuration_and_latency_passed':deep_report['passed'] and benchmark['within_budget_p95'],
            'acceptance_scope':'Configuration correctness smoke; latency readiness is a separate measured condition, independent-process verification is required separately.',
            'formal_training_started':False,
            'configuration_smoke':{'steps':4,'schedule_steps':8000,'effective_batch':16,
                 'val':metrics,'resume':resume,'reload':reloaded},
            'deep_overfit':deep_report,'channel_sensitivity':channels,
            'benchmark':benchmark,
            'limitations':['Bounded OpenScene mini logs, not official navtrain/navtest coverage.',
                           'No official EPDMS/PDMS simulation score or leaderboard evaluation.',
                           'Small independent VAL; channel response is not semantic quality or generalization.',
                           'Raw-resolution500ms budget measured separately; no silent resize.',
                           'No full dataset or8000-step training.']}
    write_json(audit/'preflight_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise AssertionError('NAVSIM single-sample overfit criteria not met')


if __name__=='__main__':main()
