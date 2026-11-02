"""Guarded real NAVSIM configuration smoke and deterministic optimizer updates."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from fast_only.common import apply_reproducibility,predict,seed_all,to_device,write_json
from fast_only.loading import batch_stream
from fast_only.train import build_scheduler
from .collate import NavsimCollator
from .config import validate_config,read_config
from .data import NavsimDataset
from .model import pose_loss
from .runtime import dataset_fingerprints,input_fingerprint,make_model,save_checkpoint,restore_checkpoint,verify_checkpoint


def execution_limit(config,requested=None):
    stage=config['execution']['stage']
    if stage=='formal' and config['data']['cohort']=='bounded_mini_preflight':
        raise ValueError('Full formal training is refused on the bounded mini preflight cohort')
    cap=config['execution']['stop_after'] or config['train']['steps']
    if requested is not None:
        if isinstance(requested,bool) or not isinstance(requested,int) or not 0<requested<=cap:
            raise ValueError('Requested training budget exceeds declared execution cap')
        cap=requested
    return min(cap,config['train']['steps'])


def verify_cohort(train,val):
    if not train or not val or {r['split'] for r in train}!={'train'} or {r['split'] for r in val}!={'val'}:
        raise ValueError('Expected nonempty independent TRAIN/VAL rows')
    for key in ['route','token']:
        if {r[key] for r in train}&{r[key] for r in val}:
            raise ValueError(f'TRAIN/VAL {key} leakage')


def verify_build_policy(config,rows):
    report=json.loads((Path(config['data']['train_manifest']).parent/'build_report.json').read_text())
    if report.get('split_seed')!=config['split_seed']:
        raise ValueError('Manifest split seed differs from configured split_seed')
    root=Path(config['data']['root']).resolve()
    if Path(report.get('data_root','')).resolve()!=root:
        raise ValueError('Manifest data root differs from configuration')
    if any(not Path(image).resolve().is_relative_to(root) for row in rows for image in row['images']):
        raise ValueError('Manifest image escapes configured data root')


def prepare_output(output,*,resume=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'{output} contains prior run evidence; choose a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    return output


def optimizer_context(model,config):
    tc=config['train']
    groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n],
             'lr':tc['lora_lr']},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and 'lora_' not in n],
             'lr':tc['new_lr']}]
    optimizer=torch.optim.AdamW(groups,weight_decay=tc['weight_decay'])
    return optimizer,build_scheduler(optimizer,tc['steps'],tc['warmup_fraction'])


def perform_update(model,optimizer,scheduler,stream,config):
    tc=config['train'];model.train();optimizer.zero_grad(set_to_none=True)
    losses=[];indices=[];hashes=[]
    for _ in range(tc['gradient_accumulation']):
        prepared=next(stream)
        hashes.append(input_fingerprint(prepared.batch));indices.append(list(prepared.indices))
        batch=to_device(prepared.batch,config['device'])
        loss=pose_loss(predict(model,batch),batch['trajectory_gt'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pose loss')
        (loss/tc['gradient_accumulation']).backward();losses.append(loss.item())
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    norm=torch.nn.utils.clip_grad_norm_(parameters,tc['grad_clip'],error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {'train_loss_l1_pose':float(np.mean(losses)),'grad_norm':float(norm),
            'microbatch_losses':losses,'sample_indices':indices,'input_hashes':hashes,
            'sampler_state':stream.state_dict(),
            'lora_lr':optimizer.param_groups[0]['lr'],'new_lr':optimizer.param_groups[1]['lr']}


def run_training(config,*,resume=None,stop_after=None,p0_only=False):
    from .evaluate import evaluate_dataset,save_predictions
    validate_config(config);cap=execution_limit(config,stop_after)
    apply_reproducibility(config);seed_all(config['train_seed'])
    torch.set_num_threads(config['cpu_threads'])
    output=prepare_output(config['output'],resume=resume is not None)
    dataset=NavsimDataset(config['data']['train_manifest'],config['data']['stats'])
    val=NavsimDataset(config['data']['val_manifest'],config['data']['stats'])
    verify_cohort(dataset.rows,val.rows)
    verify_build_policy(config,dataset.rows+val.rows)
    identity=dataset_fingerprints(config)
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        verify_checkpoint(state)
        if state['config']!=config:raise ValueError('Resume configuration differs from checkpoint')
    write_json(output/'config.json',config);write_json(output/'identity.json',identity)
    print({'phase':'load_native_navsim_planner','output':str(output)},flush=True)
    model=make_model(config)
    tc=config['train'];collator=NavsimCollator(config['model']['path'],**config['processor'])
    augmented=NavsimCollator(config['model']['path'],**config['processor'],brightness=tc['brightness'])
    optimizer,scheduler=optimizer_context(model,config)
    inventory={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'total_parameters':sum(p.numel() for p in model.parameters()),
               'lora_targets':model.lora_targets,
               'trainable_dtypes':sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
               'continuous_target_module':any('nav_mlp' in n for n,p in model.named_parameters())}
    write_json(output/'model_inventory.json',inventory)
    start=0;best=float('inf')
    if resume:
        state=restore_checkpoint(resume,model,optimizer,scheduler);start=state['step'];best=state['best_metric']
        expected={'sampler_version':'counter_v1','seed':config['train_seed'],
                  'next_batch':start*tc['gradient_accumulation'],'microbatch':tc['microbatch'],
                  'dataset_size':len(dataset)}
        if state['sampler_state']!=expected:raise ValueError('Checkpoint sampler cursor mismatch')
    else:
        initial,pred,gt=evaluate_dataset(model,dataset,collator,config['device'])
        write_json(output/'initial_train_metrics.json',initial)
        save_predictions(output/'initial',dataset,pred,gt,cohort='train')
        model.train()
        batch=to_device(augmented([dataset[i%len(dataset)] for i in range(tc['microbatch'])],
                        seed=config['train_seed']),config['device'])
        prediction=predict(model,batch);loss=pose_loss(prediction,batch['trajectory_gt']);loss.backward()
        gradients=model.gradient_report()
        if not gradients['vision_frozen'] or gradients['vision_has_gradient']:
            raise RuntimeError('Native Vision+Merger freeze failed')
        for name in ['ego','command','queries','head','full_attention_lora','gdn_layer_ffn_lora']:
            if not gradients[name]['finite'] or not gradients[name]['nonzero']:
                raise RuntimeError(f'Missing/nonfinite trainable gradient: {name}')
        write_json(output/'p0_forward_backward.json',{'prediction_shape':list(prediction.shape),
            'loss_l1_pose':loss.item(),'gradients':gradients,'layout':model.last_layout,
            'image_audit':batch['image_audit'],'continuous_target_token':False,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None})
        optimizer.zero_grad(set_to_none=True);del batch,prediction,loss
        if config['benchmark']['enabled']:
            from fast_only.benchmark import measure
            timing=measure(model,dataset,collator,config['device'],
                  **{k:config['benchmark'][k] for k in ['warmup','repeats','modes']})
            timing.pop('within_100ms_p95',None)
            budget=config['benchmark']['p95_budget_ms'];timing['planning_cycle_budget_ms']=budget
            for mode in config['benchmark']['modes']:
                dist=timing[mode+'_ms'];dist['fraction_over_budget']=sum(x>budget for x in dist['samples'])/dist['count']
            timing['within_budget_p95']=timing['disk_e2e_ms']['p95']<=budget
            write_json(output/'p0_benchmark.json',timing)
        print({'phase':'navsim_p0_passed','output':str(output)},flush=True)
        if p0_only:return {'model':model,'p0_only':True}
    if cap<=start:raise ValueError('Execution cap does not exceed restored optimizer step')
    deep=config['execution']['stage']=='deep_overfit';selection=dataset if deep else val
    if torch.device(config['device']).type=='cuda':torch.cuda.reset_peak_memory_stats(config['device'])
    started=time.perf_counter()
    with (output/'train.jsonl').open('a' if resume else 'w') as log:
        with batch_stream(dataset,augmented,tc['microbatch'],config['train_seed'],
             start*tc['gradient_accumulation'],(cap-start)*tc['gradient_accumulation'],
             **{k:config['loader'][k] for k in ['workers','prefetch','pin_memory']}) as stream:
            for step in range(start+1,cap+1):
                row=perform_update(model,optimizer,scheduler,stream,config)
                row.update(step=step,effective_batch=tc['effective_batch'],
                   windows_seen=step*tc['effective_batch'],epoch_equivalent=step*tc['effective_batch']/len(dataset),
                   elapsed_s=time.perf_counter()-started)
                if step%tc['eval_every']==0 or step==cap:
                    metrics,pred,gt=evaluate_dataset(model,selection,collator,config['device'])
                    row['train_diagnostic_metrics' if deep else 'val_metrics']=metrics
                    if metrics['ADE']<best:
                        best=metrics['ADE']
                        save_checkpoint(output/('best_train.pt' if deep else 'best_val.pt'),model,optimizer,scheduler,
                            config,step=step,sampler_state=stream.state_dict(),identity=identity,best_metric=best)
                if step%tc['checkpoint_every']==0 or step==cap:
                    for path in [output/'last.pt',output/f'step_{step:06d}.pt']:
                        save_checkpoint(path,model,optimizer,scheduler,config,step=step,
                             sampler_state=stream.state_dict(),identity=identity,best_metric=best)
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%10==0 or step==cap:print({'phase':'train',**row},flush=True)
    final,pred,gt=evaluate_dataset(model,dataset,collator,config['device'])
    final.update(optimizer_step=cap,windows_seen=cap*tc['effective_batch'],
        epoch_equivalent=cap*tc['effective_batch']/len(dataset))
    if torch.cuda.is_available():final.update(peak_training_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                               peak_training_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_json(output/'final_train_metrics.json',final)
    save_predictions(output/'final',dataset,pred,gt,cohort='train')
    print({'phase':'navsim_training_complete','output':str(output),'final':final},flush=True)
    return {'model':model,'optimizer':optimizer,'scheduler':scheduler,'dataset':dataset,'val':val,
            'collator':collator,'final_train':final,'config':config}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/navsim_preflight.yaml')
    parser.add_argument('--resume');parser.add_argument('--stop-after',type=int)
    parser.add_argument('--p0-only',action='store_true')
    args=parser.parse_args()
    run_training(read_config(args.config),resume=args.resume,stop_after=args.stop_after,p0_only=args.p0_only)


if __name__=='__main__':main()
