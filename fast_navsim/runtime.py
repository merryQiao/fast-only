"""Experiment identity and complete optimizer/RNG/sampler checkpoint state."""
import hashlib
import importlib.metadata
import json
import random
import platform
import os
from pathlib import Path
import numpy as np
import torch
from fast_only.common import apply_reproducibility, fingerprint
from .model import NavsimPlanner


def make_model(config):
    apply_reproducibility(config)
    fields=['lora_rank','lora_alpha','lora_dropout','lora_target_suffixes','lora_parameter_dtype']
    dtype=torch.bfloat16 if config['bf16'] and torch.cuda.is_available() else torch.float32
    model=NavsimPlanner.from_pretrained(config['model']['path'],dtype=dtype,
                                      **{k:config['model'][k] for k in fields})
    return model.to(config['device'])


def dataset_fingerprints(config,*,include_images=True):
    paths={f'data/{key}':Path(config['data'][key])
           for key in ['train_manifest','val_manifest','stats','source_audit']}
    audit_path=Path(config['data']['source_audit'])
    audit=json.loads(audit_path.read_text())
    for index,item in enumerate(audit.get('sources',[])):
        path=Path(item['path'])
        if not path.is_absolute():path=audit_path.parent/path
        if fingerprint(path)!=item['sha256']:raise ValueError(f'Official source fingerprint mismatch: {path}')
        paths[f'source/{index}/{path.name}']=path
    for package in ['fast_navsim','fast_only']:
        for path in sorted((Path(__file__).parents[1]/package).glob('*.py')):
            paths[f'implementation/{package}/{path.name}']=path
    for name in ['prepare_navsim_smoke.py','run_navsim_preflight.py','verify_navsim_reload.py']:
        path=Path(__file__).parents[1]/'tools'/name
        if path.exists():paths[f'implementation/tools/{name}']=path
    model=Path(config['model']['path'])
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix in {'.safetensors','.json','.jinja','.txt'}:
            paths[f'model/{path.name}']=path
    if not any(k.endswith('.safetensors') for k in paths):raise ValueError('No local model weights')
    if include_images:
        images=set()
        for key in ['train_manifest','val_manifest']:
            rows=[json.loads(line) for line in Path(config['data'][key]).read_text().splitlines() if line]
            images.update(p for row in rows for p in row['images'])
        for path in sorted(images):paths[f'image/{path}']=Path(path)
    return {key:fingerprint(path) for key,path in paths.items()}


def verify_checkpoint(state,*,include_images=True):
    if state.get('format_version')!='navsim-v1':raise ValueError('Not a NAVSIM checkpoint')
    if state.get('versions')!=environment_versions():
        raise ValueError('Checkpoint environment versions changed; bitwise resume is unsupported')
    current=dataset_fingerprints(state['config'],include_images=include_images)
    if current!=state['identity']:
        changed=sorted(k for k in current.keys()|state['identity'].keys() if current.get(k)!=state['identity'].get(k))
        raise ValueError(f'Checkpoint fingerprint mismatch: {changed}')


def environment_versions():
    result={'dataset':'OpenDriveLab/OpenScene','torch':str(torch.__version__),
            'cuda_runtime':torch.version.cuda,'python':platform.python_version()}
    for name in ['transformers','peft','flash-linear-attention','causal-conv1d','tilelang','triton','Pillow','numpy','tokenizers']:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    for key in ['FLA_TILELANG','CUBLAS_WORKSPACE_CONFIG']:
        result[key]=os.environ.get(key)
    return result


def save_checkpoint(path,model,optimizer,scheduler,config,*,step,sampler_state,identity,best_metric):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'format_version':'navsim-v1','model':model.trainable_state(),
           'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
           'config':config,'step':step,'sampler_state':sampler_state,'identity':identity,
           'best_metric':best_metric,'accumulation_boundary':True,'grad_scaler':None,
           'windows_seen':step*config.get('train',{}).get('effective_batch',1),
           'normalization_stats':json.loads(Path(config['data']['stats']).read_text()),
           'versions':environment_versions(),
           'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                  'torch':torch.get_rng_state(),
                  'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,temporary);temporary.replace(path)


def restore_checkpoint(path,model,optimizer=None,scheduler=None):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format_version')!='navsim-v1' or state.get('accumulation_boundary') is not True:
        raise ValueError('Expected NAVSIM checkpoint at an optimizer boundary')
    model.load_trainable_state(state['model'])
    if optimizer is not None:optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'])
    if torch.cuda.is_available() and state['rng']['cuda']:torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def input_fingerprint(batch):
    digest=hashlib.sha256()
    tensors={**batch['native'],**{k:batch[k] for k in ['ego','command','trajectory_gt']}}
    for key,value in sorted(tensors.items()):
        if isinstance(value,torch.Tensor):
            array=value.detach().contiguous().cpu().numpy()
            digest.update(key.encode());digest.update(str(array.shape).encode())
            digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()
