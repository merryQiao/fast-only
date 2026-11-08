"""Authenticated direct-KV inputs; cache vision only, never language features."""
from contextlib import contextmanager
import json
from pathlib import Path

import torch
from PIL import Image
import yaml

from fast_only.common import fingerprint,to_device
from fast_navsim.collate import NavsimCollator
from slowfast_v11.model import FastWithSlowKV
from slowfast_v11.protocol import LayerKV,KVSnapshot
from slowfast_v11.validation import restore_weights,state_digest,canonical_state
from tools.train_frozen_expert import cached_vision


def read_config(path='configs/slowfast_direct_full.yaml'):
    c=yaml.safe_load(Path(path).read_text())
    fixed=dict(schema='direct_full_language_v1',train_seed=42,arms=['full_fast_only','full_correct_kv','full_fixed_kv'],
        train_windows=512,val_windows=256,optimizer_updates=1024,micro_batch=1,gradient_accumulation=4,
        language_lr=1e-5,new_lr=5e-5,weight_decay=.01,gradient_clip=1.,warmup_updates=64,minimum_lr_ratio=.1,
        gradient_checkpointing=False,normalization='preserve_source_checkpoint_train256_interface')
    if any(c[k]!=v for k,v in fixed.items()):raise ValueError('Predeclared full-language training config differs')
    return c


def source_info(c):
    source=json.loads((Path(c['source_run'])/'result.json').read_text());fast=json.loads((Path(c['fast_run'])/'result.json').read_text())
    if source['arms']['joint']['checkpoint_sha256']!=c['source_checkpoint_sha256'] or fast['checkpoint_sha256']!=c['fast_checkpoint_sha256']:
        raise ValueError('Source/latest Fast identities differ')
    if fast['protocol']['source_data']!=source['data_identities'] or fast['protocol']['assets']!=source['assets']:
        raise ValueError('Fast/Slow preprocessing and asset identities differ')
    fast_path=Path(c['fast_run'])/'final.pt'
    if fingerprint(fast_path)!=c['fast_checkpoint_sha256']:raise ValueError('Authenticated Fast checkpoint differs')
    authenticated=torch.load(fast_path,map_location='cpu',weights_only=False)
    if json.loads(json.dumps(authenticated['protocol']))!=fast['protocol']:raise ValueError('Fast embedded source protocol differs')
    del authenticated
    stats_path=Path(source['config']['data']['stats'])
    if fingerprint(stats_path)!=source['data_identities']['stats']:raise ValueError('Checkpoint normalizer differs')
    return source,fast,json.loads(stats_path.read_text())


def load_fast(c):
    source,result,stats=source_info(c);original=source['config']
    for name,sha in source['assets']['fast'].items():
        if fingerprint(Path(original['fast']['model_path'])/name)!=sha:raise ValueError('Fast assets differ')
    path=Path(c['fast_run'])/'final.pt'
    if fingerprint(path)!=c['fast_checkpoint_sha256']:raise ValueError('Fast checkpoint differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=FastWithSlowKV.from_pretrained(original['fast']['model_path'],dtype=torch.bfloat16,**original['model']).cuda()
    restore_weights(model,saved['model'])
    if state_digest(canonical_state(model))!=state_digest(saved['model']):raise ValueError('Restored Fast parameters differ')
    del saved
    collator=NavsimCollator(original['fast']['model_path'],**original['processor'])
    return model,collator,stats,source


def train_rows(c):
    rows=[]
    for path in [c['original_train_manifest'],c['promoted_holdout_manifest']]:
        rows.extend([dict(json.loads(s),split='train',cohort='train') for s in Path(path).read_text().splitlines() if s.strip()])
    if len(rows)!=512 or len({r['token'] for r in rows})!=512:raise ValueError('Expected exact expanded TRAIN512')
    return rows


@torch.no_grad()
def vision_packet(model,collator,row,stats):
    images=[]
    for path in row['images']:
        with Image.open(path) as image:images.append(image.convert('RGB'))
    sample=dict(images=images,ego=torch.tensor([(v-m)/s for v,m,s in zip(row['ego'],stats['ego_mean'],stats['ego_std'])]),
                command=torch.tensor(row['command'],dtype=torch.long),trajectory_gt=torch.empty(0))
    packed=collator([sample]);batch=to_device({k:packed[k] for k in ['native','ego','command']},'cuda:0')
    native=batch['native'];features=model.backbone.get_image_features(native['pixel_values'],native['image_grid_thw'],return_dict=True).pooler_output
    original=model(**batch).float().cpu()
    native=dict(native,pixel_values=torch.empty(0,device='cuda:0'))
    packet=dict(token=row['token'],batch=dict(batch,native=native),vision=tuple(t.detach().clone() for t in features),
                original_prediction=original,image_audit=packed['image_audit'])
    with cached_vision(model.backbone,packet['vision'],native['image_grid_thw']):after=model(**packet['batch']).float().cpu()
    if not torch.equal(original,after):raise ValueError('Frozen vision cache changes native prediction')
    return packet


class LegacySources:
    def __init__(self,c):
        self.root=Path('outputs/slowfast_readability/run_01/frozen_cache')
        self.manifest=json.loads((self.root/'manifest.json').read_text());_,fast,_=source_info(c)
        if (fingerprint(self.root/'manifest.json')!=fast['protocol']['feature_manifest_sha256']
                or self.manifest['source_checkpoint']!=c['source_checkpoint_sha256']):raise ValueError('Authenticated old raw-KV manifest differs')
        self.entries={t:(s,j) for s in self.manifest['shards'] for j,t in enumerate(s['tokens'])};self.verified=set()
    def packet(self,token,device='cuda:0'):
        shard,index=self.entries[token];path=self.root/shard['path']
        if shard['path'] not in self.verified:
            if fingerprint(path)!=shard['sha256']:raise ValueError('Old raw-KV shard differs')
            self.verified.add(shard['path'])
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        return dict(keys=saved['keys'][index:index+1].to(device),values=saved['values'][index:index+1].to(device),
                    positions=saved['positions'][:,index:index+1].to(device),valid=saved['valid'][index:index+1].to(device))


def snapshot(packet):
    b,l,t,width=packet['keys'].shape
    if b!=1 or l!=6 or width!=512:raise ValueError('Expected native six-layer2head256dim source')
    layers=tuple(LayerKV(packet['keys'][:,i].view(b,t,2,256).transpose(1,2),packet['values'][:,i].view(b,t,2,256).transpose(1,2)) for i in range(6))
    return KVSnapshot(layers,packet['positions'],packet['valid'])


def equal_source_layout(a,b):
    if not torch.equal(a['positions'],b['positions']) or not torch.equal(a['valid'],b['valid']):raise ValueError('Fixed source/target layouts differ')


def direct_forward(model,packet,arm,source=None):
    if arm not in ['full_fast_only','full_correct_kv','full_fixed_kv']:raise ValueError('Unknown direct training arm')
    kwargs={}
    if arm!='full_fast_only':
        if source is None:raise ValueError('Direct KV arm requires complete frozen source')
        kwargs=dict(snapshot=snapshot(source),metadata=torch.tensor([[0.,0.,0.,0.,1.,1.]],device=packet['batch']['ego'].device))
    with cached_vision(model.backbone,packet['vision'],packet['batch']['native']['image_grid_thw']):
        return model(**packet['batch'],**kwargs)
