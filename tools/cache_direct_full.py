"""Export frozen visual outputs and native Slow KV, never Fast language features."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from PIL import Image

from fast_only.common import apply_reproducibility,fingerprint,to_device,write_json
from fast_navsim.collate import NavsimCollator
from slowfast_v11.adapter_data import recording_group
from slowfast_v11.expanded_data import _plan_digest
from slowfast_v11.model import SlowPrefill
from slowfast_v11.validation import canonical_state,restore_weights,state_digest
from tools.direct_full_common import read_config,source_info,load_fast,train_rows,vision_packet,LegacySources,equal_source_layout
from tools.train_frozen_expert import cached_vision


def validate_rows(train,val,*,counts=(512,256),groups=(12,8)):
    if (len(train),len(val))!=counts:raise ValueError('Expanded cohort count differs')
    if len({r['token'] for r in train+val})!=sum(counts):raise ValueError('Duplicate/overlapping tokens')
    if any(r['split']!='train' or r['cohort']!='train' for r in train):raise ValueError('Invalid optimizer cohort')
    if any(r['split']!='val' or r['cohort']!='new_val' for r in val):raise ValueError('Invalid independent validation cohort')
    a,b=({recording_group(r['route']) for r in rows} for rows in [train,val])
    if (len(a),len(b))!=groups or a&b:raise ValueError('Recording group leakage/count differs')


def validate_packet(packet):
    if set(packet['batch'])!={'native','ego','command'}:raise ValueError('Cached batch contains extra inputs/targets')
    native=packet['batch']['native']
    if set(native)-{'input_ids','attention_mask','pixel_values','image_grid_thw','token_type_ids','mm_token_type_ids'}:
        raise ValueError('Only native processor fields may be cached')
    if native['pixel_values'].numel():raise ValueError('Expected frozen visual outputs, empty pixel sentinel')
    if not packet['vision'] or any(z.requires_grad for z in packet['vision']):raise ValueError('Frozen visual cache required')


def authenticated_data(c):
    root=Path(c['data_dir']);pin=json.loads(Path('configs/slowfast_direct_full_cohort.json').read_text())
    plan=json.loads((root/'plan.json').read_text());tokens=json.loads((root/'frozen_tokens.json').read_text())
    if (fingerprint(root/'plan.json')!=pin['plan_file_sha256'] or _plan_digest(plan)!=pin['plan_content_sha256']
            or fingerprint(root/'frozen_tokens.json')!=pin['tokens_file_sha256'] or not tokens['frozen_before_camera_access']):
        raise ValueError('Frozen cohort pin differs')
    train,val=([json.loads(s) for s in (root/f'{split}.jsonl').read_text().splitlines() if s.strip()] for split in ['train','val'])
    validate_rows(train,val)
    if train!=plan['rows']['train'] or val!=plan['rows']['val'] or train!=train_rows(c):raise ValueError('Prepared optimizer/evaluation rows differ')
    if [r['token'] for r in train]!=pin['train_tokens'] or [r['token'] for r in val]!=pin['val_tokens']:raise ValueError('Frozen tokens differ')
    audit=json.loads((root/'source_audit.json').read_text());build=json.loads((root/'build_report.json').read_text())
    if not audit['complete'] or audit['plan_sha256']!=plan['plan_sha256'] or audit['token_manifest_sha256']!=tokens['token_manifest_sha256']:
        raise ValueError('Incomplete/mismatched acquisition provenance')
    required={str(Path(p).resolve()) for r in train+val for p in r['images']}
    sources={str(Path(s['path']).resolve()):s for s in audit['sources']}
    if not required<=sources.keys():raise ValueError('Not every input image is authenticated')
    for item in audit['sources']:
        if fingerprint(item['path'])!=item['sha256'] or Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Retained input source changed')
    for split in ['train','val']:
        if fingerprint(root/f'{split}.jsonl')!=build['manifests'][split]['sha256']:raise ValueError('Prepared manifest identity changed')
    normalizer=audit['normalization']['selected']['sha256']
    if fingerprint(root/'stats.json')!=normalizer:raise ValueError('Selected checkpoint normalizer changed')
    return train,val,dict(plan_sha256=fingerprint(root/'plan.json'),tokens_sha256=fingerprint(root/'frozen_tokens.json'),
        source_audit_sha256=fingerprint(root/'source_audit.json'),train_manifest_sha256=fingerprint(root/'train.jsonl'),
        val_manifest_sha256=fingerprint(root/'val.jsonl'),normalizer_sha256=normalizer,
        pin_sha256=fingerprint('configs/slowfast_direct_full_cohort.json'),verified_sources=len(sources))


def raw_source(snapshot):
    return dict(keys=torch.stack([z.key.transpose(1,2).flatten(2) for z in snapshot.layers],1),
        values=torch.stack([z.value.transpose(1,2).flatten(2) for z in snapshot.layers],1),
        positions=snapshot.position_ids,valid=snapshot.valid_mask)


@torch.no_grad()
def run(config_path,output):
    c=read_config(config_path);out=Path(output)
    if out.exists():raise FileExistsError('Use a fresh complete cache export directory')
    train,val,data=authenticated_data(c);rows=train+val
    out.mkdir(parents=True);(out/'packets').mkdir();(out/'sources').mkdir();started=time.monotonic()
    original,fast_result,stats=source_info(c);apply_reproducibility(original['config']);torch.set_num_threads(4)
    if data['normalizer_sha256']!=original['data_identities']['stats']:raise ValueError('Consumer/source normalizer differs')
    joint_path=Path(c['source_run'])/'joint_final.pt'
    if fingerprint(joint_path)!=c['source_checkpoint_sha256']:raise ValueError('Slow/teacher checkpoint changed')
    joint=torch.load(joint_path,map_location='cpu',weights_only=False)
    teacher=joint['feedback_producer'];teacher_digest=state_digest(teacher)
    if teacher_digest!=joint['feedback_producer_digest'] or teacher_digest!=original['feedback_producer_digest']:raise ValueError('Causal feedback teacher differs')
    fast,collator,_,_=load_fast(c);fast.eval().requires_grad_(False);latest=canonical_state(fast);latest_digest=state_digest(latest)
    for name,sha in original['assets']['slow'].items():
        if fingerprint(Path(original['config']['slow']['model_path'])/name)!=sha:raise ValueError('Slow asset differs')
    slow=SlowPrefill.from_pretrained(original['config']['slow']['model_path'],dtype=torch.bfloat16,**original['config']['model']).cuda().eval().requires_grad_(False)
    restore_weights(slow,joint['slow']);slow_digest=state_digest(canonical_state(slow))
    slow_collator=NavsimCollator(original['config']['slow']['model_path'],**original['config']['processor'])
    legacy=LegacySources(c);reference=min(r['token'] for r in train);template=legacy.packet(reference)
    entries={};feedback_records=[];slow_records=[];old_exact=False
    for n,row in enumerate(rows,1):
        token=row['token'];packet=vision_packet(fast,collator,row,stats);validate_packet(packet)
        packet_path=out/'packets'/f'{token}.pt';torch.save(to_device(packet,'cpu'),packet_path)
        source_entry=dict(kind='legacy',token=token)
        # Reproduce one old source exactly, then use its authenticated immutable cache.
        need_source=token not in legacy.entries or n==1
        if need_source:
            with cached_vision(fast.backbone,packet['vision'],packet['batch']['native']['image_grid_thw']):
                restore_weights(fast,teacher);feedback=fast(**packet['batch']);restore_weights(fast,latest)
            if n==1 and not torch.equal(feedback[0].float().cpu(),joint['feedback'][token][0]):raise ValueError('Saved causal teacher no longer reproduces old feedback')
            images=[]
            for path in row['images']:
                with Image.open(path) as image:images.append(image.convert('RGB'))
            packed=slow_collator([dict(images=images,ego=packet['batch']['ego'][0].cpu(),command=packet['batch']['command'][0].cpu(),trajectory_gt=torch.empty(0))])
            batch=to_device({k:packed[k] for k in ['native','ego','command']},'cuda:0')
            result=slow(**batch,feedback=feedback);source=raw_source(result.snapshot);equal_source_layout(template,source)
            slow_records.append(dict(token=token,route=row['route'],split=row['split'],prediction=result.trajectory[0].float().cpu().tolist()))
            if n==1:
                old=legacy.packet(token)
                if any(not torch.equal(source[k],old[k]) for k in ['keys','values','positions','valid']):raise ValueError('Recomputed frozen Slow differs from legacy raw KV')
                old_exact=True;del old
            else:
                path=out/'sources'/f'{token}.pt';torch.save(to_device(source,'cpu'),path)
                source_entry=dict(kind='native',path=str(path.resolve()),sha256=fingerprint(path))
            feedback_records.append(dict(token=token,prediction=feedback[0].float().cpu().tolist()))
            del batch,packed,result,source,feedback
        else:
            source=legacy.packet(token);equal_source_layout(template,source);del source
        entries[token]=dict(packet=dict(path=str(packet_path.resolve()),sha256=fingerprint(packet_path)),source=source_entry)
        del packet
        if n%16==0:print(f'frozen native visual/Slow KV cache {n}/768,elapsed={time.monotonic()-started:.1f}s',flush=True)
    if state_digest(canonical_state(fast))!=latest_digest or state_digest(canonical_state(slow))!=slow_digest:raise ValueError('Frozen provider changed')
    if any(p.requires_grad or p.grad is not None for m in [fast,slow] for p in m.parameters()):raise ValueError('Frozen provider acquired gradients')
    write_json(out/'feedback.json',feedback_records)
    write_json(out/'slow_predictions.json',slow_records)
    manifest=dict(schema='direct_full_frozen_inputs_v1',completed=True,rows=rows,entries=entries,
        config_sha256=fingerprint(config_path),data=data,source_checkpoint=c['source_checkpoint_sha256'],fast_checkpoint=c['fast_checkpoint_sha256'],
        teacher_digest=teacher_digest,latest_fast_digest=latest_digest,slow_digest=slow_digest,old_raw_KV_reproduced_exact=old_exact,
        legacy_manifest_path=str((legacy.root/'manifest.json').resolve()),legacy_manifest_sha256=fingerprint(legacy.root/'manifest.json'),
        slow_predictions_sha256=fingerprint(out/'slow_predictions.json'),feedback_predictions_sha256=fingerprint(out/'feedback.json'),
        raw_shape=[1,6,8255,512],fixed_source=reference,source_layout_all_exact=True,vision_native_prediction_all_exact=True,
        source_frozen=True,visual_only_cache=True,elapsed_s=time.monotonic()-started,
        source_code_sha256={p:fingerprint(p) for p in ['tools/cache_direct_full.py','tools/direct_full_common.py']})
    write_json(out/'manifest.json',manifest);print('Complete frozen inputs exported; no new VAL metrics computed',flush=True)


class DirectCache:
    def __init__(self,root='outputs/slowfast_direct_full/cache_01',config='configs/slowfast_direct_full.yaml'):
        self.root=Path(root);self.fingerprint=fingerprint(self.root/'manifest.json');self.manifest=json.loads((self.root/'manifest.json').read_text())
        c=read_config(config);m=self.manifest
        if (not m['completed'] or m['schema']!='direct_full_frozen_inputs_v1' or m['config_sha256']!=fingerprint(config)
                or m['source_checkpoint']!=c['source_checkpoint_sha256'] or m['fast_checkpoint']!=c['fast_checkpoint_sha256']):raise ValueError('Frozen cache protocol changed')
        train,val,data=authenticated_data(c)
        if m['data']!=data or m['rows']!=train+val or set(m['entries'])!={r['token'] for r in train+val}:raise ValueError('Cache no longer matches authenticated data')
        pin=json.loads(Path('configs/slowfast_direct_full_cohort.json').read_text())
        plan=json.loads((Path(c['data_dir'])/'plan.json').read_text())
        if (m['fixed_source']!=min(r['token'] for r in train) or m['fixed_source']!=plan['fixed_source_token']
                or m['fixed_source']!=pin['fixed_source_token']):
            raise ValueError('Fixed source is not the pinned minimum TRAIN token')
        joint_path=Path(c['source_run'])/'joint_final.pt'
        if fingerprint(joint_path)!=c['source_checkpoint_sha256']:raise ValueError('Authenticated frozen provider checkpoint changed')
        joint=torch.load(joint_path,map_location='cpu',weights_only=False)
        if (state_digest(joint['feedback_producer'])!=m['teacher_digest'] or joint['feedback_producer_digest']!=m['teacher_digest']
                or state_digest(joint['slow'])!=m['slow_digest']):raise ValueError('Frozen teacher/Slow parameter digest changed')
        del joint
        self.rows=m['rows'];self.train=train;self.val=val;self.entries=m['entries'];self.legacy=LegacySources(c);self.loaded={};self.verified=set()
        if fingerprint(self.legacy.root/'manifest.json')!=m['legacy_manifest_sha256']:raise ValueError('Legacy raw source identity changed')

    def _load(self,entry):
        path=entry['path']
        if path not in self.loaded:
            if fingerprint(path)!=entry['sha256']:raise ValueError('Frozen packet/source SHA differs')
            self.loaded[path]=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        return self.loaded[path]

    def packet(self,token,device='cuda:0'):
        packet=self._load(self.entries[token]['packet']);validate_packet(packet)
        if packet['token']!=token:raise ValueError('Frozen packet token differs')
        return to_device(packet,device)

    def source(self,token,device='cuda:0'):
        entry=self.entries[token]['source']
        if entry['kind']=='legacy':return self.legacy.packet(token,device)
        return to_device(self._load(entry),device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_direct_full.yaml');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)


if __name__=='__main__':main()
