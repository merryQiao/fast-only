"""Stream unseen-log frozen features with the original causal feedback producer."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import time

import torch

from fast_only.common import apply_reproducibility,fingerprint,to_device,write_json
from fast_navsim.collate import NavsimCollator
from slowfast_v11.adapter_features import snapshot_inputs,check_layout,extend_bank
from slowfast_v11.content_diagnostics import source_groups
from slowfast_v11.diagnostic_data import observation_at
from slowfast_v11.expanded_data import _plan_digest
from slowfast_v11.model import FastWithSlowKV,SlowPrefill
from slowfast_v11.readability import KVReader
from slowfast_v11.reader_fusion import reader_features
from slowfast_v11.validation import canonical_state,restore_weights,state_digest
from tools.diagnose_slowfast_readability import state_cpu
from tools.train_adapter_study import authenticated_parent,config
from tools.train_frozen_expert import cached_vision


def new_donors(rows):
    groups={log:sorted([r for r in rows if r['route']==log],key=lambda r:r['timestamp_us']) for log in sorted({r['route'] for r in rows})}
    logs=list(groups)
    if len(logs)!=8 or any(len(rs)!=32 for rs in groups.values()):raise ValueError('Need eight32-window logs')
    return {r['token']:groups[logs[(i+4)%8]][j]['token'] for i,log in enumerate(logs) for j,r in enumerate(groups[log])}


@torch.no_grad()
def run(config_path,root):
    c=config(config_path);root=Path(root);started=time.monotonic()
    trained=json.loads((root/'training.json').read_text());p=authenticated_parent(c)
    if trained['protocol']['config_sha256']!=fingerprint(config_path):raise ValueError('Training/study config identity differs')
    if (root/'features.json').exists() or (root/'frozen_features.pt').exists():raise FileExistsError('Feature extraction already started/completed')
    data_dir=Path(c['data_dir']);plan=json.loads((data_dir/'plan.json').read_text())
    pin=json.loads(Path('configs/slowfast_adapter_cohort.json').read_text())
    frozen_tokens=json.loads((data_dir/'frozen_tokens.json').read_text())
    if (fingerprint(data_dir/'plan.json')!=pin['plan_file_sha256'] or _plan_digest(plan)!=pin['plan_content_sha256']
            or fingerprint(data_dir/'frozen_tokens.json')!=pin['tokens_file_sha256'] or frozen_tokens['tokens']!=pin['tokens']
            or not frozen_tokens['frozen_before_camera_access']):raise ValueError('Pinned new cohort/token identity differs')
    audit=json.loads((data_dir/'source_audit.json').read_text())
    if audit['plan_sha256']!=pin['plan_content_sha256'] or audit['token_manifest_sha256']!=frozen_tokens['token_manifest_sha256']:
        raise ValueError('Acquisition audit belongs to another frozen cohort')
    coverage=audit['provenance_coverage']
    actual_images={str(Path(s['path']).resolve()) for s in audit['sources'] if s['kind']=='retained_original_jpeg'}
    if (coverage['verified_images']!=1024 or coverage['camera_archives']!=8
            or actual_images!={str(Path(path).resolve()) for path in plan['required_images']}):
        raise ValueError('Source audit does not cover every fixed image/archive')
    for item in audit['sources']:
        if fingerprint(item['path'])!=item['sha256']:raise ValueError('Retained JPEG/metadata changed before input decode')
    held=[json.loads(s) for s in Path(c['holdout_manifest']).read_text().splitlines() if s.strip()]
    old_rows=json.loads((Path(c['parent_run'])/'rows.json').read_text())
    from slowfast_v11.adapter_data import recording_group
    old_groups={recording_group(r['route']) for r in old_rows}
    new_groups={recording_group(r['route']) for r in held}
    if (len(held)!=256 or len(new_groups)!=8 or old_groups&new_groups or len({r['token'] for r in held})!=256
            or any(r['split']!='val' or r['cohort']!='holdout' for r in held)
            or held!=plan['rows']['val'] or [r['token'] for r in held]!=pin['tokens']
            or any(sum(recording_group(r['route'])==g for r in held)!=32 for g in new_groups)):
        raise ValueError('Fixed independent recording-group cohort differs')
    source_result=json.loads((Path(p['config']['source_run'])/'result.json').read_text());original=source_result['config']
    apply_reproducibility(original);torch.set_num_threads(4)
    for name in ['fast','slow']:
        for filename,digest in p['assets'][name].items():
            if fingerprint(Path(original[name]['model_path'])/filename)!=digest:raise ValueError('Frozen model assets differ')
    for path,digest in [(Path(p['config']['source_run'])/'joint_final.pt',c['source_checkpoint_sha256']),
                        (Path(p['config']['fast_run'])/'final.pt',c['fast_checkpoint_sha256']),
                        (Path(c['feature_source_run'])/'kv_final.pt',c['reader_checkpoint_sha256'])]:
        if fingerprint(path)!=digest:raise ValueError('Frozen checkpoint identity differs')
    manifest_path=Path(c['feature_source_run'])/'frozen_cache/manifest.json'
    if fingerprint(manifest_path)!=p['feature_manifest_sha256']:raise ValueError('Old source KV manifest differs')
    manifest=json.loads(manifest_path.read_text());shard=manifest['shards'][0];shard_path=manifest_path.parent/shard['path']
    if fingerprint(shard_path)!=shard['sha256']:raise ValueError('Authenticated layout shard differs')
    saved=torch.load(shard_path,map_location='cpu',mmap=True,weights_only=False)
    template=dict(positions=saved['positions'][:,:1].clone(),group_ids=saved['groups'][:1].clone(),valid=saved['valid'][:1].clone());del saved
    fast=FastWithSlowKV.from_pretrained(original['fast']['model_path'],dtype=torch.bfloat16,**original['model']).cuda().eval().requires_grad_(False)
    slow=SlowPrefill.from_pretrained(original['slow']['model_path'],dtype=torch.bfloat16,**original['model']).cuda().eval().requires_grad_(False)
    joint=torch.load(Path(p['config']['source_run'])/'joint_final.pt',map_location='cpu',weights_only=False)
    teacher=joint['feedback_producer'];teacher_digest=state_digest(teacher)
    if teacher_digest!=joint['feedback_producer_digest'] or teacher_digest!=source_result['feedback_producer_digest']:
        raise ValueError('Saved causal Fast512 feedback teacher differs')
    restore_weights(slow,joint['slow']);slow_digest=state_digest(canonical_state(slow));del joint
    latest=torch.load(Path(p['config']['fast_run'])/'final.pt',map_location='cpu',weights_only=False)['model']
    restore_weights(fast,latest);latest_digest=state_digest(canonical_state(fast))
    saved=torch.load(Path(c['feature_source_run'])/'kv_final.pt',map_location='cpu',weights_only=False)
    reader=KVReader(**saved['architecture']).cuda().eval().requires_grad_(False);reader.load_state_dict(saved['state']);del saved
    reader_digest=state_digest(state_cpu(reader))
    collators={name:NavsimCollator(original[name]['model_path'],**original['processor']) for name in ['fast','slow']}
    stats=json.loads(Path(original['data']['stats']).read_text())
    if fingerprint(original['data']['stats'])!=p['source_data']['stats']:raise ValueError('TRAIN statistics changed')
    metadata={r['route']:r['source'] for r in held}
    frames={}
    for route,source_info in metadata.items():
        if fingerprint(source_info['metadata_path'])!=source_info['metadata_sha256']:raise ValueError('New metadata identity differs')
        frames[route]=pickle.loads(Path(source_info['metadata_path']).read_bytes())
    old=torch.load(Path(c['parent_run'])/'frozen_features.pt',map_location='cuda:0',weights_only=False)
    def batch(row,name):
        obs=observation_at(frames[row['route']],row['frame'],plan['sensor_root'],stats)
        packed=collators[name]([dict(obs,trajectory_gt=torch.empty(0))])
        return to_device({k:packed[k] for k in ['native','ego','command']},'cuda:0'),packed['image_audit']
    f_list=[];r_list=[];baseline=[];reader_predictions=[];feedback_records=[];cold={}
    original_vision=fast.backbone.get_image_features
    # Bridge the entire old source path to the authenticated original feature bank.
    train_row=next(r for r in old_rows if r['cohort']=='train');ff=pickle.loads(Path(train_row['source']['metadata_path']).read_bytes())
    obs=observation_at(ff,train_row['frame'],original['data']['sensor_root'],stats)
    packed=collators['fast']([dict(obs,trajectory_gt=torch.empty(0))]);bf=to_device({k:packed[k] for k in ['native','ego','command']},'cuda:0')
    packed_s=collators['slow']([dict(obs,trajectory_gt=torch.empty(0))]);bs=to_device({k:packed_s[k] for k in ['native','ego','command']},'cuda:0')
    restore_weights(fast,teacher);old_teacher=fast(**bf)
    old_source=torch.load(Path(p['config']['source_run'])/'joint_final.pt',map_location='cpu',weights_only=False)
    if not torch.equal(old_teacher[0].float().cpu(),old_source['feedback'][train_row['token']][0]):raise ValueError('Causal teacher numerical path differs from saved producer')
    del old_source
    restore_weights(fast,latest);capture=[]
    handle=fast.head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().clone()))
    try:old_prediction=fast(**bf)
    finally:handle.remove()
    oi=old['index'][train_row['token']]
    if not torch.equal(old_prediction[0],old['baseline'][oi]) or not torch.equal(capture[0][0],old['fast'][oi]):
        raise ValueError('Latest native Fast features differ from old feature bank')
    old_slow=slow(**bs,feedback=old_teacher)
    groups=source_groups(bs['native'],slow.backbone.config.image_token_id,slow.backbone.visual.spatial_merge_size)
    rb=snapshot_inputs(old_slow.snapshot,groups);check_layout(rb,template)
    with torch.autocast('cuda',dtype=torch.bfloat16):old_r,_=reader_features(reader,rb)
    if not torch.equal(old_r[0].float(),old['reader']['correct'][oi]):raise ValueError('Old TRAIN Slow->Reader features no longer reproduce authenticated bank')
    cold[train_row['token']]=dict(batch=to_device(bf,'cpu'),images=train_row['images'],image_audit=packed['image_audit'],cohort='train')
    del rb,old_slow,old_r,old_teacher,capture,bf,bs,packed,packed_s,obs
    print('Old TRAIN teacher/Fast F/Slow->Reader R all reproduced exactly before new prefills',flush=True)
    for number,row in enumerate(held,1):
        bf,image_audit=batch(row,'fast');bs,_=batch(row,'slow');features=original_vision(bf['native']['pixel_values'],bf['native']['image_grid_thw'],return_dict=True).pooler_output
        with cached_vision(fast.backbone,features,bf['native']['image_grid_thw']):
            restore_weights(fast,teacher);feedback=fast(**bf)
            restore_weights(fast,latest)
            captured=[];handle=fast.head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
            try:pred=fast(**bf)
            finally:handle.remove()
        if len(captured)!=1 or not torch.equal(fast.head(captured[0]),pred):raise ValueError('New native Fast head feature differs')
        source=slow(**bs,feedback=feedback)
        groups=source_groups(bs['native'],slow.backbone.config.image_token_id,slow.backbone.visual.spatial_merge_size)
        rb=snapshot_inputs(source.snapshot,groups);check_layout(rb,template)
        with torch.autocast('cuda',dtype=torch.bfloat16):rf,rp=reader_features(reader,rb)
        f_list.append(captured[0]);r_list.append(rf.float());baseline.append(pred)
        reader_predictions.append(dict(token=row['token'],route=row['route'],prediction=rp[0].float().cpu().tolist(),target=row['trajectory_gt']))
        feedback_records.append(dict(token=row['token'],prediction=feedback[0].float().cpu().tolist()))
        if number==1:cold[row['token']]=dict(batch=to_device(bf,'cpu'),images=row['images'],image_audit=image_audit,cohort='holdout')
        del rb,source,features,bf,bs,captured,feedback,pred,rf,rp
        if number%16==0:print(f'frozen original-resolution new-log features {number}/256,elapsed={time.monotonic()-started:.1f}s',flush=True)
        if torch.cuda.max_memory_allocated()/2**30>c['max_allocated_gib']:raise MemoryError('New feature working allocation exceeded')
    if state_digest(canonical_state(fast))!=latest_digest or state_digest(canonical_state(slow))!=slow_digest or state_digest(state_cpu(reader))!=reader_digest:
        raise ValueError('Frozen source values changed')
    if any(param.requires_grad or param.grad is not None for m in [fast,slow,reader] for param in m.parameters()):raise ValueError('Frozen extraction created parameter gradients')
    donors=new_donors(held);bank=extend_bank(old,held,torch.cat(f_list),torch.cat(r_list),torch.cat(baseline),donors)
    cpu={**bank,'fast':bank['fast'].cpu(),'baseline':bank['baseline'].cpu(),'reader':{k:v.cpu() for k,v in bank['reader'].items()}}
    torch.save(cpu,root/'frozen_features.pt');torch.save(cold,root/'cold_inputs.pt')
    write_json(root/'rows.json',old_rows+held);write_json(root/'reader_predictions.json',reader_predictions);write_json(root/'feedback_predictions.json',feedback_records)
    ledger=json.loads((data_dir/'network_budget.json').read_text())
    info=dict(completed=True,schema='adapter_study_features_v1',config_sha256=fingerprint(config_path),
        old_feature_bank_sha256=c['old_feature_bank_sha256'],old_rows_sha256=c['old_rows_sha256'],
        fast_checkpoint_sha256=c['fast_checkpoint_sha256'],reader_checkpoint_sha256=c['reader_checkpoint_sha256'],source_checkpoint_sha256=c['source_checkpoint_sha256'],
        data_plan_sha256=fingerprint(data_dir/'plan.json'),frozen_tokens_sha256=fingerprint(data_dir/'frozen_tokens.json'),
        cohort_pin_sha256=fingerprint('configs/slowfast_adapter_cohort.json'),
        source_audit_sha256=fingerprint(data_dir/'source_audit.json'),holdout_manifest_sha256=fingerprint(c['holdout_manifest']),
        feature_bank_sha256=fingerprint(root/'frozen_features.pt'),rows_sha256=fingerprint(root/'rows.json'),cold_inputs_sha256=fingerprint(root/'cold_inputs.pt'),
        new_network_response_body_bytes=ledger.get('new_response_body_bytes',ledger.get('used')),
        teacher_digest=teacher_digest,teacher_saved_prediction_exact=True,old_Reader_features_exact=True,
        retained_source_hashes_verified_count=len(audit['sources']),latest_fast_digest=latest_digest,slow_digest=slow_digest,reader_digest=reader_digest,
        source_layout_exact_count=len(held),native_head_exact_count=len(held),donors=donors,reference_constant=True,
        old_train_prediction_exact=True,cold_tokens=list(cold),new_backbone_updates=0,new_reader_updates=0,
        source_code_sha256={str(Path(__file__).relative_to(Path.cwd())):fingerprint(__file__),'slowfast_v11/adapter_features.py':fingerprint('slowfast_v11/adapter_features.py')},
        elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(root/'features.json',info);print('New holdout frozen features complete; no holdout planning metrics computed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_adapter_study.yaml');p.add_argument('--run',required=True)
    a=p.parse_args();run(a.config,a.run)


if __name__=='__main__':main()
