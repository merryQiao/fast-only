"""Frozen whole-image x native Slow-KV factorial diagnostic, without training."""
import argparse
from collections import Counter
import gc
import json
import math
from pathlib import Path
import time

import torch
import yaml

from fast_only.common import fingerprint, to_device, write_json
from slowfast_v11.adapter_data import recording_group
from slowfast_v11.full_training import full_precision, restore_full_state
from slowfast_v11.validation import state_digest
from tools.direct_full_common import direct_forward, equal_source_layout, snapshot
from tools.trace_direct_kv import weights_digest, versions
from tools.train_direct_full import setup, load_result_checkpoint

OFFSETS=(0,1,4,7)
ARMS=('full_fast_only','full_correct_kv')


def input_cells(arm):
    if arm=='full_fast_only':
        return [(i,None,'original' if i==0 else 'swapped') for i in OFFSETS]
    if arm!='full_correct_kv':raise ValueError('Unknown frozen input diagnostic arm')
    return [(i,k,'A' if i==k==0 else 'B' if i==0 else 'C' if k==0 else
             'D_same' if i==k else 'D_independent') for i in OFFSETS for k in OFFSETS]


def validate_coverage(records,plan,arm):
    cells={(i,k):name for i,k,name in input_cells(arm)}
    expected={(token,i,k) for token in plan['tokens'] for i,k in cells}
    actual={(r['token'],r['image_offset'],r['kv_offset']) for r in records}
    if len(records)!=len(actual) or actual!=expected:raise ValueError('Factorial record coverage differs')
    for r in records:
        token=r['token'];i=r['image_offset'];k=r['kv_offset']
        source=lambda offset:token if offset==0 else plan['donors'][str(offset)][token]
        if (r['arm']!=arm or r['cell']!=cells[i,k] or r['image_source_token']!=source(i)
                or r['kv_source_token']!=(None if k is None else source(k))):
            raise ValueError('Factorial cell/source label differs')
    return len(records)


def read_config(path):
    c=yaml.safe_load(Path(path).read_text())
    expected=dict(schema='frozen_cross_inputs_v1',arms=list(ARMS),donor_recording_offsets=[1,4,7],
        donor_rule='sorted_recording_groups_cyclic_offset_same_manifest_index',cartesian_joint_inputs=True,
        val_windows=256,recording_groups=8,raw_replay_targets_per_recording=1,
        raw_replay_image_sources=list(OFFSETS),metadata=[0.,0.,0.,0.,1.,1.],moving_speed_mps=1.,
        high_speed_mps=10.,practical_ade_reference_m=.02,analysis_seed=20261109,
        bootstrap_replicates=20000,max_allocated_gib=32,new_training_updates=0,new_response_body_bytes=0)
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen cross-input protocol changed')
    return c


def prepare_protocol(config_path,output):
    from slowfast_v11.cross_inputs import donor_maps
    c=read_config(config_path);out=Path(output)
    manifest_path=Path(c['cache'])/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest['completed'] or not manifest['source_frozen'] or not manifest['vision_native_prediction_all_exact']:
        raise ValueError('Authenticated frozen image/KV cache required')
    rows=[r for r in manifest['rows'] if r['split']=='val']
    groups=Counter(recording_group(r['route']) for r in rows)
    if len(rows)!=256 or len(groups)!=8 or set(groups.values())!={32}:
        raise ValueError('Expected exact8x32 canonical VAL')
    maps=donor_maps(rows,tuple(c['donor_recording_offsets']))
    root=Path(c['run_root']);prior_path=root/'correct_checkpoint_diagnostics.json'
    original=json.loads((root/'result.json').read_text());prior=json.loads(prior_path.read_text())
    if fingerprint(prior_path)!=original['diagnostics_file_sha256'] or maps[4]!=prior['donors']:
        raise ValueError('Offset4 or authenticated prior diagnostic differs')
    hashes=dict(cache_manifest=fingerprint(manifest_path),prior_diagnostic=fingerprint(prior_path))
    for arm in ARMS:
        r=json.loads((root/arm/'result.json').read_text());v=json.loads((root/arm/'verification.json').read_text())
        if (not r['completed'] or not v['passed'] or r['arm']!=arm or
                r['checkpoint_sha256']!=c['checkpoint_sha256'][arm] or v['checkpoint_sha256']!=r['checkpoint_sha256'] or
                fingerprint(root/arm/'final_predictions.json')!=v['predictions_sha256']):
            raise ValueError('Verified endpoint identity differs')
        hashes[arm+'_result']=fingerprint(root/arm/'result.json')
        hashes[arm+'_verification']=fingerprint(root/arm/'verification.json')
        hashes[arm+'_predictions']=v['predictions_sha256']
    plan=dict(schema='frozen_cross_inputs_protocol_v1',config=c,config_sha256=fingerprint(config_path),
        hashes=hashes,tokens=[r['token'] for r in rows],
        targets=[{k:r[k] for k in ['token','route','command','ego']} for r in rows],
        donors={str(offset):mapping for offset,mapping in maps.items()},
        raw_targets=[next(r['token'] for r in rows if recording_group(r['route'])==group) for group in sorted(groups)],
        cells={arm:[dict(image_offset=i,kv_offset=k,cell=cell) for i,k,cell in input_cells(arm)] for arm in ARMS},
        expected_records=dict(full_fast_only=1024,full_correct_kv=4096),expected_raw_replays_per_arm=32,
        expected_saved_exact=dict(full_fast_only=256,full_correct_kv=512),
        held_fixed=['target_ego','target_command','target_native_layout','target_GT','metadata','model_weights','execution_path_within_arm'],
        moving_slice='current ego speed >=1m/s; not futureGT or model error',
        D_independent='different donor sources, not independent random draws',
        scope='observed-VAL frozen input mechanism diagnostic; mismatches can be OOD; no new-log/closed-loop claims',
        source_code_sha256={p:fingerprint(p) for p in ['tools/diagnose_cross_inputs.py','slowfast_v11/cross_inputs.py',
            'slowfast_v11/model.py','slowfast_v11/attention.py','tools/direct_full_common.py','slowfast_v11/full_training.py']})
    path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Existing immutable factorial protocol differs')
    else:
        out.mkdir(parents=True,exist_ok=True);write_json(path,plan)
    return c,plan,rows


def tensor_versions(value):
    if isinstance(value,torch.Tensor):return (value.data_ptr(),value._version,tuple(value.shape),str(value.dtype))
    if isinstance(value,dict):return {k:tensor_versions(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tensor_versions(v) for v in value]
    return value


def feature_change(a,b):
    numerator=sum((x.float()-y.float()).square().sum() for x,y in zip(a,b))
    denominator=sum(x.float().square().sum() for x in a)
    return float((numerator/denominator.clamp_min(1e-24)).sqrt())


def raw_replay(model,collator,target_packet,donor_row,source,arm,expected):
    from PIL import Image
    images=[]
    for path in donor_row['images']:
        with Image.open(path) as image:images.append(image.convert('RGB'))
    packed=collator([dict(images=images,ego=target_packet['batch']['ego'][0].cpu(),
        command=target_packet['batch']['command'][0].cpu(),trajectory_gt=torch.empty(0))])
    batch=to_device({k:packed[k] for k in ['native','ego','command']},'cuda:0')
    for name in ['ego','command']:
        if not torch.equal(batch[name],target_packet['batch'][name]):raise ValueError('Raw replay changed target sensors')
    native=target_packet['batch']['native']
    if set(native)!=set(batch['native']):raise ValueError('Raw native field layout differs')
    if any(not torch.equal(value,batch['native'][name]) for name,value in native.items() if name!='pixel_values'):
        raise ValueError('Raw replay changed target nonpixel native fields')
    kwargs={} if arm=='full_fast_only' else dict(snapshot=snapshot(source),
        metadata=torch.tensor([[0.,0.,0.,0.,1.,1.]],device='cuda:0'))
    prediction=model(**batch,**kwargs)[0].float().cpu()
    if not torch.equal(prediction,expected):raise ValueError('Raw Processor/full native forward differs from cached four-frame swap')
    return dict(image_source_token=donor_row['token'],max_abs_difference=0.,raw_prediction=prediction.tolist(),
        cached_prediction=expected.tolist(),target_ego_preserved=True,target_command_preserved=True,
        image_original_sizes=[a['source_size_wh'] for a in packed['image_audit']],
        image_audit=packed['image_audit'])


@torch.no_grad()
def run(config_path,output,arm):
    from fast_navsim.collate import NavsimCollator
    from slowfast_v11.cross_inputs import swap_vision
    c,plan,rows=prepare_protocol(config_path,output);out=Path(output)/arm
    if out.exists():raise FileExistsError('Preserve prior evidence; use a fresh arm/run directory')
    model,cache,_,identity,_,_=setup(c['direct_config'],c['cache'],c['init'])
    if cache.fingerprint!=plan['hashes']['cache_manifest'] or cache.val!=rows:
        raise ValueError('Live authenticated cache differs from pinned factorial protocol')
    root=Path(c['run_root'])/arm;r=json.loads((root/'result.json').read_text())
    if r['identity']!=identity:raise ValueError('Endpoint setup identity differs')
    saved=load_result_checkpoint(root,'final.pt',r,'checkpoint_sha256')
    if state_digest(saved['model'])!=r['model_digest']:raise ValueError('Complete endpoint state differs')
    restore_full_state(model,saved['model']);del saved;gc.collect()
    model.eval().requires_grad_(False)
    previous={z['token']:torch.tensor(z['prediction']) for z in json.loads((root/'final_predictions.json').read_text())}
    prior=json.loads((Path(c['run_root'])/'correct_checkpoint_diagnostics.json').read_text())
    prior_B4={z['token']:torch.tensor(z['prediction']) for z in prior['records'] if z['mode']=='shuffled'}
    direct=yaml.safe_load(Path(c['direct_config']).read_text())
    original=json.loads((Path(direct['source_run'])/'result.json').read_text())['config']
    collator=NavsimCollator(original['fast']['model_path'],**original['processor'])
    out.mkdir();before=weights_digest(model);marks=versions(model)
    by_token={z['token']:z for z in rows};records=[];raw=[];exact=0
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with full_precision(model):
        for n,row in enumerate(rows,1):
            token=row['token'];names={0:token,**{int(o):mapping[token] for o,mapping in plan['donors'].items()}}
            packets={o:cache.packet(source) for o,source in names.items()}
            target=packets[0];inputs_before=tensor_versions(packets)
            sources={o:cache.source(source) for o,source in names.items()} if arm=='full_correct_kv' else {}
            source_before=tensor_versions(sources)
            for source in sources.values():equal_source_layout(sources[0],source)
            visual={i:swap_vision(target,p) for i,p in packets.items()}
            differences={o:feature_change(target['vision'],p['vision']) for o,p in packets.items()}
            for image_offset,kv_offset,cell in input_cells(arm):
                source=None if kv_offset is None else sources[kv_offset]
                prediction=direct_forward(model,visual[image_offset],arm,source)[0].float().cpu()
                if not torch.isfinite(prediction).all() or prediction.shape!=(8,3):raise ValueError('Invalid complete pose prediction')
                if image_offset==0 and (kv_offset is None or kv_offset==0):
                    if not torch.equal(prediction,previous[token]):raise ValueError('Original endpoint prediction changed')
                    exact+=1
                if image_offset==0 and kv_offset==4:
                    if not torch.equal(prediction,prior_B4[token]):raise ValueError('Original offset4 B prediction changed')
                    exact+=1
                item=dict(arm=arm,token=token,route=row['route'],command=row['command'],
                    speed_mps=math.hypot(*row['ego'][:2]),image_offset=image_offset,kv_offset=kv_offset,cell=cell,
                    image_source_token=names[image_offset],kv_source_token=None if kv_offset is None else names[kv_offset],
                    image_feature_relative_l2=differences[image_offset],prediction=prediction.tolist(),target=row['trajectory_gt'])
                records.append(item)
                with (out/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(item,allow_nan=False)+'\n')
                if token in plan['raw_targets'] and (kv_offset is None or kv_offset==0):
                    check=raw_replay(model,collator,target,by_token[names[image_offset]],source,arm,prediction)
                    raw.append(dict(target_token=token,arm=arm,**check));write_json(out/'raw_replays.json',raw)
            if inputs_before!=tensor_versions(packets) or source_before!=tensor_versions(sources) or marks!=versions(model):
                raise ValueError('Frozen model/input/source mutated during factorial inference')
            if torch.cuda.max_memory_allocated()>c['max_allocated_gib']*2**30:raise MemoryError('Frozen inference exceeds32GiB cap')
            del packets,visual,target,sources,source
            if n==1 or n%16==0:print(f'{arm} cross {n}/256, records={len(records)}, exact={exact}, raw={len(raw)}, elapsed={time.monotonic()-started:.1f}s',flush=True)
    count=validate_coverage(records,plan,arm);after=weights_digest(model)
    if before!=after or marks!=versions(model) or len(raw)!=32 or exact!=plan['expected_saved_exact'][arm]:
        raise ValueError('Frozen factorial final acceptance failed')
    write_json(out/'result.json',dict(schema='frozen_cross_inputs_arm_v1',completed=True,arm=arm,
        records=count,exact_previous_forecasts=exact,raw_processor_replays_exact=len(raw),
        all_parameter_sha256_before=before,all_parameter_sha256_after=after,parameter_versions_unchanged=True,
        inputs_and_sources_unchanged=True,checkpoint_sha256=c['checkpoint_sha256'][arm],
        predictions_sha256=fingerprint(out/'predictions.jsonl'),raw_replays_sha256=fingerprint(out/'raw_replays.json'),
        protocol_sha256=fingerprint(Path(output)/'protocol.json'),
        elapsed_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        new_training_updates=0,new_response_body_bytes=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/slowfast_cross_inputs.yaml')
    p.add_argument('--arm',required=True,choices=ARMS);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output,a.arm)


if __name__=='__main__':main()
