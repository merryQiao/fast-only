"""Independent4095->4096 update replay for the frozen-Slow consumer control."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Qwen3_5TextConfig
from fast_only.common import apply_reproducibility,fingerprint,write_json,to_device
from fast_navsim.model import pose_loss
from slowfast_v11.action_expert import ActionExpert
from slowfast_v11.model import FastWithSlowKV
from slowfast_v11.validation import restore_weights,set_stage,make_optimizer,canonical_state
from slowfast_v11.expanded_protocol import WarmupCosineScheduler
from tools.diagnose_slowfast_readability import state_cpu
from tools.train_frozen_expert import cached_vision,snapshot_for
from tools.train_slowfast_joint import restore_rng,capture_rng,compare_trees


def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a new replay output path')
    result=json.loads((root/'result.json').read_text())
    for name,key in [('before_final.pt','before_final_sha256'),('final.pt','checkpoint_sha256'),('replay_input.pt','replay_input_sha256')]:
        if fingerprint(root/name)!=result[key]:raise ValueError('Consumer replay artifact SHA mismatch')
    before=torch.load(root/'before_final.pt',map_location='cpu',weights_only=False)
    after=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    for key in ['schema','arm','config','protocol','text_config']:
        if before[key]!=after[key]:raise ValueError('Before/after protocol mismatch')
    if (before['schema']!='frozen_slow_expert_control_v1' or before['arm']!=result['arm']
            or json.loads(json.dumps(before['protocol']))!=result['protocol']):raise ValueError('Result/checkpoint identity mismatch')
    if before['next_index']!=4095 or after['next_index']!=4096 or before['order']!=after['order']:
        raise ValueError('Replay boundary/order mismatch')
    c=before['config'];arm=before['arm'];protocol=before['protocol']
    source_result=json.loads((Path(c['source_run'])/'result.json').read_text())
    original=source_result['config']
    if protocol['source_checkpoint']!=source_result['arms']['joint']['checkpoint_sha256']:
        raise ValueError('Frozen provider checkpoint identity mismatch')
    apply_reproducibility(original);torch.set_num_threads(4)
    if fingerprint(Path(c['feature_run'])/'frozen_cache/manifest.json')!=protocol['feature_manifest_sha256']:
        raise ValueError('Frozen feature manifest identity differs')
    if arm=='expert':
        model=ActionExpert(text_config=Qwen3_5TextConfig(**before['text_config']),**c['expert']).cuda()
        model.load_state_dict(before['model'])
        blocks=[p for n,p in model.named_parameters() if n.startswith('blocks.')]
        new=[p for n,p in model.named_parameters() if not n.startswith('blocks.')]
        optimizer=torch.optim.AdamW([dict(params=blocks,lr=c['block_lr']),dict(params=new,lr=c['conditioning_head_lr'])],weight_decay=c['weight_decay'])
        get_state=state_cpu
    else:
        for name,digest in protocol['assets']['fast'].items():
            if fingerprint(Path(original['fast']['model_path'])/name)!=digest:raise ValueError('Fast base asset SHA differs')
        model=FastWithSlowKV.from_pretrained(original['fast']['model_path'],dtype=torch.bfloat16,**original['model']).cuda()
        restore_weights(model,before['model']);set_stage(model,'fast_consumer')
        optimizer=make_optimizer(model,dict(lora_lr=c['block_lr'],new_lr=c['conditioning_head_lr'],weight_decay=c['weight_decay']))
        get_state=canonical_state
    schedule=WarmupCosineScheduler(optimizer,4096,c['warmup_steps'],c['min_lr_ratio'])
    optimizer.load_state_dict(before['optimizer']);schedule.load_state_dict(before['scheduler'])
    packet=torch.load(root/'replay_input.pt',map_location='cuda:0',weights_only=False)
    token=before['order'][-1]
    if packet['token']!=token or packet['tokens']!=[token]:raise ValueError('Replay target token differs')
    manifest_path=Path(c['feature_run'])/'frozen_cache/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    shard=next(s for s in manifest['shards'] if token in s['tokens'])
    shard_path=manifest_path.parent/shard['path']
    if fingerprint(shard_path)!=shard['sha256']:raise ValueError('Replay source shard identity mismatch')
    frozen=torch.load(shard_path,map_location='cpu',mmap=True,weights_only=False);j=frozen['tokens'].index(token)
    for name in ['keys','values','groups','valid']:
        if not torch.equal(packet[name].cpu(),frozen[name][j:j+1]):raise ValueError('Replay source payload differs from frozen shard')
    if not torch.equal(packet['positions'].cpu(),frozen['positions'][:,j:j+1]):raise ValueError('Replay source positions differ')
    if fingerprint(original['data']['train_manifest'])!=protocol['source_data']['train_manifest']:
        raise ValueError('Replay TRAIN manifest identity mismatch')
    row=next(json.loads(line) for line in Path(original['data']['train_manifest']).read_text().splitlines() if json.loads(line)['token']==token)
    if packet['target']!=row['trajectory_gt']:raise ValueError('Replay TRAIN target differs')
    # Consumer cache.final is an unused dummy field; never pass it to a model.
    cache=SimpleNamespace(keys=packet['keys'],values=packet['values'],positions=packet['positions'],
                          valid=packet['valid'],index={token:0})
    def forward():
        if arm=='expert':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(packet['keys'],packet['values'],packet['positions'],packet['groups'],packet['valid'],
                             packet['ego'],packet['command'],packet['metadata'])
        with cached_vision(model.backbone,packet['vision'],packet['native']['image_grid_thw']):
            return model(native=packet['native'],ego=packet['ego'],command=packet['command'],
                         snapshot=snapshot_for(cache,token,token),metadata=packet['metadata'])
    model.train();restore_rng(before['rng']);optimizer.zero_grad(set_to_none=True)
    prediction=forward();target=torch.tensor([packet['target']],device='cuda:0')
    loss=pose_loss(prediction,target);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['grad_clip'])
    optimizer.step();schedule.step();model.eval()
    with torch.no_grad():prediction=forward()[0].float().cpu()
    actual=dict(model=get_state(model),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                rng=capture_rng(),reference_prediction=prediction)
    comparisons={name:compare_trees(after[name],value) for name,value in actual.items()}
    passed=all(r['passed'] for r in comparisons.values())
    evidence=dict(passed=passed,arm=arm,step_before=4095,step_after=4096,token=token,
        loss=float(loss),grad_norm=float(norm),comparisons=comparisons,
        scope='same_H800_runtime_actual_single_update',source_provider_frozen=True)
    write_json(output,evidence)
    print(json.dumps(evidence),flush=True)
    if not passed:raise ValueError('Actual consumer update replay did not match')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.run,args.output)


if __name__=='__main__':main()
