"""Cold original-JPEG inference: one Fast model and three constant bundles only."""
import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image

from fast_only.common import apply_reproducibility,fingerprint,to_device,write_json
from fast_navsim.collate import NavsimCollator
from slowfast_v11.fixed_adapter import load_fixed_adapter,bind_fixed_adapter
from slowfast_v11.model import FastWithSlowKV,SlowPrefill
from slowfast_v11.validation import restore_weights,canonical_state,state_digest


@torch.no_grad()
def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh deployment-proof destination')
    result=json.loads((root/'result.json').read_text());p=result['protocol'];c=p['config'];parent=p['parent_protocol'];f=result['features']
    if fingerprint(root/'cold_expected.json')!=result['cold_expected_sha256'] or fingerprint(root/'cold_inputs.pt')!=f['cold_inputs_sha256']:
        raise ValueError('Expected fixed-source inputs/output identity differs')
    original=json.loads((Path(parent['config']['source_run'])/'result.json').read_text())['config']
    apply_reproducibility(original);torch.set_num_threads(4)
    for name,digest in parent['assets']['fast'].items():
        if fingerprint(Path(original['fast']['model_path'])/name)!=digest:raise ValueError('Fast model/processor assets differ')
    latest_path=Path(parent['config']['fast_run'])/'final.pt'
    if fingerprint(latest_path)!=c['fast_checkpoint_sha256']:raise ValueError('Fast checkpoint SHA differs')
    # Instrument the prohibited constructor; no Reader module is imported at all.
    def forbidden(*args,**kwargs):raise AssertionError('Cold fixed baseline tried to construct Slow')
    SlowPrefill.__init__=forbidden
    if 'slowfast_v11.readability' in sys.modules:raise ValueError('Reader imported before fixed-only proof')
    fast=FastWithSlowKV.from_pretrained(original['fast']['model_path'],dtype=torch.bfloat16,**original['model']).cuda().eval().requires_grad_(False)
    saved=torch.load(latest_path,map_location='cpu',weights_only=False);restore_weights(fast,saved['model']);del saved
    before=state_digest(canonical_state(fast));collator=NavsimCollator(original['fast']['model_path'],**original['processor'])
    stats=json.loads(Path(original['data']['stats']).read_text())
    if fingerprint(original['data']['stats'])!=parent['source_data']['stats']:raise ValueError('TRAIN statistics changed')
    if fingerprint(root/'rows.json')!=f['rows_sha256']:raise ValueError('Fixed proof row identity differs')
    rows={r['token']:r for r in json.loads((root/'rows.json').read_text())}
    expected={(r['seed'],r['token']):torch.tensor(r['prediction']) for r in json.loads((root/'cold_expected.json').read_text())}
    native_reference=torch.load(root/'cold_inputs.pt',map_location='cpu',weights_only=False);checks=[]
    for seed in c['train_seeds']:
        item=result['fixed_bundles'][str(seed)]
        adapter=load_fixed_adapter(item['path'],expected_sha256=item['sha256'],fast_checkpoint_sha256=c['fast_checkpoint_sha256'],head=fast.head,map_location='cuda:0')
        for token in f['cold_tokens']:
            row=rows[token];images=[]
            for path in row['images']:
                with Image.open(path) as image:images.append(image.convert('RGB'))
            sample=dict(images=images,ego=torch.tensor([(v-m)/s for v,m,s in zip(row['ego'],stats['ego_mean'],stats['ego_std'])]),
                        command=torch.tensor(row['command'],dtype=torch.long),trajectory_gt=torch.empty(0))
            packed=collator([sample]);batch={k:packed[k] for k in ['native','ego','command']}
            old=native_reference[token]['batch']
            native_exact=all(torch.equal(v,old['native'][k]) for k,v in batch['native'].items()) and torch.equal(batch['ego'],old['ego']) and torch.equal(batch['command'],old['command'])
            if not native_exact:raise ValueError('Cold processor output differs from frozen original-JPEG input')
            calls=[];handle=fast.head.register_forward_hook(lambda module,args,out:calls.append(True))
            try:
                with bind_fixed_adapter(fast.head,adapter):prediction=fast(**to_device(batch,'cuda:0'))[0].float().cpu()
            finally:handle.remove()
            diff=float((prediction-expected[seed,token]).abs().max());exact=torch.equal(prediction,expected[seed,token])
            if not exact or len(calls)!=1:raise ValueError('Cold fixed-source adapter changed reference-arm output')
            checks.append(dict(seed=seed,token=token,cohort=row['cohort'],exact=exact,max_abs_difference=diff,native_processor_exact=native_exact,
                               original_head_calls=len(calls),visual_tokens=[a['visual_tokens'] for a in packed['image_audit']]))
    if 'slowfast_v11.readability' in sys.modules or state_digest(canonical_state(fast))!=before:raise ValueError('Fixed proof imported Reader or changed Fast')
    write_json(output,dict(passed=True,no_slow_or_reader=True,models_constructed=dict(Fast=1,Slow=0,Reader=0),checks=checks,
        scope='fresh_process_original_JPEG_and_native_Processor_full_Fast_with_fixed_bundle;two_predeclared_rows_per_seed',
        fast_checkpoint_sha256=c['fast_checkpoint_sha256'],new_network_bytes=0,source_code_sha256=fingerprint(__file__)))
    print('Cold original-image Fast-only fixed-adapter inference exact for all six cases',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.run,a.output)


if __name__=='__main__':main()
