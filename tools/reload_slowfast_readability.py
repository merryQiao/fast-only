"""Independent process reload of both frozen-feature readers on saved raw inputs."""
import argparse
import json
from pathlib import Path
import torch
from fast_only.common import apply_reproducibility,fingerprint,write_json
from slowfast_v11.readability import KVReader,FinalFeatureReader


@torch.no_grad()
def run(root,output):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Use a new reload evidence path')
    result=json.loads((root/'result.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    apply_reproducibility(protocol);torch.set_num_threads(4)
    if fingerprint(root/'reload_inputs.pt')!=result['reload_inputs_sha256']:raise ValueError('Reload packet SHA differs')
    packet=torch.load(root/'reload_inputs.pt',map_location='cuda:0',weights_only=False)
    index={t:i for i,t in enumerate(packet['tokens'])};comparisons={}
    for name,cls in [('kv',KVReader),('final',FinalFeatureReader)]:
        path=root/f'{name}_final.pt'
        if fingerprint(path)!=result['probes'][name]['checkpoint_sha256']:raise ValueError('Readout checkpoint SHA differs')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['source_checkpoint']!=protocol['source_checkpoint_sha256']:raise ValueError('Source checkpoint differs')
        model=cls(**saved['architecture']).cuda().eval();model.load_state_dict(saved['state'])
        maximum=0.;count=0
        # Preserve original evaluation mini-batch shape/order, including tail batches.
        for references in result['reload_batches'][name]:
            i=torch.tensor([index[r['token']] for r in references],device='cuda:0')
            j=torch.tensor([index[r['source']] for r in references],device='cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if name=='kv':
                    actual=model(packet['keys'][j],packet['values'][j],packet['positions'][:,i],
                                 packet['groups'][i],packet['valid'][i])
                else:actual=model(packet['final'][j])
            expected=torch.tensor([r['prediction'] for r in references],device='cuda:0')
            diff=float((actual.float()-expected).abs().max());maximum=max(maximum,diff);count+=len(references)
        comparisons[name]=dict(count=count,max_abs_difference=maximum,passed=maximum==0.)
    passed=all(r['passed'] for r in comparisons.values())
    evidence=dict(passed=passed,comparisons=comparisons,scope='same_H800_runtime_readout_only')
    write_json(output,evidence)
    if not passed:raise ValueError(f'Independent readout reload mismatch: {comparisons}')
    print(json.dumps(evidence))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.run,args.output)


if __name__=='__main__':main()
