"""Export small portable evidence and figures; never copy weights or raw data."""
import argparse
import json
from pathlib import Path
import platform
from importlib.metadata import version
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fast_only.common import fingerprint, write_json


def portable(value):
    if isinstance(value,dict): return {portable(k):portable(v) for k,v in value.items()}
    if isinstance(value,list): return [portable(v) for v in value]
    if isinstance(value,str) and value.startswith('/'): return '<local>/'+Path(value).name
    return value


def run(root, output, test_xml):
    root=Path(root); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'result.json').read_text())
    reload=json.loads((root/'reload'/'verification.json').read_text())
    if not reload['passed']: raise ValueError('Do not publish passing evidence for a failed reload')
    for name in ['result.json','protocol.json','predictions.json']:
        write_json(output/name,portable(json.loads((root/name).read_text())))
    write_json(output/'reload.json',portable(reload))
    rows=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    with (output/'train.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(portable(row),allow_nan=False)+'\n')
    tests=ET.parse(test_xml).getroot()
    suites=list(tests.iter('testsuite'))
    counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ['tests','errors','failures','skipped']}
    probes=[dict(stage=r['stage'],step=r['step'],probe=r['probe']) for r in rows if r.get('probe')]
    write_json(output/'verification.json',dict(tests=counts,test_xml_sha256=fingerprint(test_xml),
        reload_passed=reload['passed'],no_kv_detached_slow_exactly_equal=result['no_kv_detached_slow_exactly_equal'],
        joint_slow_differs=result['joint_slow_differs'],gradient_probes=probes,
        python=platform.python_version(),packages={name:version(name) for name in ['torch','transformers','peft','flash-linear-attention','tilelang']},
        test_used=False,val_used_for_selection=False,
        scope='same-runtime single-H800 mini training; not full training, async delivery or closed loop'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for arm in result['config']['arms']:
        selected=[r for r in rows if r['stage']==arm]
        for ax,key in zip(axes,['loss_fast','loss_slow']):
            values=np.array([r[key] for r in selected]); width=20
            ax.plot(np.arange(width,len(values)+1),np.convolve(values,np.ones(width)/width,mode='valid'),label=arm)
    for ax,title in zip(axes,['Fast training loss','Slow training loss']):
        ax.set(title=title,xlabel='Paired iteration',ylabel='Mean L1 pose (mixed m/rad)')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/'training_loss.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for arm in result['config']['arms']:
        steps=result['config']['train_curve_steps']
        for ax,mode in zip(axes,['none','fresh']):
            ax.plot(steps,[result['summary'][f'{arm}_step{s}']['train'][mode]['ADE'] for s in steps],marker='o',label=arm)
    for ax,title in zip(axes,['TRAIN no-KV path','TRAIN fresh-KV path']):
        ax.set(title=title,xlabel='Paired iteration',ylabel='XY ADE (m)')
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/'train_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    arms=result['config']['arms']; modes=['none','fresh','shuffled']; x=np.arange(len(arms))
    for ax,split in zip(axes,['train','val']):
        for i,mode in enumerate(modes):
            ax.bar(x+(i-1)*.24,[result['summary'][f'{a}_final'][split][mode]['ADE'] for a in arms],.24,label=mode)
        ax.set(xticks=x,xticklabels=arms,ylabel='XY ADE (m)',title=f'{split.upper()} fixed-budget final')
        ax.set_ylim(top=ax.get_ylim()[1]*1.25)
        ax.legend(); ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'paired_ade.png',dpi=160); plt.close(fig)
    index={p.name:fingerprint(p) for p in output.iterdir() if p.is_file() and p.name!='SHA256.json'}
    write_json(output/'SHA256.json',index)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--test-xml',required=True)
    args=parser.parse_args(); run(args.run,args.output,args.test_xml)


if __name__=='__main__': main()
