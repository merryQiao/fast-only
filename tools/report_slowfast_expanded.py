"""Export portable expanded-cohort evidence and fixed-diagnostic figures."""
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

from fast_only.common import fingerprint,write_json
from slowfast_v11.validation import paired_summary
from tools.report_slowfast_joint import portable


def run(root,output,test_xml):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'result.json').read_text())
    reload=json.loads((root/'reload/verification.json').read_text())
    if not reload['passed']:raise ValueError('Independent reload/replay has not passed')
    records=json.loads((root/'predictions.json').read_text())
    training=[json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    for name in ['result.json','protocol.json','predictions.json']:
        write_json(output/name,portable(json.loads((root/name).read_text())))
    write_json(output/'reload.json',portable(reload))
    with (output/'train.jsonl').open('w') as stream:
        for row in training:stream.write(json.dumps(portable(row),allow_nan=False)+'\n')
    data=Path(result['config']['data']['train_manifest']).parent
    for name in ['build_report.json','stats.json']:
        write_json(output/('data_'+name),portable(json.loads((data/name).read_text())))
    plan=json.loads((data/'plan.json').read_text())
    write_json(output/'data_selection.json',portable({key:value for key,value in plan.items()
        if key not in ['rows','required_images','legacy_copies']}))
    audit=json.loads((data/'source_audit.json').read_text())
    write_json(output/'acquisition_summary.json',portable(dict(
        schema=audit['schema'],plan_sha256=audit['plan_sha256'],openscene_revision=audit['openscene_revision'],
        network_response_body_bytes=audit['network_response_body_bytes'],full_archives_verified=False,
        completed_archive_response_body_bytes=audit['completed_archive_response_body_bytes'],
        other_charged_response_body_bytes=audit['other_charged_response_body_bytes'],
        provenance_coverage=audit['provenance_coverage'],
        source_count=len(audit['sources']),reports=[{key:value for key,value in row.items() if key!='files'}
                                                  for row in audit['acquisition_reports']])))
    strata=dict(by_log={},by_speed_bin={})
    for key,field in [('by_log','route'),('by_speed_bin','speed_bin')]:
        for value in sorted({r[field] for r in records}):
            strata[key][value]=paired_summary([r for r in records if r[field]==value])
    write_json(output/'stratified_summary.json',strata)
    tests=ET.parse(test_xml).getroot()
    counts={key:sum(int(s.get(key,'0')) for s in tests.iter('testsuite'))
            for key in ['tests','errors','failures','skipped']}
    health={}
    for arm in result['config']['arms']:
        rows=[r for r in training if r['stage']==arm]
        health[arm]={}
        for key in ['fast_grad_norm','slow_grad_norm']:
            values=np.array([r[key] for r in rows])
            health[arm][key]=dict(finite=bool(np.isfinite(values).all()),median=float(np.median(values)),
                maximum=float(values.max()),clip_fraction=float((values>1.).mean()))
    write_json(output/'verification.json',dict(tests=counts,test_xml_sha256=fingerprint(test_xml),
        independent_reload_passed=True,gradient_health=health,
        gradient_probes=[dict(stage=r['stage'],step=r['step'],probe=r['probe']) for r in training if r.get('probe')],
        python=platform.python_version(),packages={n:version(n) for n in ['torch','transformers','peft','flash-linear-attention','tilelang']},
        test_used=False,val_used_for_selection=False,scope='one-H800 expanded mini;TRAIN curves use fixed32 rows'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for arm in result['config']['arms']:
        steps=result['config']['train_curve_steps']
        for ax,mode in zip(axes,['none','fresh']):
            ax.plot(steps,[result['summary'][f'{arm}_step{s}']['train'][mode]['ADE'] for s in steps],marker='o',label=arm)
    for ax,mode in zip(axes,['none','fresh']):
        ax.set(title=f'Fixed TRAIN32: {mode}',xlabel='Paired iteration',ylabel='XY ADE (m)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'train_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    arms=result['config']['arms'];modes=['none','fresh','shuffled'];x=np.arange(len(arms))
    for ax,cohort,title in zip(axes,['new_val','legacy_val'],['New VAL64 (two logs)','Legacy VAL4 (one log)']):
        for i,mode in enumerate(modes):
            ax.bar(x+(i-1)*.24,[result['summary'][f'{a}_final'][cohort][mode]['ADE'] for a in arms],.24,label=mode)
        ax.set(xticks=x,xticklabels=arms,ylabel='XY ADE (m)',title=title)
        ax.set_ylim(top=ax.get_ylim()[1]*1.25);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.savefig(output/'paired_ade.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for arm in arms:
        rows=[r for r in training if r['stage']==arm]
        for ax,key in zip(axes,['loss_fast','loss_slow']):
            values=np.array([r[key] for r in rows]);width=40
            ax.plot(np.arange(width,len(values)+1),np.convolve(values,np.ones(width)/width,mode='valid'),label=arm)
    for ax,title in zip(axes,['Fast training loss','Slow training loss']):
        ax.set(title=title,xlabel='Paired iteration',ylabel='Mean L1 pose (mixed m/rad)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'training_loss.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    for stage in ['fast_warmup','slow_adapt','no_kv','joint']:
        rows=[r for r in training if r['stage']==stage]
        rates=[r['learning_rates']['fast'][0] if isinstance(r['learning_rates'],dict) else r['learning_rates'][0] for r in rows]
        ax.plot(range(1,len(rows)+1),rates,label=stage)
    ax.set(xlabel='Phase update',ylabel='LoRA learning rate',title='Actual update learning rates');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'learning_rates.png',dpi=160);plt.close(fig)
    write_json(output/'SHA256.json',{p.name:fingerprint(p) for p in output.iterdir() if p.is_file() and p.name!='SHA256.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);parser.add_argument('--test-xml',required=True)
    args=parser.parse_args();run(args.run,args.output,args.test_xml)


if __name__=='__main__':main()
