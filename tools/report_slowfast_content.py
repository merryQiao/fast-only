"""Portable evidence and plots for the fixed-checkpoint KV content diagnostic."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fast_only.common import fingerprint,write_json
from tools.report_slowfast_joint import portable


def attention_summary(rows):
    result={}
    for mode in rows[0]['attention']:
        if not rows[0]['attention'][mode]:continue
        layers=[]
        for index in range(6):
            selected=[r['attention'][mode][index] for r in rows]
            groups={}
            for name in selected[0]['groups']:
                group=[r['groups'][name] for r in selected]
                groups[name]={key:float(np.mean([g[key] for g in group])) for key in
                              ['tokens','visible_tokens','mass','mass_per_token']}
            layers.append(dict(layer_one_based=selected[0]['layer_one_based'],groups=groups,
                **{key:float(np.mean([r[key] for r in selected])) for key in
                   ['external_mass','self_mass','external_value_norm','self_value_norm','combined_value_norm']}))
        result[mode]=layers
    return result


def donor_summary(rows):
    result={}
    for mode in ['shuffled','contrast','reference']:
        donors=[r['donors'][mode] for r in rows]
        result[mode]=dict(count=len(rows),cross_log=sum(d['cross_log'] for d in donors),
            different_command=sum(d['different_command'] for d in donors),unique_donors=len({d['token'] for d in donors}),
            mean_abs_speed_delta_mps=float(np.mean([abs(r['speed_mps']-d['speed_mps']) for r,d in zip(rows,donors)])),
            layers=[])
        for index in range(6):
            entries=[r['kv_differences'][mode][index] for r in rows]
            result[mode]['layers'].append(dict(layer_one_based=entries[0]['layer_one_based'],
                **{kind:dict(equal_count=sum(e[kind]['equal'] for e in entries),
                              **{metric:float(np.mean([e[kind][metric] for e in entries])) for metric in
                                 ['relative_l2','cosine','rms_difference']}) for kind in ['key','value']}))
    return result


def run(root,output,test_xml):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=json.loads((root/'result.json').read_text())
    if (not result['completed'] or result['max_reference_difference']!=0 or not result['masked_content_invariance']['passed']
            or not result['weights_unchanged'] or not result['source_tensors_unchanged']):
        raise ValueError('Incomplete or failed diagnostic evidence')
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    for name in ['result.json','protocol.json']:
        write_json(output/name,portable(json.loads((root/name).read_text())))
    with (output/'records.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(portable(row),allow_nan=False)+'\n')
    attention={};donors={}
    for cohort in ['new_val','legacy_val']:
        selected=[r for r in rows if r['cohort']==cohort]
        attention[cohort]=attention_summary(selected);donors[cohort]=donor_summary(selected)
    write_json(output/'attention_summary.json',attention);write_json(output/'donor_summary.json',donors)
    tests=ET.parse(test_xml).getroot()
    counts={key:sum(int(s.get(key,'0')) for s in tests.iter('testsuite')) for key in ['tests','errors','failures','skipped']}
    if counts['errors'] or counts['failures'] or counts['skipped']:raise ValueError('Regression checks not clean')
    write_json(output/'verification.json',dict(tests=counts,test_xml_sha256=fingerprint(test_xml),
        exact_original_predictions=result['verified_reference_predictions'],max_original_difference=result['max_reference_difference'],
        masked_content_invariance=result['masked_content_invariance'],weights_unchanged=result['weights_unchanged'],
        no_training=result['new_training_updates']==0,no_download=result['new_download_bytes']==0))
    modes=['none','masked','correct','zero_v','reference','shuffled','contrast']
    labels=['No cache','Mask, meta fixed','Correct','V=0','Fixed reference','Old shuffle','State contrast']
    fig,axes=plt.subplots(1,2,figsize=(13,4.3),layout='constrained')
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        summary=result['summary'][cohort]['all']
        ax.bar(range(len(modes)),[summary['modes'][m]['ADE'] for m in modes],color=['C0','C1','C2','C3','C4','C5','C6'])
        ax.axhline(summary['slow']['ADE'],ls=':',color='black',label='Slow own head')
        ax.set(title=f'{cohort} ({summary["count"]} windows)',ylabel='XY ADE (m)',xticks=range(len(modes)),xticklabels=labels)
        ax.tick_params(axis='x',labelrotation=32);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.savefig(output/'paired_ade.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    selected=attention['new_val']
    for mode in ['correct','masked','zero_v','reference','shuffled','contrast']:
        values=selected[mode];axes[0].plot([r['layer_one_based'] for r in values],[r['external_mass'] for r in values],marker='.',label=mode)
    axes[0].set(title='New VAL: external attention mass',xlabel='Layer',ylabel='Mean probability');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    names=list(selected['correct'][0]['groups'])
    for ax,key,title in zip(axes[1:],['mass','mass_per_token'],['Correct KV: group mass','Correct KV: mass per source token']):
        values=[np.mean([r['groups'][name][key] for r in selected['correct']]) for name in names]
        ax.bar(range(len(names)),values);ax.set(title=title,xticks=range(len(names)),xticklabels=names)
        ax.tick_params(axis='x',labelrotation=55);ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'attention_profile.png',dpi=160);plt.close(fig)
    def ade(prediction,target):return float(np.linalg.norm(np.array(prediction)[:,:2]-np.array(target)[:,:2],axis=-1).mean())
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for cohort,marker in [('new_val','o'),('legacy_val','x')]:
        selected=[r for r in rows if r['cohort']==cohort]
        x=[ade(r['predictions']['none'],r['target'])-ade(r['slow_prediction'],r['target']) for r in selected]
        y=[ade(r['predictions']['masked'],r['target'])-ade(r['predictions']['correct'],r['target']) for r in selected]
        ax.scatter(x,y,label=cohort,marker=marker,alpha=.75)
    ax.axhline(0,color='grey',ls='--');ax.axvline(0,color='grey',ls='--')
    ax.set(xlabel='Slow own-head advantage vs Fast no-cache ADE (m)',ylabel='Correct KV gain vs fixed-metadata masked ADE (m)',
           title='Per-window diagnosis (retrospective VAL slices)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'slow_advantage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,kind in zip(axes,['key','value']):
        for mode in ['shuffled','contrast','reference']:
            values=donors['new_val'][mode]['layers']
            ax.plot([r['layer_one_based'] for r in values],[r[kind]['relative_l2'] for r in values],marker='.',label=mode)
        ax.set(title=f'New VAL actual {kind} difference',xlabel='Layer',ylabel='Relative L2 vs correct');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'kv_differences.png',dpi=160);plt.close(fig)
    write_json(output/'SHA256.json',{p.name:fingerprint(p) for p in output.iterdir() if p.is_file() and p.name!='SHA256.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);parser.add_argument('--test-xml',required=True)
    args=parser.parse_args();run(args.run,args.output,args.test_xml)


if __name__=='__main__':main()
