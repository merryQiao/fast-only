"""Render frozen-read diagnostics from authenticated records; no model loading."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from fast_only.common import fingerprint, write_json

LAYERS = [4, 8, 12, 16, 20, 24]
GROUPS = ['image_0','image_1','image_2','image_3','native_other','ego','command','feedback','planning_queries']


def stats(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Expected finite diagnostic values')
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def render(root, output):
    root=Path(root);out=Path(output)
    if out.exists():raise FileExistsError('Preserve previous reports')
    result=json.loads((root/'result.json').read_text())
    if not result['completed'] or result['exact_prior_predictions']!=768 or result['single_layer_interventions']!=1152:
        raise ValueError('Read localization did not complete its fixed protocol')
    for name in ['protocol','summary','records','interventions']:
        suffix='.jsonl' if name in ['records','interventions'] else '.json'
        if fingerprint(root/(name+suffix))!=result[name+'_sha256']:raise ValueError('Diagnostic artifact changed')
    records=read_jsonl(root/'records.jsonl');interventions=read_jsonl(root/'interventions.jsonl')
    if len(records)!=256 or len({r['token'] for r in records})!=256 or len(interventions)!=1152:
        raise ValueError('Diagnostic row coverage differs')
    out.mkdir(parents=True)
    cohorts={'all':records,'moving_ge_1mps':[r for r in records if r['speed_mps']>=1]}
    portable={}
    for cohort,rows in cohorts.items():
        part=dict(count=len(rows),layers=[],natural={},trajectory={})
        for mode in ['correct','reference','shuffled']:
            p=np.asarray([r['predictions'][mode] for r in rows]);gt=np.asarray([r['target'] for r in rows])
            distances=np.linalg.norm(p[...,:2]-gt[...,:2],axis=-1)
            correct=np.asarray([r['predictions']['correct'] for r in rows])
            changes=np.linalg.norm(p[...,:2]-correct[...,:2],axis=-1)
            part['trajectory'][mode]=dict(ADE_m=float(distances.mean()),FDE_m=float(distances[:,-1].mean()),
                mean_XY_change_m=float(changes.mean()),median_point_XY_change_m=float(np.median(changes)),
                p90_window_XY_change_m=float(np.quantile(changes.mean(-1),.9)),max_point_XY_change_m=float(changes.max()))
        for index,number in enumerate(LAYERS):
            probes=[r['probes']['correct'][index] for r in rows]
            item=dict(layer=number,external_mass=stats([p['float32_reference']['external_mass'] for p in probes]),
                external_mass_per_head=np.mean([p['float32_reference']['external_mass_per_head'] for p in probes],axis=0).tolist(),
                gate_mean=stats([p['actual']['gate_mean'] for p in probes]),
                gate_mean_per_head=np.mean([p['actual']['gate_mean_per_head'] for p in probes],axis=0).tolist(),
                local={},raw_source_difference={},source_position_groups={})
            for mode in ['reference','shuffled']:
                item['raw_source_difference'][mode]={field:stats([r['source_differences'][mode][index][field]['relative_l2'] for r in rows]) for field in ['key','value']}
            for variant in probes[0]['local_variants']:
                item['local'][variant]={stage:{field:stats([p['local_variants'][variant][stage][field] for p in probes])
                    for field in ['rms_difference','rms_reference','relative_l2','equal_fraction']}
                    for stage in ['pre_gate','post_gate','post_projection']}
            for group in GROUPS:
                item['source_position_groups'][group]=dict(
                    mass=stats([p['float32_reference']['groups'][group]['mass'] for p in probes]),
                    mass_per_head=np.mean([p['float32_reference']['groups'][group]['mass_per_head'] for p in probes],axis=0).tolist(),
                    contribution={stage:stats([p['float32_reference']['groups'][group]['contribution'][stage+'_rms'] for p in probes])
                        for stage in ['pre_gate','post_gate','linear_projection']})
            part['layers'].append(item)
        for mode in ['reference','shuffled']:
            part['natural'][mode]={name:{field:stats([r['natural_differences'][mode][name][field] for r in rows])
                for field in ['rms_difference','reference_rms','relative_l2','equal_fraction']}
                for name in rows[0]['natural_differences'][mode]}
        portable[cohort]=part
    by_condition=defaultdict(list)
    for r in interventions:by_condition[r['mode'],r['layer_one_based']].append(r)
    portable['single_layer']={f'{mode}_{layer}':{field:stats([r[field] for r in rows])
        for field in ['mean_plan_xy_change_m','max_plan_xy_change_m','ADE_gain_vs_correct_m']}
        for (mode,layer),rows in sorted(by_condition.items())}
    portable['numeric_reference_errors']={
        'sdpa_relative_l2':stats([p['float32_reference']['sdpa_reconstruction']['relative_l2'] for r in records for mode in ['correct','reference','shuffled'] for p in r['probes'][mode]]),
        'projected_split_relative_l2':stats([p['float32_reference']['projection_reconstruction']['relative_l2'] for r in records for mode in ['correct','reference','shuffled'] for p in r['probes'][mode]]),
        'query_vs_native_fullM_projection_max_abs':max(v['max_abs_difference'] for r in records for mode in ['correct','reference','shuffled'] for v in r['projection_shape_audit'][mode].values())}
    portable['scope']='Observed VAL diagnostic; positional groups are not independent information modalities; relative changes and norms do not establish task usefulness.'
    portable['source_result_sha256']=fingerprint(root/'result.json')
    write_json(out/'report_summary.json',portable)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(16,4.4),layout='constrained')
    for cohort,style in [('all','-o'),('moving_ge_1mps','--s')]:
        rows=portable[cohort]['layers']
        axes[0].plot(LAYERS,[r['external_mass']['mean'] for r in rows],style,label=cohort)
        axes[1].plot(LAYERS,[r['gate_mean']['mean'] for r in rows],style,label=cohort)
    for stage,style in [('pre_gate','-o'),('post_gate','--s'),('post_projection',':^')]:
        axes[2].plot(LAYERS,[r['local']['reference'][stage]['relative_l2']['mean'] for r in portable['all']['layers']],style,label=stage)
    for ax in axes:ax.set_xticks(LAYERS);ax.set_xlabel('Full-attention layer');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean joint-softmax external mass');axes[0].set_title('External KV attention mass (FP32 reference)')
    axes[1].set_ylabel('Mean sigmoid gate');axes[1].set_title('Native gate level')
    axes[2].set_ylabel('Relative L2 vs current-source local read');axes[2].set_title('Fixed Q / Fast KV / gate: reference swap, all256')
    fig.savefig(out/'six_layer_reads.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    for mode,color in [('reference','tab:blue'),('shuffled','tab:orange')]:
        for cohort,style in [('all','-'),('moving_ge_1mps','--')]:
            values=portable[cohort]['natural'][mode]
            axes[0].plot(range(1,25),[values[f'layer_{n}']['relative_l2']['mean'] for n in range(1,25)],style,color=color,label=f'{mode}, {cohort}')
        names=['post_projection_24','layer_24','head_input','head_stage_0','head_stage_1','head_stage_2','head_stage_3']
        axes[1].plot(range(len(names)),[portable['moving_ge_1mps']['natural'][mode][n]['relative_l2']['mean'] for n in names],'-o',label=mode)
    axes[0].set_xlabel('Native block');axes[0].set_ylabel('Relative L2 of Planning rows');axes[0].set_title('Natural whole-source swap through residual stream')
    axes[1].set_xticks(range(len(names)),['o_proj24','block24','final norm','head norm','linear256','GELU','pose']);axes[1].tick_params(axis='x',rotation=25)
    axes[1].set_ylabel('Relative L2 (stage-specific reference scale)');axes[1].set_title('Source changes across the head: moving91')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(out/'downstream_propagation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');x=np.arange(6)
    for offset,mode in [(-1,'reference'),(0,'shuffled'),(1,'masked')]:
        values=[portable['single_layer'][f'{mode}_{n}'] for n in LAYERS]
        axes[0].bar(x+offset*.25,[r['mean_plan_xy_change_m']['mean']*1000 for r in values],width=.25,label=mode)
        axes[1].bar(x+offset*.25,[r['ADE_gain_vs_correct_m']['mean']*1000 for r in values],width=.25,label=mode)
    for ax in axes:ax.set_xticks(x,LAYERS);ax.set_xlabel('Only this layer changed; five remain correct');ax.grid(axis='y',alpha=.25);ax.legend()
    axes[0].set_ylabel('Mean XY trajectory change (mm)');axes[0].set_title('64 metadata-selected windows / eight logs')
    axes[1].set_ylabel('ADE gain vs correct (mm); positive is better');axes[1].axhline(0,color='black',lw=.8);axes[1].set_title('Exploratory intervention effect; mask is not content-only')
    fig.savefig(out/'single_layer_interventions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,field,title in [(axes[0],'mass','Joint-softmax probability mass'),(axes[1],'projection','Contribution RMS after gate and linear projection')]:
        matrix=np.asarray([[r['source_position_groups'][group]['mass']['mean'] if field=='mass' else r['source_position_groups'][group]['contribution']['linear_projection']['mean'] for r in portable['all']['layers']] for group in GROUPS])
        im=ax.imshow(np.log10(np.maximum(matrix,1e-10)),aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='log10 mean')
        ax.set_xticks(range(6),LAYERS);ax.set_yticks(range(len(GROUPS)),GROUPS);ax.set_xlabel('Full-attention layer');ax.set_title(title)
    fig.suptitle('Groups denote source token positions; contextualized KV is not isolated modality information')
    fig.savefig(out/'source_position_groups.png',dpi=180);plt.close(fig)
    write_json(out/'provenance.json',dict(source_result_sha256=fingerprint(root/'result.json'),script_sha256=fingerprint(__file__),
        files={p.name:fingerprint(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(output=str(out),all_metrics=portable['all']['trajectory'],reference_errors=portable['numeric_reference_errors']),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();render(a.run,a.output)


if __name__=='__main__':main()
