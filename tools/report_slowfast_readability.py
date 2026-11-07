"""CPU-only portable evidence, matched metrics and figures for frozen readability.

Reads JSON/JSONL/XML only. It never loads weights/cache shards or imports Torch.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def portable(value):
    if isinstance(value,dict):return {portable(k):portable(v) for k,v in value.items()}
    if isinstance(value,list):return [portable(v) for v in value]
    if isinstance(value,str) and value.startswith('/'):return '<local>/'+Path(value).name
    return value


def read_json(path):return json.loads(Path(path).read_text())
def read_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path,value):
    Path(path).write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')


def pose_metrics(predictions,targets):
    p,t=np.asarray(predictions,dtype=np.float64),np.asarray(targets,dtype=np.float64)
    if p.shape!=t.shape or p.ndim!=3 or p.shape[1:]!=(8,3) or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Expected matching nonempty finite poses[N,8,3]')
    delta=p-t;xy=np.linalg.norm(delta[...,:2],axis=-1)
    yaw=np.abs(np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2])))
    return dict(ADE=float(xy.mean()),FDE=float(xy[:,-1].mean()),heading_mae_rad=float(yaw.mean()),
        heading_final_mae_rad=float(yaw[:,-1].mean()),loss_l1_pose=float(np.abs(delta).mean()),samples=len(p),
        **{f'L2@{(i+1)*.5:g}s':float(xy[:,i].mean()) for i in range(8)})


def window_ade(prediction,target):return pose_metrics([prediction],[target])['ADE']
def speed_bin(speed):return 'low' if speed<1 else 'mid' if speed<10 else 'high'


def matched_summary(rows):
    names=sorted(rows[0]['ADE'])
    metrics={name:pose_metrics([r['poses'][name] for r in rows],[r['target'] for r in rows]) for name in names}
    pairs={}
    for name in names:
        if name=='kv_correct':continue
        gains=np.array([r['ADE'][name]-r['ADE']['kv_correct'] for r in rows])
        pairs[name]=dict(mean_gain_m=float(gains.mean()),median_gain_m=float(np.median(gains)),
            better_windows=int((gains>0).sum()),equal_windows=int((gains==0).sum()),worse_windows=int((gains<0).sum()))
    return dict(windows=len(rows),tokens=[r['token'] for r in rows],metrics=metrics,kv_correct_advantage=pairs)


def assemble(predictions,paths,baselines,result):
    lookup={};by_token={}
    for row in predictions:
        key=row['probe'],row['token'],row['mode']
        if key in lookup:raise ValueError('Duplicate readout condition')
        lookup[key]=row
        if row['token'] in by_token and by_token[row['token']]['target']!=row['target']:raise ValueError('Target identity changed')
        by_token[row['token']]=row
    expected={'train':256,'new_val':64,'legacy_val':4}
    if Counter(r['cohort'] for r in by_token.values())!=expected:raise ValueError('Cohort coverage differs')
    if len(lookup)!=324*2*3:raise ValueError('Expected all 1,944 reader/source conditions')
    for token in by_token:
        for reader in ['kv','final']:
            for mode in ['correct','shuffled','reference']:
                if (reader,token,mode) not in lookup:raise ValueError('Missing reader/source condition')
    max_error=0.
    for reader,cohorts in result['summary'].items():
        for cohort,modes in cohorts.items():
            for mode,stored in modes.items():
                rows=[r for r in predictions if (r['probe'],r['cohort'],r['mode'])==(reader,cohort,mode)]
                actual=pose_metrics([r['prediction'] for r in rows],[r['target'] for r in rows])
                for name in ['ADE','FDE','heading_mae_rad','loss_l1_pose']:
                    error=abs(actual[name]-stored[name]);max_error=max(max_error,error)
                    if error>1e-5:raise ValueError(f'Stored/recomputed metric differs: {reader}/{cohort}/{mode}/{name}')
    path_lookup={r['token']:r for r in paths}
    if len(paths)!=68 or len(path_lookup)!=68 or set(path_lookup)!={t for t,r in by_token.items() if r['cohort']!='train'}:
        raise ValueError('Expected exactly 68 matched path windows')
    if set(baselines)!=set(by_token):raise ValueError('Frozen baseline coverage differs')
    windows=[]
    for token,row in sorted(by_token.items()):
        poses={f'{reader}_{mode}':lookup[reader,token,mode]['prediction'] for reader in ['kv','final']
               for mode in ['correct','shuffled','reference']}
        poses.update(original_slow=baselines[token]['slow'],teacher_feedback=baselines[token]['feedback'])
        if token in path_lookup:
            path=path_lookup[token]
            if path['target']!=row['target']:raise ValueError('Path/reader target differs')
            poses.update({f'fast_{mode}':prediction for mode,prediction in path['predictions'].items()})
        errors={name:window_ade(prediction,row['target']) for name,prediction in poses.items()}
        windows.append(dict(token=token,cohort=row['cohort'],route=row['route'],speed_mps=row['speed_mps'],
            speed_bin=speed_bin(row['speed_mps']),target=row['target'],poses=poses,ADE=errors,
            kv_correct_advantage_m={name:value-errors['kv_correct'] for name,value in errors.items() if name!='kv_correct'}))
    strata={}
    for cohort in expected:
        rows=[r for r in windows if r['cohort']==cohort]
        strata[cohort]=dict(all=matched_summary(rows),
            by_log={log:matched_summary([r for r in rows if r['route']==log]) for log in sorted({r['route'] for r in rows})},
            by_speed={name:matched_summary([r for r in rows if r['speed_bin']==name]) for name in ['low','mid','high']
                      if any(r['speed_bin']==name for r in rows)})
    return windows,strata,max_error


def path_summary(paths):
    result={}
    for cohort in ['new_val','legacy_val']:
        rows=[r for r in paths if r['cohort']==cohort];targets=[r['target'] for r in rows]
        modes={name:pose_metrics([r['predictions'][name] for r in rows],targets) for name in ['none','empty','masked','correct']}
        comparisons={}
        for before,after in [('none','empty'),('empty','masked'),('masked','correct')]:
            a=np.array([r['predictions'][before] for r in rows]);b=np.array([r['predictions'][after] for r in rows])
            comparisons[f'{before}_to_{after}']=dict(delta_mean_ADE_m=modes[after]['ADE']-modes[before]['ADE'],
                max_abs_xy_coordinate_change_m=float(np.abs(b[...,:2]-a[...,:2]).max()),
                max_abs_heading_change_rad=float(np.abs(b[...,2]-a[...,2]).max()),
                mean_plan_xy_change_m=float(np.linalg.norm(b[...,:2]-a[...,:2],axis=-1).mean()))
        layers={}
        for comparison in rows[0]['differences']:
            layers[comparison]={name:dict(max_abs_difference=max(r['differences'][comparison][name]['max_abs_difference'] for r in rows
                                                                if name in r['differences'][comparison]),
                compared_windows=sum(name in r['differences'][comparison] for r in rows))
                for name in sorted(set().union(*(r['differences'][comparison].keys() for r in rows)))}
        result[cohort]=dict(windows=len(rows),modes=modes,comparisons=comparisons,layer_maxima=layers,
            empty_exact_windows=sum(r['empty_path_comparison']['exact'] for r in rows),
            empty_within_1e_4_windows=sum(r['empty_path_comparison']['within_1e_4'] for r in rows))
    return result


def figures(output,result,training,strata,path,sdpa):
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for reader,label in [('kv','Full-KV reader'),('final','Final-feature MLP')]:
        rows=[r for r in training if r['probe']==reader];loss=np.array([r['loss'] for r in rows])
        axes[0].plot(np.arange(32,len(loss)+1),np.convolve(loss,np.ones(32)/32,mode='valid'),label=label)
        curve=result['probes'][reader]['curves'];steps=sorted(map(int,curve))
        axes[1].plot(steps,[curve[str(s)]['ADE'] for s in steps],marker='o',label=label)
    axes[0].set(xlabel='Probe update',ylabel='Mean L1 pose (mixed m/rad)',title='TRAIN loss: 32-step running mean')
    axes[1].set(xlabel='Probe update',ylabel='XY ADE (m)',title='Same predetermined TRAIN32 windows')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        metrics=strata[cohort]['all']['metrics'];x=np.arange(3)
        for reader,color,offset in [('kv','C0',-.18),('final','C1',.18)]:
            ax.bar(x+offset,[metrics[f'{reader}_{m}']['ADE'] for m in ['correct','shuffled','reference']],width=.35,color=color,
                   label='KV reader' if reader=='kv' else 'Final-feature MLP')
        ax.axhline(metrics['original_slow']['ADE'],ls=':',color='black',label='Original Slow head')
        ax.axhline(metrics['fast_correct']['ADE'],ls='--',color='C2',label='Original Fast: correct KV')
        ax.set(xticks=x,xticklabels=['Correct source','Shuffled source','Fixed TRAIN source'],yscale='log',ylabel='XY ADE (m; log scale)',
            title=f'{cohort}: {strata[cohort]["all"]["windows"]} windows')
        ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'source_conditions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    modes=['none','empty','masked','correct'];values=path['new_val']['modes']
    axes[0].plot(range(4),[values[m]['ADE'] for m in modes],'o-')
    axes[0].set(xticks=range(4),xticklabels=['Native none','Fusion + no-cache','Fusion + metadata','Correct KV'],
                ylabel='XY ADE (m)',title='New VAL64: controlled path sequence')
    axes[0].tick_params(axis='x',labelrotation=28)
    comparisons=path['new_val']['comparisons'];names=list(comparisons)
    axes[1].bar(range(3),[1000*comparisons[n]['mean_plan_xy_change_m'] for n in names])
    axes[1].set(xticks=range(3),xticklabels=['None → empty','Empty → metadata','Masked → correct'],
                ylabel='Mean trajectory XY change (mm)',title='Output change is distinct from ADE gain')
    axes[1].tick_params(axis='x',labelrotation=28)
    selected={r['label']:r for r in sdpa['comparisons']}
    numbers=[sdpa['native_auto_vs_fp32_math_crop']['max_abs_difference'],
             selected['bf16_default_crop_repeat']['vs_fp32_math_crop_gqa']['max_abs_difference'],
             selected['fp32_math_crop_repeat']['vs_fp32_math_crop_gqa']['max_abs_difference'],
             selected['fp32_math_crop_repeat_masked_external']['vs_fp32_math_crop_gqa']['max_abs_difference']]
    axes[2].bar(range(4),[max(v,1e-10) for v in numbers])
    for i,value in enumerate(numbers):axes[2].text(i,max(value,1e-10)*1.3,'0 (exact)' if value==0 else f'{value:.2g}',ha='center',fontsize=7)
    axes[2].set(yscale='log',ylim=(5e-11,.1),xticks=range(4),xticklabels=['Native BF16','Cropped BF16','FP32 repeat','FP32 masked prefix'],
        ylabel='Max |attention difference| vs FP32 math',title='One FA4 replay; tensor values, not metres')
    axes[2].tick_params(axis='x',labelrotation=28)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'path_backend_summary.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,4.1),layout='constrained')
    for cohort,marker in [('new_val','o'),('legacy_val','x')]:
        metrics=strata[cohort]['all'];rows=strata[cohort]['by_log']
        for log,group in rows.items():
            ax.scatter([group['metrics']['final_correct']['ADE']],[group['metrics']['kv_correct']['ADE']],marker=marker,s=65,
                       label=f'{cohort}: {log[:10]} ({group["windows"]})')
    ax.plot([0,.95],[0,.95],ls=':',color='grey')
    ax.set(xlabel='Final-feature MLP ADE (m)',ylabel='Full-KV reader ADE (m)',title='Log-level correct-source comparison')
    ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(output/'per_log_readability.png',dpi=160);plt.close(fig)


def run(root,output,reload_path,sdpa_path,test_xml,document):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    result=read_json(root/'result.json');protocol=read_json(root/'protocol.json');export=read_json(root/'export.json')
    reload=read_json(reload_path);sdpa=read_json(sdpa_path)
    if (not result['completed'] or not result['frozen_cache_unchanged'] or not export['source_parameters_unchanged']
            or result['new_base_training_updates']!=0 or result['new_probe_updates']!=2048 or result['new_download_bytes']!=0
            or result['val_used_for_selection'] or result['test_used'] or not reload['passed']):
        raise ValueError('Incomplete/failed frozen readability run')
    if export['original_head_exact_count']!=324 or export['path_exact_references']!=204:raise ValueError('Reproduction coverage differs')
    if (sdpa['source_checkpoint_sha256']!=protocol['source_checkpoint_sha256']
            or sdpa['assets']!=protocol['assets'] or sdpa['data_identities']!=protocol['data_identities']):
        raise ValueError('Backend audit checkpoint/assets/data differ')
    if (not sdpa['native_helper_replay']['post_gate_exact'] or not sdpa['source_parameters_unchanged']
            or not sdpa['original_none_vs_checkpoint_reference']['exact']
            or not sdpa['native_helper_replay']['recomputed_rope_query_exact']
            or not sdpa['native_helper_replay']['recomputed_rope_key_exact']):
        raise ValueError('Backend audit did not reproduce native helper')
    predictions=read_json(root/'predictions.json');paths=read_lines(root/'path_records.jsonl')
    baselines=read_json(root/'baselines.json');training=read_lines(root/'train.jsonl')
    if Counter(r['probe'] for r in training)!={'kv':1024,'final':1024}:raise ValueError('Probe training budget differs')
    for reader in ['kv','final']:
        rows=[r for r in training if r['probe']==reader]
        if ([r['step'] for r in rows]!=list(range(1,1025)) or any(r['windows_seen']!=r['step']*8 for r in rows)
                or not np.isfinite([[r[k] for k in ['loss','grad_norm','lr']] for r in rows]).all()):
            raise ValueError('Probe update order/exposures/gradient health differs')
    if len(protocol['train_order'])!=8192 or set(Counter(protocol['train_order']).values())!={32}:
        raise ValueError('Expected 32 complete uniform TRAIN256 epochs')
    expected_train=set(protocol['train_order'])
    if any(set(protocol['train_order'][start:start+256])!=expected_train for start in range(0,8192,256)):
        raise ValueError('Expected sampling without replacement inside every TRAIN epoch')
    windows,strata,max_error=assemble(predictions,paths,baselines,result)
    path=path_summary(paths)
    for name in ['result.json','protocol.json','export.json','predictions.json','baselines.json']:
        write_json(output/name,read_json(root/name))
    for name,rows in [('path_records.jsonl',paths),('train.jsonl',training)]:
        (output/name).write_text(''.join(json.dumps(portable(row),allow_nan=False)+'\n' for row in rows))
    write_json(output/'reload.json',reload);write_json(output/'sdpa_audit.json',sdpa)
    write_json(output/'stratified_summary.json',strata);write_json(output/'path_summary.json',path)
    write_json(output/'window_metrics.json',[{k:v for k,v in row.items() if k not in ['poses','target']} for row in windows])
    cache_manifest=root/'frozen_cache/manifest.json'
    if cache_manifest.exists():write_json(output/'frozen_cache_manifest.json',read_json(cache_manifest))
    suite=ET.parse(test_xml).getroot();counts={k:sum(int(s.get(k,'0')) for s in suite.iter('testsuite'))
                                           for k in ['tests','errors','failures','skipped']}
    if counts['errors'] or counts['failures'] or counts['skipped']:raise ValueError('Historical regression record has failures')
    reload_coverage={}
    for reader,batches in result['reload_batches'].items():
        rows=[row for batch in batches for row in batch]
        reload_coverage[reader]=dict(conditions=len(rows),unique_targets=len({r['token'] for r in rows}),
            cohorts=dict(Counter(r['cohort'] for r in rows)),modes=dict(Counter(r['mode'] for r in rows)))
        if len(rows)!=reload['comparisons'][reader]['count'] or reload['comparisons'][reader]['max_abs_difference']!=0:
            raise ValueError('Independent reload condition count/result differs')
    source_files=[root/name for name in ['result.json','protocol.json','export.json','predictions.json','path_records.jsonl','baselines.json','train.jsonl']]
    source_files += [Path(reload_path),Path(sdpa_path),Path(test_xml)]
    if cache_manifest.exists():source_files.append(cache_manifest)
    captured=dict(protocol['source_code_sha256']);captured.update(sdpa['source_code_sha256'])
    installed=distribution('transformers')
    library_paths={'transformers/modeling_qwen3_5.py':installed.locate_file('transformers/models/qwen3_5/modeling_qwen3_5.py'),
                   'transformers/sdpa_attention.py':installed.locate_file('transformers/integrations/sdpa_attention.py')}
    current={name:sha256(library_paths.get(name,Path(name))) for name in captured
             if Path(library_paths.get(name,name)).is_file()}
    sources=dict(input_files={portable(str(p)):sha256(p) for p in source_files},
        captured_implementation_sha256=captured,current_implementation_sha256=current,
        captured_matches_current={name:current[name]==digest if name in current else None for name,digest in captured.items()},
        report_tool_sha256=sha256(__file__),source_checkpoint_sha256=protocol['source_checkpoint_sha256'])
    write_json(output/'source_hashes.json',sources)
    verification=dict(schema='slowfast_readability_report_verification_v1',historical_regression=dict(
        counts=counts,xml_sha256=sha256(test_xml),suite_timestamps=[s.get('timestamp') for s in suite.iter('testsuite')],
        scope='Historical regression snapshot; subsequent checkpoint-helper/control changes are not certified by this record.'),
        reload=reload,reload_coverage=reload_coverage,reload_scope='16 new_VAL targets per reader, 3 source modes; not all68 VAL windows',
        original_head_exact_count=324,path_reference_predictions_exact=204,metric_recompute_max_abs_error=max_error,
        recompute_dtype='float64 NumPy; stored metrics were float32 Torch',frozen_cache_unchanged=True,
        no_base_model_updates=True,no_downloads=True,report_cpu_only=True,recorded_runtime=sdpa['runtime'],
        source_condition_counts=dict(Counter(f'{r["probe"]}/{r["cohort"]}/{r["mode"]}' for r in predictions)))
    write_json(output/'verification.json',verification)
    figures(output,result,training,strata,path,sdpa)
    Path(document).write_text(report_text(result,protocol,export,strata,path,verification,sdpa))
    write_json(output/'SHA256.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps(dict(exported_files=len(list(output.iterdir())),historical_tests=counts['tests'],
                         matched_windows=len(windows),metric_recompute_max_error=max_error)))


def report_text(result,protocol,export,strata,path,verification,sdpa):
    def metric(cohort,name,key='ADE'):return strata[cohort]['all']['metrics'][name][key]
    table=[]
    for cohort,title in [('new_val','新 VAL64'),('legacy_val','原 VAL4')]:
        for name,label in [('kv_correct','完整 KV Reader'),('final_correct','最终特征 MLP（含 LayerNorm）'),
                           ('original_slow','原 Slow Head'),('fast_correct','原 Fast：正确 KV'),('teacher_feedback','固定 Fast 反馈生产者')]:
            table.append(f'| {title} | {label} | {metric(cohort,name):.6f} | {metric(cohort,name,"FDE"):.6f} | {metric(cohort,name,"heading_mae_rad"):.6f} |')
    sources=[]
    for reader,title in [('kv','KV Reader'),('final','最终特征 MLP')]:
        for cohort,label in [('new_val','新 VAL64'),('legacy_val','原 VAL4')]:
            sources.append(f'| {title} | {label} | '+ ' | '.join(f'{metric(cohort,reader+"_"+mode):.6f}' for mode in ['correct','shuffled','reference'])+' |')
    logs=[]
    for cohort in ['new_val','legacy_val']:
        for log,group in strata[cohort]['by_log'].items():
            logs.append(f'| `{log}` | {group["windows"]} | {group["metrics"]["kv_correct"]["ADE"]:.6f} | {group["metrics"]["final_correct"]["ADE"]:.6f} |')
    speed_rows=[]
    for name,group in strata['new_val']['by_speed'].items():
        gain=group['kv_correct_advantage']['fast_correct']
        speed_rows.append(f'| {name} | {group["windows"]} | {group["metrics"]["kv_correct"]["ADE"]:.6f} | '
            f'{group["metrics"]["fast_correct"]["ADE"]:.6f} | {gain["better_windows"]}/{group["windows"]} |')
    gain=strata['new_val']['all']['kv_correct_advantage']['fast_correct']
    p=path['new_val'];c=p['comparisons'];max_xy=c['none_to_empty']['max_abs_xy_coordinate_change_m']
    prefix=next(r for r in sdpa['comparisons'] if r['label']=='fp32_math_crop_repeat_masked_external')
    return f'''# 冻结 Slow 的可读性与空路径诊断

本轮验证了：**既有冻结 Slow 的完整六层 KV，可以被重新训练的小型 Reader 读出与正确来源有关的轨迹信息。** 新 VAL64 的 KV Reader ADE 为 {metric('new_val','kv_correct'):.6f}m，最终 Planning 特征新 Head 为 {metric('new_val','final_correct'):.6f}m。更换来源令两者误差显著增加。这没有证明新视觉推理，也没有证明原 Fast 已充分消费 KV；充分可训练的专用 ActionExpert 是另一个独立实验，当前结果不代表其结论或架构上限。

## 范围与固定训练协议

源为 expanded joint step800，SHA256 `{protocol['source_checkpoint_sha256']}`。保留 TRAIN256/四个日志、新 VAL64/两个日志、原 VAL4/一个日志。两组 VAL 已用于先前诊断，不能重新称为未观察过的 TEST；同一日志相邻窗口也不是独立路线。新 VAL 与原 VAL始终分开。

Slow/Fast 权重冻结且没有更新；原始图像、Ego、导航和固定历史 Fast 数值反馈不变。专家未来轨迹只进入读出训练监督和指标，没有作为缓存或 Reader 输入。324 次 Slow prefill 导出的原 Head 输入能逐位重现原预测；完整 K/V 形状 `{export['raw_KV_shape']}`、BF16、原始 K/V 共 {export['raw_KV_bytes']:,} bytes（{export['raw_KV_bytes']/2**30:.2f}GiB）。九类源 token、三轴位置和 mask 均保留，没有池化或截断。RMS 缩放只由 TRAIN256 推导，使用逐层正标量可逆缩放，不对每个 V 做去均值或归一化。

两个新初始化的诊断读出分别为：完整 KV 的六路 Attention Reader（{result['probes']['kv']['parameters']:,} 参数），以及与原 Slow Head 相同拓扑的 `LayerNorm→Linear(256)→GELU→Linear(3)` 最终八个 Planning 特征 MLP（{result['probes']['final']['parameters']:,} 参数）。后者没有继承原 Head 权重。

每个 Reader 训练1024步、batch8、8192个样本曝光，即 TRAIN256 的32个完整 epoch；相同 seed42/目标顺序。AdamW LR5e-4、WD0.01、clip1，32步预热后 cosine 衰减到10%。Loss 是既有 mean L1 pose，混合 m/rad；ADE/FDE 只计算 XY，heading 单独报告。固定 TRAIN32 在256/512/1024步观察曲线，最终固定预算才评估 VAL；不据 VAL 选择或延长训练。两者不是此前 Fast 的560次 fresh 条件训练的等预算对照，也没有证明收敛。

![读出训练曲线](../evidence/slowfast_readability/learning_curves.png)

## 正确来源的绝对规划误差

下表由逐窗口预测以 NumPy float64 重算；`result.json` 保留原始 float32 Torch 汇总，核对的最大数值差为 {verification['metric_recompute_max_abs_error']:.3g}，不影响这里的判断。

| Cohort | 读出 / 基线 | ADE (m) | FDE (m) | heading MAE (rad) |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(table)}

原 Fast 无缓存的新 VAL ADE 为 {metric('new_val','fast_none'):.6f}m。KV Reader 相对原 Fast 正确 KV 的平均 ADE 改善为 {metric('new_val','fast_correct')-metric('new_val','kv_correct'):.6f}m；逐窗口胜负、配对差值以及速度分层见 [window_metrics.json](../evidence/slowfast_readability/window_metrics.json) 和 [stratified_summary.json](../evidence/slowfast_readability/stratified_summary.json)。不能只用总体平均替代逐日志检查。

平均收益没有覆盖多数窗口：新 VAL64 中 KV Reader 相对原 Fast 正确 KV 改善 {gain['better_windows']} 个、退化 {gain['worse_windows']} 个；逐窗口 ADE 改善的中位数为 {gain['median_gain_m']:.6f}m（负值表示 KV Reader 更差）。因此只能说部分误差较大的窗口拉低了平均误差，不能写成逐窗口一致改善。原 VAL4 也只有一个移动窗口改善，其余三个静止窗口略退化。

| 新 VAL 速度组（m/s） | 窗口数 | KV Reader ADE | 原 Fast 正确 KV ADE | KV Reader 改善窗口 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(speed_rows)}

low 为<1、mid 为[1,10)、high 为≥10m/s；ADE 单位均为米。速度组仍来自两个日志，不能当作独立场景实验。

| 日志 | 窗口数 | KV Reader ADE (m) | 最终特征 MLP ADE (m) |
| --- | ---: | ---: | ---: |
{chr(10).join(logs)}

![逐日志读出比较](../evidence/slowfast_readability/per_log_readability.png)

## 固定 Reader 后替换来源

| Reader | Cohort | 正确来源 ADE | 打乱来源 ADE | 固定 TRAIN 来源 ADE |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(sources)}

单位均为米。固定参考是预先指定的最小 TRAIN token `{protocol['reference_token']}`；打乱来源沿用 split/cohort 内确定性错配。KV 替换保留目标的位置、类型和布局；最终特征 MLP 使用相同来源选择。错配与固定 TRAIN 来源均为刻意无效的离线输入干预，不能当作合法在线缓存。

![来源条件比较](../evidence/slowfast_readability/source_conditions.png)

这些差异支持“来源相关的信息能被读出”。完整 KV 含 Ego、导航、固定旧 Fast 候选轨迹，以及视觉/context/规划 query；没有去掉这些非视觉线索的消融，因此**不能把收益归因为新视觉推理或增量世界知识**。固定参考只有一个，错配可能有分布差异。32个 epoch 的监督也能教新 Head 利用已有运动线索，不能据此宣称更强泛化或完整 KV 架构上限。

## 空路径、metadata 与数值后端

对全部68个 VAL 窗口比较四条路径：native 无缓存；运行六层 fusion、屏蔽所有外部列且强制 no_cache token；相同 fusion/mask 保留正常 metadata；正确 KV。两条屏蔽路径都保持 enabled=True，因此 metadata 效应是在同一 fusion 实现内比较。

| 新 VAL64 路径 | ADE (m) |
| --- | ---: |
| native none | {p['modes']['none']['ADE']:.9f} |
| fusion + masked + no_cache token（empty） | {p['modes']['empty']['ADE']:.9f} |
| fusion + masked + 正常 metadata | {p['modes']['masked']['ADE']:.9f} |
| fusion + 正确 KV | {p['modes']['correct']['ADE']:.9f} |

none→empty 的最大单个 XY 坐标变化为 {max_xy:.8f}m，故不能称空路径逐位等价；但整体平均 ADE 变化仅 {c['none_to_empty']['delta_mean_ADE_m']*1000:.6f}mm。empty→metadata 的平均 ADE 变化为 {c['empty_to_masked']['delta_mean_ADE_m']*1000:.6f}mm；这是同一路径内的 metadata 对照。本轮全部是fresh缓存，metadata为同一常量；这个对照识别缓存模式Token的切换，不证明年龄或相对位姿建模已有收益。上述量分别是最大坐标变化与平均误差变化，不能混为一个“数值噪声”大小。原 VAL4 的独立数值及逐层差异见 [path_summary.json](../evidence/slowfast_readability/path_summary.json)。完整 Context 张量的对照只在首个预声明窗口采集，不能把该检查外推成68次全 Context 验证。

单个预声明窗口的 FA4 数值审计重放原生 helper，post-gate 输出逐位相同，重建 RoPE Q/K 也逐位相同。原生完整 query 使用 FLASH；裁剪到八个 query、显式重复 KV 的 bridge 几何使用 efficient SDPA。FP32 MATH 下 cropped GQA 与显式 repeat 逐位相同；增加全部被屏蔽的外部前缀后，最大 attention 张量差为 {prefix['vs_fp32_math_crop_gqa']['max_abs_difference']:.12g}。这提供了一个真实的数值路径差异来源，没有发现缺失 Q/K norm 或重复 gate 的证据；但没有执行全长 query FP32/MATH 或全模型后端消融，不能把完整规划差异全部归因于已测的单层后端，也不能直接把 attention 张量误差称为米。

![空路径与后端摘要](../evidence/slowfast_readability/path_backend_summary.png)

## 重现、测试与边界

原 none/masked/correct 的204个保存预测均精确重现，324个原 Slow Head 输出精确重现，源参数和完整缓存未变。独立进程重载两个新 Reader，对每个 Reader 的16个新 VAL 窗口×3种来源条件复核，合计96个预测条件，最大差0；这个重载覆盖不包括全部68个 VAL，不能描述为全 VAL 逐位恢复。

历史回归记录为{verification['historical_regression']['counts']['tests']} passed、0 failures/errors/skips，XML 时间为 `{verification['historical_regression']['suite_timestamps'][0]}`。它只证明该次测试快照；其后 checkpoint helper 或独立控制代码的修改不由这条历史记录自动覆盖。详见 [verification.json](../evidence/slowfast_readability/verification.json)、[reload.json](../evidence/slowfast_readability/reload.json)、[SDPA 审计](../evidence/slowfast_readability/sdpa_audit.json) 和 [源身份](../evidence/slowfast_readability/source_hashes.json)。原始训练/诊断 JSON 和逐窗口记录已脱去机器绝对路径，数值保留；完整 K/V、图像、`.pt` 权重和3GB reload 输入未提交。导出文件校验见 [SHA256.json](../evidence/slowfast_readability/SHA256.json)。

本轮没有启动 Stage3 Fast 通信训练。用户另行授权的145M专用24层 ActionExpert/完整 Fast 读出控制使用独立协议和报告；它们的训练状态、结果与本报告分开，不能由这里的小型 Reader 成绩代替。可读性、正确来源依赖、绝对规划质量和原 Fast 能否有效消费 KV 是四个不同问题。

CPU 导出命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
python -m tools.report_slowfast_readability \\
  --run outputs/slowfast_readability/run_01 \\
  --output evidence/slowfast_readability
```
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/slowfast_readability/run_01')
    parser.add_argument('--output',default='evidence/slowfast_readability')
    parser.add_argument('--reload',default='outputs/slowfast_readability/reload_01.json')
    parser.add_argument('--sdpa-audit',default='outputs/slowfast_readability/sdpa_audit_01.json')
    parser.add_argument('--test-xml',default='outputs/slowfast_readability/final_tests.xml')
    parser.add_argument('--document',default='docs/SLOWFAST_KV_READABILITY.md')
    args=parser.parse_args()
    run(args.run,args.output,args.reload,args.sdpa_audit,args.test_xml,args.document)


if __name__=='__main__':main()
