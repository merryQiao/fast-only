"""CPU report for completed frozen-Slow Expert/Fast controls; no weight loading."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.report_slowfast_readability import (sha256,portable,read_json,read_lines,
    write_json,pose_metrics,window_ade,speed_bin)

ARMS=('expert','fast')
MODES=('correct','masked','reference','shuffled')
TRAINER_REVISIONS={'expert':'d82fac8','fast':'fc4cb32'}


def methods_text():
    return '''## 固定方法与比较边界

固定 expanded joint step800 的 Slow KV、同一 TRAIN256（四日志）及归一化统计；新 VAL64（两日志）和原 VAL4（一日志）分开报告。它们已被之前实验观察，是诊断 VAL，不能重新称为独立 TEST。没有下载数据，也没有更新 Slow。

每个消费者均新增4096步、micro-batch1，即16个完整 TRAIN epoch；seed42，相同无放回 epoch shuffle 顺序。训练只使用正确源 KV。Loss 为8×3 SE(2) mean L1 pose，混合 m/rad；ADE/FDE 只衡量 XY。AdamW block/LoRA LR1e-4、条件模块/Head LR5e-4，WD0.01、clip1；128步预热后 cosine 至10%。固定 TRAIN32 在1024/2048/3072/4096步观察；最终固定预算后才评估 VAL，没有按 VAL 调参、早停或延长。

Expert 是新随机初始化的145,145,859参数、24层全参数 dense decoder：残差宽512、SwiGLU1536、8 Query heads/2 KV heads/head dim256；每四层依次读取 Slow4/8/12/16/20/24 的原始 K/V，因此每次共24次源 Attention。源 K/V 不做适配投影或重新 norm，K 使用原生 Qwen3.5 partial mRoPE 一次；当前 Ego、command、metadata 和八个 planning queries 构成11个因果自身 token。源全部屏蔽时仍保留自身 Attention 和相同有效 metadata。

完整 Fast 消费者保留0.8B模型、当前四张原图的视觉输入、18层 GDN 和六层融合，更新语言 LoRA及现有新增模块。它从已经训练的 joint Fast 起步，已有1312次曝光；Expert 没有这份任务初始化。两者同新增样本数，不同初始状态、参数数、输入路径、源读取次数和 FLOPs；不能解释为只有“全部参数 vs LoRA”这一因素的消融。

Fast 缓存的仅为已冻结视觉塔/merger 对原始图像的原生输出。图像和 Processor 不变；语言主干每次完整重算。两个预声明实例（一个 TRAIN、一个新 VAL）核对缓存前后预测最大差0，并有 CPU 单测；这不等于逐一验证全部324个窗口。运行耗时及峰值包含准备、模型/缓存加载、Fast 视觉特征生成和评估成本，不是部署延迟或推理频率。

主要位置协议保持现有 Fast 的4096 reserved offset：Expert 第一个 sensor 位置为源 Slow18-token suffix 首位置+4096，随后11个位置递增。它不是论文连续位置协议。此模型是QwenDrive启发的直接KV/native几何/全dense消费者，不是完整论文配方：state/nav/metadata只由三个输入token提供，没有每block的AdaLN条件注入，没有noisy waypoint或flow-time Fourier编码，也没有50点flow matching。保留的是8×3回归、L1、11个因果token和reserved4096位置。这些差异可能影响轨迹预测质量，在TRAIN256下不能排除，故不声称论文复现或论文正控失效。

终点评估正确来源、全部外部列 masked、固定最小 TRAIN 来源、原 split/cohort-local shuffled 来源。保留当前输入、目标位置布局及 metadata，只替换源内容。四种条件中只有 correct 被本轮训练；其余是无效/分布外推理反事实，并非单独训练的无 KV 基线。正确来源优于反事实说明来源依赖，不能独自证明绝对规划良好或新视觉推理；缓存本身也包含 Ego、导航和旧 Fast 数值候选。
'''


def validate_arm(root,arm,replay_path):
    root=Path(root)
    for name in ['result.json','protocol.json','predictions.json','train.jsonl']:
        if not (root/name).is_file():raise ValueError(f'{arm} has not completed: missing {name}')
    result=read_json(root/'result.json');protocol=read_json(root/'protocol.json');replay=read_json(replay_path)
    c=protocol['config']
    locked=dict(schema='frozen_slow_expert_control_v1',steps=4096,micro_batch=1,seed=42,train_size=256,
        warmup_steps=128,min_lr_ratio=.1,weight_decay=.01,grad_clip=1.,block_lr=1e-4,conditioning_head_lr=5e-4,
        curve_steps=[1024,2048,3072,4096],expert=dict(width=512,ffn_width=1536,layers=24,query_heads=8,kv_heads=2,head_dim=256,position_offset=4096))
    if any(c[key]!=value for key,value in locked.items()):raise ValueError('Locked consumer recipe differs')
    if (not result['completed'] or result['arm']!=arm or result['protocol']!=protocol
            or not result['frozen_source_unchanged'] or result['new_consumer_updates']!=4096
            or result['new_Slow_updates']!=0 or result['new_download_bytes']!=0
            or result['val_used_for_selection'] or result['test_used']):raise ValueError(f'{arm} incomplete/invalid result')
    if (not replay['passed'] or replay['arm']!=arm or replay['step_before']!=4095 or replay['step_after']!=4096
            or not replay['source_provider_frozen'] or set(replay['comparisons'])!={'model','optimizer','scheduler','rng','reference_prediction'}
            or any(not x['passed'] or x['max_abs_difference']!=0
                                                         for x in replay['comparisons'].values())):
        raise ValueError(f'{arm} lacks an exact actual-update replay')
    training=read_lines(root/'train.jsonl');order=protocol['order']
    if len(training)!=4096 or [r['step'] for r in training]!=list(range(1,4097)) or [r['token'] for r in training]!=order:
        raise ValueError(f'{arm} sampler/update order differs')
    if replay['token']!=order[-1]:raise ValueError('Replayed target differs from the final sampler step')
    if len(set(order))!=256 or set(Counter(order).values())!={16} or any(len(set(order[i:i+256]))!=256 for i in range(0,4096,256)):
        raise ValueError(f'{arm} does not contain sixteen uniform epochs')
    if not np.isfinite([[r['loss'],r['grad_norm'],*r['lr']] for r in training]).all():raise ValueError('Nonfinite training record')
    from slowfast_v11.expanded_protocol import lr_factor
    for row in training:
        factor=lr_factor(row['step'],4096,128,.1)
        if row['lr']!=[1e-4*factor,5e-4*factor]:raise ValueError('Actual update LRs differ from the predeclared schedule')
    records=read_json(root/'predictions.json');conditions={(r['token'],r['mode']) for r in records}
    tokens={r['token'] for r in records}
    if len(records)!=272 or len(conditions)!=272 or len(tokens)!=68 or Counter(r['cohort'] for r in records)!={'new_val':256,'legacy_val':16}:
        raise ValueError(f'{arm} final condition coverage differs')
    for token in tokens:
        if {r['mode'] for r in records if r['token']==token}!=set(MODES):raise ValueError('Missing source intervention')
    error=0.
    for cohort in ['new_val','legacy_val']:
        for mode in MODES:
            rows=[r for r in records if (r['cohort'],r['mode'])==(cohort,mode)]
            actual=pose_metrics([r['prediction'] for r in rows],[r['target'] for r in rows])
            for key in ['ADE','FDE','heading_mae_rad','loss_l1_pose']:
                error=max(error,abs(actual[key]-result['summary'][cohort][mode][key]))
    if error>1e-5:raise ValueError('Stored and recomputed final metrics differ')
    norms=np.array([r['grad_norm'] for r in training])
    health=dict(updates=4096,nonfinite_records=0,preclip_gradient_norm=dict(minimum=float(norms.min()),median=float(np.median(norms)),
        maximum=float(norms.max()),clipped_updates=int((norms>protocol['config']['grad_clip']).sum()),
        clip_fraction=float((norms>protocol['config']['grad_clip']).mean())))
    probes=[dict(step=r['step'],blocks=r['expert_grads']) for r in training if r['expert_grads']]
    if arm=='expert':
        if [r['step'] for r in probes]!=[1,1024,2048,3072,4096] or any(
            len(r['blocks'])!=24 or {b['block'] for b in r['blocks']}!=set(range(1,25))
            or not all(b['finite'] and b['nonzero'] for b in r['blocks']) for r in probes):
            raise ValueError('Missing finite nonzero 24-block gradient probes')
    health['block_gradient_probes']=probes
    return dict(result=result,protocol=protocol,replay=replay,records=records,training=training,health=health,metric_error=error)


def summarize(arms):
    lookups={arm:{(r['token'],r['mode']):r for r in data['records']} for arm,data in arms.items()}
    if lookups['expert'].keys()!=lookups['fast'].keys():raise ValueError('Arm condition identities differ')
    windows=[]
    for token in sorted({token for token,mode in lookups['expert']}):
        reference=lookups['expert'][token,'correct'];target=reference['target'];errors={}
        for arm in ARMS:
            for mode in MODES:
                row=lookups[arm][token,mode]
                if any(row[key]!=reference[key] for key in ['target','route','cohort','speed_mps']):raise ValueError('Matched-window identity differs')
                errors[f'{arm}_{mode}']=window_ade(row['prediction'],target)
        windows.append(dict(token=token,cohort=reference['cohort'],route=reference['route'],speed_mps=reference['speed_mps'],
            speed_bin=speed_bin(reference['speed_mps']),ADE=errors,
            fast_correct_advantage_vs_expert_m=errors['expert_correct']-errors['fast_correct'],
            correct_source_advantage_m={arm:{mode:errors[f'{arm}_{mode}']-errors[f'{arm}_correct']
                                          for mode in MODES if mode!='correct'} for arm in ARMS}))
    def group(rows):
        metrics={}
        for arm in ARMS:
            metrics[arm]={mode:pose_metrics([lookups[arm][r['token'],mode]['prediction'] for r in rows],
                [lookups[arm][r['token'],mode]['target'] for r in rows]) for mode in MODES}
        def differences(values):
            a=np.array(values);return dict(mean_gain_m=float(a.mean()),median_gain_m=float(np.median(a)),
                improved_windows=int((a>0).sum()),equal_windows=int((a==0).sum()),worse_windows=int((a<0).sum()))
        return dict(windows=len(rows),tokens=[r['token'] for r in rows],metrics=metrics,
            fast_correct_advantage_vs_expert=differences([r['fast_correct_advantage_vs_expert_m'] for r in rows]),
            correct_source_advantage={arm:{mode:differences([r['correct_source_advantage_m'][arm][mode] for r in rows])
                for mode in MODES if mode!='correct'} for arm in ARMS})
    summary={}
    for cohort in ['new_val','legacy_val']:
        rows=[r for r in windows if r['cohort']==cohort]
        summary[cohort]=dict(all=group(rows),by_log={log:group([r for r in rows if r['route']==log]) for log in sorted({r['route'] for r in rows})},
            by_speed={name:group([r for r in rows if r['speed_bin']==name]) for name in ['low','mid','high']
                      if any(r['speed_bin']==name for r in rows)})
    return windows,summary


def code_identity(arms):
    result={}
    for arm,revision in TRAINER_REVISIONS.items():
        blob=subprocess.check_output(['git','show',f'{revision}:tools/train_frozen_expert.py'])
        digest=hashlib.sha256(blob).hexdigest()
        recorded=next(value for name,value in arms[arm]['protocol']['source_code_sha256'].items() if Path(name).name=='train_frozen_expert.py')
        if digest!=recorded:raise ValueError(f'{arm} trainer does not match recorded historical revision')
        result[arm]=dict(revision=revision,trainer_sha256=digest,recorded_matches_git=True,
            recorded_source_sha256=portable(arms[arm]['protocol']['source_code_sha256']))
    diff=subprocess.check_output(['git','diff','d82fac8','fc4cb32','--','tools/train_frozen_expert.py'],text=True)
    return dict(arms=result,trainer_diff=diff,diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        interpretation='unused cache.final empty->zeros; direct Fast ego/command replay packet; TRAIN curve logging. No forward/loss/optimizer change.',
        current_files={name:sha256(name) for name in ['tools/train_frozen_expert.py','tools/reload_frozen_expert.py','slowfast_v11/action_expert.py']})


def source_identity(arms):
    """Rehash small pinned inputs; do not reread the 33GB cache or base weights."""
    from slowfast_v11.expanded_protocol import epoch_plan,scalable_mismatches,select_train_diagnostics
    protocol=arms['expert']['protocol'];c=protocol['config']
    original=read_json(Path(c['source_run'])/'result.json')
    if (original['arms']['joint']['checkpoint_sha256']!=protocol['source_checkpoint']
            or original['data_identities']!=protocol['source_data'] or original['assets']!=protocol['assets']):
        raise ValueError('Frozen provider/result model or data identity differs')
    paths=original['config']['data'];digests={}
    for name in ['train_manifest','val_manifest','stats']:
        digests[name]=sha256(paths[name])
        if digests[name]!=protocol['source_data'][name]:raise ValueError('Current manifest/statistics identity differs')
    train=read_lines(paths['train_manifest']);val=read_lines(paths['val_manifest']);all_rows=train+val
    if epoch_plan(train,4096,42)!=protocol['order']:raise ValueError('Recorded sampler does not reconstruct from pinned TRAIN')
    by_token={r['token']:r for r in all_rows}
    for data in arms.values():
        for record in data['records']:
            if record['target']!=by_token[record['token']]['trajectory_gt']:raise ValueError('Saved VAL target differs from pinned manifest')
    manifest=Path(c['feature_run'])/'frozen_cache/manifest.json'
    if sha256(manifest)!=protocol['feature_manifest_sha256']:raise ValueError('Current frozen feature manifest SHA differs')
    saved=read_json(manifest)
    if saved['source_checkpoint']!=protocol['source_checkpoint'] or saved['data_identities']!=protocol['source_data']:
        raise ValueError('Frozen feature manifest provider/data differ')
    donors,coverage=scalable_mismatches(all_rows);reference=min(r['token'] for r in train)
    source_map={r['token']:dict(correct=r['token'],masked=r['token'],reference=reference,shuffled=donors[r['token']]) for r in val}
    return dict(source_checkpoint_sha256=protocol['source_checkpoint'],source_model_assets=protocol['assets'],
        source_data=protocol['source_data'],small_inputs_rehashed=digests,feature_manifest_sha256=sha256(manifest),
        feature_cache_shape=saved['raw_KV_shape'],feature_shards=len(saved['shards']),
        report_rechecks_cache_manifest_only=True,shard_bodies_authenticated_by_train_and_replay=True,
        reconstructed_diagnostic_tokens=select_train_diagnostics(train),counterfactual_sources=source_map,
        donor_coverage=coverage,source_map_scope='Reconstructed from pinned manifests and saved forward rule; predictions do not separately store donor IDs.')


def plots(output,arms,summary,windows,readability):
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    for arm in ARMS:
        data=arms[arm];training=data['training'];steps=sorted(map(int,data['result']['curves']))
        values=np.array([r['loss'] for r in training]);axes[0].plot(np.arange(64,4097),np.convolve(values,np.ones(64)/64,mode='valid'),label=arm)
        axes[1].plot(steps,[data['result']['curves'][str(step)]['ADE'] for step in steps],'o-',label=arm)
        axes[2].plot(range(1,4097),[r['lr'][0] for r in training],label=arm)
    for ax,title,y in zip(axes,['TRAIN loss: 64-update mean','Same fixed TRAIN32','Actual block/LoRA update LR'],
                         ['Mean L1 pose (mixed m/rad)','XY ADE (m)','Learning rate']):
        ax.set(title=title,xlabel='New consumer update',ylabel=y);ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        metrics=summary[cohort]['all']['metrics'];x=np.arange(4)
        for arm,offset in [('expert',-.18),('fast',.18)]:
            ax.bar(x+offset,[metrics[arm][mode]['ADE'] for mode in MODES],width=.35,label=arm)
        ax.axhline(readability['summary']['kv'][cohort]['correct']['ADE'],ls=':',color='grey',label='1M reader: unequal recipe')
        ax.set(xticks=x,xticklabels=MODES,ylabel='XY ADE (m)',title=f'{cohort} ({summary[cohort]["all"]["windows"]} windows)')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'source_conditions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        metrics=summary[cohort]['all']['metrics']['fast'];modes=['masked','reference','shuffled']
        gains=[1000*(metrics[mode]['ADE']-metrics['correct']['ADE']) for mode in modes]
        ax.bar(range(3),gains,color=['C0','C1','C2']);ax.axhline(0,color='grey',ls=':')
        ax.set(xticks=range(3),xticklabels=modes,ylabel='Correct-source ADE advantage (mm)',
               title=f'Fast: {cohort} (positive = correct better)');ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'fast_source_effects.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        rows=[r for r in windows if r['cohort']==cohort]
        for name,color in [('low','C0'),('mid','C1'),('high','C2')]:
            selected=[r for r in rows if r['speed_bin']==name]
            if selected:ax.scatter([r['ADE']['expert_correct'] for r in selected],[r['ADE']['fast_correct'] for r in selected],color=color,label=name,alpha=.75)
        limit=max(max(r['ADE']['expert_correct'],r['ADE']['fast_correct']) for r in rows)*1.05
        ax.plot([0,limit],[0,limit],ls=':',color='grey');ax.set(xlabel='Expert correct ADE (m)',ylabel='Fast correct ADE (m)',title=cohort)
        ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'matched_windows.png',dpi=160);plt.close(fig)


def final_document(arms,summary,verification,readability):
    rows=[]
    for cohort in ['new_val','legacy_val']:
        for arm in ARMS:
            metrics=summary[cohort]['all']['metrics'][arm]
            rows.append(f'| {cohort} | {arm} | '+' | '.join(f'{metrics[m]["ADE"]:.6f}' for m in MODES)+' |')
    health=[]
    for arm in ARMS:
        data=arms[arm];r=data['result'];p=data['protocol'];h=data['health']['preclip_gradient_norm']
        health.append(f'| {arm} | {p["trainable_parameters"]:,}/{p["total_parameters"]:,} | {r["elapsed_s"]:.2f} | '
                      f'{r["peak_allocated_gib"]:.3f} | {h["median"]:.6f} | {h["clip_fraction"]:.2%} |')
    logs=[]
    for cohort in ['new_val','legacy_val']:
        for log,s in summary[cohort]['by_log'].items():
            logs.append(f'| `{log}` | {s["windows"]} | {s["metrics"]["expert"]["correct"]["ADE"]:.6f} | {s["metrics"]["fast"]["correct"]["ADE"]:.6f} |')
    curves=[f'| {step} | {arms["expert"]["result"]["curves"][str(step)]["ADE"]:.6f} | '
            f'{arms["fast"]["result"]["curves"][str(step)]["ADE"]:.6f} |' for step in [1024,2048,3072,4096]]
    paired=summary['new_val']['all']['fast_correct_advantage_vs_expert']
    source_gains=summary['new_val']['all']['correct_source_advantage']['fast']
    legacy_gain=summary['legacy_val']['all']['correct_source_advantage']['fast']['masked']
    correct_rows=[]
    for cohort in ['new_val','legacy_val']:
        for arm in ARMS:
            m=summary[cohort]['all']['metrics'][arm]['correct']
            correct_rows.append(f'| {cohort} | {arm} | {m["ADE"]:.6f} | {m["FDE"]:.6f} | {m["heading_mae_rad"]:.6f} |')
    return f'''# 冻结 Slow：专用 ActionExpert 与完整 Fast 控制

两条预声明消费者训练及4095→4096实际更新重放均完成。新VAL上Expert正确KV ADE为1.455m，完整Fast＋LoRA为0.888m；Expert学到来源依赖，Fast仍没有稳定的正确KV增量收益。本报告区分“是否消费来源”与“绝对规划质量”，不把来源敏感度当作性能上限或论文复现。

{methods_text()}

## 固定终点结果

下表由保存预测以 NumPy float64 重算；原始 float32汇总保留在每臂 result.json。新/原 VAL 分开，单位为米。

| Cohort | 消费者 | correct ADE | masked ADE | reference ADE | shuffled ADE |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Expert 正确来源明显优于其反事实条件，说明它在这份配方中学到了来源依赖；但其新 VAL正确来源 ADE 为 {summary['new_val']['all']['metrics']['expert']['correct']['ADE']:.6f}m，原 VAL为 {summary['legacy_val']['all']['metrics']['expert']['correct']['ADE']:.6f}m，不能描述为绝对规划已良好或已经建立上限。它在这个适配配方中没有赢过Fast，不等于完整论文正控失败，也不证明小LoRA通常优于全参数Expert。冷启动145M模型在固定16 epoch配方下的表现，不是否定专用 Expert 架构的证据。

**Fast的正确内容增益仍弱。** 新VAL中 correct 比 masked 的ADE反而高 {-source_gains['masked']['mean_gain_m']*1000:.6f}mm，只在{source_gains['masked']['improved_windows']}/64窗口更好；相对shuffled的平均改善仅 {source_gains['shuffled']['mean_gain_m']*1000:.6f}mm。原VAL correct也比masked高 {-legacy_gain['mean_gain_m']*1000:.6f}mm。这是当前固定配方检查点的XY结果，不是充分训练的能力上限或KV信息不存在的证据；也不能把错误源/屏蔽源的退化直接当作独立训练的无KV架构成绩。

![Fast的配对来源增益](../evidence/slowfast_expert_control/fast_source_effects.png)

正确来源的FDE与航向误差一并保留，避免把ADE当作所有指标：

| Cohort | 消费者 | ADE(m) | FDE(m) | heading MAE(rad) |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(correct_rows)}

Fast 相对 Expert 的新 VAL correct 配对平均 ADE 改善为 {paired['mean_gain_m']:.6f}m，中位数为 {paired['median_gain_m']:.6f}m；{paired['improved_windows']}/{summary['new_val']['all']['windows']}窗口改善、{paired['worse_windows']}退化。正负均保留。速度分层、每窗口 correct 对 masked/reference/shuffled 的配对差值和两臂比较见 [stratified_summary.json](../evidence/slowfast_expert_control/stratified_summary.json) 与 [window_metrics.json](../evidence/slowfast_expert_control/window_metrics.json)。

![来源条件](../evidence/slowfast_expert_control/source_conditions.png)

| 日志 | 窗口数 | Expert correct ADE | Fast correct ADE |
| --- | ---: | ---: | ---: |
{chr(10).join(logs)}

![逐窗口匹配](../evidence/slowfast_expert_control/matched_windows.png)

此前1,002,553参数 KV Reader 的新 VAL ADE 为 {readability['summary']['kv']['new_val']['correct']['ADE']:.6f}m；它训练1024×batch8，即32 epoch，另有TRAIN-RMS缩放、学习型K/V投影、显式位置特征和group-count prior。Expert直接消费原生K/V并使用native RoPE，二者不只是参数量不同。小Reader列为不等预算/不同结构参考，不能由它和145M Expert的差异推断参数越大越差。最终特征 MLP与原 Slow Head 的对照见[可读性报告](SLOWFAST_KV_READABILITY.md)。

## 优化、冻结源与资源记录

| 消费者 | 可训练/总参数 | 总耗时(s，含准备) | peak allocated GiB | 梯度norm中位数（clip前） | 被clip比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(health)}

所有4096步 loss/梯度norm/LR记录均有限；Expert在1/1024/2048/3072/4096步对全部24块记录有限、非零梯度。这只证明这些检查点各块存在梯度，不证明每个参数每步都获得有效语义学习。源 KV/位置/group/mask版本未变；两臂源模型、数据/统计、feature manifest和全部4096个目标顺序身份逐一核对相同，详见[verification.json](../evidence/slowfast_expert_control/verification.json)。

![训练曲线与实际LR](../evidence/slowfast_expert_control/training_curves.png)

| 新增更新步 | 固定 TRAIN32 Expert ADE | 固定 TRAIN32 Fast ADE |
| --- | ---: | ---: |
{chr(10).join(curves)}

曲线不是完整 TRAIN256 终点汇总，也不证明已收敛；4096步预算没有依据曲线或 VAL 结果再延长。

## 恢复、代码身份与科学边界

两臂均从4095步前的参数、AdamW、scheduler、RNG恢复，执行第4096步后比较模型、优化器、scheduler、RNG和保存预测。全部精确差0；它是同H800/runtime单步训练重放，非跨GPU保证，也不是完整VAL逐位重载。Expert重放包含280个模型张量、840个AdamW张量；Fast为209个模型/627个AdamW张量。

第一次Expert重放在构造/更新前因checkpoint tuple与JSON list表示不一致而拒绝；随后只修正读取比较的规范化并重新重放，没有重训或更改训练。失败日志SHA及成功证据均保留在[replay_history.json](../evidence/slowfast_expert_control/replay_history.json)、[expert_replay.json](../evidence/slowfast_expert_control/expert_replay.json)、[fast_replay.json](../evidence/slowfast_expert_control/fast_replay.json)。

Expert历史trainer对应`d82fac8`，Fast对应`fc4cb32`；记录SHA与Git blob核对一致。两版差异仅是未用dummy final缓存初始化、直接保存Fast ego/command重放包和TRAIN曲线输出，不涉及forward、loss或optimizer；保留[代码身份与diff](../evidence/slowfast_expert_control/code_identity.json)。新鲜回归记录为 {verification['tests']['counts']['tests']} passed，时间`{verification['tests']['timestamps'][0]}`，XML SHA256 `{verification['tests']['xml_sha256']}`；它不是此前readability的测试记录，也不能自动覆盖其后修改。

当前入口固定TRAIN256及8255-token缓存形状，GPU全量缓存只适合本批小样本。迁移到完整navtrain需核对官方Scene Filter、重新计算TRAIN统计量，并实现流式数据/KV读取、DDP采样恢复和按真实epoch设置LR日程，再在目标GPU测试。当前入口不能只替换manifest就启动全量训练；已有[全量迁移指南](FULL_DATA_TRAINING.md)仍适用。

本报告只覆盖有界诊断数据、一个seed、两个新VAL日志。来源敏感度可能依赖缓存里的运动/导航/旧反馈信息，未隔离视觉增量，也不构成在线异步或闭环成绩。没有上传GitHub成功的声明。权重、33GB源缓存、图像和replay输入不提交；portable JSON、曲线、预测和hash索引见[evidence目录](../evidence/slowfast_expert_control/SHA256.json)。
'''


def run(base,output,expert_replay,fast_replay,test_xml,document):
    base,output=Path(base),Path(output)
    # Validate both completed arms before creating any final evidence output.
    arms={arm:validate_arm(base/f'{arm}_01',arm,path) for arm,path in zip(ARMS,[expert_replay,fast_replay])}
    common_keys=['config','source_checkpoint','source_data','assets','order','feature_manifest_sha256']
    if any(arms['expert']['protocol'][key]!=arms['fast']['protocol'][key] for key in common_keys):
        raise ValueError('Consumers differ in locked sampler/source/model/data/stats/feature identities')
    if (arms['expert']['protocol']['trainable_parameters']!=145145859
            or arms['expert']['protocol']['total_parameters']!=145145859
            or arms['fast']['protocol']['prior_Fast_exposures']!=1312):raise ValueError('Trainability/initialization differs')
    checks=arms['fast']['protocol']['vision_checks']
    if len(checks)!=2 or any(r['max_abs_difference']!=0 for r in checks):raise ValueError('Cached Fast vision checks did not reproduce outputs')
    identity=code_identity(arms);provider=source_identity(arms);windows,summary=summarize(arms)
    readability=read_json(Path(arms['expert']['protocol']['config']['feature_run'])/'result.json')
    tests=ET.parse(test_xml).getroot();counts={key:sum(int(s.get(key,'0')) for s in tests.iter('testsuite')) for key in ['tests','errors','failures','skipped']}
    if counts['errors'] or counts['failures'] or counts['skipped']:raise ValueError('Regression snapshot is not clean')
    output.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for arm,data in arms.items():
        root=base/f'{arm}_01'
        for name in ['result.json','protocol.json','predictions.json']:
            write_json(output/f'{arm}_{name}',read_json(root/name));inputs.append(root/name)
        (output/f'{arm}_train.jsonl').write_text(''.join(json.dumps(portable(r),allow_nan=False)+'\n' for r in data['training']))
        inputs.append(root/'train.jsonl');write_json(output/f'{arm}_replay.json',data['replay'])
    verification=dict(schema='frozen_expert_report_verification_v1',both_completed=True,exact_shared_identities=common_keys,
        tests=dict(counts=counts,xml_sha256=sha256(test_xml),timestamps=[s.get('timestamp') for s in tests.iter('testsuite')],
            scope='Historical regression snapshot; subsequent changes require their own checks.'),
        matched_conditions=544,matched_windows=68,source_frozen=True,no_Slow_updates=True,no_downloads=True,
        sampler=dict(updates_per_arm=4096,micro_batch=1,epochs=16,identical_order=True),
        optimizer_health={arm:data['health'] for arm,data in arms.items()},cached_vision_checks=checks,
        replay={arm:data['replay'] for arm,data in arms.items()},metric_recompute_max_abs_error=max(data['metric_error'] for data in arms.values()))
    write_json(output/'verification.json',verification);write_json(output/'code_identity.json',identity)
    write_json(output/'provider_identity.json',provider)
    write_json(output/'stratified_summary.json',summary);write_json(output/'window_metrics.json',windows)
    write_json(output/'train_curves.json',{arm:data['result']['curves'] for arm,data in arms.items()})
    write_json(output/'readability_reference.json',dict(summary=readability['summary'],probes=readability['probes'],
        training=dict(updates=1024,batch=8,epochs=32),scope='Unequal initialization/architecture/batch/epochs; not a model-size ablation'))
    failed=base/'expert_replay_01.log'
    history=dict(expert_first_attempt=dict(log_sha256=sha256(failed) if failed.exists() else None,
        failed_before_update=True,reason='Result/checkpoint tuple-versus-JSON-list comparison; reader-only normalization fix',training_changed=False),
        successful_replays=dict(expert=Path(expert_replay).name,fast=Path(fast_replay).name))
    write_json(output/'replay_history.json',history)
    inputs.extend([Path(expert_replay),Path(fast_replay),Path(test_xml)])
    write_json(output/'source_hashes.json',dict(input_files={portable(str(p)):sha256(p) for p in inputs},
        recorded_checkpoints={arm:{key:data['result'][key] for key in ['checkpoint_sha256','before_final_sha256','replay_input_sha256']} for arm,data in arms.items()},
        report_tool_sha256=sha256(__file__),replay_reader_sha256=sha256('tools/reload_frozen_expert.py')))
    plots(output,arms,summary,windows,readability)
    Path(document).write_text(final_document(arms,summary,verification,readability))
    write_json(output/'SHA256.json',{p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps(dict(completed_arms=list(arms),matched_windows=68,tests=counts,exported_files=len(list(output.iterdir())))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='outputs/slowfast_expert_control')
    parser.add_argument('--output',default='evidence/slowfast_expert_control')
    parser.add_argument('--expert-replay',default='outputs/slowfast_expert_control/expert_replay_02.json')
    parser.add_argument('--fast-replay',default='outputs/slowfast_expert_control/fast_replay_01.json')
    parser.add_argument('--test-xml')
    parser.add_argument('--document',default='docs/SLOWFAST_ACTION_EXPERT_CONTROL.md')
    parser.add_argument('--draft-methods',action='store_true')
    args=parser.parse_args()
    if args.draft_methods:
        Path(args.document).write_text('# 冻结 Slow：专用 ActionExpert 与完整 Fast 控制\n\n'
            '**状态：Expert训练及单步重放已完成；Fast终点、重放与测试记录等待主任务确认。本稿只记录方法，未执行最终两臂导出。**\n\n'+methods_text())
        return
    if not args.test_xml:parser.error('--test-xml is required for final export')
    run(args.base,args.output,args.expert_replay,args.fast_replay,args.test_xml,args.document)


if __name__=='__main__':main()
