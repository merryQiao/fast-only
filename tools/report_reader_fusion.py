"""CPU-only portable evidence and paired analysis of frozen residual fusion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.report_slowfast_readability import sha256,portable,read_json,read_lines,write_json,pose_metrics,window_ade,speed_bin

ARMS=('correct','reference')
MODES=('correct','off','reference','shuffled')


def gains(values):
    x=np.asarray(values,dtype=np.float64)
    return dict(mean_gain_m=float(x.mean()),median_gain_m=float(np.median(x)),
                better_windows=int((x>0).sum()),equal_windows=int((x==0).sum()),worse_windows=int((x<0).sum()))


def summarize(rows):
    metrics={a:{m:pose_metrics([r['predictions'][a][m] for r in rows],[r['target'] for r in rows]) for m in MODES} for a in ARMS}
    paired={m:gains([r['ADE']['correct'][m]-r['ADE']['correct']['correct'] for r in rows]) for m in ['off','reference','shuffled']}
    paired['reference_trained_reference_eval']=gains([r['ADE']['reference']['reference']-r['ADE']['correct']['correct'] for r in rows])
    return dict(windows=len(rows),metrics=metrics,correct_trained_advantage=paired)


def run(root,output,document,tests,reload):
    root=Path(root);output=Path(output);document=Path(document);output.mkdir(parents=True,exist_ok=True)
    result=read_json(root/'result.json');protocol=read_json(root/'protocol.json');verify=read_json(reload)
    if result['protocol']!=protocol or not result['completed'] or not verify['passed']:raise ValueError('Incomplete fusion experiment')
    if (not result['frozen_Fast_Reader_unchanged'] or result['new_Fast_updates'] or result['new_Reader_updates'] or result['new_Slow_updates']
            or result['new_download_bytes'] or result['val_used_for_selection'] or result['test_used']):raise ValueError('Invalid freeze/data-selection boundary')
    for name,digest in [('frozen_features.pt',protocol['frozen_features_sha256']),('rows.json',protocol['rows_sha256'])]:
        if sha256(root/name)!=digest:raise ValueError('Frozen feature/row identity differs')
    rows=read_json(root/'rows.json');by_token={r['token']:r for r in rows};train={r['token'] for r in rows if r['split']=='train'}
    logs={cohort:{r['route'] for r in rows if r['cohort']==cohort} for cohort in ['train','new_val','legacy_val']}
    if (len(train)!=256 or {k:len(v) for k,v in logs.items()}!={'train':4,'new_val':2,'legacy_val':1}
            or logs['train']&logs['new_val'] or logs['train']&logs['legacy_val'] or logs['new_val']&logs['legacy_val']):raise ValueError('TRAIN/VAL log boundary differs')
    all_predictions={};training={};health={};max_metric_error=0.
    for arm in ARMS:
        sub=root/arm;r=result['arms'][arm]
        if sha256(sub/'final.pt')!=r['checkpoint_sha256'] or sha256(sub/'before_final.pt')!=r['before_final_sha256']:raise ValueError('Fusion checkpoint SHA differs')
        entries=read_lines(sub/'train.jsonl');training[arm]=entries
        if len(entries)!=512 or [x['step'] for x in entries]!=list(range(1,513)):raise ValueError('Wrong optimizer budget')
        order=[t for x in entries for t in x['tokens']]
        if order!=r['order'] or Counter(order)!=Counter({t:16 for t in train}):raise ValueError('Wrong training sampler/order')
        if any(not np.isfinite([x['loss'],x['grad_norm'],x['lr']]).all() for x in entries):raise ValueError('Nonfinite optimization')
        health[arm]=dict(updates=512,exposures=len(order),grad_norm_median=float(np.median([x['grad_norm'] for x in entries])),
                        clip_fraction=float(np.mean([x['grad_norm']>1 for x in entries])))
        replay=verify['comparisons'][arm]
        if replay['conditions']!=272 or not replay['full_VAL_predictions']['passed'] or any(not x['passed'] or x['max_abs_difference'] for x in replay['actual_step_replay'].values()):raise ValueError('Missing exact reload/replay')
        predictions=read_json(sub/'predictions.json');all_predictions[arm]=predictions
        for cohort in ['new_val','legacy_val']:
            for mode in MODES:
                selected=[x for x in predictions if x['cohort']==cohort and x['mode']==mode]
                m=pose_metrics([x['prediction'] for x in selected],[x['target'] for x in selected])
                max_metric_error=max(max_metric_error,*[abs(m[k]-r['summary'][cohort][mode][k]) for k in ['ADE','FDE','heading_mae_rad']])
    if not np.isfinite(max_metric_error) or max_metric_error>1e-5:raise ValueError('Saved/recomputed metrics disagree')
    if result['arms']['correct']['initial_digest']!=result['arms']['reference']['initial_digest'] or result['arms']['correct']['order']!=result['arms']['reference']['order']:
        raise ValueError('Arm initialization/order not identical')
    windows=[]
    for row in rows:
        if row['split']=='train':continue
        t=row['token'];record=dict(token=t,cohort=row['cohort'],route=row['route'],command=row['command'],
            speed_mps=float(np.hypot(*row['ego'][:2])),speed_bin=speed_bin(np.hypot(*row['ego'][:2])),target=row['trajectory_gt'],predictions={},ADE={})
        for arm in ARMS:
            selected=[x for x in all_predictions[arm] if x['token']==t]
            if len(selected)!=4 or {x['mode'] for x in selected}!=set(MODES) or any(x['target']!=row['trajectory_gt'] for x in selected):raise ValueError('Prediction conditions/target differ')
            record['predictions'][arm]={x['mode']:x['prediction'] for x in selected}
            record['ADE'][arm]={m:window_ade(p,row['trajectory_gt']) for m,p in record['predictions'][arm].items()}
        if record['predictions']['correct']['off']!=record['predictions']['reference']['off']:raise ValueError('Off baselines differ across arms')
        windows.append(record)
    summary={}
    for cohort in ['new_val','legacy_val']:
        selected=[r for r in windows if r['cohort']==cohort]
        summary[cohort]=dict(all=summarize(selected),by_log={log:summarize([r for r in selected if r['route']==log]) for log in sorted({r['route'] for r in selected})},
            by_speed={speed:summarize([r for r in selected if r['speed_bin']==speed]) for speed in ['low','mid','high'] if any(r['speed_bin']==speed for r in selected)})
    claim_checks=dict(new_VAL_mean={k:g['mean_gain_m']>0 for k,g in summary['new_val']['all']['correct_trained_advantage'].items()},
        new_VAL_by_log={log:{k:g['mean_gain_m']>0 for k,g in s['correct_trained_advantage'].items()} for log,s in summary['new_val']['by_log'].items()})
    matched_direction=all(claim_checks['new_VAL_mean'].values()) and all(v for log in claim_checks['new_VAL_by_log'].values() for v in log.values())
    suites=list(ET.parse(tests).getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','errors','failures','skipped']}
    if counts['tests']<658 or any(counts[k] for k in ['errors','failures','skipped']):raise ValueError('Regression record incomplete/failing')
    verification=dict(passed=True,optimizer_health=health,source_frozen=True,same_init_and_order=True,no_downloads=True,
        metric_recompute_max_abs_error=max_metric_error,matched_windows=len(windows),matched_conditions=544,
        tests=dict(counts=counts,xml_sha256=sha256(tests),timestamps=[s.attrib.get('timestamp') for s in suites]),
        live=result['live'],reload=verify,export=protocol['export'],matched_source_direction_checks=claim_checks,
        matched_source_direction_all_pass=matched_direction)
    for name,data in [('result.json',result),('protocol.json',protocol),('verification.json',verification),('reload.json',verify),
                      ('window_metrics.json',windows),('stratified_summary.json',summary),('reader_b1_predictions.json',read_json(root/'reader_b1_predictions.json'))]:
        write_json(output/name,data)
    for arm in ARMS:
        write_json(output/f'{arm}_predictions.json',all_predictions[arm])
        (output/f'{arm}_train.jsonl').write_text(''.join(json.dumps(portable(x),allow_nan=False)+'\n' for x in training[arm]))
    source_paths=['configs/slowfast_reader_fusion.yaml','slowfast_v11/reader_fusion.py','tools/train_reader_fusion.py','tools/reload_reader_fusion.py',
                  'tests_slowfast/test_reader_fusion.py','tools/report_reader_fusion.py']
    source_hashes={p:sha256(p) for p in source_paths}
    if any(source_hashes[p]!=h for p,h in protocol['source_code_sha256'].items()):raise ValueError('Runtime source changed since training')
    write_json(output/'source_hashes.json',source_hashes)
    # Figures show both absolute quality and paired content effect.
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,cohort in zip(axes,['new_val','legacy_val']):
        for a,offset in [('correct',-.18),('reference',.18)]:
            ax.bar(np.arange(4)+offset,[summary[cohort]['all']['metrics'][a][m]['ADE'] for m in MODES],width=.35,label=a+'-trained')
        ax.set(xticks=np.arange(4),xticklabels=MODES,ylabel='ADE (m)',title=cohort);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'source_conditions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for arm in ARMS:
        entries=training[arm];loss=np.array([r['loss'] for r in entries]);steps=np.arange(1,len(loss)+1)
        axes[0].plot(steps[31:],np.convolve(loss,np.ones(32)/32,mode='valid'),label=arm)
        curves=result['arms'][arm]['curves'];axes[1].plot([int(s) for s in curves],[v['ADE'] for v in curves.values()],marker='o',label=arm)
    axes[0].set(xlabel='Optimizer updates',ylabel='Mean L1 pose, moving average32');axes[1].set(xlabel='Optimizer updates',ylabel='Fixed TRAIN32 ADE (m)')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    selected=[r for r in windows if r['cohort']=='new_val']
    for label,vals in [('off minus correct',[r['ADE']['correct']['off']-r['ADE']['correct']['correct'] for r in selected]),
                       ('fixed-trained/reference minus correct',[r['ADE']['reference']['reference']-r['ADE']['correct']['correct'] for r in selected])]:
        ax.plot(range(len(selected)),vals,marker='.',label=label)
    ax.axhline(0,color='grey',ls=':');ax.set(xlabel='Fixed new VAL window order',ylabel='ADE gain (m), positive favors correct fusion');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'paired_gains.png',dpi=160);plt.close(fig)
    hashes={f.name:sha256(f) for f in sorted(output.iterdir()) if f.is_file() and f.name!='SHA256.json'};write_json(output/'SHA256.json',hashes)
    table=[];logs=[]
    for cohort in ['new_val','legacy_val']:
        for arm in ARMS:
            ms=summary[cohort]['all']['metrics'][arm]
            table.append(f'| {cohort} | {arm} | '+' | '.join(f'{ms[m]["ADE"]:.6f}' for m in MODES)+' |')
        for log,s in summary[cohort]['by_log'].items():
            ms=s['metrics'];logs.append(f'| `{log}` | {s["windows"]} | '+ ' | '.join(f'{ms["correct"][m]["ADE"]:.6f}' for m in MODES)+f' | {ms["reference"]["reference"]["ADE"]:.6f} |')
    quality=[]
    for cohort in ['new_val','legacy_val']:
        for arm,mode in [('correct','off'),('correct','correct'),('reference','reference')]:
            m=summary[cohort]['all']['metrics'][arm][mode]
            quality.append(f'| {cohort} | {arm}/{mode} | {m["ADE"]:.6f} | {m["FDE"]:.6f} | {m["heading_mae_rad"]:.6f} |')
    new=summary['new_val']['all'];paired=new['correct_trained_advantage'];exp=protocol['export']
    verdict=('本批新VAL总体及两个日志均满足预声明的来源增量方向判据；支持这份小样本诊断中的增量，尚不是跨种子稳定性证据。'
             if matched_direction else '本批结果未同时满足预声明的来源增量方向判据，不能宣称已经建立稳定的匹配来源收益。')
    failed=[f'总体/{k}' for k,v in claim_checks['new_VAL_mean'].items() if not v]
    failed += [f'{log}/{k}' for log,checks in claim_checks['new_VAL_by_log'].items() for k,v in checks.items() if not v]
    verdict += (' 未通过的比较：'+', '.join(failed)+'。') if failed else ''
    reader_b1=read_json(root/'reader_b1_predictions.json')
    reader_b1_new=[r for r in reader_b1 if r['cohort']=='new_val' and r['mode']=='correct']
    reader_b1_ade=pose_metrics([r['prediction'] for r in reader_b1_new],[r['target'] for r in reader_b1_new])['ADE']
    fixed_reproduced=(new['metrics']['correct']['off']['ADE']-new['metrics']['reference']['reference']['ADE'])/(new['metrics']['correct']['off']['ADE']-new['metrics']['correct']['correct']['ADE'])
    reversals=[]
    for log,s in summary['new_val']['by_log'].items():
        for name,g in s['correct_trained_advantage'].items():
            if g['mean_gain_m']<=0:reversals.append(f'日志`{log}`中{name}比correct平均好{-g["mean_gain_m"]*1000:.3f}mm')
    paired_table='\n'.join(f'| {name} | {g["mean_gain_m"]:.6f} | {g["median_gain_m"]:.6f} | {g["better_windows"]}/{g["equal_windows"]}/{g["worse_windows"]} |' for name,g in paired.items())
    text=f'''# 冻结 Fast＋Reader 的最小残差融合

完成预声明两臂：correct-source 与 fixed-reference-source，均为512步×batch8=4096新增曝光、16个TRAIN epoch。Fast、Slow、既有Reader和原轨迹Head均未更新，只训练{result['arms']['correct']['trainable_parameters']:,}参数融合模块。完整结果和逐窗口证据均保留，不按VAL挑检查点或延长。

**本轮结论：** {verdict} 新VAL原Fast/off ADE={new['metrics']['correct']['off']['ADE']:.6f}m，正确训练/correct={new['metrics']['correct']['correct']['ADE']:.6f}m，固定训练/reference={new['metrics']['reference']['reference']['ADE']:.6f}m。分日志和legacy退化均保留在下表。

固定来源训练对照复现了相对off平均ADE改善的{fixed_reproduced:.2%}；这是结果差值的描述，不是可加的因果分解。正确来源相对该对照的平均优势只有{paired['reference_trained_reference_eval']['mean_gain_m']*1000:.3f}mm，逐窗口中位数{paired['reference_trained_reference_eval']['median_gain_m']*1e6:.3f}微米，不能把近零浮点差算成有实际意义的多数窗口改善。{'；'.join(reversals)}。legacy的平均收益由一个移动窗口贡献，另三个窗口略退化。可见新增任务适配有效，但匹配内容的稳定贡献仍未建立。

## 输入、唯一输出与关闭语义

使用最新完整Fast检查点`{protocol['fast_checkpoint']}`，保留原始1920×1080四帧2Hz、官方Ego4和离散导航command。Fast此轮始终原生无外部KV、原no_cache token。取原Head前八个planning特征F；冻结Reader的六路注意力拼接特征R经小MLP、sigmoid gate、零初始化投影形成delta；唯一输出为原Head(F+delta)。Reader旧轨迹输出不做混合/选择；Reader包含此前学到的K/V投影与位置/group处理，本轮全部冻结，不等价于原生直接KV拼接。

关闭分支跳过Reader和融合hook，直接运行原Fast；不是给Reader全masked，也不是此前0.885010m的Fast屏蔽外部列路径。原Head参数冻结，但训练时不使用no_grad包裹Head，梯度可传入残差。F/R作为冻结特征不接收梯度；融合模块根据当前F与R联合形成残差，未加入导航切换/年龄/动作相容性门控。

两臂相同seed42、融合初始化、无放回epoch顺序和LR；AdamW5e-4、WD.01、clip1、32步warmup后cosine至10%。沿用mean L1 pose(m/rad)，XY ADE/FDE和航向分开。固定TRAIN32于128/256/384/512步观察，两个VAL日志仅终点评估。正确臂训练只用correct来源，参考臂只用固定最小TRAIN来源；各自其他推理来源是干预条件。主额外容量对照为correct-trained/correct-eval对reference-trained/reference-eval，不选择后者的错配条件。

## 终点ADE与来源干预

单位m。off为相同冻结Fast，两臂全部68个窗口的off预测精确相同。

| Cohort | 训练来源 | correct | off | reference | shuffled |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

![两臂四条件](../evidence/slowfast_reader_fusion/source_conditions.png)

新VAL正确训练臂的配对收益如下：比较条件ADE减correct ADE，正值表示correct较好。均保留均值、中位数和胜/平/负窗口数，不能把大幅错配退化视作成功。

| 比较条件 | 平均收益(m) | 中位数收益(m) | 胜/平/负 |
| --- | ---: | ---: | ---: |
{paired_table}

![逐窗口收益](../evidence/slowfast_reader_fusion/paired_gains.png)

| 日志 | 窗口 | 正确训练/correct | 正确训练/off | 正确训练/reference | 正确训练/shuffled | 固定训练/reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(logs)}

航向和末端误差避免被ADE隐藏：

| Cohort | 训练/推理条件 | ADE(m) | FDE(m) | heading MAE(rad) |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(quality)}

![训练曲线](../evidence/slowfast_reader_fusion/training_curves.png)

## 正确性、恢复与边界

原Fast Head输入在全部{exp['native_Fast_head_exact_count']}个窗口精确重现原生预测。Reader先按历史batch8/final4复现{exp['historical_reader_exact_count']}个保存预测，再以batch1生成融合特征；本批Reader两种batch几何的全部保存预测也精确相同，最大差{exp['reader_b1_vs_historical_max_abs_prediction_difference']:.9g}。没有比较跨batch的中间特征；融合缓存、最终VAL及在线Reader均固定batch1。本轮B1 standalone Reader新VAL ADE={reader_b1_ade:.6f}m，残差融合尚未超过这个Reader自身的平均位置精度。固定reference特征在全部324窗口逐字节相同；目标索引/位置没有悄悄携带样本条件。原始KV manifest SHA固定为`{protocol['feature_manifest_sha256']}`，来自已认证Fast检查点，而非只信当前manifest。

一个预声明TRAIN和一个VAL样本，每臂实际执行原图Fast＋raw-KV Reader＋原Head：共{len(result['live'])}个检查，off与原Fast精确一致，on与冻结特征计算最大差0。冻结模型参数、tensor版本和参数梯度均核对未变。独立进程两臂全68 VAL×4条件重载及511→512实际训练更新重放均精确一致（参数、AdamW、scheduler、RNG、最终训练batch预测）；范围仅同H800/runtime，不是跨硬件保证。

回归测试{counts['tests']} passed，0失败/错误/跳过，XML SHA`{sha256(tests)}`。报告以NumPy float64重算指标，与原float32汇总最大差{max_metric_error:.3g}。训练耗时correct={result['arms']['correct']['elapsed_s']:.2f}s、reference={result['arms']['reference']['elapsed_s']:.2f}s；全流程含加载/原图提取/验证{result['elapsed_s']:.2f}s，peak allocated {result['peak_allocated_gib']:.3f}GiB，不是部署延迟。初次pytest默认导入模式因两个test_model同名导致收集冲突；改用--import-mode=importlib后完整通过，没有修改模型。

单seed、TRAIN256/四日志、新VAL64/两日志、legacyVAL4/一日志；这些是已观察过的诊断VAL，不是TEST。Reader已有32epoch任务读出训练，最新Fast已有自身任务适配历史；本轮只在两条新增融合臂间统一预算，不将历史Reader/Expert结果当等预算架构排名。KV包含视觉、Ego、导航、旧候选轨迹，未隔离视觉增量；fresh同步条件不支持缓存年龄、异步闭环结论。只有同时优于off、reference、shuffled和固定训练对照，且逐日志方向一致，才支持本次诊断中的来源匹配增量；广泛稳定收益仍需更多独立日志与训练种子。

本轮没有新下载或backbone训练。此固定324窗口特征缓存入口用于小样本机制验证，不可直接替换manifest用于完整navtrain。源配置、逐步训练、逐窗口预测、速度/日志分层和hash见[evidence索引](../evidence/slowfast_reader_fusion/SHA256.json)。权重、rawKV和冻结特征包仅保存在本地outputs，不提交。
'''
    document.write_text(text)
    print(json.dumps(dict(completed=True,summary={k:v['all'] for k,v in summary.items()},verification_counts=counts,files=len(hashes))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',default='evidence/slowfast_reader_fusion')
    p.add_argument('--document',default='docs/SLOWFAST_READER_FUSION.md');p.add_argument('--tests',required=True);p.add_argument('--reload',required=True)
    a=p.parse_args();run(a.run,a.output,a.document,a.tests,a.reload)


if __name__=='__main__':main()
