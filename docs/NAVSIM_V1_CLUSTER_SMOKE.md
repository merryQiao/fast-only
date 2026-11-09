# 集群迁移代码与本机Smoke验收

日期：2026-11-09。实现位于独立 `fast_cluster/`，基于已固定的原六层Direct-KV与唯一轨迹Head。**本机Smoke已通过；16张H200和全量数据的实际运行仍以目标机预检为准。** 迁移步骤见 [交接说明](NAVSIM_V1_CLUSTER_HANDOFF.md)。

## 已执行的真实检查

| 检查 | 本机结果 |
|---|---|
| 环境 | 单H800 80GB；Torch2.8.0+cu128 / Transformers5.10.2 / FLA0.5.1 / TileLang0.1.15 |
| 最终回归 | 1014通过／1项CUDA可见性跳过；集群专项55项全部通过 |
| 数据 | 本地8 TRAIN / 4 VAL；48张1920×1080原JPEG，四帧2Hz，读取时SHA校验 |
| 原图输入 | 原生Processor，pad32、不resize、不重压缩；视觉RoPE保持原生FP32 buffer |
| 两臂实际训练 | Fast-only和Slow＋Fast各4次optimizer更新；MB1、累积2、各看8窗口 |
| 配对初始化 | Fast参数摘要相同：`06fb74efb39411ae98069b300ac9bad4760c36f60a5e7294fb0e8f5f64dedd47`；数据身份相同 |
| 梯度 | Joint的24个语言层Dense基础参数、Embedding、最终Norm均有限非零；视觉冻结 |
| 冻结 | Joint训练前后视觉、Slow、反馈教师参数及buffer摘要一致 |
| 来源导出前向 | 一题原生四帧输入：导出来源的六层K/V、位置、valid mask与已认证旧缓存逐位一致，最大差0 |
| 显存 | Fast-only峰值allocated约35.00GiB；Joint约41.92GiB；仅本机MB1结果 |
| 实际中断恢复 | 第2步写磁盘退出，另起进程恢复到第4步，与连续4步模型／optimizer／scheduler／各RNG／游标／buffer逐位一致 |
| 文件与重载 | 两个最终PT SHA均为`9cd4b31be809be9f4f60e3655d156a0a9788b5f3f154e818182b7ceeb238442c`；独立进程重载预测最大差0 |
| 2-rank Gloo | CPU真实进程：13有效样本，padding／尾组全局均值正确，参数同步、逐rank RNG恢复、单写者与保留2份checkpoint通过 |
| 异步真实模型 | 独立Slow子进程＋Fast主进程，同一H800显式colocated模式；12次规划，11次读取缓存，消费版本1/3/4/5/6 |
| 快照传输 | 最大101,643,815字节；64MiB shm环境使用文件映射，完整到Fast设备后发布；使用版本均满足s≤t、r≤c |
| 生命周期 | Fast/Slow参数与buffer不变；worker退出码0，正常关闭、传输临时文件清理通过 |

本轮一共12次optimizer更新（两臂各4＋恢复对照4），没有训练Slow、Reader或新增模型分支。恢复对照不是将内存中状态重新装回同一进程。

小样本末端VAL ADE：Fast-only 1.246303m，Joint fresh 1.402931m，Joint固定来源1.407859m。这里只训练4步且是新的8/4窗口工程fixture，**这些数值只证明评测链路可运行，不支持模型质量、快慢增益或收敛结论**，也不能与此前256/512窗口实验直接比较。正式代码还报告8个时域L2及wrapped Heading MAE。

异步源时间按真实2Hz时间戳回放。该次实测释放频率约2.020Hz、有效规划约0.609Hz，源到提交延迟P50/P95约1.485/5.027秒；视觉＋语言Head计算时间P50/P95约0.267/1.344秒。数据包含启动后首次计算开销，使用同卡及共享盘传输；不是H200、多GPU或严格2Hz实时性能成绩。此轮没有闭环评价，也没有建立陈旧KV的规划收益。

## 全量V1前发现并修复的来源泄漏

官方基础TRAIN/VAL是978/214日志（101/61记录组）。现有冻结来源的旧TRAIN256训练清单SHA是 `5e956ada251fb32c98206a38bb706e77b7f916a15ff7ed9d0f951ae90f1dfb04`；其中 `2021.06.08.12.54.54_veh-26` 在官方V1 VAL中包含6日志。两个独立审核都从原文件与官方固定配置确认了交叉。

当前全量准备仍扫描所有1192日志和103,288允许token，TRAIN978保持全部有效数据。选checkpoint的VAL排除上述整个已见记录组，成为208日志／60组；完整官方VAL作为 `val_official.jsonl` 保留，标明污染范围、只供显式报告。登记文件是 `fast_cluster/source_provenance.json`，与source bundle内嵌上游TRAIN hash核对。正式入口在加载模型前拒绝没有登记或未隔离的数据包。没有修改冻结来源、暗中重划TRAIN或把该规则称为官方默认。

## 可复查的证据

完整本地证据位于 `outputs/slowfast_cluster/`（Git忽略大文件）：

- `smoke_data/identity.json`：身份`ebf319a4e58cdb570ac1d01e26704ee79b06c8825b628ef9d26eaaed520d5e11`。
- `source/source_bundle.{pt,json}`：约79MB冻结任务状态；bundle SHA `cd16b8aa6ea907814b1906f9dbb6f9ad3a133eeec13393ad1eab67c59d29e11e`。
- `native_fast_01/`、`native_joint_01/`、`native_resume_01/`：训练日志、完整checkpoint、数据／模型身份与冻结审计。
- `native_resume_verification.json`：独立状态及预测重载验证。
- `source_forward_parity.json`：导出来源原生前向与原冻结来源的逐层数值一致性。
- `gloo_01/receipt.json`：实际2-rank算法回归。
- `native_async_01/result.json`、`async_causal_audit.json`：真实异步版本／时间事件与独立因果检查。
- `regression_final.xml`：最终完整回归；`final_cluster_tests.xml`：集群专项测试。

紧凑数值摘要随源码提交于 `reports/navsim_v1_cluster_smoke.json`。实际PT、图像和基座权重不上传Git。

## 重跑与打包

本地Smoke配置为 `configs/navsim_v1_smoke.yaml`。在具备迁移fixture和两个基座路径后，使用新的输出目录：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLA_TILELANG=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m fast_cluster.train --config configs/navsim_v1_smoke.yaml \
  --arm fast_only --seed 42 --output outputs/new_smoke/fast
python -m fast_cluster.train --config configs/navsim_v1_smoke.yaml \
  --arm slow_fast --seed 42 --output outputs/new_smoke/joint
python -m fast_cluster.train --config configs/navsim_v1_smoke.yaml \
  --arm slow_fast --seed 42 --output outputs/new_smoke/resume --stop-after-updates 2
python -m fast_cluster.train --config configs/navsim_v1_smoke.yaml \
  --arm slow_fast --seed 42 --output outputs/new_smoke/resume \
  --resume outputs/new_smoke/resume/latest.pt
python -m tools.verify_cluster_resume --config configs/navsim_v1_smoke.yaml \
  --uninterrupted outputs/new_smoke/joint/latest.pt \
  --resumed outputs/new_smoke/resume/latest.pt --output outputs/new_smoke/reload.json
```

提交全部源码后可生成携带冻结来源和小型原图fixture的迁移包（不包含两个基座或全量数据）：

```bash
python -m tools.package_cluster_handoff \
  --output outputs/slowfast_cluster/handoff/navsim_v1_cluster.tar.gz
```

解压后进入 `source/`，为 `models/Qwen3.5-0.8B` 和 `models/Qwen3.5-2B` 建立指向目标机基座的链接；Mini fixture可以直接重跑Smoke。正式训练则按交接文档导出环境变量、生成全量数据包并先执行16-rank预检。包内 `HANDOFF.json` 给出实际源码commit和每个外部复制文件的SHA。

当前不需要继续在旧服务器加训；后续重点是目标机完整数据准备、16-rank预检和正式两臂训练。
