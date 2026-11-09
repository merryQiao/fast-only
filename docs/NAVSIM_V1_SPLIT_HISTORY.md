# NAVSIM V1：Fast 单帧／Slow 四帧训练与迁移

当前默认配置为 `configs/navsim_v1_split_history_h200_2x8.yaml`，schema 为 `navsim_split_history_v2`。目标是完整 NAVSIM V1、2 节点×8 H200；当前服务器只做有上限的原图 Smoke。此前 Fast 与 Slow 都读四帧的实验保留为历史结果，本协议需要重新训练，不能在旧检查点推理时删掉三帧就当作单帧模型。

## 输入、输出和三个训练臂

| 训练臂 | Fast 输入 | Slow 输入 | 用途 |
|---|---|---|---|
| `fast_single` | 当前 CAM_F0 一帧＋当前 Ego＋导航 | 无 | 单帧独立规划基线 |
| `fast_history` | CAM_F0 四帧＋当前 Ego＋导航 | 无 | 自己处理历史的强基线 |
| `slow_fast` | 当前 CAM_F0 一帧＋当前 Ego＋导航 | 四帧历史＋源 Ego／导航＋已完成候选轨迹 | 原六层 Direct-KV 协作 |

真实数据为 2Hz。四帧时刻固定 `[-1.5,-1.0,-0.5,0]s`，单帧只取 `0s`，在读取／解码前选择，不先算四帧再丢弃。官方场景资格仍是四历史＋十未来元数据，不改变三臂的训练/验证窗口集合；未来图像不是模型输入。未来监督仍是前八点，即当前后轴 SE(2) 中 `[0.5,1.0,…,4.0]s` 的 `[B,8,3] (x,y,heading)`。

Ego 使用当前官方 `[vx,vy,ax,ay]`，只用本次 TRAIN 的统计量归一化。导航使用官方四维 one-hot 的类别索引，不新增 Target Point，也不从未来轨迹构造导航。保持原图 1920×1080，只 pad 到 1920×1088，沿用原生 Processor／多模态位置布局，不缩放或重新压缩。此尺寸每帧 2,040 个视觉 token，四帧 8,160；这不是端到端加速倍数。

Fast 保持完整 0.8B、八个 Planning Queries、原有唯一轨迹 Head。对应六层注意力读取 Slow 2B 的 Direct-KV；只有 Planning Queries 可见外部列，当前视觉/Ego/导航 Context 不读外部列。每轮原生语言 KV/GDN 状态重新开始。没有 Reader、额外轨迹 Head、Future 损失、自适应刷新或动作匹配。

## 从原始资产初始化；不需要 source_bundle.pt

两个模型都直接加载本地原始预训练资产，新增模块按明确种子初始化。Fast 使用训练种子 42／43，Slow 来源种子固定 142。新协议没有旧 Mini 检查点、外部 source bundle 或冻结反馈教师依赖。

| 参数范围 | 本协议训练方式 |
|---|---|
| Fast 视觉塔＋Merger | 冻结 |
| Fast 全部语言基础参数、已有 LoRA、Ego/导航/Queries/轨迹 Head | 更新 |
| Slow 视觉塔、Merger、语言基础参数及其 LoRA | 冻结 |
| Slow 原有 Ego/导航/反馈编码器、反馈/缺失标记、Planning Queries | 更新，当前实现 1,084,160 参数 |
| Slow 原有轨迹 Head | 冻结、不作为训练监督或输出决策 |

**“冻结 Slow”在这里准确指冻结预训练主干，不是冻结随机初始化的条件模块。** Slow 新增模块通过 Fast 的规划损失学习。虽然 Slow 主干参数冻结，训练时仍须允许梯度穿过其运算；不能把整个 Slow Prefill 放在 `no_grad` 内。没有额外 Slow 轨迹损失。

训练每个合作样本的顺序是：一次 Fast 当前图视觉计算 → 同一个 Fast 在 `no_grad` 下生成同观测、无 KV 的候选轨迹并 detach → Slow 对该观测四帧完整 Prefill → Fast 使用 live KV 输出最终轨迹。候选和最终 Fast 前向共用已计算的冻结视觉特征。KV dropout=0.3、反馈 dropout=0.1 只在 TRAIN 使用；验证关闭随机丢弃。缺缓存时仍监督真实轨迹。

Slow 新增参数及原生 buffers 随合作检查点保存/恢复；独立推理进程从同一合作检查点加载它们。这里的检查点内状态不是以前必须手工转移的旧 `source_bundle.pt`。

## 完整数据与统计量

固定 NAVSIM V1 官方代码身份 `3e8291bfa89ff247231e0227778840cd0a036896`，官方 Scene Filter／token 列表和默认日志角色随仓库保存。使用完整官方 TRAIN 978 日志、VAL 214 日志；新模型没有旧 Mini 的任务训练暴露，不沿用旧来源配方隔离的六条 VAL 日志。

```bash
python -m tools.prepare_cluster_data official \
  --metadata-root /datasets/openscene/trainval_logs \
  --sensor-root /datasets/openscene/sensor_blobs/trainval \
  --output /datasets/prepared/navsim_v1_split_history \
  --official-default-val
```

工具只读已有本地数据，不下载数据集。保留所有官方资格检查、token/log/group 覆盖、图像身份、时间戳拒绝原因与比例；官方 103,288 个允许 token 横跨 TRAIN/VAL，不能全部计为 TRAIN。实际有效窗口数由扫描报告确定。时间间隔规则为 500000±1000µs，默认项目有效性拒绝率上限为零；先查看 `audit.json`，不能为了准备通过而静默放宽或丢弃大量场景。缺官方日志／token 直接失败。

准备结果包含 TRAIN/VAL Manifest、TRAIN-only Stats 及身份记录。Fast 和新 Slow 都使用本次完整 TRAIN 的统计量；不复用 Mini Stats，不使用 VAL/TEST 计算统计量。新协议验证 `val.jsonl` 与完整 `val_official.jsonl` 的字节身份一致，拒绝把旧 208 日志选择集静默用于此实验。准备目录保留相对传感器路径，更换 `sensor_root` 而图像字节不变可以迁移。

原图 I/O 仍可能是瓶颈。优先使用节点本地 NVMe 或足够吞吐的共享存储；默认每 rank 四个 workers、prefetch=2。三臂训练无持久视觉特征缓存、无全量 KV 导出。合作训练在线计算候选轨迹和 Slow Prefill，两者均消耗训练资源，不应只比较 optimizer step 数而省略计算量。

## 两节点环境、短训与启动

按仓库根 README 的已测版本安装 PyTorch/CUDA、Transformers、PEFT、FLA 和 TileLang；CUDA 扩展需匹配目标环境。单 H800 证据不能代替 H200/NCCL/节点网络预检。

两个节点需能访问相同内容的数据、基座与共享输出目录；设置 `NODE_RANK` 为 0／1，其余如下：

```bash
export FAST_BASE_MODEL=/models/Qwen3.5-0.8B
export SLOW_BASE_MODEL=/models/Qwen3.5-2B
export NAVSIM_MANIFEST_DIR=/datasets/prepared/navsim_v1_split_history
export NAVSIM_SENSOR_ROOT=/datasets/openscene/sensor_blobs/trainval
export TRAIN_OUTPUT_ROOT=/runs/navsim_v1_split_history
export MASTER_ADDR=<node0地址>
export MASTER_PORT=29500
export NODE_RANK=0
export CLUSTER_PYTHON=/path/to/environment/bin/python
export CLUSTER_CONFIG=configs/navsim_v1_split_history_h200_2x8.yaml
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLA_TILELANG=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

先做身份预检，无旧任务权重依赖：

```bash
"$CLUSTER_PYTHON" -m tools.preflight_cluster --config "$CLUSTER_CONFIG" \
  --output /runs/navsim_v1_split_history/preflight_identity.json
```

然后在两个节点同时做正式拓扑、正式原图和 batch 的短训。例为合作臂；两个 Fast-only 臂也应各做一次，输出目录分别使用新名字：

```bash
"$CLUSTER_PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m fast_cluster.train --config "$CLUSTER_CONFIG" --arm slow_fast --seed 42 \
  --output /runs/navsim_v1_split_history/preflight_slow_fast --preflight-updates 2
```

用相同命令增加 `--resume /runs/navsim_v1_split_history/preflight_slow_fast/latest.pt`，将 `--preflight-updates` 改为 3，核对实际恢复、梯度、OOM、吞吐、rank 日志及冻结参数。短训保留完整八 epoch 日程，部分 VAL 仅作检查、不参与模型选择。正式配置强制 world_size=16、每节点八进程；不要为在单卡运行而伪装其拓扑。

预检通过后，在两个节点分别启动：

```bash
bash scripts/train_navsim_v1_2x8.sh
```

脚本默认已指向新配置，依次运行种子 42／43 的三臂，共六次训练，并非在 16 卡上同时运行六组。训练每 GPU MB1、累积 2，有效全局 batch=32；各八 epoch、无放回 epoch shuffle，同种子三臂使用相同初始 Fast 参数和样本顺序。更新数由实际 TRAIN 长度确定，最后不足批次按真实有效样本归一化。语言 lr=1e-5、新模块 lr=5e-5、weight decay=0.01；现有官方不加权 pose L1 不变。

中断后用 `RESUME_MATRIX=1 bash scripts/train_navsim_v1_2x8.sh`。完整状态包括模型、Slow 新增模块、buffers、optimizer、scheduler、各 rank RNG 和采样游标；相同 world_size/batch/预算才能恢复。模型/数据身份、帧数或协议变化必须新开输出目录。目标 GPU 或 CUDA/FLA 变化时，不承诺跨硬件逐位一致。旧协议的已完成结果也不能被脚本当作新协议跳过。

每 epoch 按完整 VAL ADE 选择 best；报告 FDE、各时域 L2、heading MAE 和 windows-seen／等效 epoch。合作臂终点还在同一权重上评估正确与固定 TRAIN 来源；固定来源为 TRAIN token 字典序第一项，不新增训练臂。不使用 navtest/navhard 选择模型或调参。训练后再接官方 Agent／评分，当前运行器不是闭环评分器。

## 同一检查点的同步与固定周期异步

四帧 Fast-only 基线允许复用历史 RGB 及冻结 ViT＋Merger 特征。缓存为有界八帧 LRU，按文件身份、基座／视觉 buffers／处理策略区分；缓存开关使用相同的逐图原生视觉计算。四帧 Processor 和完整语言历史仍处理，未实现语言状态缓存。不能把图像数从四减一直接报成四倍加速。

新服务器可从准备好的 Manifest 导出 input-only 连续片段，不依赖原服务器 archive 或 `outputs/`：

```bash
python -m tools.export_cluster_fragment \
  --bundle "$NAVSIM_MANIFEST_DIR" --sensor-root "$NAVSIM_SENSOR_ROOT" \
  --split val --windows 64 --output /runs/runtime_fragment
```

该工具只引用本地原图，不复制、压缩或读取未来标注到运行片段。它要求同日志的真实连续 2Hz 节点；官方筛选后找不到足够长的连续片段会显式失败，不能插值图像补齐。片段中的 sensors 链接指向该服务器传感器根目录，路径变化后重新导出。片段只用于运行协议/耗时检查，不冒充完整 VAL 或闭环质量评估。

```bash
# 四帧 Fast-only：保留历史图像/视觉特征缓存。
python -m tools.benchmark_split_history --config "$CLUSTER_CONFIG" \
  --checkpoint /runs/navsim_v1_split_history/seed_42/fast_history/best_weights.pt \
  --fragment /runs/runtime_fragment/fragment.json --output /runs/bench_fast_history \
  --mode fast_only --ticks 12 --fast-device cuda:0

# 同步 fresh 参考与异步使用同一份 slow_fast 权重。
python -m tools.benchmark_split_history --config "$CLUSTER_CONFIG" \
  --checkpoint /runs/navsim_v1_split_history/seed_42/slow_fast/best_weights.pt \
  --fragment /runs/runtime_fragment/fragment.json --output /runs/bench_sync \
  --mode sync_fresh --ticks 12 --fast-device cuda:0 --slow-device cuda:1

python -m tools.run_cluster_async --config "$CLUSTER_CONFIG" \
  --checkpoint /runs/navsim_v1_split_history/seed_42/slow_fast/best_weights.pt \
  --fragment /runs/runtime_fragment/fragment.json --output /runs/bench_async \
  --fast-device cuda:0 --slow-device cuda:1 --ticks 16 --refresh-period-s 1 \
  --transport-root /local_nvme/fast_slow_transport
```

推理是同一主机两个独立 GPU 的 spawn 进程；训练支持两节点 DDP，但当前推理运输不是跨节点 RPC。Slow 完整快照先经 CPU mmap，再完整传到 Fast 设备并同步，之后记录 ready time、发布不可变版本；六层在一次 Fast 前向中固定同一版本。默认使用最新可用观测，记录丢帧，不阻塞等旧图。遵守 `source_time≤image_time`、`ready_time≤selection_time`，并检查 episode、导航版本及模型身份。约 102MB 的实际快照不能假设能放进容器常见 64MiB `/dev/shm`。

训练/同步参考的反馈是**同观测 no-KV Fast 候选**；异步 Slow 使用**最近已经完成的主 Fast 轨迹，可能带 KV**，并锁定它对应的原观测和参考时间，不能把旧轨迹冒记为较新观测的未来轨迹。启动时无反馈使用缺失标记。两种反馈分布存在差异，当前没有完成针对实际陈旧状态、持续失效或动作不相容的训练与质量验证。

记录 decode、Processor、vision、language＋Head、候选计算、Slow Prefill、CPU/设备传输、快照大小、源到提交延迟、实际规划频率、跳帧和 busy skips。Slow 的墙钟时间包含完整输入准备与 Prefill；CUDA event 只给 stream 时间跨度，不等于 GPU 活跃周期。异步反馈不额外在线运行一个 no-KV Fast，因此不能把同步路径的两次 Fast 语言前向假定为异步开销。

## 本机 Smoke 与交付边界

完整记录见 [结构化证据](../reports/navsim_v1_split_history_smoke.json)。本机没有下载完整数据集、运行全量训练或继续做旧分支机制消融。

这次使用已有 8 TRAIN／4 VAL fixture，单 H800、MB1／累积2；Smoke 配置声明一 epoch、四次更新，并按显式上限提前停止。正式八 epoch 和完整数据规模未在本机执行。

| 训练臂 | 连续 Smoke 更新数 | 峰值 allocated |
|---|---|---|
| Fast 单帧 | 1 | 14.41 GiB |
| Fast 四帧 | 1 | 29.35 GiB |
| Fast 单帧＋Slow 四帧 | 2 | 35.78 GiB |

合作恢复另以两个独立进程各运行一步，与连续两步对齐。CPU 回归最终为 **1077 passed、1 skipped、0 failed**；跳过项为 CPU 环境隐藏 CUDA 的启动检查，原生 GPU 更新另行执行。以上显存是本机测量，不替代目标机容量与吞吐预检。

- 三臂都使用原生 0.8B／2B 和原图执行真实梯度更新；合作臂 Fast 24 层及 Slow 新增模块梯度有限非零，冻结视觉/Slow 主干未改变。
- 合作模型实际中断后在新进程恢复，第二步完整模型、Slow 状态、optimizer、scheduler、RNG、采样进度和独立重载预测逐位一致，预测最大差异 0。
- CPU 两进程 Gloo 覆盖跨 `no_sync` 的不同 rank KV 丢弃以及全体无 KV 更新，与集中式全局有效样本参考一致。这不是 16 卡 NCCL 证明。
- 同一合作检查点的同步与异步真实运行已完成，异步消费多个已完整发布的版本；四帧 Fast 缓存观察到历史特征命中，推理冻结身份保持不变。

这些验收支持迁移后进行正式环境短训。当前没有充分收敛、动态 KV 增量、缓存年龄鲁棒性、严格 2Hz 实时性、加速倍数或闭环任务成绩。三臂及相同权重下的调度比较，正是接下来需要用完整数据回答的问题。
