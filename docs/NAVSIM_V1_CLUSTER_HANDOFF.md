# NAVSIM V1：2节点×8 H200训练交接

此文档保留旧四帧 Fast＋四帧 Slow、冻结来源双臂配方。**当前默认是 [Fast 单帧／Slow 四帧三臂协议](NAVSIM_V1_SPLIT_HISTORY.md)**，两模型从原始预训练资产开始，不需要 `source_bundle.pt` 或旧 Mini 权重。独立的原始四帧 Fast-only 控制入口见[此页](NAVSIM_V1_ORIGINAL_BASELINE.md)。不要为启动新协议而复制旧来源参数。

当前任务是正式训练工程与有界Smoke。训练入口是 `python -m fast_cluster.train`，配置为 `configs/navsim_v1_h200_2x8.yaml`。旧 `configs/navsim.yaml` 和各256/512窗口诊断脚本保持原实验身份，不作为本轮正式入口。

## 固定的训练协议

| 项目 | 当前实现 |
|---|---|
| 数据 | NAVSIM V1.1官方navtrain筛选与默认TRAIN/VAL日志角色 |
| 官方代码身份 | `3e8291bfa89ff247231e0227778840cd0a036896`，配置随仓库固定 |
| 划分 | 官方基础划分978 TRAIN/214 VAL；现有冻结来源已见的1组6日志从选择VAL隔离，选择VAL为208日志／60组 |
| 场景资格 | 4历史＋10未来、间隔1、当前有route；只监督前8未来点 |
| Fast输入 | 原始1920×1080前视四帧，2Hz；右/下padding到原生patch倍数32，不缩放或重压缩 |
| 导航/Ego | 官方4类命令的one-hot取argmax后Embedding；`[vx,vy,ax,ay]`；不虚构Target Point |
| 输出 | 当前rear-axle SE(2)下未来4秒的8×3 `(x,y,heading)` |
| 初始化 | Fast从Qwen3.5-0.8B预训练基座重新初始化原有规划模块；不用Smoke权重正式开训 |
| 参数 | Fast完整24层语言基础权重、已有LoRA、输入/Queries/唯一Head更新；视觉含Merger冻结；Slow 2B冻结 |
| 通信 | 原六层Direct-KV；同步fresh源；不新增Reader、Head或门控 |
| 两臂 | `fast_only`、`slow_fast`；配对种子42/43，同初始化和样本顺序 |
| 预算 | 各8 epoch；每GPU MB1、累积2；16 GPU有效全局batch32 |
| 优化 | 语言1e-5、新模块5e-5；官方不加权pose L1；warmup/cosine按完整8epoch计算 |
| 选择 | 每epoch独立VAL ADE选best；终点附正确/固定TRAIN来源评测，无额外训练臂 |

官方navtrain允许103,288个token，横跨两种日志角色。**完整有效TRAIN数量由真实元数据构建后确定**，不能把103,288全部当TRAIN。每epoch更新数为 `ceil(ceil(N_train/16)/2)`，全预算为其8倍。固定8000步不再充当正式训练终点。

需要明确一项来源相关的修正：旧冻结Slow与反馈教师训练过记录组 `2021.06.08.12.54.54_veh-26`，它在官方V1中属于VAL。仅让新Fast按官方划分训练仍然会让上游模型接触过选择数据。当前保持全部978 TRAIN和现有冻结权重，按 `fast_cluster/source_provenance.json` 排除该VAL记录组的6条日志；完整官方VAL保留为 `val_official.jsonl`，标明上游已见污染，不能用于选择或冒充完全留出的泛化成绩。隔离是明确记录的项目规则，不能把208日志结果写成完整官方VAL成绩。没有重划TRAIN、重新训练Slow或静默换来源。

## 先迁移什么

迁移仓库、两个完整基座目录、`source_bundle.pt`及其JSON收据。source bundle只有约79MB的冻结Slow任务参数、原冻结反馈教师任务参数、原统计量和基座资产SHA；基座权重不重复打包。

本地导出位置：`outputs/slowfast_cluster/source/source_bundle.pt`。其源检查点SHA为 `448571f4d543913a4d93471781a1ecf4aa392078f2aa2dafa3816be43f9f0ae9`；bundle SHA为 `cd16b8aa6ea907814b1906f9dbb6f9ad3a133eeec13393ad1eab67c59d29e11e`。源模型来自旧Mini适配，不应改称V1全量训练的Slow。Git不包含模型、图像、KV或PT检查点。

如需重新导出，原服务器运行：

```bash
python -m tools.export_cluster_source \
  --source-run outputs/slowfast_v11_expanded/run_01 \
  --output outputs/slowfast_cluster/source/source_bundle.pt \
  --expected-sha256 448571f4d543913a4d93471781a1ecf4aa392078f2aa2dafa3816be43f9f0ae9
```

重新序列化PT文件可能改变文件SHA；保留并使用同一次导出文件/收据。训练恢复同时校验内容身份与完整模型状态，不能在已有训练途中替换bundle。

## 准备完整官方V1数据

目标机需另行定位NAVSIM/OpenScene trainval元数据和对应传感器数据。Bench2Drive不等于NAVSIM。只用navtrain提供的历史图像也可以；未来10帧要求的是元数据，不要求下载未来图像。准备工具不执行网络下载。

```bash
python -m tools.prepare_cluster_data official \
  --metadata-root /datasets/openscene/trainval_logs \
  --sensor-root /datasets/openscene/sensor_blobs/trainval \
  --output /datasets/prepared/navsim_v1 \
  --source-provenance fast_cluster/source_provenance.json
```

工具扫描所有1192条官方日志/token，输出相对路径JSONL、TRAIN统计量、图像/元数据身份、日志/记录组清单和覆盖审计。来源污染隔离另记日志/组/token数量，不计作损坏窗口拒绝。缺日志、缺官方token直接失败。默认拒绝率上限为0；时间间隔项目策略是500000±1000µs。先检查 `audit.json` 的原因与比例，再基于证据显式调整 `--max-rejection-fraction`，不要为通过准备而随意放宽。若出现系统性频率变化，应重新对齐协议。

新Fast用完整TRAIN统计量；冻结Slow和反馈教师分别从同一raw Ego计算其原TRAIN256归一化。统计量不跨角色混用，VAL/TEST不参与统计。JPEG身份在准备时记录，训练实际读取图像时校验字节，迁移路径改变而字节不变可以恢复。

原图随机读和JPEG解码仍有成本。优先将图像放节点本地NVMe或足够吞吐的共享存储；起点每rank4 workers/prefetch2。默认每batch在线冻结教师＋Slow完整Prefill，**不要求生成全量多TB KV缓存**。目标机记录吞吐后再调整I/O参数。修改workers不改变恢复语义，修改batch/累积/预算会被拒绝恢复。

## 环境、目标机短训与正式启动

已验证依赖见 `requirements-cluster.txt`、`requirements-hopper.txt`；本机为Torch2.8.0+cu128、Transformers5.10.2、FLA0.5.1、TileLang0.1.15。CUDA编译器/FLA依赖需在目标环境安装，不能只复制venv。不要让pip意外替换既有CUDA栈。

两个节点使用相同源码、环境、模型和数据内容。输出放共享可写目录。正式配置所有路径通过环境变量提供：

```bash
export FAST_BASE_MODEL=/models/Qwen3.5-0.8B
export SLOW_BASE_MODEL=/models/Qwen3.5-2B
export SLOW_SOURCE_BUNDLE=/models/frozen_source/source_bundle.pt
export NAVSIM_MANIFEST_DIR=/datasets/prepared/navsim_v1
export NAVSIM_SENSOR_ROOT=/datasets/openscene/sensor_blobs/trainval
export TRAIN_OUTPUT_ROOT=/runs/navsim_v1
export MASTER_ADDR=<node0的可互通地址>
export MASTER_PORT=29500
export NODE_RANK=0 # 第二节点为1
export CLUSTER_PYTHON=/path/to/environment/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLA_TILELANG=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

先在目标环境核对身份：

```bash
python -m tools.preflight_cluster --config configs/navsim_v1_h200_2x8.yaml \
  --output /runs/navsim_v1/preflight_identity.json
```

然后在两个节点执行真实16-rank短训，以下命令以相同参数各启动一次：

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m fast_cluster.train --config configs/navsim_v1_h200_2x8.yaml \
  --arm slow_fast --seed 42 --output /runs/navsim_v1/target_preflight \
  --preflight-updates 2
```

使用相同命令加 `--resume /runs/navsim_v1/target_preflight/latest.pt` 并将预检上限设为3，检查真实NCCL/IB、多卡显存、DDP同步、部分VAL前向、checkpoint与恢复。预检沿用完整8epoch学习率日程，结果标为未完成；它不是正式成绩。若仅普通python启动正式配置，会明确拒绝world size不符。

目标预检通过后，两个节点分别设置NODE_RANK，再执行：

```bash
CLUSTER_CONFIG=configs/navsim_v1_h200_2x8.yaml bash scripts/train_navsim_v1_2x8.sh
```

它依次运行两种子×两臂，每臂使用全部16卡。首次运行用新目录；中断后显式 `RESUME_MATRIX=1` 重启。单个臂也可按上述torchrun格式去掉预检上限，用 `--resume .../latest.pt` 恢复。默认保留最近2份完整checkpoint与best权重，optimizer完整状态较大，预留输出空间。

DDP采用每epoch无放回置换，仅补zero-weight invalid slots保证各rank步数相同。每累积组按全局真实有效样本数求均值；尾组不重复计样本。日志包含windows_seen、equivalent_epochs、梯度、学习率、吞吐、显存和VAL。恢复存储每rank Python/NumPy/CPU及本rank CUDA RNG、消费游标、optimizer/scheduler和非持久buffer；只允许相同拓扑和语义配置恢复。跨GPU/CUDA/FLA版本不保证逐位一致。

## 最小异步推理

`python -m tools.run_cluster_async` 让Fast主进程和Slow spawn子进程分别持有模型；当前支持同节点独立GPU，不是跨节点RPC。后台固定周期完整Prefill，一次最多一个在途任务。快照完整传输至Fast设备并同步后记录ready并发布；Fast完成当前视觉处理后才选择版本，整轮固定一个版本。无合法缓存走原无KV路径。

完整六层快照约100MB，不能依赖容器常见的64MiB `/dev/shm`。传输使用CPU文件映射与描述符，子进程收到映射确认后释放临时文件；不把CUDA张量跨进程共享。目标机通过 `--transport-root /local_nvme/kv_transport` 指定节点本地临时盘，并将实测文件写入/读取纳入延迟。它与训练checkpoint和永久KV缓存无关。

推理反馈锁定最新已完成Fast计划**及它对应的同一观测快照**，Slow源时间、坐标和规划点时间因此一致；没有可用完成计划时使用当前观测和明确缺失标记。不把旧计划简单变换坐标后冒充新观测的未来0.5–4秒计划。训练反馈是原冻结教师；两种策略的差异保留在报告中。当前只训练fresh KV；本轮异步Smoke只验证工程协议，尚无陈旧KV/延迟闭环可靠性结论。

```bash
python -m tools.run_cluster_async \
  --config configs/navsim_v1_h200_2x8.yaml \
  --checkpoint /runs/navsim_v1/seed_42/slow_fast/best_weights.pt \
  --fragment /fixtures/continuous_2hz/fragment.json \
  --fast-device cuda:0 --slow-device cuda:1 \
  --ticks 16 --refresh-period-s 1 --output /runs/async_check
```

fragment只包含历史/当前图像、raw Ego、当前位姿、导航和时间，禁止GT/未来字段。本地Smoke片段由已保存压缩前缀提取64个连续2Hz窗口、67张原图，没有新增数据下载。报告记录t/c/u/source/ready、版本消费、源到提交延迟、图像处理/视觉/语言Head/传输耗时、刷新忙跳过和丢帧。单卡colocated模式必须显式开启，不能作为两GPU延迟成绩。

## 当前验收边界

当前服务器只有一张H800。两rank Gloo验证分布式算法；H800验证原图真实模型更新、冻结、恢复与异步进程。**2×8 H200、NCCL/IB、完整V1数据规模吞吐仍须目标机首次预检。** 本轮不下载完整数据、不启动正式8epoch、不提供新增模型质量结论，也没有完成NAVSIM Agent/闭环评分接口。

当前数值、测试和可迁移文件清单见 `docs/NAVSIM_V1_CLUSTER_SMOKE.md`。官方依据：[V1 splits](https://github.com/autonomousvision/navsim/blob/3e8291bfa89ff247231e0227778840cd0a036896/docs/splits.md)、[navtrain filter](https://github.com/autonomousvision/navsim/blob/3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml)、[默认日志划分](https://github.com/autonomousvision/navsim/blob/3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/training/default_train_val_test_log_split.yaml)。
