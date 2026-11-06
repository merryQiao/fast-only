# Slow-Fast V1.1：P0 接口原型与后续交接

本分支在已验收的 Fast-only 基线上增加可执行的 Slow→Fast 六层 KV 接口。**当前完成的是 P0 与部分 P1 的接口、梯度验证；不是完整 V1.1 系统，也没有获得 H1–H4 的实验结论。** `fast_navsim/`、`fast_only/` 及原基线配置没有修改。

## 数据与模型通道

继续复用现有 NAVSIM mini TRAIN，不下载新数据。两个模型均输入 CAM_F0 的四张历史图像，时刻为 `[-1.5,-1.0,-0.5,0]s`，以及当前官方 Ego `[vx,vy,ax,ay]`、四类导航 command 索引。Ego 沿用 TRAIN 统计量。图像仍为 1920×1080，只在底部补 8 像素，原生 Processor 生成每帧 2,040 visual tokens。

```text
Fast 完成的候选轨迹（不使用专家 GT 代替）
  + 同一个观测快照 / Ego / command
  → Slow 原生 2B 完整 prefill
  → 六层 KNorm 后、RoPE 前的 K，以及未旋转 V
  → Fast 当前原生视觉 Context + Ego + command + 缓存元数据 + 8 个 Queries
  → 六层中仅这 8 个 Queries 可以读取外部 KV
  → 唯一 Fast 轨迹 Head → [B,8,3]
```

Slow 另外输出自身 `[B,8,3]` 轨迹，用于后续辅助规划监督。两个轨迹均为未来 `[0.5,1.0,…,4.0]s`、相对当前后轴的 `(x,y,heading)`；x 前、y 左。NAVSIM 没有连续 Target Point 输入。

Slow 反馈使用八个数值 token：每点编码 `[目标相对时间,x,y,sin(heading),cos(heading)]`，加入反馈类型 embedding。无反馈时每点保留时间，使用明确的缺失 embedding，不转换为自然语言数字。当前低层 API 要求调用者提供与 Slow 观测同源、同参考位姿的已完成计划；尚未实现自动收集/重对齐异步反馈的调度器。

## KV 与位置协议

| 项目 | 当前实现 |
| --- | --- |
| 模型 | 原生完整 Qwen3.5-0.8B / Qwen3.5-2B；视觉塔及 Merger 冻结 |
| 对应层 | 一基编号 4、8、12、16、20、24 |
| KV 形状 | 每层 `[B,2,T_s,256]`，两模型维度一致 |
| K 捕获 | `k_norm` 输出，RoPE 之前；不用 native 的 post-RoPE cache 冒充 |
| V 捕获 | 原生 `v_proj` 输出，按 KV head 整形 |
| Slow 位置 | 保存完整原生三轴 mRoPE 布局及 padding mask |
| Fast 位置 | 所有本分支 Fast 调用均固定偏移 4096，包括无 KV / 屏蔽路径 |
| 越界处理 | Slow 有效位置达到 4096 则报错；不裁剪、不静默重排 |
| 元数据 token | `[age_s,dx_m,dy_m,sin(dyaw),cos(dyaw),enabled]`，原始 SI 单位 |
| 相对位姿 | `P_current^-1 P_source`，源自车在当前自车坐标系中的 SE(2) |
| 无缓存 | absent 和 disabled 均使用相同的可学习缺失 token |
| Fast suffix | Ego、command、metadata、8 queries；总序列 8,248 |
| Slow suffix | Ego、command、8 feedback tokens、8 queries；总序列 8,255 |

4096 是明确的原型位置约定，位置旋转只负责 token 布局，不补偿车辆物理运动。新增 metadata token、位置约定和未训练的模块使它成为需要重新适配的模型变体；不能把它的无 KV 输出当成原 Fast-only 检查点的逐位复现。

`attention.py` 保留原生完整 Fast Attention，随后在原生 `o_proj` 之前替换最后八行：对 `[K_s;K_f] / [V_s;V_f]` 重新执行 planning-row SDPA，保留原生 QNorm、KNorm、部分 RoPE、GQA 和一次 sigmoid gate，最后只执行一次原生输出投影。Context 行始终保留原生结果。外部不可见列通过布尔 Attention Mask 屏蔽；disabled 样本直接保留原生 planning 行。

这避免构造整个长序列的稠密 block mask，但会多算一次 planning attention，**没有声称已优化内核或按 query 数量得到加速**。hook 版本不支持 gradient checkpointing、持久 native cache、并发/重入同一模型；这些需要显式拒绝。每次完整前向重新初始化原生 KV / GDN 状态。

## 时间与训练计算图

`protocol.py` 提供基础 admission：`source_time <= image_time`、`ready_time <= selection_time`，加 episode/navigation/model identity 一致。允许 `image_time < ready_time <= selection_time`。调用者必须先把时钟映射到同一域；该模块没有实现跨设备时钟同步。

`FastWithSlowKV.forward()` 是低层数值接口，**不自动执行真实系统的调度/admission**。训练可传已对齐的 metadata；部署必须由调度器完成 admission，再锁定本轮版本及 gate。任何未来兼容性监督 mask 都不能直接成为在线有效标记。

- `snapshot.to(device)` 保留计算图，可用于单进程跨设备模型并行。它不等价于序列化后的分布式 autograd。
- `snapshot.inference_copy(device)` detach 并 clone，避免与生产者共享 tensor storage。frozen dataclass 不会让 Tensor 天然只读；发布者仍必须遵守不可修改、读者释放后再回收的规则。
- 阶段 A 可冻结 Slow 主干，但语言前向保留 autograd，新增 Ego/feedback/queries 仍能训练。只有冻结视觉路径使用 `no_grad`。
- 阶段 C 可读取 detached snapshot。阶段 D 必须传 live snapshot；测试与真实审计均仅使用 Fast 规划损失验证了 `Fast loss → 六层 Slow KV → Slow LoRA`。
- 保留现有官方 mean L1 pose loss，没有静默切换 Huber 或新增航向损失。

## 已执行证据

见 [real_audit.json](../evidence/slowfast_v11/real_audit.json)、[model_assets.json](../evidence/slowfast_v11/model_assets.json) 与 [test_summary.json](../evidence/slowfast_v11/test_summary.json)。模型身份以本地权重/config SHA256 记录；2B 的远端 revision 未独立确证，不冒填版本号。

真实审计使用**同一张 H800**，复用一条移动 TRAIN 窗口和两份真实预训练权重，新增规划模块随机初始化，执行一个联合优化器更新。没有以 Smoke 权重充当正式训练初始化，没有测量泛化 ADE 或 KV 收益。

| 检查 | 实测 |
| --- | --- |
| absent vs masked | 输出逐位相同 |
| 改变 masked KV | 输出逐位相同 |
| 固定当前输入、metadata，改变有效 KV | Context 逐位相同，规划输出发生变化 |
| Fast loss 到 Slow | 六层 K/V、Slow LoRA、反馈编码器均有有限非零梯度 |
| Vision / Merger | 两模型冻结，无梯度 |
| 真实更新 | loss 有限，梯度有限，Fast head 参数发生变化 |
| 同卡联合峰值 | allocated 49.50 GiB，reserved 50.89 GiB |
| 最终回归 | 419 passed：原基线 402 项，新增 17 项；独立复核也通过新增 17 项 |

单次 Fast forward 计时只是接口开销观察，没有重复分布、没有图像解码/processor，不能用来认定实时频率、P95 或 E2E 性能。

## 复现与迁移

沿用仓库已记录的 Python/CUDA/FLA 环境，先准备本地 0.8B、2B 权重和现有 mini manifest/stats。不要把机器特定路径写进正式配置。默认配置是两个独立 CUDA 设备；没有第二张卡会直接失败。

```bash
export FLA_TILELANG=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/path/to/matching/cuda-toolkit
export PATH="$CUDA_HOME/bin:$PATH"

python -m tools.slowfast_v11_audit \
  --fast-model /path/to/Qwen3.5-0.8B \
  --slow-model /path/to/Qwen3.5-2B \
  --manifest /path/to/deep_train.jsonl \
  --stats /path/to/train_stats.json \
  --output outputs/slowfast_v11/p0_target_gpu.json
```

当前单卡审计额外显式指定 `--slow-device cuda:0 --allow-single-gpu`。这个 override 只允许验证计算图，不会把结果标记为多 GPU 验收。入口强制一条样本、一次更新，不能直接替代正式训练入口。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests tests_navsim tests_slowfast -q --import-mode=importlib
```

历史 `tests/` 的几个真实 Processor/B2D provenance 测试需要原基线的本地 0.8B assets，以及 `outputs/fast_only/audit_v2/source_audit.json`；后者未随 Git 发布。新环境可先运行 `tests_slowfast`；不要把缺少本地证据文件误报成模型回归。

## 尚未完成与下一步

本轮没有实现 Future Head、运动分类 Head、A–F 完整训练入口/恢复、真实年龄下的 KV 收益曲线、异步刷新/发布/引用计数、双 GPU 通信测量或闭环评测。整个 V1.1 不能据此标记完成。

下一步先在目标双卡执行同一真实前反向；然后固定 Slow 辅助监督、反馈坐标/时间对齐和适配训练入口，建立同检查点新鲜/陈旧/缺失 KV 对照。只有确认真实年龄 KV 有收益，才扩大 Future 检验和刷新×屏蔽实验。NAVSIM 固定日志不能单独支持真实交互闭环结论，20Hz 也不能从 2Hz 图像插值获得监督证据。
