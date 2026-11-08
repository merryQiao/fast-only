# 直接 KV 与完整 Fast 语言主干：扩大 TRAIN 的配对实验

本轮关闭“冻结 Reader＋Head 前残差融合”的继续调参，执行原直接 KV 主线。冻结 Slow、Fast 视觉塔及 merger；更新完整 Fast 语言主干的基础权重、已有 LoRA、输入编码、Planning Queries 和唯一轨迹 Head。没有增加 Reader、Expert、投影架构、异步刷新或动作匹配门控。

当前状态：全参数真实训练验收已通过；冻结输入导出和三组固定预算训练尚在执行。此状态不代表已经得到 KV 规划增益结论。最终指标以各训练臂的固定终点与独立重载结果为准。

## 固定实验协议

| 项目 | 本轮选择 |
|---|---|
| TRAIN | 512 窗口／12 个 recording groups |
| 新 VAL | 256 窗口／8 个此前未使用的 recording groups |
| TRAIN 扩充方式 | 原 TRAIN256 全部保留；前轮 8 组诊断 holdout 的全部 256 窗口整体转为 TRAIN |
| 数据边界 | 新 VAL 排除此前见过的全部 15 组；旧 holdout 成绩从此仅作历史记录 |
| 三臂 | `full_fast_only`、`full_correct_kv`、`full_fixed_kv` |
| 共同起点 | 同一最新 Fast 检查点；P0 更新前的完整共享权重 |
| 冻结源 | 同一个 Slow 检查点，以及原保存的因果 Fast512 反馈生产者 |
| 正确 KV | 同一窗口、同步 fresh KV |
| 固定 KV | TRAIN 最小 token `006be99cbfcf5751` 的真实 KV；所有窗口恒定 |
| 训练预算 | 一个训练种子 42；每臂 1024 次 AdamW 更新，microbatch 1、累积 4，共 4096 次窗口曝光／8 个 TRAIN512 epoch |
| 采样 | 每 epoch 无放回打乱；三臂顺序、预算和初始化相同 |
| 学习率 | 语言主干 `1e-5`；新增输入／Queries／Head `5e-5`；64 更新 warmup，cosine 降至峰值的 10% |
| 视觉 | 原图 1920×1080，仅向下 padding 到 1920×1088；2040 token／帧，4 帧共 8160 visual tokens |
| 数值 | 训练权重 FP32；语言计算 BF16；LoRA A/B、传感器 MLP 和 Head FP32 |
| 验证使用 | 固定 1024 更新终点评价；不以新 VAL 选步数、调参或延长训练 |

输入保持 NAVSIM 2Hz 四帧前视图、当前 Ego `[vx,vy,ax,ay]` 和官方离散导航 command；输出未来 4 秒、间隔 0.5 秒的 `8×3` 自车坐标轨迹。Slow KV 仍通过第 4／8／12／16／20／24 层的直接拼接接口，仅供 Fast Planning Queries 读取。当前 Context 行不读取外部 KV。无 KV 的 Fast-only 保留原来的明确无缓存标记。

本轮是 NAVSIM mini 的有界研究实验，不是完整官方 `navtrain` 训练，也不是 `navtest` 或闭环成绩。三臂都是继续适配同一 Fast 起点；不能将本轮新增训练预算当成全部历史训练成本。

## 归一化与输入缓存

继续使用原 TRAIN256 统计量作为已恢复 Fast、Slow 与反馈生产者的共同输入接口，SHA 为 `35dc682328e66c31f35c671ad2a6654031b31d7d6c22927377301cceaf687bb6`。新的 TRAIN512 统计量另存为分布审计，明确标记为不用于模型输入。直接替换统计量会改变冻结生产者及已有 KV 的语义，因此本轮没有自动替换。未来从预训练权重重新开展完整训练时，应重新锁定完整 TRAIN 的统计量与所有消费者。

缓存仅包含冻结 Fast 视觉输出、原生 Processor 的 ID／mask／grid、Ego／command 以及冻结 Slow 原生 KV。没有缓存 Fast 的 `inputs_embeds`、语言 Hidden 或 Planning 特征。Fast 每次更新都重建可训练输入并完整前向／反向传播。导出的 KV 保存源自身的位置与 valid mask；固定来源不借用目标窗口的位置数组。

每个新导出文件先完成阻塞式 GPU→CPU 复制并独立 clone，再落盘和重新加载；所有 tensor、dtype、shape 与 metadata 逐项精确比较后才记录 SHA。初次导出使用的通用异步 CPU 拷贝缺少这一保证，已在训练前中止并保留不完整目录，未用于任何训练；最终目录重新生成。50MB BF16 延迟 CUDA 生产者的真实保存／重载检查也通过。原有 KV 库与完整参数检查点本身使用阻塞拷贝，不受这次修正影响。

正确与固定来源使用完全相同的 fresh 元数据 `[0,0,0,0,1,1]`，从而固定条件实验不混入元数据差别。固定来源是恒定条件训练对照，不声称来自当前真实场景。

## 已完成的 H800 工程验收

真实 P0 每臂执行两次累积四个 microbatch 的 AdamW 更新，并重放第二次更新。共 **759,590,723** 个参数参与训练，包含所有语言基础权重。24／24 层均有有限非零基础权重梯度；24／24 个抽查基础矩阵确实更新；Embedding 与最终 Norm 有有效梯度；完整视觉塔及 merger 均无梯度。

| 条件 | 实测峰值 allocated |
|---|---:|
| Fast-only | 35.40 GiB |
| 正确直接 KV | 36.42 GiB |
| 固定直接 KV | 36.55 GiB |

模型、AdamW 状态和 RNG 的更新重放全部精确一致。两条真实原图 TRAIN 探针上，原推理路径与新 FP32 master／BF16 language 策略的初始预测也精确一致。原生与桥接 RoPE 的相同位置、cos／sin、BF16 dtype 已机械核对。

独立复核还核对了共享初始化文件：529 个 FP32 tensor，320 个语言基础权重与原加载精度下的预训练资产一致，209 个既有 LoRA／外层 tensor 与恢复起点一致。全部 P0 更新丢弃，正式三臂从这份初始状态开始。

P0 原始记录 `updates[].loss` 是最后一个 microbatch 的 loss；正式训练日志记录四个 microbatch 的平均 loss。P0 的恢复检查是内存状态重放；正式训练另外验收磁盘检查点、scheduler、sampler 进度与独立进程预测。

## 数据流量与恢复记录

本轮新增 HTTP response body 上限为 2,000,000,000 字节。原定新 VAL 样本在相机访问前锁定，下载失败不会替换日志或窗口。两条分片发生 TLS 中断后，单独锁定了重传修正文件，之后改成持久保存压缩前缀、按实际落盘字节续传、本地解压。分片 27 的原 225MB 前缀只覆盖 126/128 张目标图像；最终在请求前锁定了唯一的范围例外：追加 4MB 到 229MB，核验全部 128 张原定图像。相机总流量 1.8GB、本轮总流量 2GB 上限均保持不变。

最终新增响应体为 **1,999,942,916 B**，包含 metadata 200,000,000 B 与 camera 1,799,942,916 B；低于 2GB 上限 57,084 B。累计已记账数据流量为 **5,403,028,597 B**，不含协议头等开销。529 条请求记录包含 12 条失败／零字节请求，365,692,042 B 中断流量已计入总数；持久前缀阶段没有丢失或重复字节读取。所有修正、失败报告、请求账本与前缀 SHA 均保留。

数据已由独立审阅者逐项复核：3072 张 JPEG 和 20 份 metadata 共 3092 个源文件，512／256 窗口、12／8 组记录，完整角色转移及旧 15 组排除均与访问相机前的清单一致。原图尺寸保持不变。

## 运行入口与验收边界

配置见 [slowfast_direct_full.yaml](../configs/slowfast_direct_full.yaml)，精确队列见 [slowfast_direct_full_cohort.json](../configs/slowfast_direct_full_cohort.json)。

```bash
python -m tools.cache_direct_full --output outputs/slowfast_direct_full/cache_01
python -m tools.train_direct_full --arm full_fast_only --output outputs/slowfast_direct_full/run_01/full_fast_only
python -m tools.train_direct_full --arm full_fast_only --output outputs/slowfast_direct_full/run_01/full_fast_only --replay
python -m tools.train_direct_full --arm full_fast_only --output outputs/slowfast_direct_full/run_01/full_fast_only --verify
```

同样执行其余两臂，使用同一个 cache 与 P0 shared init。完整检查点保存所有可训练基础权重、Optimizer、Scheduler、RNG 和采样游标；不能复用旧的仅 LoRA／外层 `canonical_state` 保存逻辑。

主要比较是分别训练的正确来源与固定来源；同时报告完整语言适配后的 Fast-only，以及旧固定特征适配器的分数参考。后者的历史训练数据和预算不同，不是本轮等预算训练臂。预先沿用 20mm 平均 ADE 实用参考与按 recording group 配对的区间；只有一个训练种子，区间不包含训练种子不确定性。

模型、原图、原始 KV、输入缓存和数 GB 检查点不提交 Git。新机器需要对应外部资产并核对 SHA；本实验入口有意锁定本轮 512／256 队列，扩大到完整官方训练集需要另锁配置和数据协议，无须改变直接 KV 模型架构。
