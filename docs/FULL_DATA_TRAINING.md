# 迁移后使用完整 NAVSIM 数据训练

工程迁移已通过；当前代码是经过单卡验证的研究基线。**全量训练启动为有条件通过，目标服务器还有配置、数据和分布式工程工作。** 本文描述下一阶段需要完成的事情，不把这些工作标成已实现，也不要求回到原服务器继续小样本训练。

## 1. 先明确当前的限制

| 当前实现 | 全量训练前需处理 |
| --- | --- |
| `config.py` 固定 `cohort=bounded_mini_preflight` | 增加正式cohort及其可验证的数据契约 |
| Manifest/Stats目标路径固定为 `outputs/navsim/data/*` | 允许目标机器配置，并验证各角色/来源/身份 |
| `train.py:execution_limit()` 拒绝mini上的formal训练 | 仅在完整数据契约通过后允许formal，不是删掉护栏 |
| 步数8000、MB2、累积8等由Schema钉死 | 按目标GPU、global batch和完整epoch预算配置 |
| mini准备工具选择3段log、2 TRAIN/1 VAL、每log有限窗口 | 新建正式navtrain构建流程，应用官方Scene Filter |
| sampler为有放回counter抽样 | 增加按epoch shuffle的无放回采样及恢复状态 |
| trainer只支持单进程，device默认cuda:0 | 多卡需要DDP、rank设备/采样/RNG、同步与单写入者 |
| Manifest/审计使用原机器绝对路径 | 做数据root重定位及基于内容的身份映射 |

这些接入工作涉及 `fast_navsim/config.py`、`data.py`、`runtime.py`、`train.py` 和新的全量数据准备工具；模型输入/输出协议不变时，**不需要重新设计 `model.py` 的Backbone、queries或head**。

## 2. 准备的是 NAVSIM/OpenScene，不是 Bench2Drive

官方 `navtrain` 是对 OpenScene `trainval` 应用Scene Filter后的训练场景集合，不能把 `trainval` 中所有帧做滑窗就称为 `navtrain`。目标机器有Bench2Drive不代表已有NAVSIM。

建议固定并保存：

1. NAVSIM devkit commit、数据revision、Scene Filter版本/文件hash。
2. 官方允许的scene token集合，以及最终接受/拒绝的token集合和原因。
3. scene token到log、当前帧和四张历史CAM_F0文件的映射。
4. TRAIN/VAL log分组、seed和各场景/命令分布；VAL不得与TRAIN共享窗口所属log。
5. 缺图、坏图、缺metadata、非法command、非法pose、timestamp gap等拒绝计数和比例。

不要用 `navtest` 或 `navhard_two_stage` 来选超参数、决定训练时长或选择checkpoint。先从训练来源建立固定独立VAL，再生成训练窗口；验证集固定后不要根据结果重新挑窗口。

当前500000±1000µs的间隔检查只在3段log上验证过。全量构建先统计真实gap分布与拒绝率；若阈值不适合，应记录证据、修改数据策略并建立新版本，不能静默丢掉大量官方场景。

官方来源：[数据划分与Scene Filter关系](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/docs/splits.md)。实际样本数应从所选版本的有效token统计，不依赖本文给一个固定总数。

## 3. 全量Manifest和Stats

继续使用当前协议：四张原图、当前 `[vx,vy,ax,ay]`、原生四维one-hot命令的类别、未来 `[8,3]` 后轴SE(2)标签。缺字段应拒绝并计数；不能补零target、换成未来专家轨迹或静默改图像分辨率。

只用**完整TRAIN**重算Ego均值/标准差，常量维保持std=1。VAL/TEST复用同一TRAIN统计量，不能沿用仓库小样本Stats，也不能从VAL重算。

路径迁移应将内容身份与存储位置分开，例如在清单保存相对路径并由可配置root解析。当前代码强制绝对路径，source audit和checkpoint也保留旧路径，必须一起适配。仍需拒绝路径越界、未知cohort、跨split重复token/log和身份漂移。

保存新的数据版本、清单hash、Stats hash、日志分组和source audit，作为正式实验身份。当前仓库只提供历史验证摘要，没有原服务器的原图、metadata、Stats或原始输出目录。

## 4. 采样、epoch和学习率日程一起定

当前有放回采样用于smoke的确定性恢复，不保证正式TRAIN每个epoch都被访问。建议改为epoch级shuffle、无放回遍历，并将下列状态纳入checkpoint：

- shuffle seed、epoch号、已消费位置，以及分布式rank如何分配样本。
- 梯度累积边界、global step、windows_seen和真正完成的epoch数。
- worker/augmentation的确定性种子；未消费的预取不能改变恢复位置。
- drop_last或DDP补齐策略；若存在补齐样本，明确记录，不能宣称绝对无重复。

计算global batch：

```text
global_batch = per_rank_microbatch × gradient_accumulation × world_size
steps_per_epoch ≈ effective_train_windows / global_batch
```

尾batch、补齐和采样策略会影响精确值。8卡若仍用每卡MB2、累积8，global batch变成128，不能继续把日志写成16。

根据实际窗口数、目标epoch和VAL评估频率，**在启动前**定义完整warmup/cosine日程。8000步可作为一个计划节点；它不是模型已收敛的保证。避免先把8000步cosine降到零，再临时续训而不说明调度器变化。更改预算/日程应记录为清晰的新阶段或新实验。

## 5. 多卡不是给当前命令加 torchrun

当前代码没有DDP，所有进程会使用相同seed/sampler，并写同一路径；默认还都指向cuda:0。直接多进程启动会造成重复采样、未同步梯度和输出冲突。

集群入口至少需要：

- 进程组初始化、LOCAL_RANK设备绑定、DDP封装及同步梯度。
- rank-aware sampler/RNG；按既定无放回策略分配同一epoch，记录world_size。
- 梯度累积中合理使用同步边界；global batch、LR策略及windows_seen一致。
- VAL指标跨rank正确聚合；不把某个rank的局部平均当全局指标。
- rank0写配置、日志、图示和checkpoint；必要的per-rank RNG/sampler状态集中或分片保存。
- 中断后的同拓扑恢复验证；拓扑/world_size变化时单独定义恢复语义。

先完成标准DDP接入和测量，再根据实际目标显存/通信成本决定更复杂的并行方案，不需要为了迁移重设计规划模型。

## 6. 原图显存与数据吞吐

H800上的已有记录：MB2峰值allocated39.23GiB，MB1深过拟合20.76GiB。40GB级卡的MB2余量很小，目标机先实测；必要时用MB1和对应累积维持global batch，修改配置并记录。不要悄悄resize图像来制造通过。

当前smoke包含在全量数据上成本可能很大的审计操作：

- 每次启动/恢复扫描全部引用图片并计算SHA。
- 每个microbatch对完整native pixel tensor同步计算hash后才H2D。
- 初始和结束时对整个TRAIN逐样本评测，并保留预测。
- 小规模CPU线程prefetch，Processor用锁保护可变状态。

全量阶段应把数据完整性扫描做成可复用、版本化的离线审计，正常训练保留数据身份验证与轻量抽查；昂贵逐tensor hash保留为显式诊断模式。把TRAIN诊断改为固定代表性子集，VAL保持有意义且固定的评测集合。**这是需要实现的优化，不是当前已完成的能力。**

统计JPEG读/解码、Processor、H2D、forward/backward、checkpoint和评测时间，以及GPU利用率、吞吐和峰值显存。存储位置/缓存策略可改变，但图像内容与采样协议保持可审计。

## 7. 环境、初始化和恢复分清楚

当前安装依赖共享环境；仓库提供安装清单、已测重要版本和observed inventory，不提供可直接复制的virtualenv或已验证通用容器。目标服务器需要建立自己的CUDA/toolkit/编译器/torch/FLA/TileLang组合。不要禁用上游数值保护来绕过兼容性问题。

正式模型应从预训练Qwen固定revision重新初始化LoRA、queries和head。单样本过拟合checkpoint只作验收，不是正式初始规划权重。

区分两种操作：

| 操作 | 应保证什么 |
| --- | --- |
| 严格resume | 恢复optimizer/scheduler/RNG/sampler及同一实验身份，配置/数据/环境不漂移 |
| 跨环境weights-only加载或重新初始化 | 显式开始新实验，不承诺继承优化器/采样位置或跨GPU逐位一致 |

当前checkpoint严格要求原配置、原路径和版本一致，尚不支持透明跨服务器重定位。需要实现后再用目标路径验证，不能把源H800的零差异当作任意GPU/CUDA/后端上的保证。

## 8. 目标集群首次启动验收

这些检查在**目标服务器**做，原服务器阶段已结束：

1. 对官方filter、TRAIN/VAL token/log、Stats、缺文件和timestamp拒绝比例做数据审计。
2. 用正式原图和per-rank batch做真实forward/backward，确认Vision冻结、各训练分支梯度有限、无OOM。
3. 跑少量真实optimizer updates及固定VAL，记录读盘/预处理/GPU吞吐和显存。
4. 验证中断恢复、完整checkpoint状态及新进程重载。多卡另外验证rank样本分配、同步更新和单一写入者。
5. 按有效TRAIN数量确认epoch长度、global batch、完整LR预算，再提交正式训练作业。

正式训练按VAL ADE选checkpoint，另外报告FDE、各时域L2和heading MAE。之后再接官方NAVSIM Agent/EPDMS；本仓库目前没有这些评分结果。NAVSIM2Hz任务验证也不等于未来Fast10Hz/Slow5Hz协议已验证。
