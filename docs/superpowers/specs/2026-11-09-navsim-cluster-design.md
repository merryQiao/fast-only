# NAVSIM V1全量训练与最小异步系统：实现及Smoke任务

## 目标与边界

当前服务器只实现代码与小规模Smoke；不下载完整数据、不执行正式训练。目标部署为2节点、每节点8张H200，共16GPU。保持2B冻结Slow、完整0.8B Fast、原六层Direct-KV和唯一轨迹Head；不再增加Reader、Head替换、Slow训练、逐层消融或动作匹配门控。

正式只训练fast_only/slow_fast两个臂，固定种子42、43，两对共同初始化／采样顺序；各8epoch，具体更新数由完整有效TRAIN数与全局batch计算。旧最强检查点仅为可选参照；不将Smoke权重当正式初始权重。默认Fast从预训练基座初始化原有规划模块，完整语言基础权重和已有LoRA/输入/Queries/Head训练，视觉塔含Merger冻结。

## 数据

锁官方NAVSIM `v1.1`分支commit `3e8291bfa89ff247231e0227778840cd0a036896`，不沿用旧Mini工具指向V2主线的身份。使用官方navtrain Scene Filter及默认TRAIN/VAL日志划分：978/214日志，101/61记录组，无组重叠。全量是所有有效官方TRAIN-role anchors；VAL保持独立。navtest不进入训练／模型选择。

官方资格为4history+10future、frame_interval1、anchor偏移3、anchor有route；保留完整14帧资格后，监督只取前8future poses。复用已验证的rear-axle SE(2)、四维Ego和原生四类导航；实际2Hz时间戳检查作为显式项目策略，统计全部拒绝原因。正式要求所有官方日志/允许token被考虑，缺数据不可静默改成partial；配置显式拒绝率上限，缺日志直接失败。

新增`fast_cluster/data.py`和准备CLI，输出图像相对路径的TRAIN/VAL JSONL、TRAIN-only Fast统计量、记录组/场景token清单、元数据/filter/split/source身份与覆盖／拒绝审计。内容身份独立于data.root；不把所有JPEG在每次训练启动重哈希。formal与smoke数据kind分开，正式入口拒绝Mini。普通训练从原图读取；只做原生padding，不resize／重压缩。

## 冻结来源与初始化

导出可迁移source bundle：当前冻结2B的LoRA/原输入编码器/Queries/Head状态、原冻结因果Fast反馈生产者状态、两基座资产身份、原source TRAIN256 normalizer和模型参数配置。仅导出小型任务状态，基座模型权重独立定位。bundle显式保留旧来源训练身份，不把它改称官方V1全量训练成果。

新Fast使用完整TRAIN统计量；冻结Slow及其冻结反馈生产者继续使用bundle原统计量，分别从同一raw Ego计算，不混用尺度。训练Slow输入反馈沿用既有冻结生产者，KV每batch同步完整Prefill，默认不要求存储全量约十TB的KV缓存。Fast-only不运行Slow。两臂先初始化相同Fast，再加载冻结来源，恢复初始化后的RNG，避免来源加载消耗随机数导致起点差异。

复用原FastWithSlowKV、SlowPrefill和full_precision。完整Fast权重FP32 master、语言BF16计算、传感器/Head/LoRA FP32；gradient checkpointing关闭。配置Fast后显式恢复原生FP32视觉RoPE非持久buffer；所有原图入口保留该数值协议。Slow和teacher的参数及buffer冻结并有身份验收。

## DDP、采样与恢复

新增`fast_cluster/distributed.py`与`fast_cluster/train.py`。torchrun绑定LOCAL_RANK，formal默认WORLD_SIZE16/LOCAL_WORLD_SIZE8/NCCL；普通python或单卡不能悄悄代替正式拓扑。Smoke配置允许单GPU；CPU两rank Gloo只验证分布式算法，不宣称16H200实测。

每epoch独立seed置换，无放回遍历有效样本；为了等长DDP batch仅补invalid padding slots，不把padding计入windows_seen。实际loss按整个累积组的全局有效样本数归一化，补齐样本零权重；no_sync用于非最后microbatch，所有rank保持相同同步次数。尾累积组按实际micro数量执行。Val按rank切分不重复补齐，使用未封装module前向，再归并sum/count，不因不等长评测触发DDP collective。

默认microbatch1/rank、累积2，16卡global32；两学习率保持显式设置，不自动按world_size缩放。训练预算8epoch、warmup/cosine全日程由有效样本数算出，8000不是固定终点。每epoch评VAL，按VAL ADE保存best，最终预算检查点仍报告；Joint终点附带固定TRAIN来源KV推理，不增加训练臂。日志包括有效windows_seen、epoch、globalbatch、各损失、梯度、吞吐、显存和验证指标。

checkpoint在optimizer边界原子写入完整语言/外层参数、optimizer/scheduler、所有rank的CPU/Python/NumPy及本rank CUDA RNG、sampler epoch/消费游标、buffer状态、数据/资产/语义配置身份。不得调用get_rng_state_all在每rank初始化其他GPU。游标按成功消费推进，不受预取影响；resume要求同一world_size及训练身份，路径重定位只在内容身份不变时允许。rank0写文件，rank错误不能默默挂住；epoch末与周期保存，限制历史保留数。

## 最小异步系统

新增`fast_cluster/async_runtime.py`与真实模型replay CLI。Fast主进程和独立spawn Slow进程各自拥有CUDA模型；目标支持同节点不同GPU，当前H800允许显式same-device functional模式，不把它当多GPU延迟证据。跨节点训练与跨节点异步RPC是不同能力；本次不宣称实现跨节点异步RPC。

Fast处理真实2Hz最新可用输入，积压采用最新帧并记录跳帧。每个固定绝对刷新周期最多一个Slow在途任务；忙则跳过并计数，不形成无限队列。请求锁定已完成Fast计划及对应当前/历史观测、raw Ego、导航epoch、source pose/time；不带GT或未来帧。

Slow完整Prefill后将六层pre-RoPE KV阻塞复制到自有CPU storage，完整打包后发送。Fast设备全部传输并同步完成，才记录ready_time并发布不可变版本；一次Fast前向固定同一版本。复用SourceMetadata/admit_snapshot/cache_metadata，保证s<=t且r<=c，episode/nav/model身份一致。导航epoch每次变化递增，不用command值作为可复用版本。源到当前位姿来自当前/历史真实Ego pose，不由未来GT估算。

先在当前观测上计算原生视觉特征，再选择已完整可用版本并运行完整语言/唯一Head；视觉仍本轮重新编码，缓存只避免同一次forward重复视觉。记录t/c/u/source/ready、缓存年龄、source-to-readable、图像处理/视觉/语言Head/传输及E2E时间、Slow调用/忙跳过/有效规划率/跳帧率。系统运行PASS和延迟下规划可靠性分开；只训练fresh KV并不能保证陈旧KV可靠。

训练固定反馈生产者和部署已完成Fast计划是显式不同反馈策略；runtime保存反馈模型版本，功能Smoke不据此宣称任务质量。无有效版本Fast按原无KV路径运行，不实现预测误差停用、动作匹配或自适应刷新。

## 验收与交付

1. CPU语义测试与2rank Gloo：无重复有效覆盖、尾batch正确均值、DDP参数同步、单写者、逐rank RNG与恢复后更新一致。
2. H800真实原分辨率：两臂少量更新，视觉/Slow冻结，24层Dense梯度有限非零，实际磁盘resume与独立重载。设置明确Smoke更新上限，不运行8epoch正式训练。
3. 从本地已认证压缩前缀恢复连续2Hz片段，无新增HTTP数据，实际Fast/Slow异步进程发布和消费多个版本。原图／缓存等价、无未来输入、全版本完整性与时间准入机械检查。
4. 跨路径内容身份、正式Mini拒绝、配置worldsize拒绝、两节点启动脚本语法与dry-run。16H200/NCCL/IB、完整数据吞吐及真实多GPU峰值留给目标机首次短训验证。

交付正式／Smoke配置、准备/导出/训练/恢复/评测/异步CLI、2节点启动脚本、迁移说明、依赖信息、Smoke证据、源码/源bundle打包清单。无模型权重／原图／KV大文件进Git；保留原用户工作区和既有结果。
