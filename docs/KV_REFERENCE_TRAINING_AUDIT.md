# KV 快慢系统代码核对：训练配方与当前差距

审计日期：2026-11-06。核对公开源码、训练配置与第一方论文；固定下列 revision。没有执行第三方训练代码、安装其依赖、下载权重/数据集或使用 GPU。本文不是对第三方成绩的复现。

四个仓库、17 个核心源码文件的 revision、SHA256 和来源链接记录在 [sources.json](../evidence/kv_reference_audit/sources.json)。已对本地训练结论与外部接口分类做独立复核，并修正了样本占比推断和 teacher-forced 开环对照的表述。

**结论：原生 K/V 直接拼接是已有可训练先例的接口，不应因为本轮小预算结果弱就先修改架构。当前有更明确的训练缺口：Slow 只接受了极短的任务训练，Fast 消费阶段始终冻结 Slow，持续 Stage D 联合训练还没有执行。** 这比泛泛怀疑“两个模型的语义空间不同”更值得优先处理。也不能据此宣称训练是唯一原因，或每个 KV 检查点都必然产生增益。

## 1. 先分清别人到底拼接什么

| 实现 / 固定 revision | 实际接口 | 与本项目最相关的条件 |
| --- | --- | --- |
| OpenPI / π₀ `215abfb217dbac7d5f1273282331b9b1866c0479` | 不同宽度专家投影后的 Q/K/V 拼接；推理复用视觉语言前缀 KV | 训练时 prefix+suffix 联合前向，动作损失可更新配置允许的前缀参数；动作 suffix 没有独立图像编码路径 |
| DriveVLA-W0 `a55766ce46ee55d9035b93816e57b8d933166482` | VLM 与独立 Action Expert 的 K/V 拼接；另有 cached-KV 前向 | 先做 NuPlan 任务预训练；公开 NAVSIM 脚本从该 checkpoint 起步并设置不冻结 VLM |
| Fast-in-Slow `14f73b3e6ebe5e44464e7958b4d086e7dda21941` | Slow 的 hidden sequence 与 Fast 输入拼接，由同一 LLM 的末层继续计算 | 共享主干，扩散与 AR 联合目标；不是独立 Qwen 之间六层原生 KV 移植 |
| 导航 AsyncVLA `b593ac5143287a62893159a42c88a471bf444f10` | 最后一层 action hidden states 经 projector，和当前/过去视觉特征拼接 | 从任务预训练 OmniVLA 起步；支持先训适配器、再打开 base LoRA 的联合训练 |

这些代码共同说明：拼接算子可以简单，能够学会消费它的训练过程并不简单。不能把 feature-token concat、共享主干 hidden concat 和原生 per-layer KV concat 当成完全同一种实现。

### OpenPI：直接拼接的明确代码先例

`gemma.py` 的 Attention 允许 width 不同的专家，但检查 head 几何相同；各自投影后合并序列。相同模块还把缓存 K/V 与本轮 K/V 合并。源码配置中 2B 与 300M 专家的宽度不同，这反驳了“必须原先处在同一隐藏宽度/天然语义坐标才可能训练”的推断。[Attention 与配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L58-L107)，[K/V 组合](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L164-L214)。

训练函数将视觉/语言 prefix 和 state/noisy-action/time suffix 放进同一次可微前向，动作目标驱动消费者与允许更新的前缀。推理才先填充 prefix cache，再复用它迭代生成动作。suffix 自身不编码图像，因此视觉信息必须通过 prefix 进入；我们的完整 Fast 则有自己的视觉路径。[输入与训练](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L104-L214)，[缓存推理](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L233-L266)。

需要区分计算图与参数冻结：训练器按 `trainable_filter` 选择参数，并非所有配置始终更新整个 VLM。公开 LIBERO 全量/LoRA 配置加载的是机器人任务预训练 `pi0_base`，例示 30,000 步微调；不能把它当成通用 VLM 冷启动后几十步即可成功的证据。[梯度与过滤](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/train.py#L136-L165)，[微调配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py#L650-L697)。

### DriveVLA-W0：NAVSIM 上更接近的 K/V 接口

`Emu3Pi0SharedLayer` 分别计算 VLM 和 action expert 的投影，直接拼接 K/V 后执行 Attention。`forward_with_cache` 则读取缓存 VLM K/V。该辅助类名称里的 Shared 指联合 Attention 处理，不等于所有投影权重共享；模型确实单独构造 action expert。[训练融合](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/reference/Emu3/emu3/mllm/modeling_emu3.py#L1779-L1848)，[缓存路径](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/reference/Emu3/emu3/mllm/modeling_emu3.py#L1852-L1902)。

公开 flow forward 把两支逐层联合计算，动作 flow loss 保留到 VLM 特征的图；这一 forward 的 VLM 自身 loss 权重还被设为 0，因此不能说它在每个阶段都同时使用 world loss。[联合层循环与 loss](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/reference/Emu3/emu3/mllm/modeling_emu3.py#L2093-L2132)。

其文档先列 NuPlan world-model 预训练，再做 NAVSIM 训练；flow launcher 从该任务 checkpoint 加载，配置 `freeze_vlm=False`、10,000 步、8 卡、每卡 batch12。[阶段说明](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/Training.md#L23-L43)，[launcher](https://github.com/BraveGroup/DriveVLA-W0/blob/a55766ce46ee55d9035b93816e57b8d933166482/scripts/scripts_train/train_navsim_flow_matching.sh#L12-L73)。

**公开性限制：** 仓库只发布审核后的部分代码；launcher 指向的 `train/train_pi0.py` 和 `configs/pi0_fast_video.json` 在此 revision 未找到。上述参数是发布配置，非我们核实完成的实验预算；可确认模型组件提供直接拼接与可微路径，不能声称完整训练流水线已复现。

其位置协议也不同：VLM 使用 RoPE，action expert 在这个 helper 中不加 RoPE。我们保留 Fast 原生 mRoPE、固定 4096 偏移，是另外一种接口约定。当前没有证据证明该约定错误，也不能因为另一个接口成功就认为位置约定无需校准；本次不修改它。

### Fast-in-Slow：共同训练与共享主干，不能误称原生 KV 移植

发布代码先运行 LLM 前段，把最后 hidden sequence 插入 Fast 视觉/state/action 输入，再运行同一模型末段；两次调用均显式不给 `past_key_values`。中间没有 detach。训练器把扩散 loss 与 AR loss 相加后一次反向。[前后段与 handoff](https://github.com/CHEN-H01/Fast-in-Slow/blob/14f73b3e6ebe5e44464e7958b4d086e7dda21941/models/vlms/prismatic.py#L973-L1034)，[共同反向](https://github.com/CHEN-H01/Fast-in-Slow/blob/14f73b3e6ebe5e44464e7958b4d086e7dda21941/training/strategies/base_strategy.py#L298-L329)。

launcher 打开 vision/LLM 训练，使用预训练 checkpoint、300 epochs、8 卡×每卡 batch6；这些是配置，不是实际完成日志。论文的预训练 epoch 数与公开 HF 配置并不完全一致，不能把其中任一个当成已独立审计的完成预算。可借鉴的是任务预训练、两支目标共同优化、当前视觉与 Slow 条件联合使用。[launcher](https://github.com/CHEN-H01/Fast-in-Slow/blob/14f73b3e6ebe5e44464e7958b4d086e7dda21941/train.sh#L9-L83)，[HF 配置](https://huggingface.co/chenhao01/fisvla/blob/d5f66f3accd4c043742cfe348431f58c5609e950/config.yaml)。

### 导航 AsyncVLA：任务化底座、学习型接口与联合适配

此处是论文 2602.13476 的 NHirose 实现，不是同名的 asynchronous flow-matching 项目。它抽取 action hidden states，经 learned projector 送给 Edge Adapter，与最新图像及当前/源图像对比特征融合，**不是原生逐层 K/V 导出**。[特征接口](https://github.com/NHirose/AsyncVLA/blob/b593ac5143287a62893159a42c88a471bf444f10/vla-scripts/train_asyncvla.py#L528-L559)，[视觉融合](https://github.com/NHirose/AsyncVLA/blob/b593ac5143287a62893159a42c88a471bf444f10/prismatic/models/small_head.py#L53-L76)。

发布默认配置冻结 base、训练新头；同时打开 `TRAIN_BASE` 与 `TRAIN_HEAD` 时，消费损失可以穿过 projector 回到 base LoRA。因此“冻结 Slow 训 Fast”本身并不错误，关键是它从已任务化的 OmniVLA 起步，并支持之后共同适配。[条件计算图](https://github.com/NHirose/AsyncVLA/blob/b593ac5143287a62893159a42c88a471bf444f10/vla-scripts/train_asyncvla.py#L487-L546)。其采样包含旧观测，并显式增加反应性样本；本次未核验实际移动/静止样本占比。[GNM 采样](https://github.com/NHirose/AsyncVLA/blob/b593ac5143287a62893159a42c88a471bf444f10/prismatic/vla/datasets/gnm_dataset.py#L648-L741)。发布为简化代码，完整部署与精确完成预算未在本审计复现。

### 论文补充：冻结 Slow 也可能成功，Slow 自身的轨迹 Head 不必胜过 Fast

《Think at 5 Hz, Act at 20 Hz》描述冻结任务预训练 LMDrive、训练逐层 KV action expert。它是冻结任务化 Slow 的设计先例，不应仅凭 Slow 自身的 Head 较弱就排除此路线。该论文采用约 27K clips、五个 epoch 的专家训练；但其开环专家还输入了原骨干 Head 没有使用的 teacher-forced 历史 waypoints，因此所报误差差距不能单独归因 KV 或作为公平读出对照。我们不会复制可能违反本项目未来输入约束的监督字段。本次未找到并核对与之对应的完整官方实现，故只标记为论文证据。[方法与训练说明](https://arxiv.org/html/2607.15621v1#S3.SS1)，[实验与开环限制](https://arxiv.org/html/2607.15621v1#S4.SS2)。

## 2. 对本项目实际训练路径的核对

审计本地代码 revision `f0a86cb4547c5c193a6804917f12f914c1f1eb60`，保持数字与前次报告一致。

| 实际阶段 | 参数与更新 | Slow 实际得到的监督 | Fast loss 是否持续更新 Slow |
| --- | --- | --- | --- |
| Fast 预热 | Fast 80 步 | 无 Slow forward | 否 |
| Slow 新模块预热 | 20 步，语言参数冻结、输入梯度保留 | Slow 自己的轨迹 mean L1 | 否 |
| Slow LoRA | 40 步，rank16，视觉及 Merger 冻结 | Slow 自己的轨迹 mean L1 | 否 |
| Fast 消费 | 120 步，36 absent /48 fresh /36 stale | Slow 冻结、缓存 detach | **否** |
| P0 连通性审计 | 一条样本、一个联合更新 | Fast-only loss 可回到六层 K/V 与 Slow LoRA | 仅证明这一步可通；不是持续 Stage D |

[阶段训练代码](https://github.com/merryQiao/fast-only/blob/f0a86cb4547c5c193a6804917f12f914c1f1eb60/tools/validate_slowfast_v11.py#L100-L140)，[单项 loss](https://github.com/merryQiao/fast-only/blob/f0a86cb4547c5c193a6804917f12f914c1f1eb60/slowfast_v11/validation.py#L32-L56)，[P0 联合审计](https://github.com/merryQiao/fast-only/blob/f0a86cb4547c5c193a6804917f12f914c1f1eb60/tools/slowfast_v11_audit.py#L120-L147)。

冻结缓存是这轮明确选择的 Stage C，**没有发现一个违背该配置的意外 detach bug**。但它有实际后果：这 120 步完全不能教 Slow“Fast 需要怎样的 K/V”。我们给通用 Qwen 新增规划接口，只做很短的任务适配后就冻结；这不具备成熟机器人/驾驶任务底座的前提。

以下差距均已由源码/已有记录核实：

- 没有持续联合阶段 D；不能用 P0 非零梯度替代训练。
- 尚未实现 Future、运动分类或 preserve 目标；当前只有各自的轨迹 L1。它们是完整 V1.1 的待办，但不是“所有 KV 模型都必须有 Future loss”的证据。
- Slow LoRA 只有 40 次单窗口更新；新模块 60 次。验证还没有建立其任务表征已稳定的证据。
- Fast 消费参数以 rank16 LoRA 和新模块为主；参考系统的动作专家常是完整可训练模块或已经共同预训练的消费者，更新自由度不同。
- Slow 反馈固定来自 80 步 Fast，其 VAL ADE 为 5.912m；本轮没有更新反馈生产者版本，也未训练缺失反馈分支。这是受控诊断选择，不是完整生命周期训练。
- Fast 自己有四张图像、完整 0.8B 和 Head，存在靠自身降低 loss 的路径。正确/错配 KV 的 VAL 轨迹差异 0.01135m，而 absent/fresh 差异 0.23762m，提示应区分内容消费与存在标记/metadata 的影响；这仍是推断，尚无 Attention 层面的因果测量。
- 12 TRAIN /4 VAL、唯一移动 VAL 速度15.08m/s高于 TRAIN 最大8.97m/s，使当前分数不能定位训练问题的唯一成因。

## 3. 据此调整优先级

**保持 Qwen3.5-2B /0.8B、六层原生 KV、单轨迹 Head，不先加所谓翻译层或换架构。** 下一轮应补训练协议，而不是重复同一冷启动冻结缓存流程后继续解释一个 ADE。

1. **先验证 Slow 的任务化学习。** 保留连续轨迹监督，在移动/交互片段检查学习曲线；不能设定“Slow 自身 Head 必须超过 Fast”这个无依据门槛，也不能仅凭模型更大就认定它已提供有用条件。
2. **执行持续 live-KV Stage D。** 不用 `inference_copy` 训练联合阶段；保持 Fast loss 到 Slow KV 的图，并保留 Slow 监督。每次阶段检查应单独验证 Fast loss 对 Slow 的梯度，防止 Slow 的梯度其实全部来自自身 loss。P0 已验证原始分辨率 MB1 联合反向能在单 H800 上运行。
3. **先诊断新鲜条件的消费，再扩大年龄分布。** 对同 checkpoint、同 moving target 比较 fresh/absent/shuffled；记录无 KV 能力是否退化。只有输出变化而没有任务改善，不算 KV 有用。
4. **保留消费者充分适配的机会。** 如果 B/D 仍无效，再单独考察六层消费参数开放程度、位置约定和读取分布；这些都要成为独立配置/消融，不在本次静默改变。
5. **再接 Future/刷新机制并做训练监督消融。** Future 不应成为掩盖核心接口没有学会消费的额外变量；同时也不能把已有的动作-only小预算结果当成对完整 V1.1 的评价。

本次只完成参考代码与训练路径分析，没有启动新训练或变更模型行为。后续的新配置、预算、反馈版本与 loss 权重应在运行前固定，并继续遵守不下载全量数据、TRAIN/VAL 分离、未来标签只用于监督的约束。
