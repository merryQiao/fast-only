# 冻结检查点的六层 Direct-KV 读取定位

用户最新修正优先于此前 Slow 适配建议：保持 2B、0.8B 及所有 Head 参数冻结，不训练 Slow，不增加 Reader、残差、门控或数据下载。Slow 自身轨迹 ADE 不是 KV 表征质量的代理。已有预测的互补性分析仅作辅助诊断，不作为重新训练 Slow 的门槛。

## 固定对象与问题

使用 `outputs/slowfast_direct_full/run_01/full_correct_kv/final.pt` 及其已验证原图视觉缓存／六层原生 Slow KV；使用现有 VAL256／8 组日志，全部结果为已观察 VAL 的离线定位，不作新的泛化确认。当前输入、元数据 `[0,0,0,0,1,1]`、位置、mask、权重及计算精度保持一致。

先回答三个分开的问题：源内容差异多大；固定当前 Q 与 Fast K/V 时，外部内容差异能否改变读取结果；读取差异经 sigmoid gate、o_proj、残差与后续 24 层和原 Head 后怎样变化。自然换源的后层差异包含先前层传播，不能冒充固定 Q 的局部读取差异。

## 观测和干预

1. 对全部 256 个窗口运行 correct、TRAIN 最小 token reference、原诊断 donor 映射 shuffled 三种推理。每种预测逐项精确复现既有来源诊断。记录六层实际 gate 前读取、gate 后、o_proj 后，以及 24 层 Planning 行和最终 Head 输入的变化。参数版本和完整参数 digest 不变。
2. 只在 correct 前向内增加只读诊断计算：固定本层 Q、Fast K/V、gate 和位置，分别代入正确／reference／shuffled 的外部 KV；另外 K-only／V-only 换源、V=0 与外部列 masked 作为局部区分。返回原 SDPA 结果本身，诊断计算不得替换生产输出。
3. FP32 reference softmax 分解外部／自身贡献，按源四帧视觉、native-other、Ego、导航、feedback、Planning token 报 attention mass 和 contribution；报告参考重建与真实 BF16 SDPA 的误差。未旋转 K 的源差异与 RoPE 后的读取差异分开记录。
4. 从每个记录组固定选 8 个窗口，共 64 个，使用 command／speed 分层和 token 顺序，不看 GT 或模型误差。分别只在一个层替换 reference／shuffled KV，其他五层仍正确；另分别只屏蔽一个层的外部 attention 列。报告轨迹变化与配对 ADE，不能把 mask 消融解释为语义信息收益。全部位置与元数据相同。
5. 当前 Context 行始终不读取外部 KV；对三种自然换源逐层机械核对 Context 输出逐位相同。mask 采用 attention 列屏蔽，不能用 V=0 模拟。异常／退出后所有临时 hooks 和函数 patch 恢复。

gate 与 o_proj 的绝对范数可能因维度／参数尺度自然变化，报告 RMS、参考 RMS、relative L2、cosine、相等比例和逐窗口分布。不能只看范数大小就断言信息被删除，也不能用 attention mass 直接证明对规划有用。单层干预提供局部因果证据；训练失败原因仍可能不能唯一识别。

## 代码与验收

新增 scoped inference helper `slowfast_v11/read_trace.py`、CPU 语义测试 `tests_slowfast/test_read_trace.py`、独立工具 `tools/trace_direct_kv.py`；复用已有原生 bridge、`trace_fast`、`DirectCache` 和严格 checkpoint restore，不修改模型／训练前向。

先做 meaningful CPU 测试：软最大概率分解重建、正确与 donor 完全相同时局部差异为零、V=0 与 masked 分母区别、sigmoid gate 与含 bias 投影的分解、只换一个层、只读 probe 精确返回原结果及异常清理。再进行真实 GPU 预测复现／全部六层覆盖、上下文隔离、完整参数不变验收。

交付中文报告、六层曲线与源类型分解、64 窗口单层干预汇总、完整 provenance／源 hash／协议及可重算 JSON。完整激活／原图／KV／权重不进 Git。新增训练更新与下载字节均为零；报告不得暗示新日志验证或闭环结论。
