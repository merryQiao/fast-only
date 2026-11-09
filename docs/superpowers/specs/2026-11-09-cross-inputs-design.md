# 冻结 Fast 图像 × Slow KV 交叉干预

任务来自用户给定设计：保持两模型和 Head 冻结，先测独立 Fast-only 的整段四帧换图，再测联合模型的正确／错配视觉 × 正确／错配 KV。当前 Ego、官方导航、GT、metadata、位置与执行路径固定；不训练，不下载，不更换 Head，不增加 Reader／门控，不继续深挖 attention 数值。

## 固定来源和组合

使用已有完整语言 Fast-only 与正确 KV 两个终点，SHA 写入 `configs/slowfast_cross_inputs.yaml`。使用原 VAL256／8 日志，只作已经观察日志上的机制诊断。按排序后的 recording group 循环偏移 `[1,4,7]`，在每组沿用 canonical manifest 内索引建立三个无放回跨日志映射；每个映射均为双射，无 GT／预测选择。偏移 4 必须逐项复现旧错配映射。

Fast-only 每窗口运行当前图像及三个 donor 四帧，共 1024 次。联合模型运行四种图像来源 × 四种 KV 来源的完整 16 格，共 4096 次。汇总 A=图像正确/KV正确，B=图像正确/KV错误（3），C=图像错误/KV正确（3），D_same=同一错误图像/KV（3），D_independent=不同错误图像/KV（6）。D 的两个部分分开报告，避免混入错误输入相互一致性。

整段四帧保持 donor 自己的相对时间顺序；仅交换冻结视觉编码器及 Merger 的输出。四图 token／mask／grid／dtype／shape 必须完全相容；当前 native 容器、Ego、command、metadata 不可随 donor 一起替换。任何不相容都失败，不 resize、截断或悄悄 repack。逐窗口记录原图来源 token、KV 来源 token、跨日志 offset 和视觉特征差异。

## 真实输入与不可变性验收

每个 recording group 取 manifest 第一窗口为原图核对目标，共 8 个，不看 GT。两检查点各对当前与三个 donor 四帧执行原 Processor＋原生视觉塔＋语言完整前向，保留目标 Ego／导航；32 次/模型，共 64 次，预测必须与视觉缓存替换精确相同。原图不 resize／重压缩；读取本地既有图像不产生新增 HTTP 流量。

Fast-only 当前图像 256 个预测、联合 A 的 256 个预测、B 的 offset4 共 256 个预测必须精确复现此前终点或来源诊断。完整参数 SHA／版本前后相同；输入、视觉特征和 KV 版本不变；全程 eval/no_grad/use_cache=False。联合全部格子均启用同样的六层融合与 metadata `[0,0,0,0,1,1]`，不通过 off/masked 方式代替错配。

## 汇总和结论边界

报告 ADE、FDE、逐时距误差、航向及 XY 预测变化，逐日志、逐 donor、当前速度档和导航类分别给结果。重复 donor 先在目标窗口内平均，再按日志配对，不能把 donor 重复当成独立样本。日志 bootstrap 固定 20,000 次、种子 20261109；移动子集预设当前速度≥1m/s，另报≥10m/s，记录组权重变化明确保留。

重点配对量：Fast-only 换图损伤；联合 B−A 的正确 KV 收益；C−A 的换图损伤；D−C 的正确 KV 补偿；`(D_same−C)−(B−A)` 的 ADE 交互量。正的补偿量表示错误图像下正确 KV 降低误差。20mm 仅为沿用的实用参考尺度，不将不显著自动解释为等效，也不强行套用三种理想模式。

错配可能构成 OOD。Slow KV 同时包含图像、Ego、导航和旧反馈信息，所以补偿也不能直接归为独立视觉语义收益。两个模型检查点不同，Fast-only 与联合的绝对分数差不直接归因于某一路。固定日志不支持真实闭环效果、在线补偿可靠性或状态捷径的确证；仅报告所测输入干预的依赖与交互。

## 实现和交付

新增 `slowfast_v11/cross_inputs.py`（donor 与仅视觉替换语义）、`tools/diagnose_cross_inputs.py`（固定协议／严格加载／原图核对／推理）、`tools/report_cross_inputs.py`（窗口和日志配对统计／图表），以及有意义 CPU 测试。保留全部旧模型前向和训练代码。交付中文报告、原始预测、逐日志表、四格图、输入来源与 hash、独立 NumPy 重算和轻量证据；原图／模型／KV 不入 Git。
