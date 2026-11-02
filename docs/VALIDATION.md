# 验证记录与当前边界

本页是2026-11-02源服务器实测的精选记录。发布时保持已验证源码/config/test字节不变，只整理文档和公开证据；没有重新训练，也没有在GitHub上重新执行402项测试。

结构化记录：[smoke_summary.json](../evidence/smoke_summary.json)。环境：[environment.json](../evidence/environment.json)。原测试源码hash：[original_source_sha256.json](../evidence/original_source_sha256.json)。原始归档仍保留在源服务器；本仓库不包含原始大输出或checkpoint，Git提交表示发布版本，源码hash对应原始验证版本。

## 数据与协议

- 固定NAVSIM源码 `0a380a9063d7162ec93d0f51e9990ebac585f720`，OpenScene数据revision `a76f840b65e972bc45e56c2adced897498e9a026`。
- public mini的3段独立log，先划分log再生成窗口。TRAIN12/2log/6scene，VAL4/1log/3scene；TEST0。
- 保留46张原始1920×1080前视JPEG。累计网络计费207,756,564 response-body bytes，小型发现请求含保守1MB reserve；没有下载整套数据。
- 只读取归档前缀，所以没有验证完整归档LFS checksum；保留文件和源码有单独SHA。公开仓库不再分发JPEG、metadata或上游源码快照。
- 原生命令覆盖有限：TRAIN索引{1,3}、VAL索引{1}。保留官方one-hot的数值编码，没有擅自推断语义标签；不据此声称左右转全覆盖。

## 数值证据

| 验证 | 结果 |
| --- | --- |
| 历史回归 | 402 passed，0 failure/error/skip；10条第三方弃用警告 |
| 原图MB2输出 | `[2,8,3]`，Vision+Merger冻结，各可训练分支梯度有限且非零 |
| 输入布局 | 1920×1088补边，grid `[1,68,120]`，2040 tokens/帧，序列8247 |
| 模型规模 | 总859,917,635参数；训练6,931,715；96个LoRA目标 |
| 正式策略smoke | 4步，有效batch16，完整8000步日程/240warmup；windows_seen64 |
| TRAIN pose loss | 1.687625→1.670275，只有4个warmup更新，未训练充分 |
| VAL4 | ADE8.28093m，FDE14.61028m，heading MAE.118469rad |
| 深过拟合 | 300步移动单样本，pose loss .885369→.00166931，下降99.81146% |
| 深过拟合XY | ADE.00368957m，FDE.00365451m |
| 深过拟合heading | wrapped MAE.000673048rad |
| 实际恢复2→3 | 参数、Adam状态、pose差异0；输入hash、游标、loss、scheduler/LR一致 |
| 独立进程重载 | 从本地基座+checkpoint重建，两份模型预测差异0 |

深过拟合不是从近乎静止的零GT样本中取巧：固定选择TRAIN首个**当前可观测速度≥1m/s**窗口，speed1.05168m/s，末点约(4.455m,.283m,.128rad)。只按输入速度选择，未根据未来GT误差或模型结果挑样本。

预先固定验收为pose L1≤.05、XY ADE≤.05m、heading MAE≤.02rad、loss下降≥95%，全部满足。pose L1混合m/rad分量，不能标作米。深过拟合使用MB1/累积1/关闭亮度增强及300步独立日程，但原图、导航、模型、loss和完整TRAIN Stats不变。

## 未来隔离与输入通道

真实Processor隔离测试替换未来raw pose/dynamics/command和未来JPEG，冻结TRAIN Stats与增强seed，逐字节比较native输入tensor、当前Ego与command：输入不变，只有GT改变。

VAL4的图像交换平均XY变化.074309m、heading变化.047988rad；只改`(command+1)%4`平均XY变化.236585m、heading变化.123041rad。图像交换均来自同一个VAL log，命令干预为反事实测试。这说明通道有响应，不是语义规划效果或泛化成绩。

CV仅用当前raw vx/vy外推、假设相对heading恒0：TRAIN ADE.597293m/FDE1.680179m，VAL ADE.350252m/FDE.634403m。没有用未来信息生成预测，也没有声称完成独立稳态厘米级校准。

## 显存与延迟

硬件NVIDIA H800，软件与严格确定性标志见环境JSON。MB2训练peak allocated40174.87MiB（39.23GiB），MB1深过拟合21261.67MiB（20.76GiB）。Reserved包含allocator缓存，不能等同于活跃推理显存。

各模式batch1、独立10次warmup+100次测量、GPU同步：

| 模式 | P50 ms | P95 ms | P99 ms | 超500ms |
| --- | ---: | ---: | ---: | ---: |
| Model-only | 165.31 | 166.20 | 166.60 | 0/100 |
| Streaming E2E | 433.20 | 453.71 | 464.72 | 0/100 |
| Disk E2E | 466.69 | 490.22 | 515.56 | 3/100 |

每次均完整计算Vision+Backbone+Head，无feature/KV缓存。Model-only从设备上的预处理tensor开始；streaming从已解码RGB开始；disk额外包含JPEG读取/解码。文件系统为warm OS cache，不代表冷对象存储或真实部署抖动。

## 尚未验证/实现

- 完整官方navtrain Scene Filter的数据覆盖、完整TRAIN统计量和正式模型收敛。
- 全量正式入口、epoch无放回采样、可重定位路径和多卡DDP。
- 跨GPU/CUDA/FLA后端的逐位恢复保证。
- 官方NAVSIM Agent、PDMS/EPDMS、leaderboard或闭环部署。
- 未来Fast10Hz/Slow5Hz协议。

小规模阶段已经结束，后续工程接入与目标GPU短训练按[迁移指南](FULL_DATA_TRAINING.md)在目标服务器执行。
