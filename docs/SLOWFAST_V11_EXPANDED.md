# V1.1 扩数据训练：256 TRAIN /64新VAL

本轮按用户要求扩大训练数据，保持原始图像和快慢架构。实验方案已固定，正在执行；结果将在完成真实训练和独立重载后填入，不预先假定扩大数据就能改善规划。

## 数据与划分

使用固定revision的OpenScene **mini 子集**，不是官方完整navtrain。相比原来的12个TRAIN窗口，本轮取4条日志各64个窗口，共256个；另外2条日志各32个窗口构成新VAL64。原VAL日志及其4个窗口保留为单独的legacy VAL，不进入TRAIN或新VAL。

先依据元数据固定日志、split和目标token，再下载其所需图像，缺图不会换样本。候选窗口stride5（2.5秒），历史四帧、未来4秒8点；同日志内未来轨迹仍有重叠，64个新VAL窗口不能视为64个独立环境。两个新VAL日志和原legacy日志都不参与优化、选超参数、选步数或决定续训。没有TEST参与。

TRAIN按当前速度分层选取；只用当前速度，不按未来轨迹或模型误差选样本。原定每日志低速/中速/高速名额为16/24/24，缺少的高速名额按固定规则转给其他可用分层并记录。实际TRAIN为低速64、中速141、高速51个，最高速度约13.58m/s；原来12窗口最高约8.97m/s。这里没有专门训练原VAL那条15.08m/s轨迹。

| 用途 | 相机分片 | 日志 |
| --- | --- | --- |
| TRAIN64 | 5 | 2021.06.09.14.58.55_veh-35_01894_02311 |
| TRAIN64 | 11 | 2021.06.09.14.03.17_veh-12_02584_02970 |
| TRAIN64 | 26 | 2021.08.17.16.57.11_veh-08_01200_01636 |
| TRAIN64 | 7 | 2021.06.08.12.54.54_veh-26_04262_04732 |
| 新VAL32 | 6 | 2021.10.11.02.57.41_veh-50_00352_00535 |
| 新VAL32 | 15 | 2021.05.12.23.36.44_veh-35_02035_02387 |
| legacy VAL4 | 2 | 2021.06.28.16.57.59_veh-26_00016_00484 |

原始1920×1080 JPEG保持字节内容，只在Processor中补边，不额外resize或重编码。获取所需的1293张JPEG，并保留成员SHA；不下载完整相机归档。网络响应体总预算2GB，包含元数据扩展、无关压缩成员、失败读取和重试；默认每个相机前缀最多225MB。分片7在225MB时只取全252/256张预定图片，因此明确修订为仅该分片允许256MB前缀，总预算与所有样本保持不变。元数据只扩展到120MB前缀，完整元数据归档约534MB。父归档LFS身份用于来源记录，不声称部分前缀验证了完整归档hash。

Ego `[vx,vy,ax,ay]` 的均值/标准差只从本轮TRAIN256重算。原生四维one-hot command的类别索引、后轴SE(2)坐标、真实2Hz时间戳和未来4秒8×3标签保持。legacy VAL也使用新的TRAIN统计量，不能把新旧检查点的结果差直接归因为数据量。

## 模型与训练

完整Qwen3.5-0.8B Fast和2B Slow，视觉塔及Merger冻结，语言基座冻结，训练LoRA和新增模块。六层KV接口、8个Planning Queries、唯一Fast轨迹Head保持。此次只训练Slow轨迹Head，没有Future/高层运动类别Head、异步调度或闭环。

重新从固定预训练资产和seed42初始化，**不继承12窗口上的过拟合权重**。

| 阶段 | 更新次数 | 内容 |
| --- | ---: | --- |
| Fast预热 | 512 | 无KV；随后生成固定的、已完成的数值轨迹反馈 |
| Slow任务适配 | 512 | 自身轨迹loss，读取上述固定反馈 |
| 无KV组 | 800 | Fast始终无KV；Slow每步仍接受自身监督 |
| 联合组 | 800 | 560次fresh live-KV、240次无KV；Fast损失可经KV回到Slow |

两个组从同一共享Fast/Slow状态开始，重置各自优化器和日程，目标顺序相同。目标为`L_F+L_S`，仍用mean L1 pose；损失混合m/rad，ADE/FDE只算XY。先完成两损失的反向，再分别clip1和更新。MB1、无累积、无增广/dropout，严格确定性配置。

每阶段按epoch shuffle、同epoch无放回；800步为三个完整256窗口epoch加32个窗口。每个最终模型累计512+800=1312次样本暴露，约5.125个TRAIN等效epoch。全实验2624次新增训练迭代、4224次optimizer.step，不能等同于相同FLOPs或已充分收敛。

AdamW峰值LR为LoRA1e-4、新模块5e-4，WD0.01；各阶段前32步线性warmup，随后cosine降到峰值10%。保存实际使用的LR、日程进度、两优化器、RNG和采样位置。新旧实验同时改变数据、初始化、预算和学习率日程，因此不是严格的单变量“数据量消融”。本轮没有独立detached组，也不单独估计回传梯度的增量贡献。

TRAIN曲线只在固定32个速度/日志分层诊断窗口上评测，位置为200/400/800步；不是整个TRAIN256。新VAL64和legacy VAL4只在每组800步完成后评测，分别报告；KV关闭/正确/错配共享同一检查点。恒速度基线按对应相同窗口集合计算。

## 工程和验证

- [expanded_data.py](../slowfast_v11/expanded_data.py)与[数据入口](../tools/prepare_slowfast_expanded.py)：元数据先行冻结、严格有界原图获取、成员hash及TRAIN统计量。
- [expanded_protocol.py](../slowfast_v11/expanded_protocol.py)：epoch采样、240/560独立模式洗牌、可恢复warmup/cosine、cohort内多项式错配。
- [训练入口](../tools/train_slowfast_expanded.py)：512/512预热和两路800步；评测KV快照LRU容量2，避免全源导出数十GB缓存。
- [重载入口](../tools/reload_slowfast_expanded.py)：新进程重建完整新/旧VAL条件，实际重放一次联合更新并比较参数、优化器、scheduler、RNG及预测。
- [报告入口](../tools/report_slowfast_expanded.py)：导出可迁移数值、按日志/三个速度区间分层、梯度裁剪、曲线和来源摘要，不提交JPEG或权重。

读取KV的四个预定步骤单独检查Fast-loss-only梯度，要求到达六层Slow K/V和十二个对应LoRA-B参数；无KV对照没有这条梯度。视觉塔/Merger应保持冻结。本轮验收只针对单张H800及当前依赖，不外推为多GPU、异步交付、完整resume或闭环成功。

## 执行

配置为[slowfast_v11_expanded.yaml](../configs/slowfast_v11_expanded.yaml)，实施清单见[预定计划](superpowers/plans/2026-11-06-expanded-data.md)。先准备带身份的元数据/来源快照，再运行：

```bash
python -m tools.prepare_slowfast_expanded --plan-only \
  --legacy-manifest /path/to/original/val.jsonl \
  --legacy-sensor-root /path/to/original/sensor_blobs/mini \
  --upstream-audit /path/to/original/source_audit.json

python -m tools.prepare_slowfast_expanded --acquire --archive-cap-override 7:256000000 \
  --legacy-manifest /path/to/original/val.jsonl \
  --legacy-sensor-root /path/to/original/sensor_blobs/mini \
  --upstream-audit /path/to/original/source_audit.json

export FLA_TILELANG=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_HOME=/path/to/matching/cuda-toolkit
export PATH="$CUDA_HOME/bin:$PATH"

python -m tools.train_slowfast_expanded --output outputs/slowfast_v11_expanded/run_new
python -m tools.reload_slowfast_expanded \
  --run outputs/slowfast_v11_expanded/run_new \
  --output outputs/slowfast_v11_expanded/run_new/reload
```

数据准备入口的旧mini默认路径只对应当前服务器，迁移时必须显式设置上述来源路径；元数据、原图、模型、旧VAL清单和训练checkpoint不会上传GitHub。本入口仍限制固定的小规模协议，不能作为全量navtrain或DDP训练入口。

## 结果

执行中。真实训练、完整分层评测和独立重载完成后更新本节。
