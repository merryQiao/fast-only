# 代码逻辑与协议

本说明描述当前已验证实现，不表示全量或多卡训练入口已完成。主入口为 `python -m tools.run_navsim_preflight`，模型与训练代码分别在 `fast_navsim/model.py` 和 `fast_navsim/train.py`。

## 1. 数据如何变成样本

`tools/prepare_navsim_smoke.py` 使用固定 OpenScene revision、HTTP Range 和累计字节预算获取少量 metadata/front JPEG，先划分整段 log，再取窗口。当前只实现 bounded mini；`--acquire` 才访问网络，默认只处理已经保留的文件。

`fast_navsim/data.py` 的 `build_window()` 读取：

- 历史图像：当前及之前3帧，只用 `CAM_F0`。
- 当前动态状态：官方 `ego_dynamic_state=[vx,vy,ax,ay]`。
- 当前导航：官方四维 one-hot `driving_command`，校验后保留原生 argmax 类别0…3。
- 当前/未来全局 Ego pose：仅用于生成8个监督标签，不进入模型特征。

时间读取原始微秒 timestamp；当前容许相邻帧500000±1000µs。这个阈值在3段日志上核验过，不能假设全量日志都会通过。未来图像不需要为监督而下载。

GT 用当前后轴 pose `(x0,y0,θ0)` 作为原点：

```text
dx = x_future - x0
dy = y_future - y0
x_local = cos(θ0)*dx + sin(θ0)*dy
y_local = -sin(θ0)*dx + cos(θ0)*dy
heading_local = atan2(sin(θ_future-θ0), cos(θ_future-θ0))
```

这里是右手系x前/y左，不翻转CARLA横向轴，也不套用GNSS外参。Quaternion yaw 按官方 PyQuaternion `yaw_pitch_roll[0]`，非零roll/pitch时也必须一致；测试覆盖了这一差异和角度跨±π情况。

`compute_ego_stats()` 只接收TRAIN行，使用总体标准差，常量维std=1。`NavsimDataset` 校验清单和统计量，读取四张原图，返回标准化Ego、整数命令、`trajectory_gt[8,3]`及元数据。深过拟合固定选TRAIN中首个当前速度≥1m/s的样本，选择不读未来GT或模型结果。

## 2. 原图与时间文本

`fast_navsim/collate.py`：

1. 训练时四帧共享一次亮度增强因子；评测关闭增强。
2. 保留完整空间分辨率，只在右/下补到原生patch×merge倍数32。
3. 原生Processor接收四张独立图像，`do_resize=False`；返回grid必须与补边尺寸相符，否则报错。
4. 图像前的固定文本时间为 `-1.5,-1.0,-0.5,+0.0s`，任务文本要求8个未来pose。
5. 仅stack当前Ego/command和监督GT。共享benchmark接口中的 `nav=None` 只是调用约定，不生成target向量或token。

实际1920×1080补到1920×1088，grid `[1,68,120]`，每帧2040视觉tokens。原始JPEG已有的压缩无法逆转；代码不再编码或降采样JPEG。

## 3. 原生模型与训练参数

`NavsimPlanner` 继承 `fast_only/model.py` 的已验证骨架：

- 使用真正的 `Qwen3_5Model` 原生Vision Encoder、Merger和language model。
- Vision+Merger冻结并保持eval；基础语言权重冻结。
- LoRA只注入语言Attention/FFN中核对过的投影，不改Gated DeltaNet递归/衰减门控；验证时共96个目标模块。
- 冻结基座BF16；LoRA和新增模块FP32，r16/alpha32/dropout.05。
- 删除继承骨架中B2D的 `nav_mlp`，不留下假导航参数。

Forward先用原生图像特征替换Processor的image placeholders，保留native mRoPE；随后追加：

```text
native images + time text → ego token → command token → query[0…7]
```

Ego通过MLP得到token；命令通过四类embedding得到token。8个可学习queries延续合法的3轴位置，使用原生因果mask，`use_cache=False`。一次language model forward后，取最后8个hidden state，经共享 `LayerNorm → Linear → GELU → Linear(3)` 输出 `[B,8,3]`。

不使用文本generate、额外视觉编码器、Slow条件、Cross-Attention或辅助规划损失。

## 4. Loss、采样和优化

`pose_loss()` 对齐官方EgoStatusMLP的 `functional.l1_loss`：batch、8个pose、3个分量上无权重取均值。训练loss不做只对heading的特殊wrap替换；heading指标则计算wrapped差值。ADE/FDE/L2只用xy。

`train.py` 创建AdamW两个参数组：LoRA LR1e−4、新增模块LR5e−4，weight_decay.01；cosine调度、3%warmup、梯度裁剪1。

当前使用 `fast_only/loading.py` 的有放回counter sampler。索引与增强seed仅依赖seed/batch counter，有界CPU线程预取，不在worker里创建CUDA tensor。checkpoint保存已消费游标，未消费future不会推进恢复位置。

当前 `effective_batch=microbatch×accumulation` 是单进程定义，**不包含world_size**。正式多卡接入时必须同时修改采样、累积、梯度同步和全局计数。

## 5. 检查点与验证

`runtime.py` 保存训练参数、Adam、scheduler、Python/NumPy/Torch CPU/CUDA RNG、sampler游标、Stats和身份信息，位于optimizer update边界。BF16未使用GradScaler，记录为None。

身份覆盖两套源码、准备/预检/重载工具、模型/Processor文件、清单、Stats、保留来源及全部TRAIN/VAL原图。依赖版本/后端标志改变会被拒绝。严格续训同时要求配置一致。

`tools/run_navsim_preflight.py` 的真实恢复检查不是单纯反序列化：恢复step2，执行真实带dropout/预取的step3，与不中断step3逐项比较参数、Adam、输入hash、loss、scheduler/LR、采样游标和pose。另一个进程通过 `tools/verify_navsim_reload.py` 从基座+checkpoint重建预测。

`evaluate.py` 提供XY/heading指标、分层可视化、当前vx/vy的CV参考和VAL通道干预。所有干预都是路径响应体检，不是语义效果消融或官方PDM评分。

## 文件映射

| 文件 | 职责 |
| --- | --- |
| `fast_navsim/data.py` | rear-axle SE(2)、窗口、Stats、Dataset、深过拟合样本选择 |
| `fast_navsim/acquire.py` | 有预算的HTTP/Range与部分归档读取 |
| `fast_navsim/collate.py` | 原图补边、原生Processor、2Hz时间文本 |
| `fast_navsim/model.py` | 8-query/no-target模型、官方pose loss、XY/heading指标 |
| `fast_navsim/config.py` | 显式策略Schema与mini预算护栏 |
| `fast_navsim/train.py` | 单进程训练、P0梯度/显存、固定VAL、保存/恢复 |
| `fast_navsim/runtime.py` | 身份校验、环境版本、完整checkpoint状态 |
| `fast_navsim/evaluate.py` | 预测导出、CV、图像/命令干预、分层图示 |
| `tools/run_navsim_preflight.py` | 4步同配置+300步移动单样本+真实恢复验证 |
| `tools/verify_navsim_reload.py` | 独立进程重建与预测一致性验证 |

官方代码参考：[dataclasses](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/common/dataclasses.py)、[baseline](https://github.com/autonomousvision/navsim/blob/0a380a9063d7162ec93d0f51e9990ebac585f720/navsim/agents/ego_status_mlp_agent.py)。
