# Fast＋固定特征任务适配器

固定来源训练臂是本项目后续研究的主要适配基线。输入是当前 Fast 的原始四帧图像、自车状态和导航；输出仍为未来4秒、每0.5秒一个点的8×3轨迹。它保存固定参考特征R0，无需逐窗口执行Slow或Reader。

计算为`原Head(F + fusion(F, R0))`。Fast及原Head的检查点、输入归一化、融合权重和R0是一组配套资产。固定特征可能带有此前Slow/Reader训练形成的先验；本基线消除的是在线样本相关Slow计算，没有消除历史训练成本。

## 导出与加载

本次固定预算实验完成后，每个融合种子导出一个包：

```text
outputs/slowfast_adapter_study/run_01/seed_42/fixed_adapter.pt
outputs/slowfast_adapter_study/run_01/seed_43/fixed_adapter.pt
outputs/slowfast_adapter_study/run_01/seed_44/fixed_adapter.pt
```

包包含冻结融合权重、R0[1,8,768]、融合结构和对应Fast/Head身份。包SHA及来源训练检查点记录于`run_01/result.json`的`fixed_bundles`。权重留在本地outputs，Git仅保存源码和轻量证据。

在已按项目协议加载的Fast上使用：

```python
import json
import torch
from slowfast_v11.fixed_adapter import load_fixed_adapter, bind_fixed_adapter

result = json.load(open("outputs/slowfast_adapter_study/run_01/result.json"))
item = result["fixed_bundles"]["42"]  # 预先指定示例种子；主实验汇报全部三个种子
adapter = load_fixed_adapter(
    item["path"],
    expected_sha256=item["sha256"],
    fast_checkpoint_sha256=result["protocol"]["config"]["fast_checkpoint_sha256"],
    head=fast.head,
    map_location="cuda:0",
)
fast.eval()
with torch.no_grad(), bind_fixed_adapter(fast.head, adapter):
    trajectory = fast(native=native, ego=ego, command=command)
```

`fast`使用配套完整Fast基座＋已保存LoRA/新增模块，模型版本和Processor由源实验资产固定。上述调用使用原生无外部KV路径，Adapter只在原Head入口生效；原Head每次前向执行一次。`bind_fixed_adapter(..., enabled=False)`直接恢复原Fast路径。

## 独立进程核验

```bash
python -m tools.verify_fixed_adapter \
  --run outputs/slowfast_adapter_study/run_01 \
  --output outputs/slowfast_adapter_study/deployment.json
```

该入口仅实例化一个Fast，重建预先指定TRAIN和新留出样本的原始JPEG/Processor输入，逐个加载三个固定包，与相应固定来源训练臂的缓存预测比较。它禁止Slow构造，并核对Reader模块没有被导入。已有输出文件会被保留，应为新的核验使用新输出路径。

## 与真实Slow信息比较

正式主比较为`正确来源训练/correct推理`对`固定来源训练/reference推理`；两者配对初始化、数据顺序和新增预算。原Fast/off保留为次要参照。当前样本来源的固定/错配替换属于同一正确来源检查点的干预，不能替代单独训练的固定来源基线。

本次包来自Mini TRAIN256和固定模型，只用于该机制验证及配套推理。迁移到全量训练时，应按完整TRAIN重新训练Fast/适配器、生成配套统计量并重新导出包；本次没有验证全量DDP训练或异步快慢系统。
