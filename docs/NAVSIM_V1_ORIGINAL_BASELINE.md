# 从原始预训练基座训练Fast-only

这是独立的**四帧 Fast-only 控制协议**。当前默认三臂、Fast 单帧／Slow 四帧的训练请见 [新迁移指南](NAVSIM_V1_SPLIT_HISTORY.md)。以下命令显式设置 `CLUSTER_CONFIG`，继续可用。

本入口直接加载原始Qwen3.5-0.8B基座，按固定种子初始化现有Ego/导航编码器、Planning Queries和唯一轨迹Head，再使用完整NAVSIM V1 TRAIN训练。**不需要`source_bundle.pt`、2B模型或任何旧Mini任务检查点。**

Fast完整语言参数与现有LoRA/规划模块更新，视觉塔含Merger冻结；四帧2Hz原图、原生pad32、四维Ego、官方四类导航、未来4秒8×3轨迹和官方pose L1不变。正式种子42/43，各8epoch，16卡MB1/累积2。TRAIN归一化统计量来自这次完整TRAIN。

## 数据

使用官方默认TRAIN978日志／VAL214日志。新初始化没有旧任务模型的训练暴露，因此不沿用旧冻结来源配方排除的6条VAL日志。数据准备明确选择完整官方VAL：

```bash
python -m tools.prepare_cluster_data official \
  --metadata-root /datasets/openscene/trainval_logs \
  --sensor-root /datasets/openscene/sensor_blobs/trainval \
  --output /datasets/prepared/navsim_v1_original \
  --official-default-val
```

如果之前生成的是旧208日志选择VAL的数据包，需使用上面的协议另建数据包，保留旧结果。新入口会检查`val.jsonl`与保存的完整`val_official.jsonl`字节身份相同，拒绝把旧排除规则静默带入原始baseline。

## 目标集群

先`git pull`更新`feature/navsim-v1-cluster`，在两个节点设置相同数据/模型根目录及输出目录；NODE_RANK分别为0和1：

```bash
export FAST_BASE_MODEL=/models/Qwen3.5-0.8B
export NAVSIM_MANIFEST_DIR=/datasets/prepared/navsim_v1_original
export NAVSIM_SENSOR_ROOT=/datasets/openscene/sensor_blobs/trainval
export TRAIN_OUTPUT_ROOT=/runs/navsim_v1_original
export MASTER_ADDR=<node0地址>
export MASTER_PORT=29500
export NODE_RANK=0
export CLUSTER_PYTHON=/path/to/environment/bin/python
export CLUSTER_CONFIG=configs/navsim_v1_baseline_h200_2x8.yaml
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLA_TILELANG=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

身份预检只读取Fast基座和准备好的数据：

```bash
python -m tools.preflight_cluster --config "$CLUSTER_CONFIG" \
  --output /runs/navsim_v1_original/preflight_identity.json
```

在两个节点分别运行短训，验证实际16卡通信与完整分辨率更新：

```bash
python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m fast_cluster.train --config "$CLUSTER_CONFIG" --arm fast_only --seed 42 \
  --output /runs/navsim_v1_original/preflight --preflight-updates 2
```

用相同命令增加`--resume /runs/navsim_v1_original/preflight/latest.pt`并将预检上限改成3，可检查真实恢复；预检保留完整8epoch学习率日程，部分VAL不参与模型选择。

目标机预检通过后启动两个种子的正式训练：

```bash
bash scripts/train_navsim_v1_2x8.sh
```

启动脚本按 `CLUSTER_CONFIG` 中的训练臂运行；本页须保留上面显式的 baseline 配置，脚本默认已更新为新的三臂配置。中断恢复用`RESUME_MATRIX=1`；新训练不用`--resume`，只加载预训练基座和新初始化模块。旧协议的检查点不能静默恢复到本协议。

## 本机验证范围

`configs/navsim_v1_baseline_smoke.yaml`只包含0.8B和8 TRAIN／4 VAL fixture，未配置任何Slow或source bundle。已经在单H800完成一次真实累积更新：24个语言层Dense梯度有限非零，冻结视觉参数与buffer不变；初始化摘要、第一步loss与此前相同Fast-only路径逐位一致。这是依赖修复与训练链路验证，不是收敛或质量成绩。16 H200/NCCL仍须目标机预检。

原来的`configs/navsim_v1_h200_2x8.yaml`保留为显式旧冻结来源的双臂复现配置，仍需其原bundle及来源相关VAL规则。它与本入口是不同的实验协议，不能混用结果或恢复状态。后续新 Slow 的初始化和训练范围已经在[单帧／四帧协议](NAVSIM_V1_SPLIT_HISTORY.md)中明确；本页不改变旧协议身份。
