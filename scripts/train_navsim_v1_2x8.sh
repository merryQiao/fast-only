#!/usr/bin/env bash
# Run the SAME command on both nodes after exporting local base/data roots.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MASTER_ADDR:?Set node 0 hostname/IP}"
: "${NODE_RANK:?Set 0 on first node, 1 on second node}"
: "${FAST_BASE_MODEL:?Local Qwen3.5-0.8B directory}"
: "${NAVSIM_MANIFEST_DIR:?Prepared full official V1 bundle directory}"
: "${NAVSIM_SENSOR_ROOT:?OpenScene trainval/navtrain sensor root}"
: "${TRAIN_OUTPUT_ROOT:?Shared writable output directory}"
if [[ "$NODE_RANK" != 0 && "$NODE_RANK" != 1 ]]; then
  echo 'NODE_RANK must be 0 or 1' >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FLA_TILELANG=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
CLUSTER_PYTHON="${CLUSTER_PYTHON:-python}"
CLUSTER_MASTER_PORT="${MASTER_PORT:-29500}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-configs/navsim_v1_split_history_h200_2x8.yaml}"
configured_arms="$("$CLUSTER_PYTHON" -c 'import sys; from fast_cluster.config import read_config; print(" ".join(read_config(sys.argv[1])["train"]["arms"]))' "$CLUSTER_CONFIG")"
read -r -a training_arms <<< "$configured_arms"
# The selected config controls all three new arms, the original Fast-only
# baseline, or explicit legacy experiments. read_config resolves only its assets.
for train_seed in 42 43; do
  for train_arm in "${training_arms[@]}"; do
    run_output="$TRAIN_OUTPUT_ROOT/seed_${train_seed}/${train_arm}"
    if [[ "${RESUME_MATRIX:-0}" == 1 && -f "$run_output/result.json" ]]; then
      completed="$("$CLUSTER_PYTHON" -c 'import sys; from fast_cluster.config import read_config,completed_run_matches; print(completed_run_matches(read_config(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4])))' "$CLUSTER_CONFIG" "$run_output" "$train_arm" "$train_seed")"
      if [[ "$completed" == True ]]; then
        echo "Already completed: $run_output"
        continue
      fi
    fi
    launch_args=(--config "$CLUSTER_CONFIG" --arm "$train_arm" --seed "$train_seed" --output "$run_output")
    # Explicit opt-in to resume an interrupted matrix. Never silently start over.
    if [[ "${RESUME_MATRIX:-0}" == 1 && -f "$run_output/latest.pt" ]]; then
      launch_args+=(--resume "$run_output/latest.pt")
    fi
    "$CLUSTER_PYTHON" -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
      --master-addr="$MASTER_ADDR" --master-port="$CLUSTER_MASTER_PORT" \
      -m fast_cluster.train "${launch_args[@]}"
  done
done
