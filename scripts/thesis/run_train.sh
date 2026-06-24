#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/RLinf"
CONFIG_ROOT="$PROJECT_ROOT/examples/embodiment/config"
RUNS_ROOT="/root/autodl-tmp/research/rlinf-thesis/runs"

CONFIG_NAME="${1:-}"

if [[ -z "$CONFIG_NAME" ]]; then
  echo "用法："
  echo "  $0 config_name"
  echo
  echo "示例："
  echo "  $0 libero_spatial_ppo_openpi_pi05_5epoch_lr1e6"
  exit 2
fi

CONFIG_NAME="${CONFIG_NAME%.yaml}"
CONFIG_FILE="$CONFIG_ROOT/$CONFIG_NAME.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] 配置文件不存在：$CONFIG_FILE"
  exit 1
fi

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export OPENPI_HOME=/root/autodl-tmp/cache/openpi

"$PROJECT_ROOT/scripts/thesis/cleanup_ray.sh"

RUN_ID="train_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
RECORD_DIR="$RUNS_ROOT/$RUN_ID"
mkdir -p "$RECORD_DIR"

cp "$CONFIG_FILE" "$RECORD_DIR/training_config.yaml"
git rev-parse HEAD > "$RECORD_DIR/git_commit.txt"
git status --short > "$RECORD_DIR/git_status.txt"
nvidia-smi > "$RECORD_DIR/nvidia_smi.txt"
df -h /root/autodl-tmp > "$RECORD_DIR/disk_usage.txt"

printf '%s\n' \
  "bash examples/embodiment/run_embodiment.sh $CONFIG_NAME" \
  > "$RECORD_DIR/command.txt"

echo "[INFO] 配置：$CONFIG_FILE"
echo "[INFO] 实验记录：$RECORD_DIR"
echo "[INFO] 开始训练……"

set +e

bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" \
  2>&1 | tee "$RECORD_DIR/launcher_console.log"

train_status=${PIPESTATUS[0]}
set -e

if [[ $train_status -ne 0 ]]; then
  echo "[ERROR] 训练失败，退出码：$train_status"
  echo "[INFO] 日志：$RECORD_DIR/launcher_console.log"
  exit "$train_status"
fi

METRICS_SCRIPT="$PROJECT_ROOT/scripts/thesis/collect_metrics.py"

if [[ -f "$METRICS_SCRIPT" ]]; then
  if python "$METRICS_SCRIPT" \
    "$RECORD_DIR/launcher_console.log" \
    --output "$RECORD_DIR/summary.csv"; then
    echo "[OK] 已生成训练指标：$RECORD_DIR/summary.csv"
  else
    echo "[WARN] 训练完成，但指标提取失败。"
  fi
else
  echo "[WARN] 未找到指标提取脚本：$METRICS_SCRIPT"
fi

echo "[OK] 训练命令执行完成。"
echo "[OK] 实验记录：$RECORD_DIR"
