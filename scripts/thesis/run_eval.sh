#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/RLinf"
BASE_MODEL="/root/autodl-tmp/models/pi05/RLinf-Pi05-LIBERO-SFT"
RUNS_ROOT="/root/autodl-tmp/research/rlinf-thesis/runs"

CHECKPOINT="${1:-}"
NUM_ENVS="${2:-4}"
RUN_TAG="${3:-pi05_eval}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "用法："
  echo "  $0 /path/to/full_weights.pt [num_envs] [run_tag]"
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] checkpoint 不存在：$CHECKPOINT"
  exit 1
fi

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "[ERROR] 基础模型不存在：$BASE_MODEL"
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

RUN_ID="${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
nvidia-smi > "$RUN_DIR/nvidia_smi.txt"

echo "[INFO] RUN_DIR=$RUN_DIR"
echo "[INFO] CHECKPOINT=$CHECKPOINT"
echo "[INFO] NUM_ENVS=$NUM_ENVS"

bash evaluations/run_eval.sh \
  libero \
  libero_spatial_openpi_pi05_eval \
  "rollout.model.model_path=$BASE_MODEL" \
  rollout.model.add_value_head=true \
  rollout.model.openpi.value_after_vlm=true \
  "runner.ckpt_path=$CHECKPOINT" \
  "env.eval.total_num_envs=$NUM_ENVS" \
  "runner.logger.log_path=$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/console.log"

grep -nE \
'eval/(num_trajectories|success_once|success_at_end|episode_len|reward|return)' \
"$RUN_DIR/console.log" \
> "$RUN_DIR/eval_metrics.txt" || true

grep -nEi \
'IndexError|Traceback|RuntimeError|Exception occurred|RayTaskError|Unexpected key|Missing key|CUDA out of memory|worker died|Exiting main process' \
"$RUN_DIR/console.log" \
> "$RUN_DIR/serious_errors.txt" || true

echo "[OK] 评估结束：$RUN_DIR"
cat "$RUN_DIR/eval_metrics.txt"

if [[ -s "$RUN_DIR/serious_errors.txt" ]]; then
  echo "[WARN] 检测到严重错误："
  cat "$RUN_DIR/serious_errors.txt"
  exit 3
fi
