#!/usr/bin/env bash
# Libero LEWM 嵌入评测 — 单节点、3 卡并行；逻辑自包含（不调用 run_eval_libero.sh）。
# 提交：sbatch batch_run_eval_libero.sh  [-- 追加 Hydra 覆盖，三路任务均会收到]
#SBATCH --job-name=lewm_eval_libero
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --account=peilab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=144G
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/qjl/2026/wmrl/lewm/eval/logs/slurm_eval_libero_%j.out
#SBATCH --error=/project/peilab/qjl/2026/wmrl/lewm/eval/logs/slurm_eval_libero_%j.err

set -euo pipefail

FORWARD_ARGS=("$@")

WMRL_ROOT="/project/peilab/qjl/2026/wmrl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$WMRL_ROOT"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs"

if [[ -f /home/hzhangex/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-lewm}"
fi

if command -v module >/dev/null 2>&1; then
  module load cuda12.2 2>/dev/null || true
fi

WM_ROOT="/project/peilab/qjl/2026/wmrl/playground/wm"
declare -a MODEL_CKPTS=(
  "$WM_ROOT/embodied_lewm_stage1.ckpt"
  "$WM_ROOT/embodied_lewm_stage2.ckpt"
  "$WM_ROOT/embodied_lewm_stage3.ckpt"
)
declare -a MODEL_NAMES=(
  embodied_lewm_stage1
  embodied_lewm_stage2
  embodied_lewm_stage3
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
  _GPU_LIST=(0 1 2)
fi

DATA_ROOT="${LIBERO_ROOT:-/project/peilab/qjl/2026/playground/dataset/libero}"
EVAL_TASKS="${EVAL_TASKS:-12}"
NSTEP_LIST="${NSTEP_LIST:-1,10,50,100,150,200}"

_EVAL_TASKS_N=$(echo "${EVAL_TASKS:-12}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')
case "${_EVAL_TASKS_N}" in
  12|'1,2'|both|all|1and2|1_and_2)
    _TASK_OVERRIDES=(eval_task1=true eval_task2=true)
    ;;
  1|task1)
    _TASK_OVERRIDES=(eval_task1=true eval_task2=false)
    ;;
  2|task2)
    _TASK_OVERRIDES=(eval_task1=false eval_task2=true)
    ;;
  *)
    echo "[batch_run_eval_libero] ERROR: EVAL_TASKS='${EVAL_TASKS:-}' 无法识别。使用 1 | 2 | 12 | both | 1,2（默认12）." >&2
    exit 1
    ;;
esac

_EVAL_NSTEP=()
if [[ -n "${NSTEP_LIST:-}" ]]; then
  _NB=$(echo "${NSTEP_LIST}" | tr -d '[:space:]')
  _EVAL_NSTEP=("n_step_list=[${_NB}]")
fi

echo "[batch_run_eval_libero] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A} host=$(hostname)"
echo "[batch_run_eval_libero] 3-way parallel; GPUs (per task)=${_GPU_LIST[*]}"
echo "[batch_run_eval_libero] DATA_ROOT=$DATA_ROOT LIBERO_SPLIT=${LIBERO_SPLIT:-train}"
echo "[batch_run_eval_libero] EVAL_TASKS=$EVAL_TASKS NSTEP_LIST=$NSTEP_LIST"

pids=()
for i in 0 1 2; do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${_GPU_LIST[$i]}"
    CKPT="${MODEL_CKPTS[$i]}"
    _TAG="${MODEL_NAMES[$i]}"

    LOG_DIR="${LEWM_EVAL_LOG_DIR:-$SCRIPT_DIR/logs/eval}"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/eval_libero_${_TAG}_$(date +%Y%m%d_%H%M%S).log"

    _BASE_OUT="${LEWM_EVAL_OUT:-$SCRIPT_DIR/eval_outputs/libero_embed_eval}"
    OUTPUT_DIR="${_BASE_OUT}/${_TAG}"

    echo "[batch_run_eval_libero] start i=$i CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[batch_run_eval_libero] tag=$_TAG log=$LOG_FILE ckpt=$CKPT output_dir=$OUTPUT_DIR"
    echo "[batch_run_eval_libero] 评测: ① full_episode ② n_step_open_loop (多组 n)"

    exec > >(tee -a "$LOG_FILE") 2>&1

    python -u -m lewm.eval.run \
      "checkpoint='${CKPT}'" \
      data.dataset.root="$DATA_ROOT" \
      data.dataset.split="${LIBERO_SPLIT:-train}" \
      output_dir="$OUTPUT_DIR" \
      max_episodes=100 \
      "${_TASK_OVERRIDES[@]}" \
      "${_EVAL_NSTEP[@]}" \
      "${FORWARD_ARGS[@]}"
  ) &
  pids+=($!)
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
