#!/usr/bin/env bash
# Libero LEWM 嵌入评测：日志同时输出到终端与文件（tee）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LEWM_EVAL_LOG_DIR:-$SCRIPT_DIR/logs/eval}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_libero_$(date +%Y%m%d_%H%M%S).log"

if [[ -f /home/hzhangex/anaconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-lewm}"
fi

if command -v module >/dev/null 2>&1; then
  module load cuda12.2 2>/dev/null || true
fi

CKPT="${LEWM_CKPT:-/project/peilab/qjl/2026/wmrl/playground/wm/last.ckpt}"
DATA_ROOT="${LIBERO_ROOT:-/project/peilab/qjl/2026/playground/dataset/libero}"

# 本脚本默认值（可被同名环境变量覆盖）：①+②；任务2 开环只在 n ∈ {1,5,10}
EVAL_TASKS="${EVAL_TASKS:-12}"
NSTEP_LIST="${NSTEP_LIST:-1,10,50,100}"

# EVAL_TASKS: 默认 12 表示①+②同时；1=仅 full_episode，2=仅 n_step_open_loop
# NSTEP_LIST: 任务2 开环步长，逗号分隔；留空则不在命令行传入 n_step_list（由 YAML 决定）

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
    echo "[run_eval_libero] ERROR: EVAL_TASKS='${EVAL_TASKS:-}' 无法识别。使用 1 | 2 | 12 | both | 1,2（默认12）." >&2
    exit 1
    ;;
esac

_EVAL_NSTEP=()
if [[ -n "${NSTEP_LIST:-}" ]]; then
  _NB=$(echo "${NSTEP_LIST}" | tr -d '[:space:]')
  _EVAL_NSTEP=("n_step_list=[${_NB}]")
fi

echo "[run_eval_libero] log: $LOG_FILE"
echo "[run_eval_libero] ckpt: $CKPT"
echo "[run_eval_libero] data: $DATA_ROOT"
echo "[run_eval_libero] LIBERO_SPLIT (data.dataset.split)=${LIBERO_SPLIT:-train}"
echo "[run_eval_libero] EVAL_TASKS=${EVAL_TASKS} -> eval_task1/eval_task2 (12=两套都跑)"
echo "[run_eval_libero] NSTEP_LIST=${NSTEP_LIST}"
echo "[run_eval_libero] 评测内容: LEWM 嵌入离线评测 — ① full_episode 末帧 rollout vs 编码"
echo "[run_eval_libero]           ② n_step_open_loop 在多组开环长度 n 上对 (episode,start_t) 取均值"

exec > >(tee -a "$LOG_FILE") 2>&1

# Hydra 只把「第一个 =」当分隔符；路径里若含 =（如 *-epoch=*.ckpt）须包在引号里，否则会报 mismatched input '='。
python -u -m lewm.eval.run \
  "checkpoint='${CKPT}'" \
  data.dataset.root="$DATA_ROOT" \
  data.dataset.split="${LIBERO_SPLIT:-train}" \
  output_dir="${LEWM_EVAL_OUT:-$SCRIPT_DIR/eval_outputs/libero_embed_eval}" \
  max_episodes=30 \
  "${_TASK_OVERRIDES[@]}" \
  "${_EVAL_NSTEP[@]}" \
  "$@"
