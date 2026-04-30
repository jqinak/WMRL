#!/usr/bin/env bash
set -euo pipefail

cd /project/peilab/qjl/2026/wmrl
source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate wmrl
module load cuda12.2

run_id=${RUN_ID:-debug}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/${run_id}_$(date +%Y%m%d_%H%M%S).log"

# ── 默认：多段 trajectory rollout + 下一帧 GT 对齐 + 三种轨迹奖励全开；可用环境变量改成 false ──
: "${WMRL_TRAJECTORY_ROLLOUT_ENABLED:=true}"
: "${WMRL_GT_NEXT_OBS:=true}"
: "${WMRL_TRAJ_REWARD_SPARSE_CHUNK_END:=true}"
: "${WMRL_TRAJ_REWARD_DENSE_MICRO:=true}"
: "${WMRL_TRAJ_REWARD_TERMINAL:=true}"
# wandb 必须用 trainer.wandb.*（根上 wandb_entity 会触发 struct 报错）
: "${WMRL_WANDB_ENTITY:=jqinak-hkust}"
: "${WMRL_WANDB_PROJECT:=qwenPI_rl_test}"

python3 -u -m main_wmrl_qwenpi \
  trainer.total_training_steps=100 \
  trainer.save_interval=50 \
  trainer.log_interval=1 \
  trainer.wandb.entity="${WMRL_WANDB_ENTITY}" \
  trainer.wandb.project="${WMRL_WANDB_PROJECT}" \
  trajectory_rollout.enabled="${WMRL_TRAJECTORY_ROLLOUT_ENABLED}" \
  trajectory_rollout.gt_use_next_observation="${WMRL_GT_NEXT_OBS}" \
  reward.trajectory.enable_trajectory_sparse_milestone="${WMRL_TRAJ_REWARD_SPARSE_CHUNK_END}" \
  reward.trajectory.enable_trajectory_dense_milestone="${WMRL_TRAJ_REWARD_DENSE_MICRO}" \
  reward.trajectory.enable_trajectory_terminal_bonus="${WMRL_TRAJ_REWARD_TERMINAL}" \
  "$@" 2>&1 | tee "${LOG_FILE}"

  # runtime.smoke_random_init=true \
