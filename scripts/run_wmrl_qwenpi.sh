#!/usr/bin/env bash
set -euo pipefail

cd /project/peilab/qjl/2026/wmrl
source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate wmrl
module load cuda12.2

run_id=${RUN_ID:-debug}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/${run_id}_$(date +%Y%m%d_%H%M%S).log"

python3 -m main_wmrl_qwenpi \
  trainer.total_training_steps=100 \
  trainer.save_interval=5 \
  data.num_workers=0 \
  reward.fallback_to_action_embedding=true \
  "$@" 2>&1 | tee "${LOG_FILE}"

  
  # runtime.smoke_random_init=true \