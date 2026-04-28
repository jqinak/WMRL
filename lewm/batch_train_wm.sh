#!/usr/bin/env bash
#SBATCH --job-name=lewm_libero
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --account=peilab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=47:00:00
#SBATCH --output=/project/peilab/qjl/2026/le-wm/logs/slurm-%j.out
#SBATCH --error=/project/peilab/qjl/2026/le-wm/logs/slurm-%j.err

set -euo pipefail

LOG_DIR="/project/peilab/qjl/2026/le-wm/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_wm_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] job_id=${SLURM_JOB_ID:-N/A} host=$(hostname) start_time=$(date '+%F %T')"
echo "[INFO] cwd=$(pwd)"

module load cuda12.2
nvcc -V

cd /project/peilab/qjl/2026/le-wm

source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate lewm

# Use line-buffered Python output for realtime logs in sbatch.
python train.py data=libero \
  data.dataset.root=/project/peilab/qjl/2026/playground/dataset/libero \
  data.dataset.split=train \
  data.dataset.num_steps=4 \
  data.dataset.frameskip=1 \
  'data.dataset.keys_to_load=[pixels,action,state]'

echo "[INFO] end_time=$(date '+%F %T')"
