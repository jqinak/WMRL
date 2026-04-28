#!/bin/bash
#SBATCH --job-name=qwen35_08b_pi_libero
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/project/peilab/qjl/2026/starVLA/examples/LIBERO/train_files/slurm-%x-%j.out
#SBATCH --error=/project/peilab/qjl/2026/starVLA/examples/LIBERO/train_files/slurm-%x-%j.err

set -euo pipefail

cd /project/peilab/qjl/2026/starVLA

# Conda for non-interactive batch shell
source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate starVLA

module load cuda12.2

# W&B (batch mode: no interactive login needed)
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
# Keep old var for compatibility; new var for latest torch warning
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# Optional: avoid DeepSpeed Triton cache warning on NFS home
export TRITON_CACHE_DIR=/tmp/${USER}/triton_cache
mkdir -p "${TRITON_CACHE_DIR}"

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenPI
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/qwen35vlPI_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=libero_qwen35_08b_PI_8w_batch
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# Log stdout/stderr under this script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/libero_qwen35vlPI_${run_id}_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project Qwen35_08b_PI_Libero_8w_batch \
  --wandb_entity jqinak-hkust \
  2>&1 | tee "${LOG_FILE}"

exit ${PIPESTATUS[0]}
