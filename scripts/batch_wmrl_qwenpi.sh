#!/usr/bin/env bash
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --job-name=wmrl_qwenpi
#SBATCH --output=/project/peilab/qjl/2026/wmrl/scripts/logs/slurm-%x-%j.out
#SBATCH --error=/project/peilab/qjl/2026/wmrl/scripts/logs/slurm-%x-%j.err
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#
# 集群若不用 --gres，可改为例如：#SBATCH --gpus-per-node=4
# 分区按需改：#SBATCH -p <your_gpu_partition>

set -euo pipefail

mkdir -p /project/peilab/qjl/2026/wmrl/scripts/logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# 与 run_wmrl_qwenpi.sh 一致：在项目根执行、同一 conda / module
exec bash "${SCRIPT_DIR}/run_wmrl_qwenpi.sh" "$@"
