cd /project/peilab/qjl/2026/wmrl/jepo
source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
conda activate wmrl
module load cuda12.2
CUDA_VISIBLE_DEVICES=0 python -u -m main_jepo_qwenpi \
  trainer.total_training_steps=400 \
  reward.jepo.reward_type=terminal \
  trainer.output_dir=/project/peilab/qjl/2026/wmrl/checkpoints/jepo_terminal_400 \
  trainer.wandb.project=qwenPI_jepo_terminal_400

# cd /project/peilab/qjl/2026/wmrl/jepo
# source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
# conda activate wmrl
# module load cuda12.2
# CUDA_VISIBLE_DEVICES=1 python -u -m main_jepo_qwenpi \
#   trainer.total_training_steps=400 \
#   reward.jepo.reward_type=sparse_milestone \
#   trainer.output_dir=/project/peilab/qjl/2026/wmrl/checkpoints/jepo_sparse_400 \
#   trainer.wandb.project=qwenPI_jepo_sparse_400

# cd /project/peilab/qjl/2026/wmrl/jepo
# source /home/hzhangex/anaconda3/etc/profile.d/conda.sh
# conda activate wmrl
# module load cuda12.2
# CUDA_VISIBLE_DEVICES=2 python -u -m main_jepo_qwenpi \
#   trainer.total_training_steps=400 \
#   reward.jepo.reward_type=dense_milestone \
#   trainer.output_dir=/project/peilab/qjl/2026/wmrl/checkpoints/jepo_dense_400 \
#   trainer.wandb.project=qwenPI_jepo_dense_400