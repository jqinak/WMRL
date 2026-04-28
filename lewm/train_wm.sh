#!/usr/bin/env bash

LOG_DIR="/project/peilab/qjl/2026/le-wm/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_wm_$(date +%Y%m%d_%H%M%S).log"
# exec > "$LOG_FILE" 2>&1
exec > >(tee -a "$LOG_FILE") 2>&1

module load cuda12.2
nvcc -V
cd /project/peilab/qjl/2026/le-wm


python train.py data=libero \
  data.dataset.root=/project/peilab/qjl/2026/playground/dataset/libero \
  data.dataset.split=train \
  data.dataset.num_steps=4 \
  data.dataset.frameskip=1 \
  'data.dataset.keys_to_load=[pixels,action,state]'