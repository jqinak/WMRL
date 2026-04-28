#!/usr/bin/env bash
# 常用命令备忘（手工执行）

# 安装
# cd /project/peilab/qjl/2026/wmrl
# pip install -e .

# 查看配置展开
# python -m main_wmrl_qwenpi --cfg job

# 发烟训练（2 步）
# python -m main_wmrl_qwenpi trainer.total_training_steps=2 trainer.log_interval=1 trainer.save_interval=2

# 常规启动
# bash scripts/run_wmrl_qwenpi.sh

# Git（排除 playground）
# git status
# git add . ':!playground/**'
# git commit -m "..."
# git push
