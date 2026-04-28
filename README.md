# WMRL (小写仓) 集成说明

本目录是强化学习 VLA 工程根目录，原则如下：

- 仅在本目录新增/维护 WMRL 逻辑。
- 不修改上游 `verl`、`starVLA`、`le-wm` 源码。
- 通过适配层引用上游能力（策略、世界模型、算法）。

## 目录分工

- `main_wmrl_qwenpi.py`: 训练入口。
- `wmrl/trainer`: 训练主循环与质量门禁。
- `wmrl/workers`: actor/reward/bridge 三类 worker。
- `wmrl/models`: WMRL 自有模型部件（如 sigma net）。
- `model`: starVLA 运行时引用适配层。
- `wm`: LE-WM 运行时引用适配层。
- `config`: 三个主配置（`wmrl_qwenpi.yaml`、`qwen35vlPI_libero.yaml`、`lewm.yaml`）。

## 安装

```bash
cd /project/peilab/qjl/2026/wmrl
pip install -e .
```

## 发烟测试（推荐顺序）

```bash
# 1) import + 配置解析
python -c "import wmrl; print('import ok')"
python -m main_wmrl_qwenpi --cfg job

# 2) 小步数训练
python -m main_wmrl_qwenpi trainer.total_training_steps=2 trainer.log_interval=1 trainer.save_interval=2
```

## 常用命令

见 `scripts/common_commands.sh`。
