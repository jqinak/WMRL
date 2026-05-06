#!/usr/bin/env bash
set -euo pipefail

# Repo roots (按需修改)
WMRL_ROOT="/project/peilab/qjl/2026/wmrl"
STARVLA_DIR="${STARVLA_DIR:-${WMRL_ROOT}/starVLA}"
LIBERO_HOME="${LIBERO_HOME:-/project/peilab/qjl/2026/LIBERO}"

CFG="${CFG:-${WMRL_ROOT}/simulator_rl/config/simulator_grpo_libero.yaml}"

# LIBERO 仿真依赖 robosuite（仅 PYTHONPATH 不够）。在 wmrl conda 中执行一次，例如：
#   pip install 'robosuite==1.4.0' 'bddl==1.0.1'

# 与 train_libero_grpo 内 bootstrap 一致（共享机 /tmp 不可写时用项目下 tmp）
TMP_ROOT_DEFAULT="/project/peilab/qjl/2026/tmp"
mkdir -p "${TMP_ROOT_DEFAULT}"
export TMPDIR="${TMPDIR:-${TMP_ROOT_DEFAULT}}"
export TMP="${TMP:-${TMP_ROOT_DEFAULT}}"
export TEMP="${TEMP:-${TMP_ROOT_DEFAULT}}"
export WMRL_TMP_ROOT="${WMRL_TMP_ROOT:-${TMP_ROOT_DEFAULT}}"
export ROBOSUITE_LOG_FILE="${ROBOSUITE_LOG_FILE:-${TMP_ROOT_DEFAULT}/robosuite.log}"
export PYTHONPATH="${LIBERO_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${STARVLA_DIR}:${WMRL_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

# 若 STARVLA 仓内存在补丁脚本则执行（对齐 eval_libero.sh；train 入口也会自动 patch）
_ENSURE="${STARVLA_DIR}/examples/LIBERO/eval_files/ensure_robosuite_log_path.py"
if [[ -f "${_ENSURE}" ]]; then
  python "${_ENSURE}"
fi

cd "${WMRL_ROOT}"

python "${WMRL_ROOT}/simulator_rl/train_libero_grpo.py" --config "${CFG}" "$@"
