"""LIBERO OffScreen env helpers (aligned with starVLA examples/LIBERO/eval_files/eval_libero.py)."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import numpy as np
import torch

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def ensure_libero_runtime_dependencies() -> None:
    """LIBERO 代码在 PYTHONPATH 上时，仍需同一环境内安装 robosuite/bddl 等（见 LIBERO/requirements.txt）。"""
    from simulator_rl.bootstrap_runtime_env import apply_shared_tmp_and_robosuite_patch

    apply_shared_tmp_and_robosuite_patch()

    try:
        import robosuite  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "当前 conda 环境缺少 `robosuite`（导入 libero 之后会立刻用到）。\n"
            "请在 **wmrl 所用的同一环境** 中安装，例如与官方 LIBERO 一致：\n"
            "  pip install 'robosuite==1.4.0'\n"
            "其他依赖见 LIBERO 仓库根目录 `requirements.txt`（如 bddl、gym 等）。\n"
            "注意：仅设置 LIBERO_HOME / PYTHONPATH 不会自动安装这些包。"
        ) from e


def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat[3] > 1.0:
        quat = quat.copy()
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat = quat.copy()
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / den).astype(np.float32)


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def get_libero_env(task: Any, resolution: int, seed: int):
    """Returns (env, task_description). Requires libero on PYTHONPATH."""
    ensure_libero_runtime_dependencies()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, str(task_description)


def obs_to_policy_example(
    obs: dict,
    task_description: str,
    *,
    resize_hw: tuple[int, int] | None = None,
    include_state: bool = True,
) -> dict:
    """Build wmrl / StarVLA-style example dict for ActorRolloutWorker.

    **State dim:** LIBERO Franka ``robot0_gripper_qpos`` 常为 2 维，直接与位姿拼接会得到 8 维；
    而 QwenPI 配置里 ``state_dim=7``（与 lerobot LIBERO 训练一致：6DOF + 1  gripper 标量）。
    eval_libero.py 通过 WebSocket 推理时示例里往往 **不带 state**；若 ``include_state=True``，
    则将双指夹爪压成 1 维（取均值），使总长为 7。若你希望与评测脚本完全一致，设 ``include_state=False``。
    """
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if resize_hw is not None:
        from starVLA.deployment.model_server.tools.image_tools import resize_with_pad

        h, w = int(resize_hw[0]), int(resize_hw[1])
        stack = resize_with_pad(np.stack([img, wrist_img], axis=0), h, w)
        img, wrist_img = stack[0], stack[1]

    pose6 = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3],
            quat2axisangle(obs["robot0_eef_quat"]).reshape(-1)[:3],
        )
    )
    out: dict = {
        "image": [img, wrist_img],
        "lang": str(task_description),
    }
    if include_state:
        g = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if g.size >= 2:
            g1 = np.array([float(g[:2].mean())], dtype=np.float32)
        elif g.size == 1:
            g1 = g[:1].astype(np.float32)
        else:
            g1 = np.zeros(1, dtype=np.float32)
        out["state"] = np.concatenate([pose6.reshape(-1)[:6], g1]).astype(np.float32)
    return out


def load_dataset_action_stats(policy_ckpt_path: str | pathlib.Path, unnorm_key: str | None) -> dict[str, np.ndarray]:
    """Load `{min,max,mask}` from wmrl/starVLA checkpoint (same contract as ModelClient.get_action_stats)."""
    from pathlib import Path as P

    from starVLA.model.tools import read_mode_config

    p = P(policy_ckpt_path)
    _, norm_stats = read_mode_config(p)
    if unnorm_key is None:
        if len(norm_stats) != 1:
            keys = ",".join(str(k) for k in norm_stats.keys())
            raise ValueError(f"trainer.simulator.action_unnorm_key is required when norm_stats has multiple keys: [{keys}]")
        unnorm_key = next(iter(norm_stats.keys()))
    if unnorm_key not in norm_stats:
        raise KeyError(f"action_unnorm_key={unnorm_key!r} not in norm_stats keys {list(norm_stats.keys())}")
    return norm_stats[str(unnorm_key)]["action"]


def unnormalize_action_rows(normalized: np.ndarray | torch.Tensor, action_norm_stats: dict[str, np.ndarray]) -> np.ndarray:
    """Map normalized predictions in roughly [-1,1] to physical actions (mirror ModelClient.unnormalize_actions)."""
    na = torch.as_tensor(normalized, dtype=torch.float64).detach().cpu().numpy()
    if na.ndim == 1:
        na = na[None, :]
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
    na_clip = np.clip(na.astype(np.float64), -1.0, 1.0)
    if na_clip.shape[-1] > 6:
        na_clip = na_clip.copy()
        na_clip[:, 6] = np.where(na_clip[:, 6] < 0.5, 0.0, 1.0)
    raw = np.where(
        mask,
        0.5 * (na_clip + 1.0) * (action_high - action_low) + action_low,
        na_clip,
    )
    return raw.astype(np.float32)


def policy_action_row_to_libero_delta(denormalized_row: torch.Tensor | np.ndarray) -> list[float]:
    """One denormalized [7]: xyz, euler delta, gripper open [0-1]."""
    row = torch.as_tensor(denormalized_row, dtype=torch.float32).reshape(-1).cpu().numpy()
    if row.shape[0] < 7:
        raise ValueError(f"Expected action dim >= 7, got {row.shape[0]}")
    world_vector_delta = np.asarray(row[:3], dtype=np.float32).reshape(-1)
    rotation_delta = np.asarray(row[3:6], dtype=np.float32).reshape(-1)
    open_gripper = np.asarray(row[6:7], dtype=np.float32).reshape(-1)
    gripper = binarize_gripper_open(open_gripper)
    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
    return delta_action.tolist()
