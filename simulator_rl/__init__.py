"""Online LIBERO simulation GRPO helpers (wmrl + ActorRolloutWorker)."""

from __future__ import annotations

from simulator_rl.compat_starvla_paths import ensure_starvla_deployment_aliases

ensure_starvla_deployment_aliases()

__all__ = ["libero_env_utils", "online_episode", "grpo_libero_trainer"]
