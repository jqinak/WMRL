"""One LIBERO episode: chunked policy rollout with wmrl ActorRolloutWorker."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from simulator_rl import libero_env_utils as lu

if TYPE_CHECKING:
    from wmrl.workers.actor_rollout_worker import ActorRolloutWorker


@dataclass
class OnlineEpisodeResult:
    chunk_examples: list[dict]
    """Deep-copied observation examples (one per policy chunk)."""

    x_chains: torch.Tensor
    """[S, k+1, H, action_dim] CPU float — one chain per chunk."""

    old_log_probs: torch.Tensor
    """[S, H * action_dim] CPU float."""

    success: float
    """1.0 if env terminated with success (`done`), else 0.0."""

    episode_env_steps_after_wait: int


def run_one_libero_rollout(
    actor: ActorRolloutWorker,
    env,
    task_description: str,
    initial_state_vec: np.ndarray,
    *,
    action_norm_stats: dict[str, np.ndarray],
    rollout_noise_seed: int,
    max_steps: int,
    num_steps_wait: int,
    policy_image_resize: tuple[int, int] | None = None,
) -> OnlineEpisodeResult:
    """Run one stochastic episode using the same chunked execution as STARVLA ModelClient."""

    env.reset()
    obs = env.set_init_state(initial_state_vec)

    horizon = int(actor.action_horizon)
    action_dim = int(actor.action_dim)
    per = horizon * action_dim

    chunks_ex: list[dict] = []
    chains_sl: list[torch.Tensor] = []
    lp_sl: list[torch.Tensor] = []

    noise_gen = torch.Generator(device=torch.device("cpu"))
    noise_gen.manual_seed(int(rollout_noise_seed) % (2**31))

    dummy = lu.LIBERO_DUMMY_ACTION
    t = 0
    env_steps_used = 0
    terminated_success = False
    pending: torch.Tensor | None = None
    h_exec = 0

    done = False
    while t < max_steps + num_steps_wait and not done:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(dummy)
            t += 1
            continue

        if pending is None or h_exec >= horizon:
            example = copy.deepcopy(
                lu.obs_to_policy_example(obs, task_description, resize_hw=policy_image_resize)
            )
            noise = torch.randn(1, horizon, action_dim, generator=noise_gen, dtype=torch.float32)
            roll = actor.generate_actions([example], noise=noise.to(actor.device))

            pend = roll["predicted_actions"][0].detach().float().cpu().numpy()
            pend_raw = lu.unnormalize_action_rows(pend, action_norm_stats)
            pending = torch.from_numpy(pend_raw)
            h_exec = 0

            xc = roll["x_chain"][0].detach().cpu().float()

            chunks_ex.append(example)
            chains_sl.append(xc.unsqueeze(0))
            old_lp_row = actor.compute_log_prob([example], roll["x_chain"][0 : 1].cpu())
            lp_sl.append(old_lp_row)

        row = pending[h_exec]
        h_exec += 1
        lu_delta = lu.policy_action_row_to_libero_delta(row)
        obs, _rew, done, info = env.step(lu_delta)
        env_steps_used += 1
        t += 1

        if done:
            terminated_success = True
            break

    if not chunks_ex:
        raise RuntimeError("LIBERO rollout produced zero policy chunks (unexpected).")

    x_chains = torch.cat(chains_sl, dim=0)
    old_lp = torch.cat(lp_sl, dim=0)
    assert old_lp.shape == (len(chunks_ex), per)

    succ = 1.0 if terminated_success else 0.0
    return OnlineEpisodeResult(
        chunk_examples=chunks_ex,
        x_chains=x_chains,
        old_log_probs=old_lp,
        success=succ,
        episode_env_steps_after_wait=int(env_steps_used),
    )


def grpo_outcome_advantages_1d(successes: list[float], epsilon: float = 1e-6) -> torch.Tensor:
    """GRPO-style normalization over a single group (length rollout_n). Returns [rollout_n] CPU float."""
    t = torch.tensor(successes, dtype=torch.float32)
    mean = t.mean()
    std = t.std(unbiased=False)
    std_eff = torch.clamp(std, min=epsilon)
    adv = (t - mean) / (std_eff + epsilon)
    return adv
