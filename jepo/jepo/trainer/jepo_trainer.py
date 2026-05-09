from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from wmrl.trainer.ray_wmrl_trainer import (
    RayWMRLTrainer as _BaseRayWMRLTrainer,
    _chunks_to_micro_tensor,
    _gt_chunks_to_micro_tensor,
    _log,
)
from wmrl.workers import TokenizerBridge

from jepo.data import JEPOFullExpertTrajectoryIterable
from jepo.workers.actor_rollout_worker import JEPOActorRolloutWorker
from jepo.workers.lewm_reward_worker import JEPOLewmRewardWorker


class JEPORayTrainer(_BaseRayWMRLTrainer):
    """Trajectory-level JEPO trainer with variable-length base batches."""

    def __init__(self, config):
        self.config = config
        self._set_seed(int(config.runtime.seed))
        self.bridge = TokenizerBridge()
        self.actor_worker = JEPOActorRolloutWorker(config)
        self.reward_worker = JEPOLewmRewardWorker(config)
        self.use_trajectory_rollout = bool(OmegaConf.select(config, "trajectory_rollout.enabled") or False)
        if not self.use_trajectory_rollout:
            raise ValueError("JEPO requires trajectory_rollout.enabled=true")
        _log("[jepo] trajectory rollout enabled: batched full episodes + masked JEPO rewards")
        self.rollout_cycle = itertools.cycle(self._build_rollout_stream())
        self.total_steps = int(config.trainer.total_training_steps)
        self.log_interval = int(config.trainer.log_interval)
        self.save_interval = int(config.trainer.save_interval)
        self.output_dir = Path(config.trainer.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_step = 1
        if bool(config.trainer.get("auto_resume", False)):
            self.start_step = self._try_resume()
        self._wandb_initialized = False
        self._monitor_preamble_logged = False
        self.metrics_jsonl_enabled = bool(config.trainer.get("metrics_jsonl", True))
        self._metrics_jsonl_fh = None
        self._metrics_jsonl_path_announced = False
        self._maybe_init_wandb()

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_trajectory_dataloader(self):
        starvla_cfg = OmegaConf.load(self.config.data.starvla_cfg)
        base = get_vla_dataset(data_cfg=starvla_cfg.datasets.vla_data)
        traj_cfg = self.config.trajectory_rollout
        a = int(self.actor_worker.action_horizon)
        tb = int(traj_cfg.get("train_batch_size", self.config.data.train_batch_size))
        _log(f"[jepo] building JEPOFullExpertTrajectoryIterable (full episodes, batch={tb}) ...")
        it_ds = JEPOFullExpertTrajectoryIterable(
            base,
            chunk_actions=a,
            seed=int(self.config.runtime.seed),
            max_sample_tries=int(traj_cfg.get("max_sample_tries", 512)),
            action_take_dim=int(traj_cfg.get("action_take_dim", self.actor_worker.action_dim)),
            gt_use_next_observation=bool(traj_cfg.get("gt_use_next_observation", True)),
            train_batch_size=tb,
        )
        return DataLoader(it_ds, batch_size=None, num_workers=0, pin_memory=False)

    @staticmethod
    def _build_jepo_group_index(base_batch_size: int, repeat_n: int) -> np.ndarray:
        ids = [f"traj-{i}" for i in range(int(base_batch_size)) for _ in range(int(repeat_n))]
        return np.asarray(ids, dtype=object)

    def _run_trajectory_training_step(self, traj_batch: list):
        repeat_n = int(self.config.algorithm.rollout_n)
        base_b = len(traj_batch)
        expected_b = int(self.config.trajectory_rollout.get("train_batch_size", self.config.data.train_batch_size))
        if base_b != expected_b:
            raise ValueError(f"JEPO expects {expected_b} base trajectories per batch, got {base_b}")
        if repeat_n < 1:
            raise ValueError(f"Invalid rollout_n={repeat_n}")

        a = int(self.actor_worker.action_horizon)
        d = int(self.actor_worker.action_dim)
        per = a * d
        metas = [item.get("meta") or {} for item in traj_batch]
        n_micro = [int(meta.get("n_micro", meta.get("micro_steps", -1))) for meta in metas]
        if any(n < 1 for n in n_micro):
            raise ValueError(f"Invalid JEPO n_micro values: {n_micro}")
        s_chunks = [len(item["chunk_examples"]) for item in traj_batch]
        s_max = max(s_chunks)
        t_max = s_max * a
        padded_tokens = t_max * d

        traj_roll_cfg = OmegaConf.select(self.config, "trajectory_rollout") or OmegaConf.create({})
        use_next_gt_obs = bool(traj_roll_cfg.get("gt_use_next_observation", True))
        expert_views_base: list[list] = []
        gt_micro_base = torch.zeros(base_b, t_max, d, dtype=torch.float32)
        base_masks = torch.zeros(base_b, padded_tokens, dtype=torch.float32)

        for i, item in enumerate(traj_batch):
            expect_views = n_micro[i] + (1 if use_next_gt_obs else 0)
            views = item.get("expert_views")
            if views is None:
                raise ValueError(f"traj_batch[{i}] missing expert_views")
            if len(views) != expect_views:
                raise ValueError(
                    f"traj_batch[{i}] expert_views length {len(views)} != expected {expect_views} "
                    f"(n_micro={n_micro[i]}, gt_use_next_observation={use_next_gt_obs})"
                )
            expert_views_base.append(list(views))
            mic_full = _gt_chunks_to_micro_tensor(item, s_chunks=s_chunks[i], chunk_actions=a, n_micro=n_micro[i])
            gt_micro_base[i, : n_micro[i]] = mic_full
            base_masks[i, : n_micro[i] * d] = 1.0

        pred_micro_rows: list[torch.Tensor] = []
        logprob_rows: list[torch.Tensor] = []
        flat_chunk_examples_ordered: list[dict] = []
        chain_slices_ordered: list[torch.Tensor] = []
        compact_old_log_probs: list[torch.Tensor] = []
        row_chunk_counts: list[int] = []

        for bi, item in enumerate(traj_batch):
            chunk_flat = list(item["chunk_examples"])
            for _rn in range(repeat_n):
                noise = self.actor_worker.sample_noise_for_chunks(chunk_flat)
                roll = self.actor_worker.generate_actions_chunk_flat(chunk_flat, noise)
                pred_c = roll["predicted_actions"].float()
                x_chain = roll["x_chain"]
                micro_1d = _chunks_to_micro_tensor(pred_c, s_chunks=s_chunks[bi], chunk_actions=a, n_micro=n_micro[bi])
                pred_pad = torch.zeros(t_max, d, dtype=torch.float32)
                pred_pad[: n_micro[bi]] = micro_1d
                pred_micro_rows.append(pred_pad)

                old_lp_flat = self.actor_worker.compute_log_prob(chunk_flat, x_chain).float()
                logp_pad = torch.zeros(padded_tokens, dtype=torch.float32)
                for j in range(s_chunks[bi]):
                    logp_pad[j * per : (j + 1) * per] = old_lp_flat[j]
                    flat_chunk_examples_ordered.append(chunk_flat[j])
                    chain_slices_ordered.append(x_chain[j : j + 1])
                    compact_old_log_probs.append(old_lp_flat[j : j + 1])
                logprob_rows.append(logp_pad)
                row_chunk_counts.append(int(s_chunks[bi]))

        predicted_micro = torch.stack(pred_micro_rows, dim=0)
        logprob_ordered = torch.stack(logprob_rows, dim=0)
        response_mask_bt = base_masks.repeat_interleave(repeat_n, dim=0).contiguous()
        token_rows = response_mask_bt.shape[0]
        if predicted_micro.shape != (base_b * repeat_n, t_max, d):
            raise RuntimeError(f"predicted_micro shape {tuple(predicted_micro.shape)} != ({base_b * repeat_n}, {t_max}, {d})")
        if logprob_ordered.shape != (token_rows, padded_tokens):
            raise RuntimeError(f"logprob shape {tuple(logprob_ordered.shape)} != ({token_rows}, {padded_tokens})")

        reward_out = self.reward_worker.compute_jepo_trajectory_rewards(
            expert_views_base,
            predicted_micro,
            response_mask_bt,
            gt_micro_actions=gt_micro_base,
            chunk_actions=a,
            rollout_n=repeat_n,
        )
        token_ordered = reward_out["token_level_rewards"].detach().cpu().float()
        if token_ordered.shape != response_mask_bt.shape:
            raise RuntimeError(f"reward shape {tuple(token_ordered.shape)} != response_mask {tuple(response_mask_bt.shape)}")

        gid = self._build_jepo_group_index(base_b, repeat_n)
        self._assert_finite("token_level_rewards", token_ordered)
        advantages, returns = self._compute_advantage(token_ordered, gid, response_mask=response_mask_bt)
        if advantages.shape != logprob_ordered.shape:
            raise RuntimeError(f"advantages vs log_probs shape mismatch: {advantages.shape}, {logprob_ordered.shape}")

        min_rs_traj = float(self.config.trainer.get("min_reward_std_trajectory", 0.0))
        if min_rs_traj > 0.0 and float(token_ordered.std(unbiased=False)) <= min_rs_traj:
            _log(
                f"[jepo] warning: token reward std low {float(token_ordered.std(unbiased=False)):.6e} "
                f"<= {min_rs_traj:.6e}"
            )

        chained_chains = torch.cat(chain_slices_ordered, dim=0)
        old_log_probs_chunkwise = torch.cat(compact_old_log_probs, dim=0)
        update_metrics = self.actor_worker.update_actor_variable_trajectory_chunks(
            row_chunk_counts=row_chunk_counts,
            advantages=advantages,
            flat_chunk_examples=flat_chunk_examples_ordered,
            chains=chained_chains,
            old_log_probs=old_log_probs_chunkwise,
            response_mask=response_mask_bt,
        )

        rollout_meta = {
            "predicted_micro_reference": predicted_micro.detach().cpu(),
            "micro_tokens": float(sum(n_micro) * d / max(1, base_b)),
            "padded_response_tokens": float(padded_tokens),
            "s_chunks": float(sum(s_chunks) / max(1, base_b)),
            "s_chunks_max": float(s_max),
            "chunks_total": float(sum(row_chunk_counts)),
            "n_micro": float(sum(n_micro) / max(1, base_b)),
            "n_micro_min": float(min(n_micro)),
            "n_micro_max": float(max(n_micro)),
            "pad_tail": float(sum(int(meta.get("pad_tail", 0)) for meta in metas) / max(1, base_b)),
            "base_trajectories": float(base_b),
            "rollout_rows": float(base_b * repeat_n),
        }
        return (
            token_ordered,
            advantages,
            returns,
            chained_chains,
            logprob_ordered,
            update_metrics,
            reward_out,
            rollout_meta,
        )
