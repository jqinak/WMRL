from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from wmrl.workers.actor_rollout_worker import ActorRolloutWorker as _BaseActorRolloutWorker
from wmrl.workers.actor_rollout_worker import core_algos


class JEPOActorRolloutWorker(_BaseActorRolloutWorker):
    """Actor worker extension for variable-length trajectory batches."""

    def update_actor_variable_trajectory_chunks(
        self,
        *,
        row_chunk_counts: list[int],
        advantages: torch.Tensor,
        flat_chunk_examples: list[dict[str, Any]],
        chains: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict[str, float]:
        bt = advantages.shape[0]
        if len(row_chunk_counts) != bt:
            raise RuntimeError(f"row_chunk_counts length {len(row_chunk_counts)} != batch rows {bt}")
        if response_mask.shape != advantages.shape:
            raise ValueError("response_mask must match advantages shape in JEPO trajectory PPO update.")
        expected_chunks = int(sum(int(x) for x in row_chunk_counts))
        if chains.shape[0] != expected_chunks:
            raise RuntimeError(f"chains rows {chains.shape[0]} != expected chunks {expected_chunks}")
        if old_log_probs.shape[0] != expected_chunks:
            raise RuntimeError(f"old_log_probs rows {old_log_probs.shape[0]} != expected chunks {expected_chunks}")
        if len(flat_chunk_examples) != expected_chunks:
            raise RuntimeError(f"flat examples {len(flat_chunk_examples)} != expected chunks {expected_chunks}")

        per = int(self.action_horizon * self.action_dim)
        row_metrics: list[dict[str, float]] = []
        cursor = 0

        for bi, s_chunks in enumerate(row_chunk_counts):
            self.optimizer.zero_grad(set_to_none=True)
            self.action_model.train()
            self.sigma_net.train()
            total_loss: torch.Tensor | None = None
            ec = float(self.config.algorithm.get("entropy_coeff", 0.0))
            chunk_pg: list[float] = []
            chunk_entropy_loss: list[float] = []
            chunk_kl: list[float] = []
            chunk_clip: list[float] = []
            chunk_clip_lo: list[float] = []
            chunk_ratio_mean: list[float] = []
            chunk_ratio_max: list[float] = []
            chunk_ratio_raw_max: list[float] = []

            for j in range(int(s_chunks)):
                ci = cursor + j
                single_ex = [flat_chunk_examples[ci]]
                x_ch = chains[ci : ci + 1].to(self.device)
                ol = old_log_probs[ci : ci + 1].to(self.device)
                adv_slice = advantages[bi : bi + 1, j * per : (j + 1) * per].to(self.device)
                resp_mask_slice = response_mask[bi : bi + 1, j * per : (j + 1) * per].to(self.device)

                new_lp, entropy = self._compute_log_prob(single_ex, x_chain=x_ch, return_entropy=True, require_grad=True)
                assert entropy is not None
                log_ratio = (new_lp - ol).float()
                max_ratio_guard = float(self.config.algorithm.get("max_ratio_guard", 20.0))
                log_cap = math.log(max_ratio_guard + 1e-12)
                ratio = torch.exp(torch.clamp(log_ratio, min=-log_cap, max=log_cap))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                        old_log_prob=ol,
                        log_prob=new_lp,
                        advantages=adv_slice,
                        response_mask=resp_mask_slice,
                        cliprange=float(self.config.algorithm.clip_ratio),
                        cliprange_low=float(self.config.algorithm.clip_ratio),
                        cliprange_high=float(self.config.algorithm.clip_ratio),
                        clip_ratio_c=3.0,
                    )
                entropy_loss = core_algos.agg_loss(entropy, resp_mask_slice, loss_agg_mode="token-mean")
                chunk_loss = pg_loss - ec * entropy_loss
                chunk_pg.append(float(pg_loss.detach().cpu()))
                chunk_entropy_loss.append(float(entropy_loss.detach().cpu()))
                chunk_kl.append(float(ppo_kl.detach().cpu()))
                chunk_clip.append(float(pg_clipfrac.detach().cpu()))
                chunk_clip_lo.append(float(pg_clipfrac_lower.detach().cpu()))
                chunk_ratio_mean.append(float(ratio.mean().detach().cpu()))
                chunk_ratio_max.append(float(ratio.max().detach().cpu()))
                chunk_ratio_raw_max.append(
                    float(torch.exp(torch.clamp(log_ratio, min=-50.0, max=50.0)).max().detach().cpu())
                )
                total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss

            cursor += int(s_chunks)
            assert total_loss is not None
            n_c = max(1, len(chunk_pg))
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.action_model.parameters()) + list(self.sigma_net.parameters()),
                max_norm=float(self.config.algorithm.max_grad_norm),
            )
            if not torch.isfinite(total_loss) or not torch.isfinite(grad_norm):
                raise FloatingPointError("JEPO trajectory actor update non-finite loss/grad.")

            target_kl = float(self.config.algorithm.get("target_kl", 0.0))
            mean_kl = float(sum(chunk_kl) / n_c)
            if (
                bool(self.config.algorithm.get("kl_stop_on_exceed", False))
                and target_kl > 0.0
                and mean_kl > target_kl
                and not bool(getattr(self.config.runtime, "smoke_random_init", False))
            ):
                raise RuntimeError(f"PPO KL too large: {mean_kl:.6f} > {target_kl}")

            self.optimizer.step()
            row_metrics.append(
                {
                    "actor/loss": float(total_loss.detach().cpu()),
                    "actor/loss_avg_per_chunk": float(total_loss.detach().cpu()) / n_c,
                    "actor/pg_loss": float(sum(chunk_pg) / n_c),
                    "actor/pg_loss_sum_over_chunks": float(sum(chunk_pg)),
                    "actor/entropy": float(sum(chunk_entropy_loss) / n_c),
                    "actor/entropy_coeff_times_entropy": ec * float(sum(chunk_entropy_loss) / n_c),
                    "actor/ppo_kl": mean_kl,
                    "actor/pg_clipfrac": float(sum(chunk_clip) / n_c),
                    "actor/pg_clipfrac_lower": float(sum(chunk_clip_lo) / n_c),
                    "actor/ratio_mean": float(sum(chunk_ratio_mean) / n_c),
                    "actor/ratio_max": float(sum(chunk_ratio_max) / n_c),
                    "actor/ratio_raw_max": max(chunk_ratio_raw_max) if chunk_ratio_raw_max else 0.0,
                    "actor/grad_norm": float(grad_norm.detach().cpu()),
                }
            )

        out_metrics: dict[str, float] = {}
        if row_metrics:
            for key in row_metrics[0].keys():
                out_metrics[key] = float(sum(row[key] for row in row_metrics) / len(row_metrics))
        return out_metrics
