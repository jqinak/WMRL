"""Online LIBERO RL with GRPO (outcome = env success only) and wmrl ActorRolloutWorker."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from simulator_rl.compat_starvla_paths import ensure_starvla_deployment_aliases

ensure_starvla_deployment_aliases()

from simulator_rl import libero_env_utils as lu
from simulator_rl.online_episode import grpo_outcome_advantages_1d, run_one_libero_rollout
from simulator_rl.resolve_ckpt_paths import apply_read_mode_config_patch
from wmrl.workers.actor_rollout_worker import ActorRolloutWorker


def _log(msg: str) -> None:
    print(msg, flush=True)


def _flatten_config(cfg) -> dict:
    """OmegaConf -> JSON-serializable dict for wandb.init(config=...)."""
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return {}


class LiberoOnlineGRPOTrainer:
    """B=1 task group × ``rollout_n`` LIBERO episodes per optimizer macro-step."""

    def __init__(self, config):
        self.config = config

        ckpt_ref = OmegaConf.select(config, "paths.starvla_ckpt")
        if ckpt_ref is None or str(ckpt_ref).strip() == "":
            raise ValueError("OmegaConf paths.starvla_ckpt must be set (checkpoint containing dataset_statistics).")

        unnorm_k = OmegaConf.select(config, "simulator.action_unnorm_key")
        unk = None if unnorm_k is None else str(unnorm_k)

        sr_override = OmegaConf.select(config, "simulator.starvla_repo_root")
        apply_read_mode_config_patch(star_root_override=sr_override)

        self._action_norm_stats = lu.load_dataset_action_stats(str(ckpt_ref), unk)

        self.actor = ActorRolloutWorker(config)
        self.rollout_n = int(config.algorithm.rollout_n)
        if self.rollout_n < 2:
            raise ValueError("algorithm.rollout_n must be >= 2 for GRPO grouping.")

        self.output_dir = Path(str(config.trainer.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = int(config.trainer.total_training_steps)
        self.log_interval = int(config.trainer.get("log_interval", 10))
        self.save_interval = int(config.trainer.get("save_interval", 100))

        pis = OmegaConf.select(config, "simulator.policy_image_resize")
        self.policy_image_resize: tuple[int, int] | None = None
        if pis is not None:
            plist = OmegaConf.to_container(pis, resolve=True)
            if isinstance(plist, (list, tuple)) and len(plist) == 2:
                self.policy_image_resize = (int(plist[0]), int(plist[1]))

        suite = str(config.simulator.task_suite_name)
        from libero.libero import benchmark

        self.task_suite_name = suite
        self.task_suite = benchmark.get_benchmark_dict()[suite]()
        self.suite_max_steps = lu.max_steps_for_suite(suite)

        rs = OmegaConf.select(config, "runtime.seed")
        self._runtime_seed = int(rs if rs is not None else 0)
        random.seed(self._runtime_seed)
        np.random.seed(self._runtime_seed)
        torch.manual_seed(self._runtime_seed)
        self.sim_seed_base = int(config.simulator.base_seed)

        self._wandb_initialized = False
        self._metrics_jsonl_fh = None
        self._metrics_jsonl_path_announced = False
        self.metrics_jsonl_enabled = bool(self.config.trainer.get("metrics_jsonl", False))
        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        tw = OmegaConf.select(self.config, "trainer.wandb") or OmegaConf.create({})
        if not bool(tw.get("enabled", False)):
            return
        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            _log("[wandb] 未安装 wandb，跳过。安装: pip install wandb")
            return

        entity = OmegaConf.select(tw, "entity")
        project = str(OmegaConf.select(tw, "project") or "simulator_rl_test")
        run_name = OmegaConf.select(tw, "run_name") or OmegaConf.select(tw, "name")
        if not run_name:
            run_name = f"simulator_rl_{self.output_dir.name}_seed{self._runtime_seed}"
        tags = OmegaConf.select(tw, "tags")
        tags_list = OmegaConf.to_container(tags, resolve=True) if tags is not None else None
        if not isinstance(tags_list, list):
            tags_list = []
        tags_list = [str(t) for t in tags_list]
        notes = OmegaConf.select(tw, "notes")
        notes_str = str(notes) if notes else None

        init_kwargs: dict = {
            "project": project,
            "name": str(run_name),
            "config": _flatten_config(self.config),
            "dir": str(self.output_dir),
        }
        if entity:
            init_kwargs["entity"] = str(entity)
        if tags_list:
            init_kwargs["tags"] = tags_list
        if notes_str:
            init_kwargs["notes"] = notes_str

        wandb.init(**init_kwargs)  # type: ignore[arg-type]
        self._wandb_initialized = True
        _log(f"[wandb] initialized project={project} name={run_name}" + (f" entity={entity}" if entity else ""))

    def _wandb_log(self, global_step: int, metrics: dict[str, float | int]) -> None:
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        out = {str(k): v for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)}
        if out:
            wandb.log(out, step=int(global_step))

    def _append_metrics_jsonl(self, global_step: int, row: dict) -> None:
        if not self.metrics_jsonl_enabled:
            return
        path = self.output_dir / "train_metrics.jsonl"
        if self._metrics_jsonl_fh is None:
            self._metrics_jsonl_fh = open(path, "a", encoding="utf-8")
            if not self._metrics_jsonl_path_announced:
                _log(f"[metrics] metrics_jsonl → {path}")
                self._metrics_jsonl_path_announced = True
        rec = dict(row)
        rec["global_step"] = int(global_step)
        self._metrics_jsonl_fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        self._metrics_jsonl_fh.flush()

    def _wandb_finish(self) -> None:
        if not self._wandb_initialized:
            return
        try:
            import wandb  # type: ignore[import-untyped]

            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        self._wandb_initialized = False

    def _close_metrics_jsonl(self) -> None:
        if self._metrics_jsonl_fh is not None:
            try:
                self._metrics_jsonl_fh.flush()
                self._metrics_jsonl_fh.close()
            except Exception:
                pass
            self._metrics_jsonl_fh = None

    def fit(self):
        rollout_n = self.rollout_n
        rng = np.random.default_rng(seed=int(self.sim_seed_base))

        try:
            for step in range(1, self.total_steps + 1):
                tid = int(rng.integers(0, self.task_suite.n_tasks))
                task = self.task_suite.get_task(tid)
                init_bank = self.task_suite.get_task_init_states(tid)
                eid = int(rng.integers(0, len(init_bank)))

                env, lang = lu.get_libero_env(
                    task, int(self.config.simulator.render_resolution), self.sim_seed_base + step
                )

                successes: list[float] = []
                rollouts_meta: list = []

                for r in range(rollout_n):
                    rs = self.sim_seed_base * 100_003 + step * 1_000 + r + 17
                    res = run_one_libero_rollout(
                        self.actor,
                        env,
                        lang,
                        init_bank[eid],
                        action_norm_stats=self._action_norm_stats,
                        rollout_noise_seed=rs,
                        max_steps=self.suite_max_steps,
                        num_steps_wait=int(self.config.simulator.num_steps_wait),
                        policy_image_resize=self.policy_image_resize,
                    )
                    successes.append(float(res.success))
                    rollouts_meta.append(res)

                adv_scalar = grpo_outcome_advantages_1d(successes)

                agg: dict[str, float] | None = None
                for r in range(rollout_n):
                    er = rollouts_meta[r]
                    s_chunks = len(er.chunk_examples)
                    hor = self.actor.action_horizon
                    adim = self.actor.action_dim
                    adv_row = torch.full((1, s_chunks * hor * adim), float(adv_scalar[r].item()))
                    m = self.actor.update_actor_trajectory_chunks(
                        s_chunks=s_chunks,
                        advantages=adv_row,
                        flat_chunk_examples=er.chunk_examples,
                        chains=er.x_chains.to(torch.float32),
                        old_log_probs=er.old_log_probs.to(torch.float32),
                    )
                    if agg is None:
                        agg = dict(m)
                    else:
                        for k, v in m.items():
                            agg[k] += v

                assert agg is not None
                for k in list(agg.keys()):
                    agg[k] /= rollout_n

                succ_rate = float(np.mean(successes))
                msg = (
                    f"[LIBERO-online step {step}/{self.total_steps}] task={tid} suite={self.task_suite_name} "
                    f"pilot_ep={eid} success_grp={succ_rate:.3f} "
                    f"loss={agg.get('actor/loss', 0):.6f} kl={agg.get('actor/ppo_kl', 0):.5f}"
                )
                if step % self.log_interval == 0 or step == 1:
                    _log(msg)

                ep_lens = [float(r.episode_env_steps_after_wait) for r in rollouts_meta]
                adv_std = float(adv_scalar.std(unbiased=False)) if adv_scalar.numel() > 1 else 0.0

                wb: dict[str, float | int] = {
                    "trainer/global_step": int(step),
                    "trainer/total_training_steps": int(self.total_steps),
                    "env/success_rate": succ_rate,
                    "env/success_total": float(sum(successes)),
                    "env/rollout_n": float(rollout_n),
                    "env/episode_steps_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
                    "env/episode_steps_max": float(np.max(ep_lens)) if ep_lens else 0.0,
                    "grpo/group_advantage_std": adv_std,
                    "meta/task_id": int(tid),
                    "meta/init_episode_id": int(eid),
                }
                for k, v in agg.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        sk = str(k).replace("/", "__")
                        wb[f"actor_metrics/{sk}"] = float(v)

                self._wandb_log(step, wb)
                self._append_metrics_jsonl(step, {str(k): float(v) for k, v in wb.items() if isinstance(v, (int, float))})

                try:
                    env.close()
                except Exception:
                    pass

                if step % self.save_interval == 0 or step == self.total_steps:
                    path = self.actor.save_checkpoint(str(self.output_dir), step)
                    _log(f"[checkpoint] saved {path}")
        finally:
            self._close_metrics_jsonl()
            self._wandb_finish()


def build_trainer_from_cfg_file(cfg_file: str | Path, cli_overrides: list[str] | None = None):
    base = OmegaConf.load(cfg_file)
    if cli_overrides:
        cli_cfg = OmegaConf.from_cli(cli_overrides)
        base = OmegaConf.merge(base, cli_cfg)
    OmegaConf.resolve(base)
    return LiberoOnlineGRPOTrainer(base)
