from __future__ import annotations

import copy
import glob
import importlib
import itertools
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    from verl.trainer.ppo import core_algos
except ModuleNotFoundError:
    _wmrl_root = Path(__file__).resolve().parents[2]
    _candidates = [_wmrl_root / "verl", _wmrl_root / "verl" / "verl"]
    for _p in _candidates:
        if str(_p) not in os.sys.path:
            os.sys.path.insert(0, str(_p))
    if "verl" in os.sys.modules:
        del os.sys.modules["verl"]
    core_algos = importlib.import_module("verl.trainer.ppo.core_algos")


from wmrl.workers import ActorRolloutWorker, LewmRewardWorker, TokenizerBridge
from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset


def _flatten_config_for_wandb(cfg) -> dict:
    """OmegaConf/DictConfig → 可 JSON 化的 dict，供 wandb.init(config=...)."""
    try:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        return {}


def _tensor_scalar_stats(t: torch.Tensor, prefix: str) -> dict[str, float]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return {}
    x = t.detach().float().reshape(-1)
    return {
        f"{prefix}/mean": float(x.mean().cpu()),
        f"{prefix}/std": float(x.std(unbiased=False).cpu()),
        f"{prefix}/min": float(x.min().cpu()),
        f"{prefix}/max": float(x.max().cpu()),
        f"{prefix}/absmax": float(x.abs().max().cpu()),
    }


def _log(msg: str) -> None:
    """写到 stdout 并立即刷新；重定向到文件时避免整块缓冲、指标滞后于 stderr 的 Warning。"""
    print(msg, flush=True)


class RayWMRLTrainer:
    """WMRL 训练器：rollout -> reward -> advantage -> actor update。"""

    def __init__(self, config):
        self.config = config  # 保存配置对象
        self._set_seed(int(config.runtime.seed))  # 固定随机种子
        self.bridge = TokenizerBridge()  # 批次桥接器
        self.actor_worker = ActorRolloutWorker(config)  # 策略 worker
        self.reward_worker = LewmRewardWorker(config)  # 奖励 worker
        self.train_dataloader = self._build_dataloader()  # 训练数据流
        self.total_steps = int(config.trainer.total_training_steps)  # 总步数
        self.log_interval = int(config.trainer.log_interval)  # 日志间隔
        self.save_interval = int(config.trainer.save_interval)  # 保存间隔
        self.output_dir = Path(config.trainer.output_dir)  # 输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_step = 1
        if bool(config.trainer.get("auto_resume", False)):
            self.start_step = self._try_resume()
        self._wandb_initialized = False
        self._maybe_init_wandb()

    def _maybe_init_wandb(self) -> None:
        """读取 starvla_cfg（如 qwen35vlPI_libero.yaml）里的 trackers / wandb_*，以及 trainer.wandb 覆盖项。"""
        tw = OmegaConf.select(self.config, "trainer.wandb") or OmegaConf.create({})
        starvla_path = Path(str(self.config.data.starvla_cfg))
        starvla_root = OmegaConf.load(starvla_path) if starvla_path.is_file() else OmegaConf.create({})

        enabled = OmegaConf.select(tw, "enabled")
        if enabled is None:
            trackers = OmegaConf.select(starvla_root, "trackers") or []
            trackers_list = OmegaConf.to_container(trackers, resolve=True)
            if not isinstance(trackers_list, list):
                trackers_list = []
            enabled = any(str(x).lower() == "wandb" for x in trackers_list)
        else:
            enabled = bool(enabled)

        if not enabled:
            return

        try:
            import wandb  # type: ignore[import-untyped]
        except ImportError:
            _log("[wandb] 未安装 wandb，跳过远程记录。安装: pip install wandb")
            return

        entity = OmegaConf.select(tw, "entity") or OmegaConf.select(starvla_root, "wandb_entity")
        project = OmegaConf.select(tw, "project") or OmegaConf.select(starvla_root, "wandb_project")
        if not project:
            project = "wmrl"
            _log("[wandb] 未配置 project，使用默认值 'wmrl'")
        run_name = OmegaConf.select(tw, "run_name") or OmegaConf.select(tw, "name")
        if not run_name:
            run_name = f"wmrl_{self.output_dir.name}_seed{int(self.config.runtime.seed)}"

        tags = OmegaConf.select(tw, "tags")
        tags_list = OmegaConf.to_container(tags, resolve=True) if tags is not None else None
        if not isinstance(tags_list, list):
            tags_list = []
        tags_list = [str(t) for t in tags_list]

        notes = OmegaConf.select(tw, "notes")
        notes_str = str(notes) if notes else None

        cfg_dict = _flatten_config_for_wandb(self.config)
        init_kwargs: dict = {
            "project": str(project),
            "name": str(run_name),
            "config": cfg_dict,
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
        _log(f"[wandb] 已初始化: project={project} name={run_name}" + (f" entity={entity}" if entity else ""))

    def _wandb_log(self, global_step: int, metrics: dict) -> None:
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        out: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                continue
            try:
                x = float(v)
                if np.isfinite(x):
                    out[str(k)] = x
            except (TypeError, ValueError):
                pass
        if out:
            wandb.log(out, step=int(global_step))

    def _wandb_finish(self) -> None:
        if not self._wandb_initialized:
            return
        import wandb  # type: ignore[import-untyped]

        if wandb.run is not None:
            wandb.finish()
        self._wandb_initialized = False

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_dataloader(self):
        starvla_cfg = OmegaConf.load(self.config.data.starvla_cfg)
        _log("正在构建数据加载器...")
        dataset = get_vla_dataset(data_cfg=starvla_cfg.datasets.vla_data)
        _log("数据加载器构建完成")
        _log(f"数据集长度: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Empty dataset from starVLA dataloader.")
        return DataLoader(
            dataset,
            batch_size=int(self.config.data.train_batch_size),
            num_workers=int(self.config.data.num_workers),
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def _try_resume(self) -> int:
        pattern = str(self.output_dir / "global_step_*" / "wmrl_actor.pt")
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            return 1
        latest = ckpts[-1]
        state = torch.load(latest, map_location="cpu")
        self.actor_worker.action_model.load_state_dict(state["action_model"], strict=True)
        self.actor_worker.sigma_net.load_state_dict(state["sigma_net"], strict=True)
        self.actor_worker.optimizer.load_state_dict(state["optimizer"])
        step = int(Path(latest).parent.name.replace("global_step_", ""))
        _log(f"[resume] loaded {latest}, start from step {step + 1}")
        return step + 1

    @staticmethod
    def _repeat_examples(examples: list[dict], repeat_n: int) -> list[dict]:
        return [copy.deepcopy(ex) for ex in examples for _ in range(repeat_n)]

    @staticmethod
    def _build_group_index(base_batch_size: int, repeat_n: int) -> np.ndarray:
        ids = [f"uid-{i}" for i in range(base_batch_size) for _ in range(repeat_n)]
        return np.asarray(ids, dtype=object)

    @staticmethod
    def _assert_finite(name: str, value: torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN/Inf.")

    def _normalize_advantages(self, advantages: torch.Tensor, group_index: np.ndarray) -> torch.Tensor:
        if not bool(self.config.algorithm.get("normalize_advantage", True)):
            return advantages
        mode = str(self.config.algorithm.get("adv_norm_mode", "batch")).lower()
        eps = float(self.config.algorithm.get("adv_norm_eps", 1e-6))
        out = advantages.clone()
        if mode == "batch":
            mean = out.mean()
            std = out.std(unbiased=False)
            return (out - mean) / (std + eps)
        if mode == "group":
            for gid in np.unique(group_index):
                idx = np.where(group_index == gid)[0]
                group_adv = out[idx]
                mean = group_adv.mean()
                std = group_adv.std(unbiased=False)
                out[idx] = (group_adv - mean) / (std + eps)
            return out
        raise ValueError(f"Unsupported algorithm.adv_norm_mode: {mode}")

    def _compute_advantage(self, token_level_rewards: torch.Tensor, group_index: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if token_level_rewards.ndim != 2:
            raise ValueError(f"token_level_rewards should be 2D, got {token_level_rewards.shape}")
        if token_level_rewards.shape[0] != group_index.shape[0]:
            raise ValueError(
                f"Reward batch size {token_level_rewards.shape[0]} != group index size {group_index.shape[0]}"
            )
        response_mask = torch.ones_like(token_level_rewards)
        adv_estimator = str(self.config.algorithm.adv_estimator).lower()
        if adv_estimator == "grpo":
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=group_index,
            )
            advantages = self._normalize_advantages(advantages, group_index)
            self._assert_finite("advantages", advantages)
            self._assert_finite("returns", returns)
            return advantages, returns
        if adv_estimator == "gae":
            values = torch.zeros_like(token_level_rewards)
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=token_level_rewards,
                values=values,
                response_mask=response_mask,
                gamma=float(self.config.algorithm.gamma),
                lam=float(self.config.algorithm.lam),
            )
            advantages = self._normalize_advantages(advantages, group_index)
            self._assert_finite("advantages", advantages)
            self._assert_finite("returns", returns)
            return advantages, returns
        raise NotImplementedError(f"Unsupported adv estimator: {adv_estimator}")

    def fit(self):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
            except Exception:
                pass
        data_iter = itertools.cycle(self.train_dataloader)
        try:
            for global_step in range(self.start_step, self.total_steps + 1):
                raw_batch = next(data_iter)
                examples = self.bridge.normalize_examples(raw_batch)
                repeat_n = int(self.config.algorithm.rollout_n)
                repeated_examples = self._repeat_examples(examples, repeat_n)

                noisy_dict = self.actor_worker.sample_noisy_actions(repeated_examples)
                rollout = self.actor_worker.generate_actions(repeated_examples, noise=noisy_dict["noise"])
                old_log_probs = self.actor_worker.compute_log_prob(repeated_examples, rollout["x_chain"])
                if old_log_probs.shape[0] != len(repeated_examples):
                    raise RuntimeError(
                        f"old_log_probs batch mismatch: {old_log_probs.shape[0]} vs {len(repeated_examples)}"
                    )

                reward_out = self.reward_worker.compute_rewards(repeated_examples, rollout["predicted_actions"])
                token_level_rewards = reward_out["token_level_rewards"]
                self._assert_finite("token_level_rewards", token_level_rewards)
                group_index = self._build_group_index(len(examples), repeat_n)
                advantages, returns = self._compute_advantage(token_level_rewards, group_index)
                if advantages.shape != old_log_probs.shape:
                    raise RuntimeError(
                        f"advantages shape {advantages.shape} != old_log_probs shape {old_log_probs.shape}"
                    )

                min_reward_std = float(self.config.trainer.get("min_reward_std", 1e-6))
                if reward_out["reward_std"] <= min_reward_std:
                    raise RuntimeError(
                        f"Reward std too small ({reward_out['reward_std']:.6e}) <= min_reward_std ({min_reward_std:.6e})."
                    )

                update_metrics = self.actor_worker.update_actor(
                    {
                        "examples": repeated_examples,
                        "x_chain": rollout["x_chain"],
                        "old_log_probs": old_log_probs,
                        "advantages": advantages,
                    }
                )
                for key in ("actor/loss", "actor/ppo_kl", "actor/grad_norm"):
                    self._assert_finite(key, torch.tensor(update_metrics[key]))

                ratio_mean = float(update_metrics.get("actor/ratio_mean", 1.0))
                smoke = bool(getattr(self.config.runtime, "smoke_random_init", False))
                # if not smoke and not (0.2 <= ratio_mean <= 5.0):
                #     raise RuntimeError(f"Suspicious PPO ratio_mean={ratio_mean:.4f}, likely unstable update.")

                if global_step == self.start_step and self.log_interval != 1:
                    _log(
                        f"[sanity step {global_step}] "
                        f"reward_mean={reward_out['reward_mean']:.4f} "
                        f"reward_mean_raw={reward_out['reward_mean_raw']:.4f} "
                        f"step_reward_std={reward_out['step_reward_std']:.4f} "
                        f"actor_loss={update_metrics['actor/loss']:.4f} "
                        f"ppo_kl={update_metrics['actor/ppo_kl']:.4f} "
                        f"ratio_mean={update_metrics['actor/ratio_mean']:.4f} "
                        f"grad_norm={update_metrics['actor/grad_norm']:.4f}"
                    )

                if global_step % self.log_interval == 0:
                    _log(
                        f"[step {global_step}] "
                        f"reward_mean={reward_out['reward_mean']:.4f} "
                        f"reward_mean_raw={reward_out['reward_mean_raw']:.4f} "
                        f"reward_std={reward_out['reward_std']:.4f} "
                        f"reward_std_raw={reward_out['reward_std_raw']:.4f} "
                        f"step_reward_mean={reward_out['step_reward_mean']:.4f} "
                        f"step_reward_mean_raw={reward_out['step_reward_mean_raw']:.4f} "
                        f"step_reward_std={reward_out['step_reward_std']:.4f} "
                        f"step_reward_std_raw={reward_out['step_reward_std_raw']:.4f} "
                        f"actor_loss={update_metrics['actor/loss']:.4f} "
                        f"ppo_kl={update_metrics['actor/ppo_kl']:.4f} "
                        f"ratio_mean={update_metrics['actor/ratio_mean']:.4f} "
                        f"ratio_max={update_metrics['actor/ratio_max']:.4f}"
                    )

                if global_step % self.save_interval == 0 or global_step == self.total_steps:
                    ckpt = self.actor_worker.save_checkpoint(str(self.output_dir), global_step)
                    _log(f"[step {global_step}] saved checkpoint: {ckpt}")

                wb: dict[str, float | torch.Tensor] = {}
                for rk, rv in reward_out.items():
                    if rk == "token_level_rewards":
                        continue
                    wb[f"reward/{rk}"] = rv
                wb.update(update_metrics)
                wb.update(_tensor_scalar_stats(token_level_rewards.float(), "reward/token_level_tensor"))
                wb.update(_tensor_scalar_stats(advantages.float(), "advantage"))
                wb.update(_tensor_scalar_stats(returns.float(), "returns"))
                wb.update(_tensor_scalar_stats(rollout["predicted_actions"].float(), "rollout/predicted_actions"))
                wb.update(_tensor_scalar_stats(old_log_probs.float(), "rollout/old_log_prob"))
                pa = rollout["predicted_actions"]
                wb["meta/base_batch"] = float(len(examples))
                wb["meta/repeated_batch"] = float(len(repeated_examples))
                wb["meta/rollout_n"] = float(repeat_n)
                wb["meta/action_horizon"] = float(pa.shape[1])
                wb["meta/action_dim"] = float(pa.shape[-1])
                self._wandb_log(global_step, wb)
        finally:
            self._wandb_finish()
