from __future__ import annotations

import copy
import glob
import importlib
import itertools
import os
import random
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

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_dataloader(self):
        starvla_cfg = OmegaConf.load(self.config.data.starvla_cfg)
        print("正在构建数据加载器...")
        dataset = get_vla_dataset(data_cfg=starvla_cfg.datasets.vla_data)
        print("数据加载器构建完成")
        print(f"数据集长度: {len(dataset)}")
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
        print(f"[resume] loaded {latest}, start from step {step + 1}")
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
        data_iter = itertools.cycle(self.train_dataloader)
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
            advantages, _returns = self._compute_advantage(token_level_rewards, group_index)
            if advantages.shape != old_log_probs.shape:
                raise RuntimeError(f"advantages shape {advantages.shape} != old_log_probs shape {old_log_probs.shape}")

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
            # actor 侧 ratio 已与 max_ratio_guard 对齐截断到 [1/guard, guard]；均值可能 >5，随机初始化 smoke 无意义
            smoke = bool(getattr(self.config.runtime, "smoke_random_init", False))
            if not smoke and not (0.2 <= ratio_mean <= 5.0):
                raise RuntimeError(f"Suspicious PPO ratio_mean={ratio_mean:.4f}, likely unstable update.")

            if global_step == self.start_step:
                print(
                    f"[sanity step {global_step}] "
                    f"reward_mean={reward_out['reward_mean']:.4f} "
                    f"step_reward_std={reward_out['step_reward_std']:.4f} "
                    f"actor_loss={update_metrics['actor/loss']:.4f} "
                    f"ppo_kl={update_metrics['actor/ppo_kl']:.4f} "
                    f"ratio_mean={update_metrics['actor/ratio_mean']:.4f} "
                    f"grad_norm={update_metrics['actor/grad_norm']:.4f}"
                )

            if global_step % self.log_interval == 0:
                print(
                    f"[step {global_step}] "
                    f"reward_mean={reward_out['reward_mean']:.4f} "
                    f"reward_std={reward_out['reward_std']:.4f} "
                    f"step_reward_mean={reward_out['step_reward_mean']:.4f} "
                    f"step_reward_std={reward_out['step_reward_std']:.4f} "
                    f"actor_loss={update_metrics['actor/loss']:.4f} "
                    f"ppo_kl={update_metrics['actor/ppo_kl']:.4f} "
                    f"ratio_mean={update_metrics['actor/ratio_mean']:.4f} "
                    f"ratio_max={update_metrics['actor/ratio_max']:.4f}"
                )

            if global_step % self.save_interval == 0 or global_step == self.total_steps:
                ckpt = self.actor_worker.save_checkpoint(str(self.output_dir), global_step)
                print(f"[step {global_step}] saved checkpoint: {ckpt}")
