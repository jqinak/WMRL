from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lewm import ARPredictor, Embedder, JEPA, MLP
from wmrl.workers.tokenizer_bridge import TokenizerBridge


class LewmRewardWorker:
    """LE-WM 奖励 worker：默认 pred_vs_gt_pixel — predictor(像素上下文+预测动作) 与 GT 像素 encoder 表征的余弦相似度。"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.runtime.device if torch.cuda.is_available() else "cpu")
        self.bridge = TokenizerBridge()
        self.smoke_random_init = bool(config.runtime.get("smoke_random_init", False))
        self.model = None
        self.history_size = 3
        self.num_preds = 1
        self.expected_action_dim = None
        self.image_size = 224
        self.use_fallback = bool(config.reward.fallback_to_action_embedding)
        self.strict_load = bool(config.reward.get("strict_lewm_load", True))
        self.min_load_ratio = float(config.reward.get("min_param_load_ratio", 0.90))
        self._try_load_lewm()

    def _try_load_lewm(self):
        if self.smoke_random_init:
            try:
                self.model = self._build_jepa_from_config()
                self.model.to(self.device).eval()
                print("[smoke] lewm uses random initialization (no lewm_ckpt load).")
            except Exception as e:
                self.model = None
                self.use_fallback = True
                print(f"[smoke] lewm random init failed, fallback reward enabled: {e}")
            return

        ckpt_path = Path(self.config.paths.lewm_ckpt)
        if not ckpt_path.exists():
            if self.use_fallback:
                print(f"[LEWM] checkpoint missing, using fallback: {ckpt_path}")
                return
            raise FileNotFoundError(f"LE-WM checkpoint not found: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[LEWM] torch.load failed ({ckpt_path}): {e}")
            ckpt = None

        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt.to(self.device).eval()
            print(f"[LEWM] loaded full nn.Module from {ckpt_path}")
            return
        if ckpt is not None and hasattr(ckpt, "encode") and hasattr(ckpt, "predict"):
            self.model = ckpt.to(self.device).eval()
            print(f"[LEWM] loaded object with encode/predict from {ckpt_path}")
            return

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            try:
                self.model = self._build_jepa_from_config()
                filtered = {}
                for k, v in ckpt["state_dict"].items():
                    if k.startswith("model."):
                        filtered[k[len("model.") :]] = v
                self._load_state_dict_checked(self.model, filtered)
                self.model.to(self.device).eval()
                print(f"[LEWM] loaded state_dict (model.* keys) from {ckpt_path}, params matched checked.")
                return
            except Exception as e:
                self.model = None
                print(f"[LEWM] state_dict branch failed: {e}")

        if isinstance(ckpt, dict) and any(str(k).startswith("encoder.") for k in ckpt.keys()):
            try:
                self.model = self._build_jepa_from_config()
                self._load_state_dict_checked(self.model, ckpt)
                self.model.to(self.device).eval()
                print(f"[LEWM] loaded flat encoder.* state_dict from {ckpt_path}")
                return
            except Exception as e:
                self.model = None
                print(f"[LEWM] encoder.* branch failed: {e}")

        if not self.use_fallback:
            raise RuntimeError(
                "Failed to load LE-WM checkpoint in supported formats. "
                "Set reward.fallback_to_action_embedding=true to allow fallback."
            )
        print("[LEWM] all load branches failed; using fallback reward (cosine actions).")

    def _load_state_dict_checked(self, model: torch.nn.Module, state_dict: dict[str, torch.Tensor]):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        total = len(model.state_dict().keys())
        loaded = total - len(missing)
        ratio = 0.0 if total == 0 else loaded / total
        if self.strict_load and ratio < self.min_load_ratio:
            raise RuntimeError(
                f"LE-WM checkpoint load ratio too low: loaded={loaded}/{total} ({ratio:.2%}), "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def _build_jepa_from_config(self):
        from omegaconf import OmegaConf
        import stable_pretraining as spt

        override_cfg = self.config.paths.get("lewm_cfg_path", None)
        if override_cfg:
            cfg_path = Path(override_cfg)
        else:
            cfg_path = Path(self.config.paths.lewm_repo) / "config/train/lewm.yaml"
        cfg = OmegaConf.load(str(cfg_path))
        self.history_size = int(cfg.wm.history_size)
        self.num_preds = int(cfg.wm.num_preds)
        self.image_size = int(cfg.img_size)

        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_dim = encoder.config.hidden_size
        embed_dim = cfg.wm.get("embed_dim", hidden_dim)
        # 某些 lewm 配置没有 wm.action_dim/data.dataset.frameskip，这里兜底为 7。
        action_dim = int(cfg.wm.get("action_dim", 7))
        frameskip = int(cfg.get("data", {}).get("dataset", {}).get("frameskip", 1)) if hasattr(cfg, "get") else 1
        effective_act_dim = frameskip * action_dim
        self.expected_action_dim = int(effective_act_dim)

        predictor = ARPredictor(
            num_frames=cfg.wm.history_size,
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **cfg.predictor,
        )
        action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
        projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
        return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=projector, pred_proj=predictor_proj)

    @staticmethod
    def _to_chw_float(image_like: Any) -> torch.Tensor:
        if isinstance(image_like, torch.Tensor):
            x = image_like.detach().float()
            if x.ndim != 3:
                raise ValueError(f"Image tensor must be 3D, got shape={tuple(x.shape)}")
            chw = x if x.shape[0] in (1, 3) else x.permute(2, 0, 1)
        else:
            arr = np.asarray(image_like if not isinstance(image_like, Image.Image) else image_like)
            if arr.ndim != 3:
                raise ValueError(f"Image array must be HWC, got shape={arr.shape}")
            chw = torch.from_numpy(arr).permute(2, 0, 1).float()
        if chw.max() > 1.0:
            chw = chw / 255.0
        return chw

    def _resize_chw(self, chw: torch.Tensor) -> torch.Tensor:
        x = chw.unsqueeze(0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def _extract_pixels_sequence(self, examples: list[dict], seq_len: int) -> torch.Tensor:
        seq_batch = []
        for ex in examples:
            image_field = ex["image"]
            if isinstance(image_field, list):
                frames = image_field[:seq_len] if len(image_field) >= seq_len else [image_field[0]] * seq_len
            else:
                frames = [image_field] * seq_len
            frame_tensors = [self._resize_chw(self._to_chw_float(frm)) for frm in frames]
            seq_batch.append(torch.stack(frame_tensors, dim=0))
        return torch.stack(seq_batch, dim=0).to(self.device)

    def _match_action_dim(self, actions: torch.Tensor) -> torch.Tensor:
        if self.expected_action_dim is None:
            return actions
        cur_dim = actions.shape[-1]
        exp_dim = int(self.expected_action_dim)
        if cur_dim == exp_dim:
            return actions
        if exp_dim % cur_dim == 0:
            factor = exp_dim // cur_dim
            return actions.repeat_interleave(factor, dim=-1)
        if cur_dim < exp_dim:
            return F.pad(actions, (0, exp_dim - cur_dim))
        return actions[..., :exp_dim]

    def _prepare_action_sequence(self, actions: torch.Tensor, seq_len: int) -> torch.Tensor:
        bsz, t, dim = actions.shape
        if t >= seq_len:
            out = actions[:, :seq_len, :]
        else:
            last = actions[:, -1:, :].expand(bsz, seq_len - t, dim)
            out = torch.cat([actions, last], dim=1)
        return self._match_action_dim(out)

    def _compute_step_similarity(self, examples: list[dict], pred_actions: torch.Tensor, gt_actions: torch.Tensor) -> torch.Tensor:
        if self.model is not None and hasattr(self.model, "encode") and hasattr(self.model, "predict"):
            with torch.no_grad():
                seq_len = int(self.history_size + self.num_preds)
                pixels = self._extract_pixels_sequence(examples, seq_len=seq_len)
                pred_seq = self._prepare_action_sequence(pred_actions.to(self.device), seq_len=seq_len)
                mode = str(self.config.reward.get("lewm_compare_mode", "pred_vs_gt_pixel")).lower()

                # 默认 pred_vs_gt_pixel：GT 像素序列仅过 encode(+projector) 得 emb_gt（表征与动作无关）。
                # 预测侧：历史 ctx_emb + 预测动作经 action_encoder 再经 predictor 得 pred_emb。
                # 奖励：cos(pred_emb, tgt_emb)，tgt_emb 为 GT 像素在对应时间片的 encoder 表征。
                if mode == "pred_vs_gt_pixel":
                    emb_info = self.model.encode({"pixels": pixels})
                    print(f"pred_vs_gt_pixel: pixels shape: {pixels.shape}")
                    emb_gt = emb_info["emb"]
                elif mode == "joint_encode_gt_action":
                    gt_seq = self._prepare_action_sequence(gt_actions.to(self.device), seq_len=seq_len)
                    emb_info = self.model.encode({"pixels": pixels, "action": gt_seq})
                    emb_gt = emb_info["emb"]
                else:
                    raise ValueError(
                        f"Unsupported reward.lewm_compare_mode: {mode!r} "
                        "(use 'pred_vs_gt_pixel' or 'joint_encode_gt_action')."
                    )

                ctx_emb = emb_gt[:, : self.history_size]
                tgt_emb = emb_gt[:, self.num_preds :]
                pred_act_emb = self.model.action_encoder(pred_seq[:, : self.history_size])
                pred_emb = self.model.predict(ctx_emb, pred_act_emb)
                return F.cosine_similarity(pred_emb, tgt_emb, dim=-1)

        if not self.use_fallback:
            raise RuntimeError("LE-WM model unavailable and fallback disabled.")
        return F.cosine_similarity(pred_actions.to(self.device), gt_actions.to(self.device), dim=-1)

    @staticmethod
    def _align_step_reward_to_horizon(step_reward: torch.Tensor, horizon: int) -> torch.Tensor:
        """将 LEWM 相似度 [B, T_wm] 与策略 rollout 步数 horizon 对齐（fallback 路径已为 [B, horizon]）。"""
        if step_reward.dim() != 2:
            raise ValueError(f"step_reward must be 2D, got shape={tuple(step_reward.shape)}")
        b, t = step_reward.shape
        if t == horizon:
            return step_reward
        if t == 1:
            return step_reward.expand(b, horizon)
        x = step_reward.unsqueeze(1).float()
        x = F.interpolate(x, size=horizon, mode="linear", align_corners=False)
        return x.squeeze(1).to(step_reward.dtype)

    def _aggregate(self, step_reward: torch.Tensor) -> torch.Tensor:
        mode = str(self.config.reward.aggregate).lower()
        if mode == "last":
            return step_reward[:, -1]
        if mode == "discount":
            gamma = float(self.config.reward.discount)
            t = step_reward.size(1)
            weight = torch.pow(
                torch.tensor(gamma, device=step_reward.device, dtype=step_reward.dtype),
                torch.arange(t - 1, -1, -1, device=step_reward.device, dtype=step_reward.dtype),
            )
            return (step_reward * weight.unsqueeze(0)).sum(dim=1) / weight.sum()
        return step_reward.mean(dim=1)

    def compute_rewards(self, examples: list[dict], predicted_actions: torch.Tensor) -> dict[str, Any]:
        pred = predicted_actions.to(self.device)
        horizon = pred.shape[1]
        gt = self.bridge.extract_actions(examples, self.device, horizon=horizon).to(self.device)
        step_reward = self._compute_step_similarity(examples, pred, gt)
        step_reward = self._align_step_reward_to_horizon(step_reward, horizon)
        # 归一化前统计：用于日志观察「真实相似度/回报」是否随训练上升（normalize 后 reward_mean≈0 无信息量）
        sample_reward_raw = self._aggregate(step_reward)
        reward_mean_raw = float(sample_reward_raw.mean().detach().cpu())
        reward_std_raw = float(sample_reward_raw.std(unbiased=False).detach().cpu())
        step_reward_mean_raw = float(step_reward.mean().detach().cpu())
        step_reward_std_raw = float(step_reward.std(unbiased=False).detach().cpu())

        if bool(self.config.reward.normalize):
            step_reward = (step_reward - step_reward.mean()) / (step_reward.std(unbiased=False) + 1e-6)

        sample_reward = self._aggregate(step_reward)
        action_dim = pred.shape[-1]
        reward_mode = str(self.config.reward.get("tokenization_mode", "time_action")).lower()
        if reward_mode == "time_action":
            token_level_rewards = step_reward.unsqueeze(-1).expand(-1, -1, action_dim).reshape(pred.shape[0], horizon * action_dim)
        elif reward_mode == "sample_broadcast":
            token_level_rewards = sample_reward[:, None].repeat(1, horizon * action_dim)
        else:
            raise ValueError(f"Unsupported reward.tokenization_mode: {reward_mode}")

        return {
            "token_level_rewards": token_level_rewards.detach().cpu(),
            "reward_mean": float(sample_reward.mean().detach().cpu()),
            "reward_std": float(sample_reward.std(unbiased=False).detach().cpu()),
            "step_reward_mean": float(step_reward.mean().detach().cpu()),
            "step_reward_std": float(step_reward.std(unbiased=False).detach().cpu()),
            "reward_mean_raw": reward_mean_raw,
            "reward_std_raw": reward_std_raw,
            "step_reward_mean_raw": step_reward_mean_raw,
            "step_reward_std_raw": step_reward_std_raw,
        }
