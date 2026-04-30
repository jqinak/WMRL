"""LEWM autoregressive micro-step embedding rollout (initialized from first observation only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image


def _pad_time_left(seq: torch.Tensor, target_len: int, pad_pattern: str = "repeat_first") -> torch.Tensor:
    """seq: [B, T, D] -> last target_len slices, left-padded if needed."""
    b, t, d = seq.shape
    if t >= target_len:
        return seq[:, -target_len:, :]
    need = target_len - t
    if pad_pattern == "repeat_first":
        left = seq[:, 0:1, :].expand(b, need, d)
        return torch.cat([left, seq], dim=1)
    raise ValueError(f"Unknown pad_pattern: {pad_pattern}")


def encode_pixels_bt(model: Any, pixels_btc_hw: torch.Tensor) -> torch.Tensor:
    """Encode (B*T) frames → emb (B*T, Din) then rearrange → (B, T, Din)."""
    b, tm, c, h, w = pixels_btc_hw.shape
    x = rearrange(pixels_btc_hw.float(), "b t ... -> (b t) ...")
    raw = {"pixels": x}
    out = model.encode(raw)
    emb_flat = out["emb"]
    return rearrange(emb_flat, "(b t) d -> b t d", b=b)


def predict_micro_emb_sequence_open_loop(
    model: Any,
    first_frame_chw: torch.Tensor,
    micro_actions_bnad: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """Roll LEWM predictor for n micro-steps.

    Uses only encoder(first_frame); subsequent latents come from predictor. Action history padded on the left.

    Args:
        model: JEPA-like with encode({"pixels"}), predict(emb_hist, action_emb_hist), action_encoder.
        first_frame_chw: [B, 3, H, W].
        micro_actions_bnad: [B, n, A] matched to Embedder expected dim.

    Returns:
        pred_embs [B, n, D_emb] predictor outputs at steps 0..n-1 (alignment with milestone index k).
    """
    device = first_frame_chw.device
    b, n_micro, act_dim = micro_actions_bnad.shape
    if n_micro < 1:
        raise ValueError("n_micro must be >= 1")

    inp = rearrange(first_frame_chw.unsqueeze(1), "b t c h w -> b t c h w").contiguous()
    emb0_seq = encode_pixels_bt(model, inp)
    emb0 = emb0_seq[:, 0, :]
    emb_seq = emb0.unsqueeze(1)
    preds: list[torch.Tensor] = []

    for tau in range(n_micro):
        act_slice = micro_actions_bnad[:, : tau + 1, :].contiguous()
        act_hist = _pad_time_left(act_slice, history_size)
        act_emb_hist = model.action_encoder(act_hist)
        emb_hist = _pad_time_left(emb_seq, history_size)
        pred_step = model.predict(emb_hist, act_emb_hist)[:, -1:, :]
        preds.append(pred_step.squeeze(1))
        emb_seq = torch.cat([emb_seq, pred_step], dim=1)

    return torch.stack(preds, dim=1)


def pil_batch_to_pixels_btc(
    batches_of_lists: list[list[Any]],
    image_size: int,
    device: torch.device,
    dtype=torch.float32,
) -> torch.Tensor:
    """batches_of_lists[b][t] PIL or ndarray -> FloatTensor [B, T, C, H, W] in [0,1]."""
    out = []
    for views in batches_of_lists:
        chw_rows = []
        for im in views:
            arr = np.asarray(im if not isinstance(im, Image.Image) else im.convert("RGB"))
            if arr.ndim != 3:
                raise ValueError(f"Expected HWC image, shape={arr.shape}")
            tchw = torch.from_numpy(arr).permute(2, 0, 1).float()
            if tchw.max() > 1.0:
                tchw /= 255.0
            tchw = F.interpolate(tchw.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
            chw_rows.append(tchw.squeeze(0))
        out.append(torch.stack(chw_rows, dim=0))
    return torch.stack(out, dim=0).to(device=device, dtype=dtype)
