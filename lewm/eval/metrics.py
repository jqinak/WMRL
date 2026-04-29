"""Embedding distance metrics for LEWM evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return ((pred - gt) ** 2).mean()


def mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs().mean()


def l2_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - gt, p=2, dim=-1).mean()


def cosine_dist(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1, eps=eps)
    gt_n = F.normalize(gt, dim=-1, eps=eps)
    return (1.0 - (pred_n * gt_n).sum(dim=-1)).mean()


def all_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    return {
        "mse": float(mse(pred, gt).item()),
        "mae": float(mae(pred, gt).item()),
        "l2": float(l2_dist(pred, gt).item()),
        "cosine_dist": float(cosine_dist(pred, gt).item()),
    }
