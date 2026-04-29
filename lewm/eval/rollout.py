"""Open-loop rollout aligned with training (take predictor last timestep)."""

from __future__ import annotations

import torch


def predict_next_embedding(model, emb_window: torch.Tensor, act_emb_window: torch.Tensor) -> torch.Tensor:
    """One open-loop step: (B, H, D) + (B, H, A) -> (B, 1, D)."""
    out = model.predict(emb_window, act_emb_window)
    return out[:, -1:]


def rollout_full_episode_to_last(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
) -> torch.Tensor:
    """
    Start from ground-truth embeddings for the first `history_size` frames, then
    open-loop until the sequence has length L. Returns the final predicted
    embedding (B, 1, D) matching the last frame index.

    emb, act_emb: (B, L, D) with L >= history_size.
    """
    b, l, _ = emb.shape
    h = history_size
    if l < h:
        raise ValueError(f"Sequence length L={l} < history_size={h}")

    chain = emb[:, :h].clone()
    for s in range(l - h):
        w_e = chain[:, -h:]
        w_a = act_emb[:, s : s + h]
        nxt = predict_next_embedding(model, w_e, w_a)
        chain = torch.cat([chain, nxt], dim=1)
    return chain[:, -1:]


def rollout_n_steps_open_loop(
    model,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    start_t: int,
    n: int,
) -> torch.Tensor:
    """
    Bootstrap with ground-truth emb[:, start_t : start_t + H], then n open-loop
    steps. Returns embedding at index start_t + H + n - 1 (B, 1, D).
    """
    b, l, _ = emb.shape
    h = history_size
    if start_t + h + n > l:
        raise ValueError("start_t + history_size + n exceeds sequence length")
    chain = emb[:, start_t : start_t + h].clone()
    for i in range(n):
        w_e = chain[:, -h:]
        w_a = act_emb[:, start_t + i : start_t + i + h]
        nxt = predict_next_embedding(model, w_e, w_a)
        chain = torch.cat([chain, nxt], dim=1)
    return chain[:, -1:]
