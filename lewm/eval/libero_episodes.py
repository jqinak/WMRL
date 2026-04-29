"""Load full Libero episodes for offline evaluation (aligned with LiberoParquetDataset)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from libero_dataset import LiberoParquetDataset


def list_episode_indices(dataset: LiberoParquetDataset) -> List[int]:
    return list(dataset.episode_indices)


def load_raw_episode(
    dataset: LiberoParquetDataset, episode_index: int
) -> Dict[str, Any]:
    """Load one full episode as numpy (before transform). Same layout as __getitem__ slices."""
    episode = dataset._load_episode(episode_index)
    length = int(dataset.episodes[episode_index]["length"])
    frameskip = dataset.frameskip
    out: Dict[str, Any] = {}
    for key in dataset.keys_to_load:
        col = dataset.key_mapping.get(key, key)
        if key == "action":
            rows = []
            for base in range(length):
                chunk = episode[col][base : base + frameskip]
                rows.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            out[key] = np.stack(rows, axis=0)
        else:
            values = [episode[col][i] for i in range(length)]
            if key == "pixels":
                out[key] = np.stack(
                    [dataset._decode_image_bytes(v) for v in values], axis=0
                )
            else:
                out[key] = np.asarray(values, dtype=np.float32)
    return out


def apply_transform_to_episode(
    sample: Dict[str, Any], transform
) -> Dict[str, torch.Tensor]:
    """Run dataset transform on full-length sample (T, ...)."""
    if transform is None:
        return {
            k: torch.as_tensor(v)
            for k, v in sample.items()
            if k in ("pixels", "action", "state")
        }
    return transform(sample)


def episode_dict_to_batch(
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Add batch dim B=1 and move to device."""
    batch: Dict[str, torch.Tensor] = {}
    pixels = tensors["pixels"].float()
    batch["pixels"] = pixels.unsqueeze(0).to(device, non_blocking=True)
    batch["action"] = torch.nan_to_num(
        tensors["action"].float(), 0.0
    ).unsqueeze(0).to(device, non_blocking=True)
    if "state" in tensors:
        batch["state"] = torch.nan_to_num(
            tensors["state"].float(), 0.0
        ).unsqueeze(0).to(device, non_blocking=True)
    return batch


def encode_pixels_in_chunks(
    model,
    pixels_bthwc: torch.Tensor,
    actions_bta: torch.Tensor,
    chunk_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode (B,L,...) in temporal chunks; returns emb (B,L,D), act_emb (B,L,Da)."""
    b, l = pixels_bthwc.shape[:2]
    embs = []
    act_embs = []
    device = pixels_bthwc.device
    for start in range(0, l, chunk_t):
        end = min(start + chunk_t, l)
        sub = {"pixels": pixels_bthwc[:, start:end], "action": actions_bta[:, start:end]}
        out = model.encode(sub)
        embs.append(out["emb"])
        act_embs.append(out["act_emb"])
    emb = torch.cat(embs, dim=1)
    act_emb = torch.cat(act_embs, dim=1)
    return emb, act_emb
