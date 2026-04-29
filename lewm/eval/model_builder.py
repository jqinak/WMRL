"""Build JEPA (LEWM) from Hydra config and load Lightning checkpoints."""

from __future__ import annotations

from pathlib import Path

import stable_pretraining as spt
import torch

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def build_jepa_model(cfg) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )


def load_jepa_checkpoint(model: JEPA, ckpt_path: str | Path, device: torch.device) -> None:
    path = Path(ckpt_path).expanduser().resolve()
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"]
    model_sd = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict strict failed: missing={missing} unexpected={unexpected}")
