import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from libero_dataset import LiberoParquetDataset
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack

DEFAULT_ROOT = "/project/peilab/qjl/2026/playground/dataset/libero"


def _get_libero_defaults():
    """Read Libero defaults, allowing command-time env overrides."""
    keys_env = os.getenv("LIBERO_KEYS_TO_LOAD", "pixels,action,state")
    keys_to_load = [k.strip() for k in keys_env.split(",") if k.strip()]
    if not keys_to_load:
        keys_to_load = ["pixels", "action", "state"]

    return {
        "root": os.getenv("LIBERO_ROOT", DEFAULT_ROOT),
        "split": os.getenv("LIBERO_SPLIT", "train"),
        "num_steps": int(os.getenv("LIBERO_NUM_STEPS", "4")),
        "frameskip": int(os.getenv("LIBERO_FRAMESKIP", "1")),
        "keys_to_load": keys_to_load,
    }


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def build_dataset(dataset_cfg):
    dataset_cfg_dict = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_type = dataset_cfg_dict.pop("type", "hdf5")

    if dataset_type == "hdf5":
        return swm.data.HDF5Dataset(**dataset_cfg_dict, transform=None)

    if dataset_type == "libero":
        # Keep behavior aligned with test_dataset.py defaults when fields are omitted.
        libero_defaults = _get_libero_defaults()
        for key, value in libero_defaults.items():
            dataset_cfg_dict.setdefault(key, value)
        return LiberoParquetDataset(**dataset_cfg_dict, transform=None)

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def print_first_samples(dataset, n: int = 2):
    """Load and print the first n samples before training."""
    num_to_show = min(max(int(n), 0), len(dataset))
    print(f"[DATA PREVIEW] dataset_len={len(dataset)}, preview_count={num_to_show}")
    for i in range(num_to_show):
        sample = dataset[i]
        print(f"[DATA PREVIEW] sample_index={i}")
        for key, value in sample.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            print(
                f"  - {key}: type={type(value).__name__}, shape={shape}, dtype={dtype}"
            )
            print(f"    value={value}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = build_dataset(cfg.data.dataset)
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    print_first_samples(dataset, n=2)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if loader_cfg.get("num_workers", 0) == 0:
        loader_cfg["persistent_workers"] = False
        loader_cfg["prefetch_factor"] = None

    train = torch.utils.data.DataLoader(
        train_set, **loader_cfg, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **loader_cfg, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

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

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        try:
            logger = WandbLogger(**cfg.wandb.config)
            logger.log_hyperparams(OmegaConf.to_container(cfg))
        except Exception as e:
            # Continue training when W&B auth/project permissions are unavailable.
            print(f"[WARN] WandB disabled due to initialization error: {e}")
            logger = None

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
