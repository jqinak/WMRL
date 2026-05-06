"""Resolve relative ``framework.qwenvl.base_vlm`` in ckpt config to an absolute local path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_PATCH_INSTALLED = False
_EFFECTIVE_STAR_ROOT: Path | None = None


def _detect_starvla_repo_root() -> Path | None:
    env = os.environ.get("STARVLA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    wmrl = Path(__file__).resolve().parents[1]
    cand = (wmrl / "starVLA").resolve()
    return cand if cand.is_dir() else None


def _make_absolute_base_vlm(raw: str, star_root: Path) -> str:
    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())
    cleaned = raw.lstrip("./")
    return str((star_root / cleaned).resolve())


def _install_read_mode_config_wrapper() -> None:
    global _PATCH_INSTALLED

    try:
        from starVLA.model.framework import share_tools as st
    except ImportError:
        return

    if _PATCH_INSTALLED:
        return

    st._read_mode_config_orig = st.read_mode_config  # type: ignore[attr-defined]

    def _wrapped(pretrained_checkpoint: str | Path) -> tuple[dict[str, Any], dict]:
        star = _EFFECTIVE_STAR_ROOT
        orig_fn = st._read_mode_config_orig  # type: ignore[attr-defined]
        global_cfg, norm_stats = orig_fn(pretrained_checkpoint)
        if star is None:
            return global_cfg, norm_stats
        try:
            cfg = OmegaConf.create(global_cfg)
            bv = OmegaConf.select(cfg, "framework.qwenvl.base_vlm")
            if bv is None or str(bv).strip() == "":
                return global_cfg, norm_stats
            bvs = str(bv).strip()
            if Path(bvs).is_absolute():
                return global_cfg, norm_stats
            abs_bv = _make_absolute_base_vlm(bvs, star)
            OmegaConf.update(cfg, "framework.qwenvl.base_vlm", abs_bv)
            return OmegaConf.to_container(cfg, resolve=True), norm_stats
        except Exception:
            return global_cfg, norm_stats

    st.read_mode_config = _wrapped
    _PATCH_INSTALLED = True


def apply_read_mode_config_patch(star_root_override: str | Path | None = None) -> None:
    """Patch ``read_mode_config`` before ``baseframework.from_pretrained``.

    Checkpoints often store ``framework.qwenvl.base_vlm`` as ``playground/Pretrained_models/...``
    (relative to StarVLA repo). Hugging Face then mis-treats it as a Hub repo id.

    Args:
        star_root_override: If set and is an existing directory, use as repo root; else ``STARVLA_DIR`` / ``wmrl/starVLA``.
    """
    global _EFFECTIVE_STAR_ROOT

    root: Path | None = None
    if star_root_override is not None and str(star_root_override).strip() not in {"", "null", "~"}:
        p = Path(str(star_root_override)).expanduser().resolve()
        root = p if p.is_dir() else None
    if root is None:
        root = _detect_starvla_repo_root()

    _EFFECTIVE_STAR_ROOT = root
    _install_read_mode_config_wrapper()
