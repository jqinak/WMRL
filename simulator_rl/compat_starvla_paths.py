"""Resolve ``import starVLA.deployment.*`` when the repo exposes a top-level ``deployment`` package."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

_installed = False


class _AliasLoader(importlib.abc.Loader):
    __slots__ = ("_real",)

    def __init__(self, real_fullname: str) -> None:
        self._real = real_fullname

    def create_module(self, spec):
        return importlib.import_module(self._real)

    def exec_module(self, module) -> None:
        return


class _StarVLADeploymentFinder(MetaPathFinder):
    _prefix = "starVLA.deployment"

    def find_spec(self, fullname: str, path=None, target=None):
        prefix = self._prefix
        if fullname == prefix:
            real = "deployment"
        elif fullname.startswith(prefix + "."):
            real = "deployment." + fullname[len(prefix) + 1 :]
        else:
            return None
        rspec = importlib.util.find_spec(real)
        if rspec is None:
            return None
        loader = _AliasLoader(real)
        spec = importlib.util.spec_from_loader(fullname, loader, origin=getattr(rspec, "origin", None))
        subs = getattr(rspec, "submodule_search_locations", None)
        if subs is not None:
            spec.submodule_search_locations = list(subs)
        return spec


_FINDER_INSTANCE: _StarVLADeploymentFinder | None = None


def ensure_starvla_deployment_aliases() -> None:
    """Idempotent: prepend StarVLA repo to ``sys.path`` and register import hook for ``starVLA.deployment``."""
    global _installed, _FINDER_INSTANCE
    if _installed:
        return

    env = os.environ.get("STARVLA_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser().resolve())
    wmrl_root = Path(__file__).resolve().parents[1]
    candidates.append((wmrl_root / "starVLA").resolve())

    star_repo: Path | None = None
    for c in candidates:
        if c.is_dir() and (c / "deployment").is_dir() and (c / "starVLA").is_dir():
            star_repo = c
            break
    if star_repo is None:
        return

    sr = str(star_repo)
    if sr not in sys.path:
        sys.path.insert(0, sr)

    if _FINDER_INSTANCE is None:
        _FINDER_INSTANCE = _StarVLADeploymentFinder()
    if _FINDER_INSTANCE not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER_INSTANCE)

    _installed = True
