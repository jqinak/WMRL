#!/usr/bin/env python3
"""Entry: OmegaConf YAML (+ optional argparse overrides ``key=value``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WMRL_ROOT = Path(__file__).resolve().parents[1]
if str(_WMRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_WMRL_ROOT))

from simulator_rl.bootstrap_runtime_env import apply_shared_tmp_and_robosuite_patch

apply_shared_tmp_and_robosuite_patch()

from simulator_rl.compat_starvla_paths import ensure_starvla_deployment_aliases

ensure_starvla_deployment_aliases()


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="LIBERO online GRPO via simulator_rl")
    p.add_argument(
        "--config",
        required=True,
        help="YAML path e.g. simulator_rl/config/simulator_grpo_libero.yaml",
    )
    cfg_path, overrides = p.parse_known_args(argv)
    from simulator_rl.grpo_libero_trainer import build_trainer_from_cfg_file

    trainer = build_trainer_from_cfg_file(cfg_path.config, overrides or None)
    trainer.fit()


if __name__ == "__main__":
    main()
