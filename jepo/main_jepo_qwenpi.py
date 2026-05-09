from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra

_THIS_DIR = Path(__file__).resolve().parent
_WMRL_ROOT = _THIS_DIR.parent
for _p in (_THIS_DIR, _WMRL_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# WMRL 依赖 ``import starVLA.deployment.*``，但 vendored starVLA 里 ``deployment`` 是仓库根下的独立包；
# ``simulator_rl.compat_starvla_paths`` 会注册 meta_path 别名并把 starVLA 仓库根 prepend 到 sys.path。
from simulator_rl.compat_starvla_paths import ensure_starvla_deployment_aliases

ensure_starvla_deployment_aliases()


@hydra.main(config_path="config", config_name="jepo_qwenpi", version_base=None)
def main(config):
    run_jepo(config)


def run_jepo(config):
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    use_ray = bool(config.runtime.get("use_ray", False))
    if use_ray:
        import ray

        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                    }
                }
            )

    from jepo.trainer import JEPORayTrainer

    trainer = JEPORayTrainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
