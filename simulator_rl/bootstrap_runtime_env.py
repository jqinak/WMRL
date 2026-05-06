"""可写 tmp 根目录 + robosuite 日志（须在首次 ``import robosuite`` 之前执行）。"""

from __future__ import annotations

import os
import pathlib
import site
import sys


def default_tmp_root() -> str:
    return os.environ.get("WMRL_TMP_ROOT", "/project/peilab/qjl/2026/tmp")


def _site_package_roots() -> list[pathlib.Path]:
    roots: list[pathlib.Path] = []
    for d in site.getsitepackages():
        roots.append(pathlib.Path(d))
    try:
        u = site.getusersitepackages()
        if u:
            roots.append(pathlib.Path(u))
    except Exception:
        pass
    return roots


def patch_robosuite_log_utils_site_packages() -> bool:
    """将 robosuite 中硬编码 ``/tmp/robosuite.log`` 改为读取 ``ROBOSUITE_LOG_FILE``（与 StarVLA ensure_robosuite_log_path.py 一致）。"""
    needle = 'fh = logging.FileHandler("/tmp/robosuite.log")'
    replacement = 'fh = logging.FileHandler(os.environ.get("ROBOSUITE_LOG_FILE", "/tmp/robosuite.log"))'

    for base in _site_package_roots():
        path = base / "robosuite" / "utils" / "log_utils.py"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "ROBOSUITE_LOG_FILE" in text and "os.environ.get" in text:
            return True
        if needle not in text:
            print(
                f"[bootstrap_runtime_env] no hardcoded /tmp line in {path}; skip.",
                file=sys.stderr,
            )
            return False
        if "\nimport os\n" not in text and not text.lstrip().startswith("import os\n"):
            if "import logging\n" not in text:
                print(f"[bootstrap_runtime_env] unexpected {path} layout", file=sys.stderr)
                return False
            text = text.replace("import logging\n", "import logging\nimport os\n", 1)
        text = text.replace(needle, replacement, 1)
        path.write_text(text, encoding="utf-8")
        print(f"[bootstrap_runtime_env] patched {path}")
        return True

    print("[bootstrap_runtime_env] robosuite/utils/log_utils.py not found.", file=sys.stderr)
    return False


def apply_shared_tmp_and_robosuite_patch(tmp_root: str | None = None) -> None:
    """创建共享 tmp、设置 TMPDIR、以及 ROBOSUITE_LOG_FILE，并尽量 patch robosuite（仅当尚未 patch）。"""
    root = os.path.expanduser((tmp_root or default_tmp_root()).strip())
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", root)
    os.environ.setdefault("TMP", root)
    os.environ.setdefault("TEMP", root)

    log_file = str(pathlib.Path(root) / "robosuite.log")
    os.environ.setdefault("ROBOSUITE_LOG_FILE", log_file)

    patch_robosuite_log_utils_site_packages()
