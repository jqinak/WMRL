"""
wmrl 强化学习核心包。

该包只放项目自定义逻辑，不修改上游 verl/starVLA/le-wm 代码。
"""

import importlib
import sys

# 兼容导入：允许通过 wmrl.model / wmrl.wm 访问顶层适配包。
for _pkg in ("model", "wm"):
    try:
        _mod = importlib.import_module(_pkg)
        sys.modules[f"{__name__}.{_pkg}"] = _mod
    except Exception:
        pass
"""
wmrl RL core package.
"""

import importlib
import sys

# Compatibility alias: allow imports using wmrl.model / wmrl.wm
for _pkg in ("model", "wm"):
    try:
        _mod = importlib.import_module(_pkg)
        sys.modules[f"{__name__}.{_pkg}"] = _mod
    except Exception:
        pass
