"""JEPO trajectory-level RL package.

The package lives beside the original WMRL implementation and reuses its
upstream adapters without modifying them in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WMRL_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_PATHS = (
    _WMRL_ROOT,
    _WMRL_ROOT / "starVLA",
    _WMRL_ROOT / "lewm",
    _WMRL_ROOT / "verl",
)
for _path in _VENDORED_PATHS:
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
