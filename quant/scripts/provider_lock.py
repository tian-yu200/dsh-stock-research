"""Compatibility import for scripts and tests that load modules from this directory."""

import sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.provider_lock import ProviderLock

__all__ = ["ProviderLock"]
