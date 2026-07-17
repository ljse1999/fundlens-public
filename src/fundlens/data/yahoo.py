"""Shared, Streamlit-safe access to :mod:`yfinance`.

Streamlit Community Cloud's Python 3.14 image can resolve the default user
cache directory inside its read-only virtual environment.  yfinance stores
timezone, cookie, and ISIN SQLite databases there unless told otherwise.
Configure those databases under the operating system's writable temporary
directory before making any Yahoo request.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_CONFIGURED_DIR: Path | None = None


def get_yfinance(cache_dir: str | Path | None = None) -> Any:
    """Return yfinance after pointing all of its caches at a writable path."""
    global _CONFIGURED_DIR

    configured = cache_dir or os.environ.get("FUNDLENS_YFINANCE_CACHE_DIR")
    target = Path(configured or Path(tempfile.gettempdir()) / "fundlens-yfinance-cache")
    target = target.expanduser().resolve()

    with _LOCK:
        import yfinance as yf

        if _CONFIGURED_DIR != target:
            target.mkdir(parents=True, exist_ok=True)
            if not os.access(target, os.R_OK | os.W_OK):
                raise PermissionError(f"Yahoo cache directory is not writable: {target}")
            # Despite its historic name, this configures yfinance's timezone,
            # cookie, and ISIN databases together.
            yf.set_tz_cache_location(str(target))
            _CONFIGURED_DIR = target
        return yf
