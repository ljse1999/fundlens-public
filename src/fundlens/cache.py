"""Disk-backed cache for dataframes and JSON blobs with TTL expiry.

Each cache key maps to a parquet file (for dataframes) or a json file (for
json-serialisable objects), plus a small sidecar ``.meta.json`` file that
records a UTC write timestamp used to enforce a caller-supplied
time-to-live (TTL) in days.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.\-]+")


def _sanitise_key(key: str) -> str:
    """Convert a logical cache key (e.g. ``navs/GB00B41YBW71/monthly``) into a safe filename stem."""
    safe = key.replace("/", "__")
    safe = _UNSAFE_CHARS.sub("_", safe)
    return safe


class DiskCache:
    """A simple TTL-based disk cache for pandas DataFrames and JSON objects."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- internal path helpers -------------------------------------------------

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{_sanitise_key(key)}.meta.json"

    def _df_path(self, key: str) -> Path:
        return self.cache_dir / f"{_sanitise_key(key)}.parquet"

    def _json_path(self, key: str) -> Path:
        return self.cache_dir / f"{_sanitise_key(key)}.json"

    def _is_fresh(self, key: str, ttl_days: float) -> bool:
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            written_at = datetime.fromisoformat(meta["written_at"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return False
        if written_at.tzinfo is None:
            written_at = written_at.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - written_at).total_seconds() / 86400.0
        return age_days <= ttl_days

    def _write_meta(self, key: str) -> None:
        meta_path = self._meta_path(key)
        meta = {"written_at": datetime.now(timezone.utc).isoformat()}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # -- dataframe API ----------------------------------------------------------

    def get_df(self, key: str, ttl_days: float) -> pd.DataFrame | None:
        """Return the cached DataFrame for ``key`` if present and not older than ``ttl_days``, else None."""
        df_path = self._df_path(key)
        if not df_path.exists():
            return None
        if not self._is_fresh(key, ttl_days):
            return None
        try:
            return pd.read_parquet(df_path)
        except (OSError, ValueError):
            return None

    def put_df(self, key: str, df: pd.DataFrame) -> None:
        """Write ``df`` to the cache under ``key`` (parquet) with a fresh UTC timestamp sidecar."""
        df_path = self._df_path(key)
        df.to_parquet(df_path)
        self._write_meta(key)

    # -- json API -----------------------------------------------------------------

    def get_json(self, key: str, ttl_days: float) -> Any | None:
        """Return the cached JSON-decoded object for ``key`` if present and not older than ``ttl_days``, else None."""
        json_path = self._json_path(key)
        if not json_path.exists():
            return None
        if not self._is_fresh(key, ttl_days):
            return None
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put_json(self, key: str, obj: Any) -> None:
        """Write ``obj`` (JSON-serialisable) to the cache under ``key`` with a fresh UTC timestamp sidecar."""
        json_path = self._json_path(key)
        json_path.write_text(json.dumps(obj), encoding="utf-8")
        self._write_meta(key)
