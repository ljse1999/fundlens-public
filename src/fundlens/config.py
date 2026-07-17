"""Project settings and environment loading for fundlens.

Loads environment variables (API keys, etc.) from a small ordered chain of
.env files and exposes a cached ``Settings`` object with the project's
canonical directories (project root, disk-cache dir, reports dir).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Resolve the project root: the parent of the ``src`` directory.

    This file lives at ``<project_root>/src/fundlens/config.py``, so
    ``parents[0]`` is ``fundlens``, ``parents[1]`` is ``src``, and
    ``parents[2]`` is ``<project_root>``.
    """
    return Path(__file__).resolve().parents[2]


def _load_env_chain(project_root: Path) -> None:
    """Populate ``os.environ`` from Streamlit Cloud secrets, then local ``.env`` files.

    Order (later sources do not override already-set vars):

    1. ``st.secrets`` — used when running under Streamlit Community Cloud.
       The maintainer pastes API keys into the Cloud dashboard's "Secrets" panel;
       they never live in git.
    2. ``<project_root>/.env`` — local development.
    3. ``~/.env`` — local development fallback.
    """
    # Streamlit Cloud: read st.secrets into os.environ (highest priority).
    try:
        import streamlit as st  # type: ignore[import-not-found]

        if hasattr(st, "secrets"):
            for key, value in dict(st.secrets).items():
                os.environ.setdefault(key, str(value))
    except Exception:
        # Not running under Streamlit, or streamlit not installed.
        pass

    # Local development fallback.
    for candidate in (project_root / ".env", Path.home() / ".env"):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)


@dataclass
class Settings:
    """Runtime settings for fundlens.

    Attributes:
        project_root: Root directory of the fundlens project (parent of ``src``).
        cache_dir: Directory used by :class:`fundlens.cache.DiskCache` for
            on-disk caching of dataframes and JSON blobs. Defaults to
            ``<project_root>/.cache``.
        reports_dir: Directory where generated reports are written. Defaults
            to ``<project_root>/reports``.
    """

    project_root: Path
    cache_dir: Path = field(default=None)  # type: ignore[assignment]
    reports_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = self.project_root / ".cache"
        if self.reports_dir is None:
            self.reports_dir = self.project_root / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    On first call, loads environment variables from the ``st.secrets`` chain
    (when running under Streamlit Cloud), then the ``.env`` chain
    (project root, then home directory) before constructing the settings object.
    """
    root = _project_root()
    _load_env_chain(root)
    return Settings(project_root=root)
