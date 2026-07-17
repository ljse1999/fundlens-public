"""Project settings and environment loading for fundlens.

Loads environment variables (API keys, etc.) from a small ordered chain of
.env files and exposes a cached ``Settings`` object with the project's
canonical directories (project root, disk-cache dir, reports dir).
"""
from __future__ import annotations

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
    """Load .env files in order; earlier files win (do not override already-set vars).

    Order:
      1. ``<project_root>/.env``
      2. ``C:/Users/hp/research_tools/.env``
      3. ``~/.env``
    """
    candidates = [
        project_root / ".env",
        Path("C:/Users/hp/research_tools/.env"),
        Path.home() / ".env",
    ]
    for candidate in candidates:
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

    On first call, loads environment variables from the ``.env`` chain
    (project root, then ``research_tools``, then home directory) before
    constructing the settings object.
    """
    root = _project_root()
    _load_env_chain(root)
    return Settings(project_root=root)
