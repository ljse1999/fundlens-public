from __future__ import annotations

import tempfile
from pathlib import Path

from fundlens import config
from fundlens.cache import DiskCache


def test_installed_package_finds_checkout_from_cwd(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fundlens'\n")
    (tmp_path / "app.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config,
        "__file__",
        "/home/adminuser/venv/lib/python3.14/site-packages/fundlens/config.py",
    )

    assert config._project_root() == tmp_path.resolve()


def test_runtime_directories_default_to_writable_temp(monkeypatch):
    monkeypatch.delenv("FUNDLENS_CACHE_DIR", raising=False)
    monkeypatch.delenv("FUNDLENS_REPORTS_DIR", raising=False)
    settings = config.Settings(
        project_root=Path("/home/adminuser/venv/lib/python3.14")
    )

    temp_root = Path(tempfile.gettempdir()).resolve()
    assert settings.cache_dir == temp_root / "fundlens-cache"
    assert settings.reports_dir == temp_root / "fundlens-reports"
    # Exercise the same mkdir that previously raised PermissionError.
    assert DiskCache(settings.cache_dir).cache_dir == settings.cache_dir
