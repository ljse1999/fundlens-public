"""Tests for the fundlens config env-chain (deployment-hardened version)."""
from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch


def _reload_config():
    import fundlens.config
    importlib.reload(fundlens.config)
    return fundlens.config


def test_env_chain_has_no_hardcoded_research_tools_path():
    """The Windows-specific C:/Users/hp/research_tools path must be gone."""
    config = _reload_config()
    import inspect
    body = inspect.getsource(config._load_env_chain)
    assert "research_tools" not in body, "hardcoded research_tools path leaked into public repo"
    assert "C:/" not in body and "C:\\" not in body, "hardcoded Windows path leaked"


def test_env_chain_reads_st_secrets_when_streamlit_available(tmp_path, monkeypatch):
    """When st.secrets is populated, its keys land in os.environ."""
    import os
    monkeypatch.delenv("FUNDLENS_TEST_KEY", raising=False)

    fake_secrets = {"FUNDLENS_TEST_KEY": "from-secrets-panel"}

    class _FakeSecrets:
        def __iter__(self):
            return iter(fake_secrets)

        def keys(self):
            return fake_secrets.keys()

        def items(self):
            return fake_secrets.items()

        def __getitem__(self, k):
            return fake_secrets[k]

    fake_streamlit = type("M", (), {"secrets": _FakeSecrets()})

    with patch.dict("sys.modules", {"streamlit": fake_streamlit}):
        config = _reload_config()
        monkeypatch.delenv("FUNDLENS_TEST_KEY", raising=False)
        config._load_env_chain(tmp_path)
        assert os.environ.get("FUNDLENS_TEST_KEY") == "from-secrets-panel"


def test_env_chain_falls_back_to_dotenv_without_streamlit(tmp_path, monkeypatch):
    """With streamlit import failing, .env files still load."""
    env_file = tmp_path / ".env"
    env_file.write_text("FUNDLENS_LOCAL_KEY=from-dotenv\n")

    import builtins
    real_import = builtins.__import__

    def _block_streamlit(name, *args, **kwargs):
        if name == "streamlit":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.delenv("FUNDLENS_LOCAL_KEY", raising=False)
    with patch("builtins.__import__", side_effect=_block_streamlit):
        config = _reload_config()
        config._load_env_chain(tmp_path)

    import os
    assert os.environ.get("FUNDLENS_LOCAL_KEY") == "from-dotenv"
