"""Tests for the mstarpy Chromium patch (fundlens._cloud_chrome)."""
from __future__ import annotations

import importlib

import mstarpy.utils as mu


def _reload(modname: str):
    mod = importlib.import_module(modname)
    return importlib.reload(mod)


def _unpatch():
    """Restore the original mstarpy browser_options between tests."""
    importlib.reload(mu)


def test_patch_is_noop_when_no_binary_and_not_cloud(monkeypatch):
    """With no binary arg and not on cloud linux, mstarpy is untouched."""
    monkeypatch.setattr("sys.platform", "win32")
    _unpatch()
    mod = _reload("fundlens._cloud_chrome")
    assert mod._is_cloud_linux() is False
    mod._install_patch()  # no-op
    assert not getattr(mu.browser_options, "_fundlens_patched", False)


def test_patch_sets_binary_and_flags_when_binary_given(tmp_path):
    """When a binary path is supplied, browser_options is patched to use it
    and add the cloud-safe flags + headless."""
    _unpatch()
    fake_binary = str(tmp_path / "fake-chromium")
    mod = _reload("fundlens._cloud_chrome")
    mod._install_patch(binary=fake_binary)

    opts = mu.browser_options()
    assert opts.binary_location == fake_binary
    args = opts.arguments if hasattr(opts, "arguments") else []
    assert "--no-sandbox" in args
    assert "--disable-dev-shm-usage" in args
    assert "--remote-debugging-pipe" in args
    assert "--headless=new" in args


def test_patch_uses_explicit_system_driver(monkeypatch, tmp_path):
    """The cloud path bypasses Selenium Manager for Debian's driver."""
    _unpatch()
    fake_browser = str(tmp_path / "fake-chromium")
    fake_driver = str(tmp_path / "fake-chromedriver")
    mod = _reload("fundlens._cloud_chrome")

    captured = {}

    class FakeWebDriver:
        def quit(self):
            captured["quit"] = True

    def fake_chrome(*, service, options):
        captured["service"] = service
        captured["options"] = options
        return FakeWebDriver()

    monkeypatch.setattr("selenium.webdriver.Chrome", fake_chrome)
    mod._install_patch(binary=fake_browser, driver_binary=fake_driver)

    with mu.get_webdriver() as active:
        assert isinstance(active, FakeWebDriver)

    assert captured["service"].path == fake_driver
    assert captured["options"].binary_location == fake_browser
    assert any(
        arg.startswith("--user-data-dir=") for arg in captured["options"].arguments
    )
    assert "--remote-debugging-pipe" in captured["options"].arguments
    assert captured["quit"] is True


def test_patch_is_idempotent(tmp_path):
    """Calling _install_patch twice must not double-register or error."""
    _unpatch()
    fake_binary = str(tmp_path / "fake-chromium")
    mod = _reload("fundlens._cloud_chrome")
    mod._install_patch(binary=fake_binary)
    mod._install_patch(binary=fake_binary)
    assert getattr(mu.browser_options, "_fundlens_patched", False)
