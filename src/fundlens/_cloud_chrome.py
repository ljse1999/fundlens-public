"""Patch mstarpy to use a system Chromium on Streamlit Community Cloud.

Why this exists
---------------
``mstarpy.search.MorningstarSession.__init__`` unconditionally launches Chrome
via Selenium to scrape WAF cookies from ``global.morningstar.com``. On a clean
Streamlit Cloud container there is no Chrome binary and no GUI, so the default
Selenium path fails with "chromedriver unexpectedly exited. Status code: 127"
(missing shared libraries) or "cannot find Chrome binary".

This module monkeypatches ``mstarpy.utils.browser_options`` so that, **only on
Linux when a system Chromium is present**, it:

  1. Sets ``binary_location`` to the Chromium binary (``/usr/bin/chromium``).
  2. Prepends the cloud-safe flags ``--no-sandbox --disable-dev-shm-usage
     --disable-gpu`` so headless Chrome can run inside the container.
  3. Enables headless mode (mstarpy leaves it off by default).

On non-Linux platforms (local dev) it does nothing, so the original behaviour
— using whatever Chrome the developer has installed — is preserved.

How to use
----------
Import this module once, as early as possible, before any fundlens data module
constructs a ``MorningstarSession``. ``app.py`` does this on startup::

    import fundlens._cloud_chrome  # noqa: F401  (register patch)
    from fundlens.ui.streamlit_app import main

The import has side effects (registers the patch) and that is the point.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


#: Candidate Chromium binary locations on a Linux container.
_CHROMIUM_BINARIES = ("/usr/bin/chromium", "/usr/bin/chromium-browser")
_CHROMEDRIVER_BINARIES = ("/usr/bin/chromedriver", "/usr/lib/chromium/chromedriver")


def _find_chromium_binary() -> str | None:
    """Return the first existing Chromium binary path, or None."""
    for candidate in _CHROMIUM_BINARIES:
        if Path(candidate).exists():
            return candidate
    return None


def _find_chromedriver_binary() -> str | None:
    """Return the first existing system ChromeDriver path, or None."""
    for candidate in _CHROMEDRIVER_BINARIES:
        if Path(candidate).exists():
            return candidate
    return None


def _is_cloud_linux() -> bool:
    """True when running on Linux with a system Chromium installed.

    Used to gate the patch so local development on macOS/Windows is unaffected.
    """
    if not sys.platform.startswith("linux"):
        return False
    return _find_chromium_binary() is not None


def _install_patch(binary: str | None = None, driver_binary: str | None = None) -> None:
    """Monkeypatch mstarpy.utils.browser_options for the Cloud Chromium path.

    Args:
        binary: Override the Chromium binary path (used by tests). When None,
            auto-detected via :func:`_is_cloud_linux`; on non-cloud platforms
            this function is a no-op.
        driver_binary: Override the ChromeDriver path. In production this is
            auto-detected from the Debian ``chromium-driver`` package.

    Idempotent: safe to call multiple times.
    """
    if binary is None and not _is_cloud_linux():
        return

    try:
        from mstarpy import utils as _mstarpy_utils
    except Exception:
        # mstarpy not importable yet; nothing to patch.
        return

    if getattr(_mstarpy_utils.browser_options, "_fundlens_patched", False):
        return

    if binary is None:
        binary = _find_chromium_binary()
    if binary is None:
        return
    if driver_binary is None:
        driver_binary = _find_chromedriver_binary()

    # Cloud-safe flags. These are required in sandboxed containers:
    #   --no-sandbox            Chrome refuses to run as root without this.
    #   --disable-dev-shm-usage Avoid /dev/shm exhaustion on small containers.
    #   --disable-gpu           No GPU available in the container.
    cloud_flags = [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-extensions",
        "--no-first-run",
        "--no-default-browser-check",
        "--remote-debugging-pipe",
    ]

    # Merge in any caller-supplied flags via SELENIUM_CHROME_FLAGS (mstarpy's
    # own mechanism) so we don't clobber intentional configuration.
    user_flags = os.environ.get("SELENIUM_CHROME_FLAGS", "").split()
    all_flags = cloud_flags + [f for f in user_flags if f not in cloud_flags]

    original = _mstarpy_utils.browser_options

    def browser_options():  # type: ignore[no-redef]
        opts = original()
        opts.binary_location = binary
        for flag in all_flags:
            opts.add_argument(flag)
        # mstarpy leaves headless off; the Cloud container has no display.
        opts.add_argument("--headless=new")
        return opts

    browser_options._fundlens_patched = True  # type: ignore[attr-defined]
    _mstarpy_utils.browser_options = browser_options

    # Bypass Selenium Manager when Debian's matching system driver is present.
    # Otherwise Selenium Manager may select a stale cached Chrome-for-Testing
    # driver under ~/.cache/selenium instead of /usr/bin/chromedriver.
    if driver_binary is not None:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service

        @contextmanager
        def get_webdriver():
            active_driver = None
            # Chromium 136+ requires remote debugging to use a non-default
            # profile. A unique directory also prevents profile-lock collisions
            # between simultaneous Streamlit sessions.
            with TemporaryDirectory(prefix="fundlens-chrome-") as profile_dir:
                try:
                    options = _mstarpy_utils.browser_options()
                    options.add_argument(f"--user-data-dir={profile_dir}")
                    active_driver = webdriver.Chrome(
                        service=Service(
                            executable_path=driver_binary,
                            service_args=["--verbose"],
                            log_output=sys.stderr,
                        ),
                        options=options,
                    )
                    active_drivers = getattr(_mstarpy_utils, "_active_webdrivers", None)
                    if active_drivers is not None:
                        active_drivers.add(active_driver)
                    yield active_driver
                finally:
                    if active_driver is not None:
                        try:
                            active_driver.quit()
                        except Exception:
                            pass

        get_webdriver._fundlens_patched = True  # type: ignore[attr-defined]
        _mstarpy_utils.get_webdriver = get_webdriver

        # mstarpy.search imports get_webdriver by value, so update that binding
        # as well as the source function in mstarpy.utils.
        try:
            from mstarpy import search as _mstarpy_search

            _mstarpy_search.get_webdriver = get_webdriver
        except Exception:
            pass


# Register the patch on import. ``app.py`` imports this module before any
# fundlens.data.* module touches mstarpy, so the patch is in place by the time
# MorningstarSession is first constructed.
_install_patch()
