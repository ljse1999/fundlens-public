"""Streamlit entrypoint for FundLens (Streamlit Community Cloud main file)."""
# Register the mstarpy/Chromium patch BEFORE importing any fundlens.data module,
# since those modules trigger mstarpy's MorningstarSession (which launches
# Chrome) at call time. On non-Linux platforms the patch is a no-op.
import fundlens._cloud_chrome  # noqa: F401  (side-effect import)

from fundlens.ui.streamlit_app import main

if __name__ == "__main__":
    main()
