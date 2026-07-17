"""The Codex button only renders when FUNDLENS_CODEX_AVAILABLE is set."""
from __future__ import annotations

import inspect


def test_codex_button_is_gated_by_env_var():
    """Button source must reference the gate env var.

    The Codex CLI is not available on Streamlit Community Cloud; without this
    gate the button renders, fails, and confuses visitors.
    """
    import fundlens.ui.streamlit_app as m

    src = inspect.getsource(m)
    assert "FUNDLENS_CODEX_AVAILABLE" in src, (
        "Codex button must be gated behind FUNDLENS_CODEX_AVAILABLE so it "
        "doesn't render on Streamlit Cloud where the codex CLI is absent."
    )
