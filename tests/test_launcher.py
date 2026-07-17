from __future__ import annotations

from pathlib import Path


def test_windows_streamlit_launcher_points_at_project_app():
    launcher = Path("Launch FundLens Streamlit.bat")
    text = launcher.read_text(encoding="utf-8")

    assert "C:\\Python314\\python.exe" in text
    assert "src\\fundlens\\ui\\streamlit_app.py" in text
    assert "cd /d \"%PROJECT_DIR%\"" in text
    assert "-m streamlit run" in text
    assert "pause" in text.lower()
