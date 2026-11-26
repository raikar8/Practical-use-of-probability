"""Smoke tests: every page renders without raising, at default settings."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "app_pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page):
    at = AppTest.from_file(str(page), default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_entry_point_renders():
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("page", ["d_speech.py", "d_enhance.py"])
def test_record_mode_waits_for_audio(page):
    """Switching to 'Your own voice' shows the recorder and keeps the rest of the page working."""
    at = AppTest.from_file(str(ROOT / "app_pages" / page), default_timeout=120)
    at.run()
    radio = next(r for r in at.radio if any("Your own voice" in o for o in r.options))
    radio.set_value(next(o for o in radio.options if "Your own voice" in o)).run()
    assert not at.exception, [e.message for e in at.exception]
