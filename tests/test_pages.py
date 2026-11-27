"""Smoke tests: every page renders without raising, opened through the real navigation."""
import re
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "app_pages").rglob("*.py"))


def _open(page: str) -> "AppTest":
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    at.run()
    at.switch_page(page)
    at.run()
    return at


def test_every_page_file_is_in_the_navigation():
    from ssp.ui import TOPICS
    registered = {"app_pages/home.py", "app_pages/glossary.py"}
    for _, _, theory, lab in TOPICS.values():
        registered |= {theory, lab}
    assert set(PAGES) == registered


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _open(page)
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("page", ["app_pages/d_speech.py", "app_pages/d_enhance.py"])
def test_record_mode_waits_for_audio(page):
    """Switching to 'Your own voice' shows the recorder and keeps the rest of the page working."""
    at = _open(page)
    radio = next(r for r in at.radio if any("Your own voice" in o for o in r.options))
    radio.set_value(next(o for o in radio.options if "Your own voice" in o)).run()
    assert not at.exception, [e.message for e in at.exception]


def test_no_multiline_display_math_in_markdown():
    """Streamlit's Markdown can't parse $$...$$ blocks that span lines (shows a KaTeX error).
    Display equations must use st.latex / ui.eq, or keep $$...$$ on one line."""
    bad = []
    for path in [*(ROOT / "app_pages").rglob("*.py"), ROOT / "ssp" / "ui.py"]:
        text = path.read_text()
        for m in re.finditer(r"\$\$(.*?)\$\$", text, flags=re.S):
            if "\n" in m.group(1):
                bad.append(f"{path.relative_to(ROOT)}: {m.group(0)[:60]!r}")
    assert not bad, bad
