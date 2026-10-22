import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "cs2model" / "dashboard" / "app.py").read_text()


def test_sidebar_width_only_applies_while_expanded():
    """A forced width on a collapsed sidebar leaves an empty gutter beside the main area."""
    rules = re.findall(r'section\[data-testid="stSidebar"\][^{]*\{[^}]*width:\d+px!important', SRC)
    assert rules
    assert all('[aria-expanded="true"]' in r for r in rules)


def test_responsive_containers_exist_for_css_hooks():
    for key in ('key="pdist"', 'key="pcards"', 'key=f"fitrow_{pre}"', 'key=f"fitrow_tog_{scope}_{r.name}"', 'key="kprjs"'):
        assert key in SRC, key
    for sel in (".st-key-pdist {container-type:inline-size;}", "@container (max-width:860px)", '[class*="st-key-fitrow"]',
                "min-width:max-content!important", '[data-testid="stLayoutWrapper"]:has(> .st-key-kprjs) {display:none;}'):
        assert sel in SRC, sel


def test_sidebar_autocollapse_respects_stored_choice_and_is_not_persisted():
    js = SRC[SRC.index('SIDEBAR_JS = """'):SRC.index('</script>"""')]
    assert "w.innerWidth > 900" in js
    assert "Object.keys(ls).some(k => k.startsWith(KEY))" in js
    assert "ls.removeItem(k)" in js
    assert "st.html(SIDEBAR_JS, unsafe_allow_javascript=True)" in SRC
