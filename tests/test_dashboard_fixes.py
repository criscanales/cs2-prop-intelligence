import types

import numpy as np

from cs2model.dashboard import history_filters as HF

PRE = "hf_map_s1_"

APP = """
import streamlit as st
from cs2model.dashboard import history_filters as HF
pre = "hf_map_s1_"
wk = HF.widget_prefix(st.session_state, pre)
with st.popover("Filters"):
    st.multiselect("Opponent", ["A", "B"], key=wk + "opp")
    st.selectbox("Series result", ["Any", "Won", "Lost"], key=wk + "result")
    st.slider("Total rounds", 10, 60, (10, 60), key=wk + "rounds")
st.button("Reset filters", on_click=lambda: HF.reset_state(st.session_state, pre), key=pre + "reset")
"""


def test_reset_state_clears_values_and_moves_widget_keys():
    s = {PRE + "0_opp": ["A"], PRE + "0_result": "Won", PRE + "nonce": 0, "other": 1}
    old = HF.widget_prefix(s, PRE)
    HF.reset_state(s, PRE)
    assert s == {"other": 1, PRE + "nonce": 1}
    assert HF.widget_prefix(s, PRE) != old
    HF.reset_state(s, PRE)
    assert HF.widget_prefix(s, PRE) == PRE + "2_"


def test_reset_clears_displayed_popover_controls():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string(APP).run()
    at.multiselect[0].set_value(["B"])
    at.selectbox[0].set_value("Lost")
    at.slider[0].set_value((20, 30))
    at.run()
    assert at.multiselect[0].value == ["B"] and at.selectbox[0].value == "Lost" and at.slider[0].value == (20, 30)
    at.button[0].click().run()
    assert at.multiselect[0].value == [] and at.selectbox[0].value == "Any" and at.slider[0].value == (10, 60)
    assert not [k for k in at.session_state.filtered_state if k.startswith(PRE + "0_")]
    at.run()
    assert at.multiselect[0].value == [] and at.selectbox[0].value == "Any"


def test_kpi_value_wraps_between_kda_parts():
    v = HF.kpi_value_html("123,456 / 100,234 / 45,678")
    assert v.count('white-space:nowrap') == 3 and v.count(" / ") == 2
    assert HF.kpi_value_html("1.23") == '<span style="white-space:nowrap">1.23</span>'
    assert "&lt;" in HF.kpi_value_html("<b>")


def test_portrait_cascade_without_cv2_data(tmp_path):
    from cs2model.scraper import portraits as P
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / P.CASCADE).write_text("x")
    fake = types.SimpleNamespace(__file__=str(tmp_path / "__init__.py"))
    assert P._cascade_file(fake) == str(tmp_path / "data" / P.CASCADE)
    assert P._cascade_file(types.SimpleNamespace(__file__=str(tmp_path / "nowhere" / "x.py"))) is None


def test_portrait_missing_cascade_keeps_default_crop(monkeypatch):
    from cs2model.scraper import portraits as P
    monkeypatch.setattr(P, "_detector", lambda: (None, None))
    blob = np.zeros((455, 354, 4), np.uint8)
    blob[40:300, 120:240] = 255
    assert P.frame(blob) == P.DEFAULT


def test_portrait_frames_threaded_no_errors(monkeypatch, tmp_path, caplog):
    from cs2model.scraper import portraits as P
    P._detector.cache_clear()
    monkeypatch.setattr(P, "FRAMES_PATH", tmp_path / "frames.json")
    monkeypatch.setattr(P, "_image", lambda u: np.zeros((455, 354, 4), np.uint8))
    out = P.frames({i: f"u{i}" for i in range(32)}, workers=8)
    assert set(out) == set(range(32)) and all(v == P.DEFAULT for v in out.values())
    assert "portrait u" not in caplog.text
