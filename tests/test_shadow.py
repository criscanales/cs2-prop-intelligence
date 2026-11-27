"""Shadow recalibration is logged next to production pre-match and never changes production prices."""
import json

import pandas as pd

from cs2model.tracking import shadow, store
from tests.test_core import UP
from tests.test_prematch_tracking import STARTED, _row


def _cal(tmp_path, monkeypatch):
    p = tmp_path / "shadow.json"
    p.write_text(json.dumps({"version": "t", "params": {"kills_map": {"coef": 0.5, "intercept": 0.0}}}))
    monkeypatch.setattr(shadow, "PATH", p)


def _off(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "PATH", tmp_path / "missing.json")
    monkeypatch.setattr(shadow, "BUNDLED", tmp_path / "missing.json")


def test_uncertain_quote_only_shadowed_when_production_kept_it(tmp_path, monkeypatch):
    con = store.connect(tmp_path / "t.db")
    now = pd.Timestamp.now(tz="UTC")
    stale = {frozenset((1, 2)): {"match_id": 7, "status": "upcoming", "started": False,
                                 "scheduled": (now - pd.Timedelta(minutes=20)).isoformat()}}
    bst = now + pd.Timedelta(hours=1)
    _off(tmp_path, monkeypatch)
    store.log_preds(_row(bst), con, status=UP)  # production price logged before the shadow file existed
    _cal(tmp_path, monkeypatch)
    store.log_preds(_row(bst, line=12.5), con, status=stale)  # rejected by preds, so no shadow row either
    assert con.execute("SELECT count(*) FROM shadow_preds").fetchone()[0] == 0
    store.log_preds(_row(bst + pd.Timedelta(minutes=30)), con, status=stale)  # new key: preds keeps it, shadow too
    assert con.execute("SELECT line, prematch FROM shadow_preds").fetchall() == [(13.5, "uncertain")]


def test_desktop_home_without_config_falls_back_to_bundled(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow, "PATH", tmp_path / "config" / "shadow_calibration.json")
    assert shadow.load()[0].startswith("recal-v1:")


def test_shadow_logged_beside_production_and_prod_unchanged(tmp_path, monkeypatch):
    _cal(tmp_path, monkeypatch)
    con = store.connect(tmp_path / "t.db")
    bst = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
    slate = _row(bst, p_over=0.7, be_over=0.55, be_under=0.55, p_push=0.0)
    before = slate.copy()
    store.log_preds(slate, con, status=UP)
    pd.testing.assert_frame_equal(slate, before)
    assert con.execute("SELECT p_over FROM preds").fetchone()[0] == 0.7
    prod, p, e_o, e_u, mid = con.execute("SELECT p_over_prod, p_over, edge_over, edge_under, match_id FROM shadow_preds").fetchone()
    assert prod == 0.7 and 0.5 < p < 0.7 and mid == 7
    assert abs(e_o - (p - 0.55)) < 1e-9 and abs(e_u - (1 - p - 0.55)) < 1e-9


def test_shadow_skips_started_t3_and_unconfigured(tmp_path, monkeypatch):
    _cal(tmp_path, monkeypatch)
    con = store.connect(tmp_path / "t.db")
    bst = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
    store.log_preds(_row(bst), con, status=STARTED)
    store.log_preds(_row(bst, t3=True), con, status=UP)
    store.log_preds(_row(bst, stat="headshots"), con, status=UP)  # no calibrator for this market
    assert con.execute("SELECT count(*) FROM shadow_preds").fetchone()[0] == 0
    _off(tmp_path, monkeypatch)
    store.log_preds(_row(bst, player="p11"), con, status=UP)
    assert con.execute("SELECT count(*) FROM shadow_preds").fetchone()[0] == 0


def test_repo_shadow_config_loads_all_four_markets():
    version, cal = shadow.load()
    assert version.startswith("recal-v1:") and set(cal) == {"kills_map", "kills_series", "headshots_map", "headshots_series"}
