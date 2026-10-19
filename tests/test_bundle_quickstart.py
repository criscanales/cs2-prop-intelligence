import io
import json
import zipfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from cs2model import bundle as BU
from cs2model.features.build import FEATURES
from cs2model.scraper import bo3_history as b
from cs2model.training import bootstrap as BS
from cs2model.training.calibration import Platt


def _rows(n_games=3, start=0, players=4):
    out = []
    for g in range(start, start + n_games):
        for p in range(players):
            out.append({"date": f"2026-0{1 + g % 9}-10T12:00:00Z", "match_id": 100 + g // 2, "game_id": 1000 + g,
                        "map_number": 1 + g % 2, "map_name": "de_mirage", "tier": "a", "team_id": 1 + p % 2,
                        "opp_id": 2 - p % 2, "player_id": 50 + p, "player": f"p{p}", "won": p % 2, "rounds": 22,
                        "kills": 15 + p, "headshots": 7, "weapon_kills": 15 + p, "awp_kills": 0, "awp_hs": 0})
    return pd.DataFrame(out)


@pytest.fixture
def home(tmp_path, monkeypatch):
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()
    hist = data / "player_maps.csv"
    for mod, attrs in ((b, {"DATA": data, "HIST_PATH": hist}),
                       (BS, {"DATA": data, "HIST_PATH": hist, "MODEL_DIR": models, "COVERAGE_PATH": data / "coverage.json"}),
                       (BU, {"DATA": data, "HIST_PATH": hist, "MODEL_DIR": models, "AWP_PATH": data / "awp_maps.csv",
                             "ROOT": tmp_path, "EXPORT_DIR": tmp_path / "exports"})):
        for k, v in attrs.items():
            monkeypatch.setattr(mod, k, v)
    monkeypatch.setattr(BU, "DATA_FILES", {k: data / v.name for k, v in BU.DATA_FILES.items()})
    monkeypatch.setattr(BU, "MODEL_FILES", {k: models / v.name for k, v in BU.MODEL_FILES.items()})
    monkeypatch.setattr(BU, "_app_version", lambda: "test")
    return tmp_path


def _write_models(d, features=FEATURES, trained="2026-05-01"):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(features))), columns=list(features))
    for f in ("kills_lgbm.txt", "hs_lgbm.txt"):
        lgb.train({"objective": "poisson", "verbose": -1}, lgb.Dataset(X, rng.poisson(3, 200)), 3).save_model(str(d / f))
    p = rng.uniform(0.2, 0.8, 300)
    cal = {k: Platt().fit(p, (rng.uniform(size=300) < p).astype(int)) for k in ("kills_map", "kills_series", "headshots_map", "headshots_series")}
    joblib.dump(cal, d / "calibrators.joblib")
    (d / "backtest.json").write_text(json.dumps({"split_dates": {"test": ["2026-04-01", "2026-05-01"]}}))
    (d / "backtest_rows.csv").write_text("a\n1\n")
    (d / "artifacts.json").write_text(json.dumps({"features": list(features), "trained_through": trained, "n_rows": 200,
                                                  "shrink": {}, "alpha": 0.1, "rho": 0.01, "win_ratio": 1.1, "rounds_table": {},
                                                  "history": {"days": 45, "quick_start": True}}))
    return cal


def test_platt_params_roundtrip():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 500)
    c = Platt().fit(p, (rng.uniform(size=500) < p).astype(int))
    q = np.linspace(0.02, 0.98, 50)
    assert np.allclose(Platt.from_params(json.loads(json.dumps(c.to_params()))).predict(q), c.predict(q))


def test_merge_history_dedupes_and_keeps_most_complete():
    a = _rows(3)
    c = _rows(3, start=2)
    a.loc[a.game_id == 1002, "weapon_kills"] = np.nan  # c has the more complete copy of game 1002
    m = b.merge_history(a, c)
    assert len(m) == 4 * 5 and not m.duplicated(["game_id", "player_id"]).any()
    assert m[m.game_id == 1002].weapon_kills.notna().all()


def test_crawl_reuses_known_matches(monkeypatch):
    calls = []
    m = {"id": 100, "tier": "a", "tournament": 9, "team1_id": 1, "team2_id": 2}
    monkeypatch.setattr(b, "finished_matches", lambda day: ([m], {}))
    monkeypatch.setattr(b, "tournament", lambda tid: calls.append(("t", tid)) or {})
    monkeypatch.setattr(b, "match_games", lambda mid: calls.append(("g", mid)) or [])
    known = _rows(2)  # match 100 = games 1000, 1001
    df = b.crawl(0, tiers=("a",), write=False, known=known)
    assert calls == [] and len(df) == len(known)
    b.crawl(0, tiers=("a",), write=False)
    assert ("g", 100) in calls


def test_legacy_install_counts_as_complete(home):
    assert BS.read_coverage() is None and BS.is_complete()
    BS.write_coverage(complete=False)
    assert not BS.is_complete()


def test_export_contents_checksums_and_exclusions(home):
    _rows(4).to_csv(home / "data" / "player_maps.csv", index=False)
    (home / "data" / "tracking.db").write_bytes(b"secret picks")
    (home / "data" / "pinnacle_last_good.json").write_text("{}")
    (home / "data" / "cache").mkdir()
    _write_models(home / "models")
    path = BU.export_bundle(home / "out")
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        man = json.loads(z.read("manifest.json"))
    assert names <= BU.ALLOWED and "models/calibrators.json" in names and "models/calibrators.joblib" not in names
    assert not any("tracking" in n or "cache" in n or "config" in n or "pinnacle" in n for n in names)
    assert man["history"]["rows"] == 16 and man["model"]["trained_through"] == "2026-05-01"
    assert man["model"]["history"]["quick_start"] is True and set(man["files"]) == names - {"manifest.json"}


def _bundle(home, rows, **kw):
    rows.to_csv(home / "data" / "player_maps.csv", index=False)
    _write_models(home / "models", **kw)
    data = BU.export_bundle(home / "out").read_bytes()
    for f in (home / "models").iterdir():
        f.unlink()
    (home / "data" / "player_maps.csv").unlink()
    return data


def test_import_fresh_install_then_merge_without_duplicates(home):
    data = _bundle(home, _rows(4))
    r = BU.import_bundle(data)
    assert r["models_installed"] and r["rows_added"] == 16 and (home / "models" / "calibrators.joblib").exists()
    from cs2model.inference.predict import Models
    m = Models.load(home / "models")
    assert m.art["trained_through"] == "2026-05-01" and 0 < m.calibrate("kills", 1, 0.6) < 1
    r2 = BU.import_bundle(data)  # same bundle again: nothing new, local models kept
    assert r2["rows_added"] == 0 and r2["duplicates_skipped"] == 16 and not r2["models_installed"]
    assert len(pd.read_csv(home / "data" / "player_maps.csv")) == 16


def test_import_replace_models_backs_up(home):
    data = _bundle(home, _rows(2))
    _write_models(home / "models", trained="2026-01-01")
    r = BU.import_bundle(data, replace_models=True)
    assert r["models_installed"] and r["model_backup"]
    assert json.loads((home / "models" / "artifacts.json").read_text())["trained_through"] == "2026-05-01"


def test_import_rejects_tampered_and_changes_nothing(home):
    data = _bundle(home, _rows(2))
    zin, out = zipfile.ZipFile(io.BytesIO(data)), io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for n in zin.namelist():
            body = zin.read(n)
            z.writestr(n, body.replace(b"p1", b"p9") if n == "data/player_maps.csv" else body)
    with pytest.raises(BU.BundleError, match="integrity"):
        BU.import_bundle(out.getvalue())
    assert not (home / "data" / "player_maps.csv").exists() and not (home / "models" / "artifacts.json").exists()


@pytest.mark.parametrize("name", ["../evil.py", "data/tracking.db", "config/settings.yaml"])
def test_import_rejects_unexpected_members(home, name):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("manifest.json", json.dumps({"format": BU.FORMAT, "bundle_version": 1, "files": {}}))
        z.writestr(name, b"x")
    with pytest.raises(BU.BundleError, match="Unexpected"):
        BU.import_bundle(out.getvalue())


def test_import_newer_version_and_wrong_format(home):
    for man, msg in (({"format": BU.FORMAT, "bundle_version": 99, "files": {}}, "newer"), ({"format": "zip"}, "format")):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("manifest.json", json.dumps(man))
        with pytest.raises(BU.BundleError, match=msg):
            BU.import_bundle(out.getvalue())
    with pytest.raises(BU.BundleError, match="valid bundle"):
        BU.import_bundle(b"not a zip")


def test_import_incompatible_model_keeps_history_only(home):
    data = _bundle(home, _rows(2), features=[*FEATURES[:-1], "other_feature"])
    r = BU.import_bundle(data)
    assert not r["models_installed"] and "feature set" in r["models_skipped_reason"]
    assert r["rows_added"] == 8 and not (home / "models" / "artifacts.json").exists()


def test_import_marks_partial_history_for_backfill(home):
    BS.write_coverage(complete=False, target_days=183)  # exported from an install still backfilling
    data = _bundle(home, _rows(2))
    (home / "data" / "coverage.json").unlink()
    r = BU.import_bundle(data)
    assert not r["history_complete"] and not BS.is_complete()


def test_gated_retrain_skips_when_already_running(monkeypatch):
    monkeypatch.setattr(BS, "_gated_retrain", lambda h, s: pytest.fail("ran concurrently"))
    with BS._gate_lock:
        assert BS.gated_retrain(pd.DataFrame(), {}) is None
