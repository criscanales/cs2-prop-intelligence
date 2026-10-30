"""History ID dtypes: a results update with a missing ID must not turn every ID into a float (the sh1ro 23237 failure)."""
from datetime import date

import pandas as pd
import pytest

from cs2model.scraper import bo3_history as b

SH1RO = 23237
IDS = ["match_id", "game_id", "team_id", "opp_id", "player_id"]


def _hist():
    rows = []
    for g, (pid, name) in enumerate([(SH1RO, "sh1ro"), (29219, "donk"), (SH1RO, "sh1ro")]):
        rows.append({"date": f"2026-10-2{g}T10:00:00Z", "match_id": 500 + g, "game_id": 9000 + g, "map_number": 1,
                     "map_name": "de_mirage", "tier": "s", "tournament_id": 4207, "team_id": 654, "team": "Spirit",
                     "opp_id": 757, "opponent": "TYLOO", "player_id": pid, "player_slug": name, "player": name,
                     "won": 1, "rounds": 24, "kills": 19.0 + g, "headshots": 7.0, "rating": 1.21})
    return pd.DataFrame(rows)


def _update_with_missing_ids():
    """Fresh bo3 rows: one with no team/opponent IDs and a tournament ID sent as a string, as the API sometimes does."""
    return [{"date": "2026-10-24T12:00:00Z", "match_id": 600, "game_id": 9100, "map_number": 1, "map_name": "de_nuke",
             "tier": "s", "tournament_id": "5571", "team_id": None, "team": "Spirit", "opp_id": None, "opponent": "TYLOO",
             "player_id": SH1RO, "player_slug": "sh1ro", "player": "sh1ro", "won": 0, "rounds": 21, "kills": 14.0,
             "headshots": 5.0, "rating": 0.98}]


@pytest.fixture
def disk(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "DATA", tmp_path)
    monkeypatch.setattr(b, "HIST_PATH", tmp_path / "player_maps.csv")
    monkeypatch.setattr(b, "tournament", lambda tid: {})
    monkeypatch.setattr(b, "rows_for_match", lambda m, teams, tour: _update_with_missing_ids())
    window = [{"id": i, "tier": "s", "tournament": 4207} for i in (500, 501, 502)] + [
        {"id": 600, "tier": "s", "tournament": 5571}]
    monkeypatch.setattr(b, "finished_matches", lambda day: (window, {}))
    return b.HIST_PATH


def _results_update():
    """The dashboard results update: crawl with the stored history as `known`, then save."""
    old = b.read_history()
    df = b.crawl(0, end=date(2026, 10, 24), tiers=("s",), write=False, known=old)
    b.save_history(df)
    return df


def test_update_with_missing_ids_keeps_whole_number_ids(disk):
    _hist().to_csv(disk, index=False)
    _results_update()
    text = disk.read_text()
    assert f",{SH1RO}," in text and f"{SH1RO}.0" not in text and ",654.0," not in text and ",757.0," not in text
    h = b.read_history()
    assert all(pd.api.types.is_integer_dtype(h[c]) for c in IDS + ["tournament_id"])
    new = h[h.game_id == 9100].iloc[0]
    assert pd.isna(new.team_id) and pd.isna(new.opp_id)  # missing stays missing, never 0
    assert new.tournament_id == 5571
    assert (h.team_id.dropna() > 0).all()
    assert str(h.player_id.iloc[0]) == str(SH1RO)  # pricing keys players by str(id)


def test_float_history_is_read_back_as_int_and_predictor_keys_match(disk):
    """The broken file from production: every ID written as N.0."""
    bad = _hist()
    for c in ["team_id", "opp_id", "player_id"]:
        bad[c] = bad[c].astype(float)
    bad.to_csv(disk, index=False)
    assert f"{SH1RO}.0" in disk.read_text()
    h = b.read_history()
    assert h.player_id.dtype == "int64" and h.team_id.dtype == "int64"
    assert {str(p) for p in h.player_id} == {str(SH1RO), "29219"}
    b.save_history(h)
    assert f"{SH1RO}.0" not in disk.read_text()


def test_repeated_update_save_reload_cycles_do_not_change_ids_or_stats(disk):
    _hist().to_csv(disk, index=False)
    first = _results_update()
    snap = b.read_history()
    for _ in range(3):
        _results_update()
        again = b.read_history()
        pd.testing.assert_frame_equal(again, snap)
    old = _hist()
    kept = snap[snap.game_id.isin(old.game_id)].reset_index(drop=True)
    for c in IDS + ["kills", "headshots", "rating", "rounds", "won"]:
        assert kept[c].tolist() == old[c].tolist()
    assert len(first) == len(old) + 1


def test_merge_history_float_and_int_frames_gives_int_ids():
    a = _hist()
    f = _hist().assign(game_id=lambda d: d.game_id + 10)
    f["player_id"] = f.player_id.astype(float)
    m = b.merge_history(a, f)
    assert m.player_id.dtype == "int64" and set(m.player_id) == {SH1RO, 29219}


def test_coerce_ids_rejects_fractional_ids():
    with pytest.raises(TypeError):
        b.coerce_ids(pd.DataFrame({"player_id": [1.0, 2.5]}))


def test_concurrent_saves_never_interleave(disk):
    """Refresh loop and dashboard auto-pull saving at once corrupted player_maps.csv (Oct 28: 62 fields on one row)."""
    import threading
    small, big = _hist(), pd.concat([_hist()] * 3000, ignore_index=True)
    errs = []

    def writer(df):
        try:
            for _ in range(5):
                b.save_history(df)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=writer, args=(d,)) for d in (big, small, big, small)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert len(b.read_history()) in (len(small), len(big))
    assert [p.name for p in disk.parent.iterdir() if p.suffix == ".tmp"] == []


def test_saved_history_keeps_default_file_mode(disk):
    import os
    import stat
    umask = os.umask(0o022)
    try:
        b.save_history(_hist())
    finally:
        os.umask(umask)
    assert stat.S_IMODE(disk.stat().st_mode) == 0o644
