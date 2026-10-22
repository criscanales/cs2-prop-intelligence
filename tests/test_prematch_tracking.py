"""Pre-match gating, rescheduled-copy dedupe and exclusion-aware reporting in tracking.db."""
import json
import sqlite3

import pandas as pd
import pytest

from cs2model.tracking import prematch, store
from tests.test_core import UP, _res, _series_hist

STARTED = {frozenset((1, 2)): {"match_id": 7, "status": "current", "started": True, "scheduled": None}}


def _row(st, **kw):
    r = dict(source="prizepicks", player="p10", team="T1", opponent="T2", stat="kills", maps=(3,), start_time=st,
             odds_type="standard", line=13.5, model_mean=14.0, p_over=0.55, p_over_raw=0.55, t3=False, team_id=1, opp_id=2,
             side="under", p_side=0.6, breakeven=0.55, edge=0.05, ev=0.09, tier="B")
    r.update(kw)
    return pd.DataFrame([r])


def _rec(st):
    return {"source": "pinnacle", "match_key": "1-2", "team_a_id": 1, "team_b_id": 2, "start_time": st.isoformat(),
            "best_of": 3, "t3": 0, "p_series_a": 0.6}


def _quar(con):
    return dict(con.execute("SELECT tbl || ':' || reason, count(*) FROM quarantine GROUP BY 1").fetchall())


def _pred(con, st, ts, maps="1+2", source="underdog", line=22.5, t3=0, p_over=0.6, match_id=None, prematch=None):
    con.execute("INSERT INTO preds (source, player, team, opponent, stat, maps, start_time, ts, line, model_mean, p_over, t3, "
                "match_id, prematch) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (source, "p10", "T1", "T2", "kills", maps, st.isoformat(), str(ts), line, 23.0, p_over, t3, match_id, prematch))


def _pick(con, st, created, maps="1+2", source="underdog", line=22.5, side="over", t3=0, match_id=None, prematch=None):
    return con.execute("INSERT INTO picks (created, source, player, team, opponent, stat, maps, line, side, p_model, breakeven, edge, "
                       "tier, start_time, t3, match_id, prematch) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (str(created), source, "p10", "T1", "T2", "kills", maps, line, side, 0.6, 0.5, 0.1, "B", st.isoformat(), t3,
                        match_id, prematch)).lastrowid


def _excl(con):
    return {(t, k): why for t, k, why in con.execute("SELECT tbl, row_key, reason FROM exclusions")}


@pytest.fixture
def st():
    return pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=6)


def test_book_start_moving_forward_after_map1_began_is_quarantined(tmp_path):
    con = store.connect(tmp_path / "t.db")
    now = pd.Timestamp.now(tz="UTC")
    for k in range(4):  # the platform keeps pushing the scheduled start while the series is live
        bst = now + pd.Timedelta(minutes=30 * (k + 1))
        store.log_preds(_row(bst), con, status=STARTED)
        store.log_picks(_row(bst), con, status=STARTED)
        store.log_map_odds([_rec(bst)], now, con, status=STARTED)
    assert [con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("preds", "picks", "map_odds_log")] == [0, 0, 0]
    assert _quar(con) == {"preds:started": 4, "picks:started": 4, "map_odds_log:started": 4}
    payload = json.loads(con.execute("SELECT payload FROM quarantine WHERE tbl='preds' LIMIT 1").fetchone()[0])
    assert "p10" in payload  # original record and its timestamps are kept for auditing
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM quarantine")


def test_two_hour_delay_between_maps_history_is_excluded_not_deleted(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    h = _series_hist([1, 1], st)
    h.loc[h.map_number == 2, "date"] = st + pd.Timedelta(hours=2)
    _pred(con, st, st - pd.Timedelta(hours=1))
    eligible = _pick(con, st, st - pd.Timedelta(hours=1))
    late = []
    for k, bst in enumerate((st + pd.Timedelta(hours=2), st + pd.Timedelta(minutes=150))):  # logged while maps 1-2 were live
        _pred(con, bst, st + pd.Timedelta(minutes=90 + k), maps="3", source="prizepicks", line=13.5)
        late.append(_pick(con, bst, st + pd.Timedelta(minutes=90 + k), maps="3", source="prizepicks", line=13.5, side="under"))
    store.grade(h, _res, con)
    assert con.execute("SELECT count(*) FROM preds WHERE maps='3' AND void=1").fetchone()[0] == 2  # settlement kept
    ex = _excl(con)
    assert all(ex[("picks", str(i))] == "post_start" for i in late) and ("picks", str(eligible)) not in ex
    assert sum(1 for (t, _), why in ex.items() if t == "preds" and why == "post_start") == 2
    perf = store.performance(con)[0]
    assert (perf["picks"], perf["voids"]) == (1, 0)
    assert store.bias_report(con).lines.sum() == 1
    assert con.execute("SELECT count(*) FROM picks").fetchone()[0] == 3  # nothing deleted


def test_rescheduled_copies_count_once_and_distinct_lines_stay(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    pre = st - pd.Timedelta(hours=1)
    first, later = st, st + pd.Timedelta(minutes=10)
    for bst in (first, later):
        _pred(con, bst, pre)
        con.execute("INSERT INTO map_odds_log (snapshot_ts, source, match_key, team_a_id, team_b_id, start_time, best_of, t3) "
                    "VALUES (?,?,?,?,?,?,?,?)", (pre.isoformat(), "pinnacle", "1-2", 1, 2, bst.isoformat(), 3, 0))
    old_copy, new_copy = _pick(con, first, pre), _pick(con, later, pre)
    other_line = _pick(con, later, pre, line=23.5)  # a genuinely different line is its own selection
    store.grade(_series_hist([1, 1], st), _res, con)
    ex = _excl(con)
    assert ex[("picks", str(old_copy))] == "superseded_reschedule"
    assert ("picks", str(new_copy)) not in ex and ("picks", str(other_line)) not in ex
    assert ex[("preds", store.pred_key("underdog", "p10", "kills", "1+2", first.isoformat()))] == "superseded_reschedule"
    perf = store.performance(con)[0]
    assert perf["picks"] == 2 and perf["hit_rate"] == 0.5  # 23 kills: over 22.5 wins, over 23.5 loses
    assert store.bias_report(con).lines.sum() == 1
    assert con.execute("SELECT count(*) FROM match_results").fetchone()[0] == 1  # one row per bo3 match


def test_existing_duplicate_match_results_are_excluded(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    for m in (0, 10):
        con.execute("INSERT INTO match_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("1-2", (st + pd.Timedelta(minutes=m)).isoformat(), 7, 3, 2, 0, 1, 1, None, 1, "x"))
    store.audit(_series_hist([1, 1], st), _res, con)
    assert list(_excl(con).items()) == [(("match_results", f"1-2|{st.isoformat()}"), "duplicate_match")]


def test_valid_prematch_line_changes(tmp_path):
    con = store.connect(tmp_path / "t.db")
    bst = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)
    store.log_preds(_row(bst), con, status=UP)
    store.log_preds(_row(bst, line=12.5), con, status=UP)  # the line moved before the match: keep the latest price
    assert con.execute("SELECT count(*), max(line), max(match_id) FROM preds").fetchone() == (1, 12.5, 7)
    store.log_picks(_row(bst), con, status=UP)
    store.log_picks(_row(bst, line=12.5), con, status=UP)
    assert con.execute("SELECT count(*) FROM picks WHERE match_id=7").fetchone()[0] == 2
    assert _quar(con) == {}


def test_push_over_under_and_map3_void_settlement(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    pre = st - pd.Timedelta(hours=1)
    ids = {"push": _pick(con, st, pre, maps="1", line=11.0), "over": _pick(con, st, pre, maps="1", line=10.5),
           "under": _pick(con, st, pre, maps="2", line=12.5, side="under"), "loss": _pick(con, st, pre, line=22.5, side="under"),
           "void": _pick(con, st, pre, maps="3", line=13.5)}
    _pred(con, st, pre, maps="1", line=11.0)
    store.grade(_series_hist([1, 1], st), _res, con)  # p10: 11 kills on Map 1, 12 on Map 2, 2-0 so no Map 3
    got = {k: con.execute("SELECT outcome, actual FROM picks WHERE id=?", (i,)).fetchone() for k, i in ids.items()}
    assert got == {"push": ("push", 11.0), "over": ("win", 11.0), "under": ("win", 12.0), "loss": ("loss", 23.0), "void": ("void", None)}
    perf = store.performance(con)[0]
    assert (perf["picks"], perf["pushes"], perf["voids"]) == (3, 1, 1)
    assert con.execute("SELECT actual, line FROM preds").fetchone() == (11.0, 11.0)


def test_tier3_stays_separate(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    pre = st - pd.Timedelta(hours=1)
    _pick(con, st, pre, line=22.5)
    _pick(con, st, pre, line=22.5, source="prizepicks", t3=1)
    _pred(con, st, pre, line=22.5)
    _pred(con, st, pre, line=22.5, source="prizepicks", t3=1)
    store.grade(_series_hist([1, 1], st), _res, con)
    g = {r["group"]: r["picks"] for r in store.performance(con)[0]["by_group"]}
    assert g == {"Tier 1/2": 1, "Tier 3": 1}
    assert store.bias_report(con).lines.sum() == 1
    rel = store.t3_reliability(con)
    assert rel[rel.market == "All tier-3 lines"].lines.max() == 1
    store.log_preds(_row(pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1), t3=True), con, status=STARTED)
    assert _quar(con) == {"preds:started": 1}  # the same timing gate applies to tier 3


def test_missing_or_delayed_start_information_is_never_treated_as_prematch(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    bst = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)
    store.log_preds(_row(bst), con, status=prematch.UNAVAILABLE)
    store.log_preds(_row(bst), con, status={})
    store.log_picks(_row(bst, team_id=float("nan")), con, status=UP)
    assert _quar(con) == {"preds:status_unavailable": 1, "preds:no_verified_match": 1, "picks:no_verified_match": 1}
    assert prematch.safe_status(lambda: (_ for _ in ()).throw(ConnectionError("down"))) is prematch.UNAVAILABLE
    # bo3.gg still listed the match as upcoming, but the history later shows Map 1 had begun before the record
    _pred(con, st + pd.Timedelta(minutes=30), st + pd.Timedelta(minutes=5))
    _pred(con, st + pd.Timedelta(minutes=40), "not a time")
    store.grade(_series_hist([1, 1], st), _res, con)
    assert sorted(_excl(con).values()) == ["post_start", "unverified_time"]


def test_repeated_refresh_and_grading_add_no_duplicates(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    pre = st - pd.Timedelta(hours=1)
    for bst in (st, st + pd.Timedelta(minutes=10)):
        _pick(con, bst, pre)
        _pred(con, bst, pre)
        con.execute("INSERT INTO map_odds_log (snapshot_ts, source, match_key, team_a_id, team_b_id, start_time, best_of, t3) "
                    "VALUES (?,?,?,?,?,?,?,?)", (pre.isoformat(), "pinnacle", "1-2", 1, 2, bst.isoformat(), 3, 0))
    h = _series_hist([1, 1], st)
    snaps = []
    for _ in range(3):
        store.grade(h, _res, con)
        snaps.append((con.execute("SELECT count(*) FROM exclusions").fetchone()[0],
                      con.execute("SELECT count(*) FROM match_results").fetchone()[0], store.performance(con)[0]["picks"]))
    assert snaps == [(2, 1, 1)] * 3
    assert store.audit(h, _res, con, dry_run=True).empty


def test_slips_with_post_start_legs_or_repeats_are_excluded(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    pre = st - pd.Timedelta(hours=1)
    a, b = _pick(con, st, pre, source="prizepicks"), _pick(con, st, pre, source="prizepicks", maps="1", line=10.5)
    late = _pick(con, st, st + pd.Timedelta(minutes=5), source="prizepicks", maps="2", line=11.5)
    for created, legs in (("2000-01-01T00:00:00+00:00", [a, b]), ("2000-01-01T00:30:00+00:00", [a, b]),
                          ("2000-01-01T00:40:00+00:00", [a, late])):
        con.execute("INSERT INTO slips (created, platform, play, legs, stake) VALUES (?,?,?,?,?)", (created, "prizepicks", "power_2",
                                                                                                     json.dumps(legs), 10.0))
    store.grade(_series_hist([1, 1], st), _res, con)
    assert sorted(why for (t, _), why in _excl(con).items() if t == "slips") == ["duplicate_slip", "post_start_leg"]
    assert store.performance(con)[0]["slips"] == 1


def test_fetch_match_status_reads_upcoming_live_and_finished(monkeypatch):
    from cs2model.scraper import http

    def fake(url, ttl=None, params=None):
        if url.endswith("upcoming"):
            return {"data": {"tiers": {"high_tier": {"matches": [
                {"id": 1, "team1_id": 1, "team2_id": 2, "status": "upcoming", "games": [], "start_date": "2999-01-01T00:00:00Z"},
                {"id": 2, "team1_id": 3, "team2_id": 4, "status": "upcoming", "games": [{"number": 1, "status": "current"}]}]}}}}
        if url.endswith("live"):
            return {"data": [{"id": 3, "team1_id": 5, "team2_id": 6, "status": "current", "games": []}]}
        return {"data": {"tiers": {"low_tier": {"matches": [{"id": 4, "team1_id": 7, "team2_id": 8, "status": "finished"}]}}}}
    monkeypatch.setattr(http, "fetch_json", fake)
    s = prematch.fetch_match_status()
    assert {k: [e["started"] for e in v] for k, v in s.items()} == {frozenset((1, 2)): [False], frozenset((3, 4)): [True],
                                                                  frozenset((5, 6)): [True], frozenset((7, 8)): [True]}
    assert prematch.classify(2, 1, s) == ("scheduled", 1, None) and prematch.classify(4, 3, s) == (None, 2, "started")


def test_same_teams_in_two_matches_use_the_verified_bo3_match_id(tmp_path):
    con = store.connect(tmp_path / "t.db")
    now = pd.Timestamp.now(tz="UTC")
    two = {frozenset((1, 2)): [
        {"match_id": 7, "status": "finished", "started": True, "scheduled": (now - pd.Timedelta(hours=5)).isoformat()},
        {"match_id": 8, "status": "upcoming", "started": False, "scheduled": (now + pd.Timedelta(hours=3)).isoformat()}]}
    store.log_preds(_row(now + pd.Timedelta(hours=3, minutes=10)), con, status=two)  # the rematch, not the finished series
    assert con.execute("SELECT match_id, prematch FROM preds").fetchall() == [(8, "scheduled")]
    store.log_picks(_row(now - pd.Timedelta(hours=4)), con, status=two)  # nearest to the series that already started
    close = {frozenset((1, 2)): [
        {"match_id": 8, "status": "upcoming", "started": False, "scheduled": (now + pd.Timedelta(hours=2)).isoformat()},
        {"match_id": 9, "status": "upcoming", "started": False, "scheduled": (now + pd.Timedelta(hours=3)).isoformat()}]}
    store.log_picks(_row(now + pd.Timedelta(hours=2, minutes=30)), con, status=close)
    assert _quar(con) == {"picks:started": 1, "picks:ambiguous_match": 1}
    assert con.execute("SELECT match_id FROM quarantine WHERE reason='started'").fetchone() == (7,)
    assert prematch.classify(1, 2, two) == (None, None, "ambiguous_match")  # no start time to tell them apart


def test_grading_uses_the_stored_match_id_and_skips_ambiguous_legacy_rows(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    later = st + pd.Timedelta(hours=4)
    h = pd.concat([_series_hist([1, 1], st),  # match 7: p10 has 11 + 12 kills
                   _series_hist([1, 1], later).assign(match_id=8, game_id=lambda d: d.game_id + 10, kills=lambda d: d.kills + 10)])
    _pred(con, later, later - pd.Timedelta(hours=1), match_id=8, prematch="scheduled")
    _pred(con, st, st - pd.Timedelta(hours=1), source="prizepicks")  # legacy row: both series fall in its time window
    store.grade(h, _res, con)
    assert dict(con.execute("SELECT source, actual FROM preds").fetchall()) == {"underdog": 43.0, "prizepicks": None}


def test_stale_upcoming_status_is_held_until_the_history_verifies_it(tmp_path, st):
    con = store.connect(tmp_path / "t.db")
    now = pd.Timestamp.now(tz="UTC")
    stale = {frozenset((1, 2)): {"match_id": 7, "status": "upcoming", "started": False,
                                 "scheduled": (now - pd.Timedelta(minutes=20)).isoformat()}}
    bst = now + pd.Timedelta(hours=1)  # the platform still shows a future start
    store.log_preds(_row(bst), con, status=UP)
    store.log_preds(_row(bst, line=12.5), con, status=stale)  # never overwrites the confident pre-match price
    store.log_preds(_row(bst + pd.Timedelta(minutes=30)), con, status=stale)
    assert sorted(con.execute("SELECT line, prematch FROM preds").fetchall()) == [(13.5, "scheduled"), (13.5, "uncertain")]
    assert store.log_map_odds([_rec(bst)], now, con, status=stale) == 0
    assert _quar(con) == {"map_odds_log:past_scheduled_start": 1}
    late = _pick(con, st, st + pd.Timedelta(minutes=5), prematch="uncertain")  # Map 1 had begun at st
    early = _pick(con, st, st - pd.Timedelta(minutes=30), line=21.5, prematch="uncertain")
    con.execute("UPDATE picks SET outcome='win', actual=23")  # settled but not yet audited
    assert store.performance(con)[0]["picks"] == 0  # held rows never reach reports unverified
    store.audit(_series_hist([1, 1], st), _res, con)
    assert dict(con.execute("SELECT id, prematch FROM picks").fetchall()) == {late: "uncertain", early: "verified"}
    assert _excl(con) == {("picks", str(late)): "post_start"}
    assert store.performance(con)[0]["picks"] == 1
