"""Turn today's prop lines + match odds into calibrated over/under probabilities, edges and
simulated hit indicators (for slips)."""
import dataclasses
import hashlib
import logging
import os
import pickle
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import FEATURES, build_features
from ..maps.pool import MapModel, anchored
from ..names import norm_player, team_similarity
from ..parlay.engine import load_yaml, single_breakeven
from ..scraper.bo3_history import HIST_PATH
from ..training.train import MODEL_DIR, shrunk_rate
from . import analytic as A
from . import distribution as D
from .simulate import MatchSpec, PlayerSpec, leg_totals, simulate_match

log = logging.getLogger(__name__)
ACTIVE_MAP_MIN_SHARE = 0.02


@dataclass
class Models:
    kills: lgb.Booster
    hs: lgb.Booster
    calibrators: dict
    art: dict
    version: str = ""

    @classmethod
    def load(cls, d=MODEL_DIR):
        import json
        files = ("kills_lgbm.txt", "hs_lgbm.txt", "calibrators.joblib", "artifacts.json")
        version = hashlib.sha1(b"".join((d / f).read_bytes() for f in files)).hexdigest()[:12]
        return cls(lgb.Booster(model_file=str(d / "kills_lgbm.txt")), lgb.Booster(model_file=str(d / "hs_lgbm.txt")),
                   joblib.load(d / "calibrators.joblib"), json.loads((d / "artifacts.json").read_text()), version)

    def calibrate(self, stat, n_maps, p):
        c = self.calibrators.get(f"{stat}_{'map' if n_maps == 1 else 'series'}")
        return float(c.predict([p])[0]) if c is not None else p


def series_win(p_map, best_of):
    """P(team a wins the series) from its per-map win chances (BO1/BO3; longer series use map 1's price)."""
    p = [p_map.get(j, p_map.get(1, 0.5)) for j in (1, 2, 3)]
    if best_of == 1:
        return float(p[0])
    if best_of == 3:
        return float(p[0] * p[1] + (p[0] * (1 - p[1]) + (1 - p[0]) * p[1]) * p[2])
    q, n, w = p[0], best_of, best_of // 2 + 1
    from math import comb
    return float(sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(w, n + 1)))


def line_probs(pmf, line, calibrate=lambda p: p, be_over=0.5, be_under=0.5):
    """Calibrated over/under probabilities and per-side edge for any line, priced like the slate (push voids the leg)."""
    k = np.arange(len(pmf))
    p_push = float(pmf[k == line].sum())
    p_over = calibrate(float(pmf[k > line].sum()))
    p_under = max(0.0, 1 - p_over - p_push)
    live = max(1e-9, 1 - p_push)
    c_over, c_under = min(1.0, p_over / live), min(1.0, p_under / live)
    return {"p_over": p_over, "p_under": p_under, "p_push": p_push, "edge_over": c_over - be_over, "edge_under": c_under - be_under}


class Predictor:
    def __init__(self, hist: pd.DataFrame | None = None, models: Models | None = None, settings=None, payouts=None):
        self.hist = hist if hist is not None else pd.read_csv(HIST_PATH)
        self.hist["date"] = pd.to_datetime(self.hist["date"], utc=True, format="ISO8601")
        self.m = models or Models.load()
        self.settings = settings or load_yaml("settings.yaml")
        self._payouts_fixed = payouts is not None
        self.payouts = payouts or load_yaml("payouts.yaml")
        h = self.hist.sort_values("date")
        last = h.groupby("player_id").tail(1)
        self.players = last.assign(norm=last.player.map(norm_player), slug_norm=last.player_slug.fillna("").map(norm_player))
        self.teams = h.groupby("team_id").tail(1)[["team_id", "team", "date"]]
        tt = h.drop_duplicates(["game_id", "team_id"])
        recent = tt[tt.date >= tt.date.max() - pd.Timedelta(days=30)]
        share = recent.map_name.value_counts(normalize=True)
        self.active_maps = sorted(share[share >= ACTIVE_MAP_MIN_SHARE].index)
        self.team_games = tt
        mm = self.m.art.get("map_model")
        self.map_model = MapModel(self.hist, mm) if mm else None
        top = tt[tt.tier.isin(self.settings["tiers"])]
        top = top[top.date >= top.date.max() - pd.Timedelta(days=float(self.settings.get("team_maps_days", 92)))]
        self.team_counts = top.team_id.value_counts()
        self._feat_hist = None
        self._rcache, self._feat_cache, self._price_cache, self._sim_cache = {}, {}, {}, {}
        self.hist_version = f"{len(self.hist)}:{self.hist.date.max()}:{int(self.hist.game_id.sum())}"
        self.tnames = self.teams.set_index("team_id").team.to_dict()
        self.lgb_threads = 0
        self.timings = {}

    # ---------- name resolution
    def resolve_team(self, name):
        k = ("team", name)
        if k not in self._rcache:
            self._rcache[k] = self._resolve_team(name)
        return self._rcache[k]

    def _resolve_team(self, name):
        if not name:
            return None
        sims = self.teams.team.map(lambda t: team_similarity(t, name))
        if sims.max() < 0.75:
            return None
        cands = self.teams[sims == sims.max()]
        return int(cands.sort_values("date").team_id.iloc[-1])

    def resolve_player(self, name, team=None):
        k = ("player", name, team)
        if k not in self._rcache:
            self._rcache[k] = self._resolve_player(name, team)
        return self._rcache[k]

    def _resolve_player(self, name, team=None):
        n = norm_player(name)
        c = self.players[(self.players.norm == n) | (self.players.slug_norm == n)]
        if c.empty:
            return None
        if len(c) > 1 and team:
            s = c.team.map(lambda t: team_similarity(t, team))
            c = c[s == s.max()]
        return int(c.sort_values("date").player_id.iloc[-1])

    def roster(self, team_id, k=5):
        r = self.hist[(self.hist.team_id == team_id) & (self.hist.date >= self.hist.date.max() - pd.Timedelta(days=45))]
        last = r.sort_values("date").groupby("player_id").tail(1)
        return last.sort_values("date").player_id.tail(k).tolist()

    # ---------- match context
    def map_weights(self, t1, t2):
        if self.map_model is not None and self.map_model.p["use_picks"]:
            return self.map_model.weights(t1, t2, self.active_maps)

        def freq(t):
            g = self.team_games[(self.team_games.team_id == t)]
            vc = g.map_name.value_counts()
            vc = vc.reindex(self.active_maps).fillna(0)
            return vc / vc.sum() if vc.sum() else pd.Series(1 / len(self.active_maps), index=self.active_maps)
        w = 0.85 * (freq(t1) + freq(t2)) / 2 + 0.15 / len(self.active_maps)
        return (w / w.sum()).values

    def _future_rows(self, match_id, t1, t2, tnames, players, meta):
        rows = []
        for pid, tid in players:
            pr = self.players[self.players.player_id == pid].iloc[0]
            opp = t2 if tid == t1 else t1
            rows.append({"date": None, "match_id": match_id, "game_id": -abs(hash((match_id, 1))) % 10**9 * -1 - 1,
                         "map_number": 1, "map_name": "__future__", "bo_type": meta["best_of"], "tier": meta.get("tier", "b"),
                         "event_type": meta.get("event_type"), "team_id": tid, "team": tnames[tid], "opp_id": opp,
                         "opponent": tnames[opp], "player_id": pid, "player_slug": pr.player_slug, "player": pr.player,
                         "p_win_market": meta["p_map_team1"] if tid == t1 else 1 - meta["p_map_team1"]})
        return rows

    def _map_overrides(self, X, maps):
        """Duplicate each future row once per candidate map and fill map-specific features from history."""
        h = self.hist
        pm = h.groupby(["player_id", "map_name"])[["kills", "rounds"]].sum()
        tg = self.team_games.groupby(["team_id", "map_name"]).agg(w=("won", "sum"), n=("won", "size"), r=("rounds", "sum"))
        td = h.groupby(["team_id", "map_name"])["deaths"].sum()
        mn = h.groupby(["player_id", "map_name"]).size()
        out = []
        for m in maps:
            Y = X.copy()
            Y["map_name"] = m
            Y["map_code"] = (list_index(m))
            k = pm.reindex(pd.MultiIndex.from_arrays([Y.player_id, [m] * len(Y)]))
            Y["map_kpr"] = ((k["kills"].fillna(0) + 60 * Y.kpr_shrunk.values) / (k["rounds"].fillna(0) + 60)).values
            Y["map_n"] = mn.reindex(pd.MultiIndex.from_arrays([Y.player_id, [m] * len(Y)])).fillna(0).values
            for col, tcol in (("team_map_wr", "team_id"), ("opp_map_wr", "opp_id")):
                g = tg.reindex(pd.MultiIndex.from_arrays([Y[tcol], [m] * len(Y)]))
                Y[col] = ((g["w"].fillna(0) + 2.5) / (g["n"].fillna(0) + 5)).values
            g = tg.reindex(pd.MultiIndex.from_arrays([Y.opp_id, [m] * len(Y)]))
            d = td.reindex(pd.MultiIndex.from_arrays([Y.opp_id, [m] * len(Y)]))
            Y["opp_map_dpr"] = ((d.fillna(0).values / 5 + 100 * Y.opp_dpr.fillna(D_PRIOR).values) / (g["r"].fillna(0).values + 100))
            out.append(Y)
        return pd.concat(out, ignore_index=True)

    # ---------- per-match stages (cached; heavy stages can run in worker processes)
    def _context(self, a, b, lns, books):
        tnames = self.tnames
        odds = _find_odds(books, tnames.get(a), tnames.get(b))
        bo = max(max(x.maps) for x, _, _ in lns)
        best_of = odds.best_of if odds else (3 if bo <= 3 else 5)
        p_map = {}
        if odds:
            flip = team_similarity(odds.team1, tnames[a]) < team_similarity(odds.team2, tnames[a])
            for k, v in odds.p_map_team1.items():
                p_map[k] = 1 - v if flip else v
        players = {(pid, tid) for _, pid, tid in lns}
        players |= {(p, a) for p in self.roster(a)} | {(p, b) for p in self.roster(b)}
        seen = {}  # drop players listed on both sides (roster noise)
        for p, t in sorted(players):
            seen.setdefault(p, t)
        players = sorted(seen.items())
        meta = {"best_of": best_of, "p_map_team1": float(np.mean(list(p_map.values()))) if p_map else np.nan,
                "tier": "b", "event_type": None}
        key = f"{a}-{b}"
        fut = pd.DataFrame(self._future_rows(key, a, b, tnames, players, meta))
        legs = sorted({(str(pid), ln.stat, tuple(ln.maps)) for ln, pid, _ in lns})
        line_keys = frozenset((ln.source, ln.player, ln.stat, tuple(ln.maps), ln.line, ln.odds_type) for ln, _, _ in lns)
        fkey = (self.hist_version, key, tuple(players), best_of, round(meta["p_map_team1"], 4) if p_map else None)
        okey = repr((sorted(p_map.items()), best_of, odds.rounds_ladder if odds else None, odds.p_overtime if odds else None))
        return {"key": key, "a": a, "b": b, "lns": lns, "odds": odds, "p_map": p_map, "best_of": best_of,
                "players": players, "fut": fut, "fkey": fkey, "legs": legs, "line_keys": line_keys,
                "pkey": (fkey, tuple(legs), self.m.version, okey)}

    def _features(self, ctxs):
        """One feature pass over history for all new matches (disjoint players/teams per pass), cached."""
        todo = [c for c in ctxs if c["fkey"] not in self._feat_cache]
        while todo:
            batch, rest, used_p, used_t = [], [], set(), set()
            for c in todo:
                ps, ts = {p for p, _ in c["players"]}, {c["a"], c["b"]}
                if ps & used_p or ts & used_t:
                    rest.append(c)
                    continue
                batch.append(c)
                used_p |= ps
                used_t |= ts
            F = build_features(self.hist, pd.concat([c["fut"] for c in batch], ignore_index=True))
            F = F[F.future]
            for c in batch:
                self._feat_cache[c["fkey"]] = F[F.match_id == c["key"]].reset_index(drop=True)
            todo = rest
        for c in ctxs:
            c["F"] = self._feat_cache[c["fkey"]]

    def _price(self, job):
        """Model predict + exact single-leg pmfs for one match."""
        t0 = time.perf_counter()
        F, a, b, odds, best_of = job["F"], job["a"], job["b"], job["odds"], job["best_of"]
        p_map = dict(job["p_map"])
        if not p_map:
            pe = float(F[F.team_id == a].p_win_elo.mean())
            p_map = {k: pe for k in range(1, 4)}
        maps = self.active_maps
        w = self.map_weights(a, b)
        X = self._map_overrides(F, maps)
        X["kpr_hat"] = np.exp(self.m.kills.predict(X[FEATURES], raw_score=True, num_threads=self.lgb_threads))
        X["hs_hat"] = np.clip(np.exp(self.m.hs.predict(X[FEATURES], raw_score=True, num_threads=self.lgb_threads)), 0.05, 0.95)
        sh = self.m.art.get("shrink", {})
        X["kpr_hat"] = shrunk_rate(X.kpr_hat.values, X.kpr_shrunk.values, sh.get("kills", 1.0))
        X["hs_hat"] = np.clip(shrunk_rate(X.hs_hat.values, X.hs_pct_shrunk.values, sh.get("headshots", 1.0)), 0.05, 0.95)
        rounds = {}
        for j in range(1, min(best_of, 3) + 1):
            pj = p_map.get(j, p_map.get(1, 0.5))
            rounds[j] = D.rounds_dist(self.m.art["rounds_table"], 1 - abs(2 * pj - 1),
                                      (odds.rounds_ladder.get(j) if odds else None) or None,
                                      odds.p_overtime.get(j) if odds else None)
        pspecs = []
        for pid, tid in F[["player_id", "team_id"]].itertuples(index=False):
            sub = X[X.player_id == pid]
            pspecs.append(PlayerSpec(str(pid), 1 if tid == a else 2, dict(zip(sub.map_name, sub.kpr_hat)),
                                     dict(zip(sub.map_name, sub.hs_hat)), self.m.art["alpha"], self.m.art["rho"]))
        p_win_map = {}
        if self.map_model is not None and self.map_model.p["use_wins"]:
            dd = self.map_model.strength_diff(a, b, maps)
            for j in range(1, min(best_of, 3) + 1):  # per-map shifts, re-centred on this slot's market/Elo price
                p_win_map[j] = [float(x) for x in anchored(p_map.get(j, 0.5), dd, A.slot_marginal(w, j))]
        spec = MatchSpec(job["key"], best_of, maps, list(w), p_map, rounds, self.m.art["win_ratio"], pspecs, p_win_map)
        t1 = time.perf_counter()
        by_key, cache = {ps.key: ps for ps in pspecs}, {}
        pmfs = {(pk, stat, mp): A.leg_pmf(by_key[pk], spec, stat, mp, cache) for pk, stat, mp in job["legs"]}
        p3 = A.p_map3(spec)
        for k, pmf in pmfs.items():
            if spec.best_of == 3 and tuple(k[2]) == (3,) and p3 > 1e-6:
                # a "Map 3" line is void if map 3 isn't played: price it conditional on map 3 happening
                c = pmf.copy()
                c[0] = max(0.0, c[0] - (1 - p3))
                pmfs[k] = c / c.sum()
        return {"spec": spec, "p_map": p_map, "rounds": rounds, "pmfs": pmfs,
                "timings": {"model_predict": t1 - t0, "analytic": time.perf_counter() - t1}}

    def ensure_sims(self, reqs, seed=7):
        """{match: (ctx, n)} -> simulated outcome matrices with at least n draws. Cached on
        match + lines + model version + match spec; a fixed per-match seed, and top-ups append draws."""
        T, out, jobs, keys = self.timings, {}, [], []
        for mk, (c, n) in reqs.items():
            spec = self._price_cache[c["pkey"]]["spec"]
            skey = (mk, c["line_keys"], self.m.version, self.hist_version, hashlib.sha1(pickle.dumps(spec)).hexdigest())
            prev = self._sim_cache.get(skey)
            if prev is not None and prev["_n"] >= n:
                out[mk] = prev
                T["sim_cache_hits"] = T.get("sim_cache_hits", 0) + 1
                continue
            jobs.append({"spec": spec, "n": n - (prev["_n"] if prev else 0), "seed": zlib.crc32(f"{mk}|{seed}".encode()),
                         "state": prev["_state"] if prev else None})
            keys.append((mk, skey, prev))
        t0 = time.perf_counter()
        for (mk, skey, prev), new in zip(keys, self._map(_sim_job, jobs)):
            sim = new if prev is None else _append_sim(prev, new)
            self._sim_cache[skey] = out[mk] = sim
        T["simulate"] = T.get("simulate", 0) + time.perf_counter() - t0
        T["matches_simulated"] = T.get("matches_simulated", 0) + len(jobs)
        T["sims_drawn"] = T.get("sims_drawn", 0) + sum(j["n"] for j in jobs)
        return out

    def _n_workers(self):
        w = int(self.settings.get("workers", 0))
        return w if w > 0 else max(1, min(4, (os.cpu_count() or 2) - 1))

    def _map(self, fn, jobs):
        global _WORKER
        if not jobs:
            return []
        w = self._n_workers()
        with _POOL_LOCK:  # concurrent pricing threads (several dashboard sessions) share one pool and _WORKER
            if w > 1 and len(jobs) >= int(self.settings.get("parallel_min_jobs", 3)):
                try:
                    return list(_get_pool(w, self).map(fn, jobs))
                except Exception as e:  # noqa: BLE001  any pool failure falls back to the serial path
                    log.warning("parallel pricing failed (%s); running serially", e)
                    _shutdown_pool()
            prev, _WORKER = _WORKER, self
            try:
                return [fn(j) for j in jobs]
            finally:
                _WORKER = prev

    # ---------- main entry
    def build_slate(self, lines, books, n_sims=None, seed=7, con=None, now=None):
        """Returns (slate DataFrame, hit matrix dict {row_id: bool array for recommended side}, details)."""
        n_sims = n_sims or self.settings["n_sims"]
        T = self.timings = dict.fromkeys(("resolve_filter", "features", "price_matches_wall", "model_predict", "analytic", "rows",
                                          "simulate", "matches_priced_fresh", "matches_simulated", "sims_drawn", "sim_cache_hits"), 0.0)
        if not self._payouts_fixed:
            self.payouts = load_yaml("payouts.yaml")
        t0 = time.perf_counter()
        groups = {}
        unmatched, seen_keys = [], set()
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        for ln in lines:
            k = (ln.source, ln.player, ln.team, ln.opponent, ln.stat, tuple(ln.maps), ln.line, ln.odds_type)
            if k in seen_keys:
                unmatched.append(skip_record(ln, "duplicate line on board"))
                continue
            seen_keys.add(k)
            alt = alt_line_mode(ln, self.payouts)
            if alt == "exclude":
                unmatched.append(skip_record(ln, f"PrizePicks {ln.odds_type} line (alt payout, excluded)"))
                continue
            if isinstance(alt, float):
                ln = dataclasses.replace(ln, over_decimal=alt)
            t1, t2 = self.resolve_team(ln.team), self.resolve_team(ln.opponent)
            pid = self.resolve_player(ln.player, ln.team)
            if pid is None or t1 is None or t2 is None:
                unmatched.append(skip_record(ln, "player not in bo3 history" if pid is None else "team not matched"))
                continue
            skip = None
            if max(ln.maps) > 3:
                skip = "map 4/5 lines not modelled"
            elif ln.start_time is not None and pd.Timestamp(ln.start_time) < now - pd.Timedelta(minutes=5):
                skip = "match already started (pre-match model only)"
            if skip:
                unmatched.append(skip_record(ln, skip))
                continue
            thr = self.settings["min_team_maps"]
            if ln.odds_type == "demon" and self.settings.get("demon_lower_tier", False):
                thr = min(thr, int(self.settings.get("demon_min_team_maps", 4)))
            low = [n for n, t in ((ln.team, t1), (ln.opponent, t2)) if self.team_counts.get(t, 0) < thr]
            if low:
                unmatched.append(skip_record(ln, f"not enough tier-1/2 maps: {', '.join(low)}"))
                continue
            key = tuple(sorted((t1, t2)))
            groups.setdefault(key, []).append((ln, pid, t1))

        T["resolve_filter"] += time.perf_counter() - t0
        rows, details = [], {}
        ctxs = [self._context(a, b, lns, books) for (a, b), lns in groups.items()]
        by_match = {c["key"]: c for c in ctxs}
        t0 = time.perf_counter()
        self._features(ctxs)
        T["features"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        todo = [c for c in ctxs if c["pkey"] not in self._price_cache]
        jobs = [{"key": c["key"], "a": c["a"], "b": c["b"], "F": c["F"], "p_map": c["p_map"], "best_of": c["best_of"],
                 "odds": c["odds"], "legs": c["legs"]} for c in todo]
        for c, out in zip(todo, self._map(_price_job, jobs)):
            self._price_cache[c["pkey"]] = out
            T["model_predict"] += out["timings"]["model_predict"]
            T["analytic"] += out["timings"]["analytic"]
        T["price_matches_wall"] += time.perf_counter() - t0
        T["matches_priced_fresh"] = len(todo)
        t0 = time.perf_counter()
        for c in ctxs:
            out = self._price_cache[c["pkey"]]
            spec, p_map, rounds, a, odds = out["spec"], out["p_map"], out["rounds"], c["a"], c["odds"]
            Fi = c["F"].set_index("player_id")
            p_ot = float(np.mean([(rounds[j][0] > D.OT_ROUNDS) @ rounds[j][1] for j in rounds]))
            p_m3 = A.p_map3(spec)
            p_series = series_win(p_map, c["best_of"])
            for ln, pid, tid in c["lns"]:
                pmf = out["pmfs"][(str(pid), ln.stat, tuple(ln.maps))]
                k = np.arange(len(pmf))
                cdf = np.cumsum(pmf)
                p_over_raw = float(pmf[k > ln.line].sum())
                p_push = float(pmf[k == ln.line].sum())
                p_over = self.m.calibrate(ln.stat, len(ln.maps), p_over_raw)
                p_under = max(0.0, 1 - p_over - p_push)
                f = Fi.loc[pid]
                be_over = single_breakeven(ln.source, ln.over_decimal, self.payouts)
                be_under = single_breakeven(ln.source, ln.under_decimal, self.payouts)
                # a push voids the leg on PrizePicks/Underdog, so price each side conditional on no push
                live = max(1e-9, 1 - p_push)
                c_over, c_under = min(1.0, p_over / live), min(1.0, p_under / live)
                e_over, e_under = c_over - be_over, (c_under - be_under) if not ln.over_only else -1
                side = "over" if e_over >= e_under else "under"
                row = {
                    "id": len(rows), "source": ln.source, "player": ln.player, "player_id": pid, "team": ln.team,
                    "team_id": tid, "opp_id": c["b"] if tid == a else a,
                    "opponent": ln.opponent, "match": spec.key, "start_time": ln.start_time, "stat": ln.stat,
                    "maps": ln.maps, "maps_label": (f"Map {ln.maps[0]}" if len(ln.maps) == 1 else "Maps " + "+".join(map(str, ln.maps))), "line": ln.line,
                    "odds_type": ln.odds_type, "model_mean": float(k @ pmf),
                    "team_maps": int(min(self.team_counts.get(a, 0), self.team_counts.get(c["b"], 0))),
                    "q10": float(np.searchsorted(cdf, 0.1)), "q90": float(np.searchsorted(cdf, 0.9)),
                    "p_over": p_over, "p_under": p_under, "p_over_raw": p_over_raw, "side": side,
                    "p_push": p_push, "p_side": c_over if side == "over" else c_under, "breakeven": be_over if side == "over" else be_under,
                    "be_over": be_over, "be_under": np.nan if ln.over_only else be_under,
                    "n_maps": int(f.n_maps), "tenure": int(f.tenure), "role_shift": float(f.role_shift) if f.role_shift == f.role_shift else 0.0,
                    "kpr_10": float(f.kpr_10) if f.kpr_10 == f.kpr_10 else np.nan, "hs_pct_10": float(f.hs_pct_10) if f.hs_pct_10 == f.hs_pct_10 else np.nan,
                    "opp_dpr": float(f.opp_dpr) if f.opp_dpr == f.opp_dpr else np.nan,
                    "p_map_win": float(np.mean([p_map[j] if tid == a else 1 - p_map[j] for j in p_map])),
                    "p_ot": p_ot, "p_map3": p_m3, "odds_source": odds.source if odds else "elo",
                    "best_of": int(c["best_of"]), "p_series_win": p_series if tid == a else 1 - p_series,
                }
                row["edge"] = row["p_side"] - row["breakeven"]
                row["ev"] = row["p_side"] / row["breakeven"] - 1
                rows.append(row)
                details[row["id"]] = {"pmf": pmf, "spec": out["spec"], "pkey": str(pid)}
        slate = pd.DataFrame(rows)
        T["rows"] += time.perf_counter() - t0
        if not slate.empty:
            slate = apply_rules(slate, self.settings, con)
            alt = slate.odds_type.ne("standard")
            slate.loc[alt, "why"] = slate.loc[alt, "why"] + slate.loc[alt].apply(
                lambda r: f" · {r.odds_type}: " + ("priced at your set payout" if alt_line_mode(r, self.payouts) != "standard"
                                                   else "priced at the standard payout; the real payout is higher, so EV is a lower bound"), axis=1)
            # Monte Carlo only where slips can be built: matches with at least one +EV leg.
            n0 = int(self.settings.get("n_sims_initial", 20000))
            n_full = max(n0, int(n_sims))
            need = {}
            for mk, g in slate[slate.plus_ev].groupby("match"):
                # same-match legs lean on the simulated correlation, so those matches get the full count
                need[mk] = n_full if g.groupby("source").size().max() >= 2 else n0
            sims = self.ensure_sims({mk: (by_match[mk], n) for mk, n in need.items()}, seed)
            slate["n_sims"] = slate.match.map(lambda m: sims[m]["_n"] if m in sims else 0)
            for rid in slate.index[slate.n_sims > 0]:
                r = slate.loc[rid]
                tot = leg_totals(sims[r.match], str(r.player_id), r.stat, r.maps)
                rrng = np.random.default_rng(zlib.crc32(f"{r.match}|{r.source}|{r.player_id}|{r.stat}|{r.maps}|{r.line}|{seed}".encode()))
                hit = (tot > r.line) if r.side == "over" else (tot < r.line)
                push = tot == r.line
                if push.any():  # voided leg ~ resolved at the conditional rate
                    hit = hit | (push & (rrng.random(len(tot)) < r.p_side))
                details[rid].update(totals=tot, hit=_thin(hit, r.p_side, rrng))
        return slate, details, pd.DataFrame(unmatched, columns=SKIP_COLS)


D_PRIOR = 0.68

# ---------- worker processes: each loads history + models once and runs LightGBM single-threaded
_WORKER = None
_POOL, _POOL_SIG = None, None
_POOL_LOCK = threading.RLock()


def _init_worker(settings, payouts):
    global _WORKER
    _WORKER = Predictor(settings=settings, payouts=payouts)
    _WORKER.lgb_threads = 1


def _get_pool(n, pred):
    global _POOL, _POOL_SIG
    sig = (n, pred.m.version, pred.hist_version)
    if _POOL is None or _POOL_SIG != sig:
        _shutdown_pool()
        # spawn re-imports the parent's __main__ in each worker; under `streamlit run` that is the app
        # script, so start every worker now behind a blank __main__.
        import sys
        import types
        main = sys.modules.get("__main__")
        sys.modules["__main__"] = types.ModuleType("__main__")
        try:
            _POOL = ProcessPoolExecutor(n, mp_context=get_context("spawn"), initializer=_init_worker,
                                        initargs=(pred.settings, pred.payouts))
            list(_POOL.map(_noop, [0.5] * n))
        finally:
            sys.modules["__main__"] = main
        _POOL_SIG = sig
    return _POOL


def _noop(t):
    time.sleep(t)


def _shutdown_pool():
    global _POOL, _POOL_SIG
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
    _POOL, _POOL_SIG = None, None


def _price_job(job):
    return _WORKER._price(job)


def _sim_job(job):
    rng = np.random.default_rng(job["seed"])
    if job["state"] is not None:
        rng.bit_generator.state = job["state"]
    sim = simulate_match(job["spec"], job["n"], rng)
    out = {k: (v if k == "_played" else {s: a.astype(np.int16) for s, a in v.items()}) for k, v in sim.items()}
    out["_n"], out["_state"] = job["n"], rng.bit_generator.state
    return out


def _append_sim(a, b):
    out = {k: (np.concatenate([a[k], b[k]]) if k == "_played" else {s: np.concatenate([a[k][s], b[k][s]]) for s in a[k]})
           for k in a if not k.startswith("_") or k == "_played"}
    out["_n"], out["_state"] = a["_n"] + b["_n"], b["_state"]
    return out


def list_index(m):
    from ..features.build import MAPS
    return MAPS.index(m) if m in MAPS else -1


# tie-break between books on the same match: Pinnacle (has map ladders / OT), then Polymarket (validated in
# docs/validation.md against the Elo fallback), then Thunderpick
BOOK_PRIORITY = {"pinnacle": 0.02, "polymarket": 0.01}


def _find_odds(books, n1, n2):
    best, score = None, 0.0
    for mo in books:
        s = min(max(team_similarity(mo.team1, n1), team_similarity(mo.team2, n1)),
                max(team_similarity(mo.team1, n2), team_similarity(mo.team2, n2)))
        bonus = BOOK_PRIORITY.get(mo.source, 0.0)
        if s >= 0.8 and s + bonus > score:
            best, score = mo, s + bonus
    return best


def alt_line_mode(ln, payouts):
    """How a PrizePicks demon/goblin line is priced: 'standard', a per-pick decimal payout, or 'exclude'.
    PrizePicks doesn't publish these payouts in its API: demons pay more than standard (so the standard breakeven is a
    conservative lower bound on EV), goblins pay less (so they need a payout set in payouts.yaml)."""
    if ln.source != "prizepicks" or ln.odds_type == "standard":
        return "standard"
    cfg = payouts.get("prizepicks", {})
    if ln.odds_type in cfg.get("exclude_odds_types", []):
        return "exclude"
    mode = cfg.get("alt_lines", {}).get(ln.odds_type, "exclude")
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return float(mode) if mode > 1 else "exclude"
    return mode if mode == "standard" else "exclude"


SKIP_COLS = ["platform", "player", "team", "opponent", "start_time", "stat", "maps", "maps_label", "line", "odds_type", "reason"]


def skip_record(ln, reason):
    return {"platform": ln.source, "player": ln.player, "team": ln.team, "opponent": ln.opponent,
            "start_time": ln.start_time, "stat": ln.stat, "maps": "+".join(map(str, ln.maps)),
            "maps_label": f"Map {ln.maps[0]}" if len(ln.maps) == 1 else "Maps " + "+".join(map(str, ln.maps)),
            "line": ln.line, "odds_type": ln.odds_type, "reason": reason}


def _thin(hit, p_target, rng):
    """Adjust simulated hit indicators to the calibrated marginal while keeping the joint structure."""
    p_raw = hit.mean()
    hit = hit.copy()
    if p_raw > p_target and p_raw > 0:
        flip = hit & (rng.random(len(hit)) < (p_raw - p_target) / p_raw)
        hit[flip] = False
    elif p_raw < p_target and p_raw < 1:
        flip = ~hit & (rng.random(len(hit)) < (p_target - p_raw) / (1 - p_raw))
        hit[flip] = True
    return hit


def apply_rules(s: pd.DataFrame, st, con=None) -> pd.DataFrame:
    """Exclusions, line-movement checks, confidence tiers and one-line reasons."""
    from ..tracking.store import first_line
    flags, tiers, reasons, why = [], [], [], []
    for _, r in s.iterrows():
        f = []
        if r.n_maps < st["min_maps"]:
            f.append(f"thin sample ({r.n_maps} maps)")
        if r.tenure < st["standin_tenure"]:
            f.append("stand-in / new to team")
        if r.role_shift > st["role_shift_flag"]:
            f.append("recent role change")
        if r.line > 0 and abs(r.model_mean / r.line - 1) > st["max_line_gap"]:
            f.append("model far from line (news?)")
        if con is not None:
            fl = first_line(con, r.source, r.player, r.stat, r.maps, r.start_time.isoformat() if r.start_time else None)
            if fl is not None and abs(r.line - fl) >= st["line_move_flag"]:
                moved_against = (r.line - fl) * (1 if r.side == "over" else -1) > 0
                f.append(f"line moved {fl}->{r.line}" + (" against model" if moved_against else ""))
        flags.append("; ".join(f))
        if f or r.edge < st["min_edge"]:
            tiers.append("-")
        elif r.n_maps >= 30 and r.tenure >= 10 and r.edge >= 0.06 and r.odds_source != "elo":
            tiers.append("A")
        elif r.n_maps >= 20 and r.edge >= 0.04:
            tiers.append("B")
        else:
            tiers.append("C")
        lower_tier = r.get("team_maps", 10**6) < st.get("min_team_maps", 0)  # only demons get here (opt-in setting)
        if lower_tier and tiers[-1] in ("A", "B"):
            tiers[-1] = "C"
        stat_rate = (f"{r.kpr_10:.2f} KPR L10" if r.stat == "kills" else f"{r.hs_pct_10:.0%} HS L10") if r[
            "kpr_10" if r.stat == "kills" else "hs_pct_10"] == r["kpr_10" if r.stat == "kills" else "hs_pct_10"] else "form n/a"
        reasons.append(f"{r.side.upper()} {r.line}: model {r.model_mean:.1f} ({stat_rate}, opp allows {r.opp_dpr:.2f} KPR, "
                       f"{r.p_map_win:.0%} map win, {r.p_ot:.0%} OT/map" + (f", {r.p_map3:.0%} map 3" if 3 in r.maps else "") + ")")
        if f:
            why.append(f"Skip: {f[0]}")
        elif r.edge < st["min_edge"]:
            why.append(f"No bet: edge {r.edge:+.1%} < {st['min_edge']:.0%} (mean {r.model_mean:.1f} vs {r.line:g})")
        else:
            why.append(f"{r.side.upper()} {r.line:g}: mean {r.model_mean:.1f}, {stat_rate}, P {r.p_side:.0%} vs {r.breakeven:.0%} needed"
                       + (f" · lower-tier team ({r.team_maps} tier-1/2 maps), capped at C" if lower_tier else ""))
    s = s.copy()
    s["flags"], s["tier"], s["reason"], s["why"] = flags, tiers, reasons, why
    s["plus_ev"] = s.tier != "-"
    return s
