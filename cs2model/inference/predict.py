"""Turn today's prop lines + match odds into calibrated over/under probabilities, edges and
simulated hit indicators (for slips)."""
import logging
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.build import FEATURES, build_features
from ..names import norm_player, team_similarity
from ..parlay.engine import load_yaml, single_breakeven
from ..scraper.bo3_history import HIST_PATH
from ..training.train import MODEL_DIR
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

    @classmethod
    def load(cls, d=MODEL_DIR):
        import json
        return cls(lgb.Booster(model_file=str(d / "kills_lgbm.txt")), lgb.Booster(model_file=str(d / "hs_lgbm.txt")),
                   joblib.load(d / "calibrators.joblib"), json.loads((d / "artifacts.json").read_text()))

    def calibrate(self, stat, n_maps, p):
        c = self.calibrators.get(f"{stat}_{'map' if n_maps == 1 else 'series'}")
        return float(c.predict([p])[0]) if c is not None else p


class Predictor:
    def __init__(self, hist: pd.DataFrame | None = None, models: Models | None = None, settings=None, payouts=None):
        self.hist = hist if hist is not None else pd.read_csv(HIST_PATH)
        self.hist["date"] = pd.to_datetime(self.hist["date"], utc=True)
        self.m = models or Models.load()
        self.settings = settings or load_yaml("settings.yaml")
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
        self._feat_hist = None

    # ---------- name resolution
    def resolve_team(self, name):
        if not name:
            return None
        sims = self.teams.team.map(lambda t: team_similarity(t, name))
        if sims.max() < 0.75:
            return None
        cands = self.teams[sims == sims.max()]
        return int(cands.sort_values("date").team_id.iloc[-1])

    def resolve_player(self, name, team=None):
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

    # ---------- main entry
    def build_slate(self, lines, books, n_sims=None, seed=7, con=None):
        """Returns (slate DataFrame, hit matrix dict {row_id: bool array for recommended side}, details)."""
        n_sims = n_sims or self.settings["n_sims"]
        rng = np.random.default_rng(seed)
        groups = {}
        unmatched = []
        for ln in lines:
            if ln.source == "prizepicks" and ln.odds_type in self.payouts["prizepicks"].get("exclude_odds_types", []):
                continue
            t1, t2 = self.resolve_team(ln.team), self.resolve_team(ln.opponent)
            pid = self.resolve_player(ln.player, ln.team)
            if pid is None or t1 is None or t2 is None:
                unmatched.append({"player": ln.player, "team": ln.team, "opponent": ln.opponent,
                                  "reason": "player not in bo3 history" if pid is None else "team not matched"})
                continue
            key = tuple(sorted((t1, t2)))
            groups.setdefault(key, []).append((ln, pid, t1))

        rows, details, sims = [], {}, {}
        for (a, b), lns in groups.items():
            odds = _find_odds(books, self.teams.set_index("team_id").team.get(a), self.teams.set_index("team_id").team.get(b))
            tnames = self.teams.set_index("team_id").team.to_dict()
            bo = max(max(x.maps) for x, _, _ in lns)
            best_of = odds.best_of if odds else (3 if bo <= 3 else 5)
            p_map = {}
            if odds:
                flip = team_similarity(odds.team1, tnames[a]) < team_similarity(odds.team2, tnames[a])
                for k, v in odds.p_map_team1.items():
                    p_map[k] = 1 - v if flip else v
            players = {(pid, tid) for _, pid, tid in lns}
            players |= {(p, a) for p in self.roster(a)} | {(p, b) for p in self.roster(b)}
            # drop players listed on both sides (roster noise)
            seen = {}
            for p, t in sorted(players):
                seen.setdefault(p, t)
            players = sorted(seen.items())
            meta = {"best_of": best_of, "p_map_team1": float(np.mean(list(p_map.values()))) if p_map else np.nan,
                    "tier": "b", "event_type": None}
            fut = pd.DataFrame(self._future_rows(f"{a}-{b}", a, b, tnames, players, meta))
            F = build_features(self.hist, fut)
            F = F[F.future].reset_index(drop=True)
            if not p_map:
                pe = float(F[F.team_id == a].p_win_elo.mean())
                p_map = {k: pe for k in range(1, 4)}
            maps = self.active_maps
            w = self.map_weights(a, b)
            X = self._map_overrides(F, maps)
            X["kpr_hat"] = np.exp(self.m.kills.predict(X[FEATURES], raw_score=True))
            X["hs_hat"] = np.clip(np.exp(self.m.hs.predict(X[FEATURES], raw_score=True)), 0.05, 0.95)
            rounds = {}
            for j in range(1, min(best_of, 3) + 1):
                pj = p_map.get(j, p_map.get(1, 0.5))
                rounds[j] = D.rounds_dist(self.m.art["rounds_table"], 1 - abs(2 * pj - 1),
                                          (odds.rounds_ladder.get(j) if odds else None) or None,
                                          odds.p_overtime.get(j) if odds else None)
            pspecs = []
            for (pid, tid) in players:
                sub = X[X.player_id == pid]
                pspecs.append(PlayerSpec(str(pid), 1 if tid == a else 2, dict(zip(sub.map_name, sub.kpr_hat)),
                                         dict(zip(sub.map_name, sub.hs_hat)), self.m.art["alpha"], self.m.art["rho"]))
            spec = MatchSpec(f"{a}-{b}", best_of, maps, list(w), p_map, rounds, self.m.art["win_ratio"], pspecs)
            sim = simulate_match(spec, n_sims, rng)
            sims[spec.key] = sim
            Fi = F.set_index("player_id")
            for ln, pid, tid in lns:
                tot = leg_totals(sim, str(pid), ln.stat, ln.maps)
                p_over_raw = float((tot > ln.line).mean())
                p_push = float((tot == ln.line).mean())
                p_over = self.m.calibrate(ln.stat, len(ln.maps), p_over_raw)
                p_under = max(0.0, 1 - p_over - p_push)
                f = Fi.loc[pid]
                be_over = single_breakeven(ln.source, ln.over_decimal, self.payouts)
                be_under = single_breakeven(ln.source, ln.under_decimal, self.payouts)
                e_over, e_under = p_over - be_over, (p_under - be_under) if not ln.over_only else -1
                side = "over" if e_over >= e_under else "under"
                row = {
                    "id": len(rows), "source": ln.source, "player": ln.player, "player_id": pid, "team": ln.team,
                    "opponent": ln.opponent, "match": spec.key, "start_time": ln.start_time, "stat": ln.stat,
                    "maps": ln.maps, "maps_label": "Maps " + "+".join(map(str, ln.maps)), "line": ln.line,
                    "odds_type": ln.odds_type, "model_mean": float(tot.mean()),
                    "q10": float(np.quantile(tot, 0.1)), "q90": float(np.quantile(tot, 0.9)),
                    "p_over": p_over, "p_under": p_under, "p_over_raw": p_over_raw, "side": side,
                    "p_side": p_over if side == "over" else p_under, "breakeven": be_over if side == "over" else be_under,
                    "n_maps": int(f.n_maps), "tenure": int(f.tenure), "role_shift": float(f.role_shift) if f.role_shift == f.role_shift else 0.0,
                    "kpr_10": float(f.kpr_10) if f.kpr_10 == f.kpr_10 else np.nan, "hs_pct_10": float(f.hs_pct_10) if f.hs_pct_10 == f.hs_pct_10 else np.nan,
                    "opp_dpr": float(f.opp_dpr) if f.opp_dpr == f.opp_dpr else np.nan,
                    "p_map_win": float(np.mean([p_map[k] if tid == a else 1 - p_map[k] for k in p_map])),
                    "p_ot": float(np.mean([(rounds[j][0] > D.OT_ROUNDS) @ rounds[j][1] for j in rounds])),
                    "p_map3": float(sim["_played"][:, 2].mean()) if sim["_played"].shape[1] > 2 else 0.0,
                    "odds_source": odds.source if odds else "elo",
                }
                row["edge"] = row["p_side"] - row["breakeven"]
                row["ev"] = row["p_side"] / row["breakeven"] - 1
                rows.append(row)
                details[row["id"]] = {"totals": tot, "hit": (tot > ln.line) if side == "over" else (tot < ln.line)}
        slate = pd.DataFrame(rows)
        if not slate.empty:
            slate = apply_rules(slate, self.settings, con)
            for rid, d in details.items():
                d["hit"] = _thin(d["hit"], slate.loc[rid, "p_side"], rng)
        return slate, details, pd.DataFrame(unmatched)


D_PRIOR = 0.68


def list_index(m):
    from ..features.build import MAPS
    return MAPS.index(m) if m in MAPS else -1


def _find_odds(books, n1, n2):
    best, score = None, 0.0
    for mo in books:
        s = min(max(team_similarity(mo.team1, n1), team_similarity(mo.team2, n1)),
                max(team_similarity(mo.team1, n2), team_similarity(mo.team2, n2)))
        bonus = 0.01 if mo.source == "pinnacle" else 0.0  # pinnacle has map ladders / OT
        if s >= 0.8 and s + bonus > score:
            best, score = mo, s + bonus
    return best


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
    flags, tiers, reasons = [], [], []
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
        stat_rate = (f"{r.kpr_10:.2f} KPR L10" if r.stat == "kills" else f"{r.hs_pct_10:.0%} HS L10") if r[
            "kpr_10" if r.stat == "kills" else "hs_pct_10"] == r["kpr_10" if r.stat == "kills" else "hs_pct_10"] else "form n/a"
        reasons.append(f"{r.side.upper()} {r.line}: model {r.model_mean:.1f} ({stat_rate}, opp allows {r.opp_dpr:.2f} KPR, "
                       f"{r.p_map_win:.0%} map win, {r.p_ot:.0%} OT/map" + (f", {r.p_map3:.0%} map 3" if 3 in r.maps else "") + ")")
    s = s.copy()
    s["flags"], s["tier"], s["reason"] = flags, tiers, reasons
    s["plus_ev"] = s.tier != "-"
    return s
