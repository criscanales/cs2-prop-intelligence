"""End-to-end daily run shared by the CLI and the dashboard."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .parlay import engine as E
from .scraper.books import fetch_all_books, fetch_bo3_match_meta
from .scraper.http import ROOT
from .scraper.lines import fetch_all_lines
from .tracking import store

OUT = ROOT / "data" / "out"


def line_sig(lines):
    """Identity of the current board: (source, player, stat, maps, odds_type) -> line."""
    return {(ln.source, ln.player, ln.stat, tuple(ln.maps), ln.odds_type): ln.line for ln in lines}


def diff_sig(old, new):
    added = [k for k in new if k not in old]
    moved = [k for k in new if k in old and new[k] != old[k]]
    removed = [k for k in old if k not in new]
    return {"added": added, "moved": moved, "removed": removed, "old": old}


def reconcile(n_lines, slate, skipped):
    """Every pulled line is either priced or skipped with exactly one reason."""
    cat = skipped.reason.str.split(":").str[0] if len(skipped) else pd.Series(dtype=str)
    per = {}
    for plat in sorted(set(skipped.platform) | set(slate.source if len(slate) else [])):
        sl = slate[slate.source == plat] if len(slate) else slate
        per[plat] = {"lines": int((skipped.platform == plat).sum()) + len(sl), "priced_props": len(sl),
                     "priced_matches": int(sl.match.nunique()) if len(sl) else 0}
    return {"lines": n_lines, "priced": len(slate), "skipped": len(skipped), "balanced": len(slate) + len(skipped) == n_lines,
            "by_reason": cat.value_counts().to_dict(), "by_reason_detail": skipped.reason.value_counts().to_dict(),
            "per_platform": per, "priced_matches": int(slate.match.nunique()) if len(slate) else 0}


def run_slate(predictor=None, line_sources=("underdog", "prizepicks"), book_sources=("pinnacle", "polymarket", "thunderpick"),
              n_sims=None, log=True, on_lines=None):
    from .inference.predict import Predictor
    predictor = predictor or Predictor()
    lines, e1 = fetch_all_lines(line_sources)
    if on_lines:
        on_lines(lines)
    books, e2 = fetch_all_books(book_sources)
    odds_at = datetime.now(timezone.utc)
    con = store.connect()
    if log:
        store.snapshot_lines(lines, con)
    slate, details, unmatched = predictor.build_slate(lines, books, n_sims=n_sims, con=con)
    slate = add_match_meta(slate, odds_at)
    if log:
        store.log_preds(pd.concat([slate, getattr(predictor, "t3_collected", pd.DataFrame())]), con)
    return {"slate": slate, "details": details, "unmatched": unmatched, "errors": e1 + e2,
            "recon": reconcile(len(lines), slate, unmatched),
            "n_lines": len(lines), "sig": line_sig(lines), "n_books": len(books), "predictor": predictor,
            "ts": datetime.now(timezone.utc)}


def add_match_meta(slate, odds_at, meta=None):
    """Event / tier / LAN from bo3.gg (None when unknown) and when the match odds were fetched."""
    if slate.empty:
        return slate
    if meta is None:
        try:
            meta = fetch_bo3_match_meta()
        except Exception:  # noqa: BLE001
            meta = {}
    keys = [frozenset((int(a), int(b))) if a == a and b == b else None for a, b in zip(slate.team_id, slate.opp_id)]
    for col in ("event", "match_tier", "event_type"):
        slate[col] = [(meta.get(k) or {}).get(col) for k in keys]
    slate["odds_at"] = [odds_at if s != "elo" else None for s in slate.odds_source]
    return slate


def stake_cap(settings, payouts, platform):
    ok = payouts.get(platform, {}).get("verified", False)
    return settings["max_stake_pct"] if ok else min(settings["max_stake_pct"], settings.get("unverified_max_stake_pct", 0.005))


def hit_labels(hit_dist):
    return " · ".join(f"{k} hit{'s' if k != 1 else ''}: {p:.1%}" for k, p in enumerate(hit_dist))


def hit_matrix(details, ids):
    """Leg hit draws as columns. Matches are simulated independently and can have different draw counts (top-ups only
    go to matches that need them), so every column is cut to the shortest; draws within a match stay aligned."""
    hits = [details[i]["hit"] for i in ids]
    n = min(len(h) for h in hits)
    return np.column_stack([h[:n] for h in hits])


def build_slips(slate, details, platform, settings, payouts, max_legs_pool=20, plays=None):
    """Best slips per play type from +EV legs on one platform (correlation-aware Monte Carlo)."""
    pool = slate[(slate.source == platform) & slate.plus_ev].sort_values("edge", ascending=False).head(max_legs_pool)
    if pool.empty:
        return pd.DataFrame()
    H = hit_matrix(details, pool.id)
    legs = [{"match": r.match, "player": r.player_id, "stat": r.stat, "id": r.id} for r in pool.itertuples()]
    out = []
    for name, play in (plays or E.plays(platform, payouts)).items():
        for rank, r in enumerate(E.optimize(H, legs, play, top_n=settings["top_n_slips"],
                                            max_per_match=settings["max_legs_per_match"])):
            f = E.kelly_fraction(r["hit_dist"], play)
            stake = min(settings["kelly_fraction"] * f, stake_cap(settings, payouts, platform)) * settings["bankroll"]
            ids = [legs[i]["id"] for i in r["legs"]]
            out.append({
                "platform": platform, "play": name, "legs_n": play["legs"], "type": play["type"], "rank": rank + 1,
                "leg_ids": ids,
                "legs": " | ".join(f"{slate.loc[i, 'player']} {slate.loc[i, 'side'][0].upper()} {slate.loc[i, 'line']} "
                                   f"{slate.loc[i, 'stat'][:1].upper()}{'' if slate.loc[i, 'stat'] == 'kills' else 'S'}"
                                   for i in ids),
                "p_all": r["p_all"], "hit_dist": r["hit_dist"].round(4).tolist(), "hits": hit_labels(r["hit_dist"]),
                "max_payout": float(E.payout_vector(play).max()), "ev": r["ev"], "ev_indep": r["ev_indep"],
                "corr_ratio": r["corr_ratio"], "heavy_corr": r["heavy_corr"],
                "sanity_ok": E.sanity_check(H, r["legs"], play) if play["legs"] >= 6 else True,
                "kelly_full": f, "stake": round(stake, 2),
                "matches": len({legs[i]["match"] for i in r["legs"]}),
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df[df.ev > 0].sort_values("ev", ascending=False).reset_index(drop=True)
    # daily exposure cap over the best slip of each play type
    best = df.groupby("play").head(1).index
    cap = settings["daily_exposure_pct"] * settings["bankroll"]
    tot = df.loc[best, "stake"].sum()
    df["recommended"] = df.index.isin(best)
    if tot > cap > 0:
        df.loc[best, "stake"] = (df.loc[best, "stake"] * cap / tot).round(2)
    return df


def export(slate, slips=None, tag=None, skipped=None):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    cols = ["source", "player", "team", "opponent", "start_time", "maps_label", "stat", "line", "model_mean", "q10", "q90",
            "p_over", "p_under", "side", "edge", "ev", "tier", "n_maps", "flags", "reason"]
    picks = slate[slate.plus_ev][cols] if not slate.empty else slate
    paths = [OUT / f"picks_{tag}.csv", OUT / f"picks_{tag}.json"]
    picks.to_csv(paths[0], index=False)
    paths[1].write_text(picks.to_json(orient="records", date_format="iso", indent=1))
    if slips is not None and not slips.empty:
        paths += [OUT / f"slips_{tag}.csv", OUT / f"slips_{tag}.json"]
        slips.to_csv(paths[2], index=False)
        paths[3].write_text(json.dumps(slips.to_dict("records"), default=str, indent=1))
    if skipped is not None and len(skipped):
        paths.append(OUT / f"skipped_{tag}.csv")
        skipped.drop_duplicates().to_csv(paths[-1], index=False)
    return paths
