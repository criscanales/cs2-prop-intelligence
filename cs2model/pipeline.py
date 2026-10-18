"""End-to-end daily run shared by the CLI and the dashboard."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .parlay import engine as E
from .scraper.books import fetch_all_books
from .scraper.http import ROOT
from .scraper.lines import fetch_all_lines
from .tracking import store

OUT = ROOT / "data" / "out"


def run_slate(predictor=None, line_sources=("underdog", "prizepicks"), book_sources=("pinnacle", "thunderpick"),
              n_sims=None, log=True):
    from .inference.predict import Predictor
    predictor = predictor or Predictor()
    lines, e1 = fetch_all_lines(line_sources)
    books, e2 = fetch_all_books(book_sources)
    con = store.connect()
    if log:
        store.snapshot_lines(lines, con)
    slate, details, unmatched = predictor.build_slate(lines, books, n_sims=n_sims, con=con)
    return {"slate": slate, "details": details, "unmatched": unmatched, "errors": e1 + e2,
            "n_lines": len(lines), "n_books": len(books), "predictor": predictor,
            "ts": datetime.now(timezone.utc)}


def build_slips(slate, details, platform, settings, payouts, max_legs_pool=20, plays=None):
    """Best slips per play type from +EV legs on one platform (correlation-aware Monte Carlo)."""
    pool = slate[(slate.source == platform) & slate.plus_ev].sort_values("edge", ascending=False).head(max_legs_pool)
    if pool.empty:
        return pd.DataFrame()
    H = np.column_stack([details[i]["hit"] for i in pool.id])
    legs = [{"match": r.match, "player": r.player_id, "stat": r.stat, "id": r.id} for r in pool.itertuples()]
    out = []
    for name, play in (plays or E.plays(platform, payouts)).items():
        for rank, r in enumerate(E.optimize(H, legs, play, top_n=settings["top_n_slips"],
                                            max_per_match=settings["max_legs_per_match"])):
            f = E.kelly_fraction(r["hit_dist"], play)
            stake = min(settings["kelly_fraction"] * f, settings["max_stake_pct"]) * settings["bankroll"]
            ids = [legs[i]["id"] for i in r["legs"]]
            out.append({
                "platform": platform, "play": name, "legs_n": play["legs"], "type": play["type"], "rank": rank + 1,
                "leg_ids": ids,
                "legs": " | ".join(f"{slate.loc[i, 'player']} {slate.loc[i, 'side'][0].upper()} {slate.loc[i, 'line']} "
                                   f"{slate.loc[i, 'stat'][:1].upper()}{'' if slate.loc[i, 'stat'] == 'kills' else 'S'}"
                                   for i in ids),
                "p_all": r["p_all"], "hit_dist": r["hit_dist"].round(4).tolist(),
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


def export(slate, slips=None, tag=None):
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
    return paths
