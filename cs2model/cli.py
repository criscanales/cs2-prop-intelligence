"""python -m cs2model.cli {crawl,train,slate,grade,refresh}"""
import argparse
import logging
import time

import pandas as pd

from .parlay.engine import load_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cs2model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="pull last N days of bo3.gg map stats")
    c.add_argument("--days", type=int, default=None)
    sub.add_parser("train", help="train + backtest models")
    s = sub.add_parser("slate", help="price today's lines, print +EV picks and slips, export CSV/JSON")
    s.add_argument("--platform", action="append", default=None)
    s.add_argument("--sims", type=int, default=None)
    s.add_argument("--log-picks", action="store_true", help="record +EV picks and best slips in tracking.db")
    sub.add_parser("grade", help="grade logged picks/slips against bo3 results")
    r = sub.add_parser("refresh", help="crawl recent days, grade, retrain (optional), re-price; loop")
    r.add_argument("--every", type=int, default=0, help="minutes between runs (0 = once)")
    r.add_argument("--retrain", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st = load_yaml("settings.yaml")

    if a.cmd == "crawl":
        from .scraper.bo3_history import crawl
        df = crawl(a.days or st["history_days"], tiers=tuple(st["tiers"]))
        print(f"{len(df)} player-map rows")
    elif a.cmd == "train":
        from .training.train import train
        _, rep = train()
        for name, r in rep["models"].items():
            for k, v in r.items():
                b = v["betting"]
                print(f"{name:7s} {k:18s} MAE {v['mae']:.2f}  Brier {v['brier']:.4f}  LogLoss {v['logloss']:.4f}  "
                      f"picks {b['n_picks']:5d}  hit {b['hit_rate']:.3f}  ROI {b['roi']:+.3f}")
    elif a.cmd == "slate":
        slate_cmd(a.platform, a.sims, a.log_picks, st)
    elif a.cmd == "grade":
        grade_cmd()
    elif a.cmd == "refresh":
        while True:
            from .scraper.bo3_history import crawl
            crawl(st["history_days"], tiers=tuple(st["tiers"]))  # cached days are free; only new ones hit the API
            grade_cmd()
            if a.retrain:
                from .training.train import train
                train()
            slate_cmd(None, None, True, st)
            if not a.every:
                break
            time.sleep(a.every * 60)


def slate_cmd(platforms, sims, log_picks, st):
    from .pipeline import build_slips, export, run_slate
    from .tracking import store
    res = run_slate(n_sims=sims)
    for e in res["errors"]:
        print("SOURCE ERROR:", e)
    slate = res["slate"]
    print(f"{res['n_lines']} lines, {res['n_books']} matches with odds, {len(res['unmatched'])} unmatched")
    if slate.empty:
        return
    payouts = load_yaml("payouts.yaml")
    pd.set_option("display.width", 220)
    ev = slate[slate.plus_ev].sort_values("edge", ascending=False)
    print(ev[["source", "player", "opponent", "maps_label", "stat", "line", "model_mean", "side", "p_side", "edge",
              "tier", "reason"]].head(30).to_string(index=False))
    slips = []
    for p in platforms or ["prizepicks", "underdog"]:
        s = build_slips(slate, res["details"], p, st, payouts)
        if not s.empty:
            slips.append(s)
            print(f"\n{p}: best slip per play type")
            print(s[s.recommended][["play", "legs", "p_all", "ev", "ev_indep", "corr_ratio", "sanity_ok", "stake"]].to_string(index=False))
    slips = pd.concat(slips, ignore_index=True) if slips else pd.DataFrame()
    print("exported:", *export(slate, slips))
    if log_picks:
        con = store.connect()
        store.log_picks(ev, con)
        ids = {(r.source, r.player, r.stat, "+".join(map(str, r.maps)), r.side, r.line): None for r in ev.itertuples()}
        for r in con.execute("SELECT id, source, player, stat, maps, side, line FROM picks"):
            k = tuple(r[1:])
            if k in ids:
                ids[k] = r[0]
        if not slips.empty:
            for s in slips[slips.recommended].itertuples():
                leg_db = [ids.get((slate.loc[i, "source"], slate.loc[i, "player"], slate.loc[i, "stat"],
                                   "+".join(map(str, slate.loc[i, "maps"])), slate.loc[i, "side"], slate.loc[i, "line"]))
                          for i in s.leg_ids]
                store.log_slip(s.platform, s.play, leg_db, s.p_all, s.ev, s.stake, s.hit_dist, con)


def grade_cmd():
    from .inference.predict import Predictor
    from .scraper.bo3_history import HIST_PATH
    from .tracking import store
    hist = pd.read_csv(HIST_PATH)
    pr = Predictor(hist)
    n = store.grade(pr.hist, pr.resolve_player)
    summ, _, _ = store.performance()
    print(f"graded {n} picks; performance: {summ}")


if __name__ == "__main__":
    main()
