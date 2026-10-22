"""python -m cs2model.cli {crawl,train,slate,grade,regrade,map3-report,refresh,backup,backup-daemon,restore,backup-status,record-gap}"""
import argparse
import json
import logging
import os
import sys
import time

import pandas as pd

from .parlay.engine import load_yaml


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cs2model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="pull last N days of bo3.gg map stats and merge them into the history")
    c.add_argument("--days", type=int, default=None)
    sub.add_parser("train", help="train + backtest models")
    s = sub.add_parser("slate", help="price today's lines, print +EV picks and slips, export CSV/JSON")
    s.add_argument("--platform", action="append", default=None)
    s.add_argument("--sims", type=int, default=None)
    s.add_argument("--log-picks", action="store_true", help="record +EV picks and best slips in tracking.db")
    sub.add_parser("grade", help="grade logged picks/slips against bo3 results")
    rg = sub.add_parser("regrade", help="re-check settled picks/preds against bo3 results and fix wrong settlements")
    rg.add_argument("--dry-run", action="store_true")
    sub.add_parser("map3-report", help="predicted vs actual Map 3 rate for finished BO3s, by favourite strength and odds provenance")
    r = sub.add_parser("refresh", help="crawl recent days, grade, retrain (optional), re-price; loop")
    r.add_argument("--every", type=int, default=0, help="minutes between runs (0 = once)")
    r.add_argument("--retrain", action="store_true")
    dest = dict(default=os.environ.get("KPR_BACKUP_DEST"), help="gh:owner/private-repo or dir:/path (default $KPR_BACKUP_DEST)")
    b = sub.add_parser("backup", help="snapshot tracking.db, verify, upload if changed, prune old versions")
    b.add_argument("--dest", **dest)
    b.add_argument("--reason", choices=["daily", "event", "manual"], default="manual")
    bd = sub.add_parser("backup-daemon", help="daily backups + backups after grading/collection runs")
    bd.add_argument("--dest", **dest)
    bd.add_argument("--poll-min", type=int, default=15)
    bs = sub.add_parser("backup-supervisor", help="run backup-daemon and restart it whenever it exits")
    bs.add_argument("--dest", **dest)
    bs.add_argument("--poll-min", type=int, default=15)
    rs = sub.add_parser("restore", help="download, verify and restore tracking.db from a backup")
    rs.add_argument("--dest", **dest)
    rs.add_argument("--name", default=None, help="backup file name (default newest)")
    rs.add_argument("--target", default=None)
    rs.add_argument("--force", action="store_true", help="replace a non-empty tracking.db (old copy is kept)")
    sub.add_parser("backup-status", help="last backup attempt / success / error")
    g = sub.add_parser("record-gap", help="permanently record a period with no tracking data")
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True, help="ISO time, or first-observation = earliest record in tracking.db")
    g.add_argument("--scope", required=True)
    g.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    st = load_yaml("settings.yaml")

    if a.cmd == "crawl":
        from .scraper.bo3_history import crawl
        df = crawl(a.days or st["history_days"], tiers=tuple(st.get("history_tiers", st["tiers"])),
                   keep_days=st["history_days"])
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
    elif a.cmd == "regrade":
        regrade_cmd(a.dry_run)
    elif a.cmd == "grade":
        grade_cmd()
    elif a.cmd == "map3-report":
        from .tracking import store
        rep = store.map3_report()
        print(rep.to_string(index=False) if len(rep) else "no finished BO3s with logged map odds yet")
    elif a.cmd in ("backup", "backup-daemon", "backup-supervisor", "restore"):
        from .tracking import backup as BK
        if not a.dest:
            raise SystemExit("no backup destination: pass --dest or set KPR_BACKUP_DEST")
        if a.cmd == "backup":
            try:
                name = BK.backup(a.dest, a.reason)
            except Exception as e:
                raise SystemExit(f"BACKUP FAILED: {e}")
            print(f"uploaded {name}" if name else "unchanged since last backup; nothing uploaded")
        elif a.cmd == "backup-daemon":
            BK.daemon(a.dest, a.poll_min)
        elif a.cmd == "backup-supervisor":
            BK.supervise([sys.executable, "-m", "cs2model.cli", "backup-daemon", "--dest", a.dest, "--poll-min", str(a.poll_min)])
        else:
            print(json.dumps(BK.restore_latest(a.dest, a.target, a.force, a.name), indent=2))
    elif a.cmd == "backup-status":
        from .tracking import backup as BK
        print(json.dumps(BK.status(), indent=2) or "no backups yet")
    elif a.cmd == "record-gap":
        from .tracking import store
        con = store.connect()
        end = store.first_observation(con) if a.end == "first-observation" else a.end
        if end is None:
            raise SystemExit("no records in tracking.db yet; cannot take the gap end from the first observation")
        store.record_gap(a.start, end, a.scope, a.reason, con)
        print(store.data_gaps().to_string(index=False))
    elif a.cmd == "refresh":
        while True:
            from .scraper.bo3_history import crawl
            crawl(st["history_days"], tiers=tuple(st.get("history_tiers", st["tiers"])), keep_days=st["history_days"])  # cached days are free; only new ones hit the API
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
    print("exported:", *export(slate, slips, skipped=res["unmatched"]))
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


def regrade_cmd(dry_run=False):
    from .inference.predict import Predictor
    from .scraper.bo3_history import HIST_PATH
    from .tracking import store
    pr = Predictor(pd.read_csv(HIST_PATH))
    ch = store.resettle(pr.hist, pr.resolve_player, dry_run=dry_run)
    print(f"{'would change' if dry_run else 'changed'} {len(ch)} settlements")
    if len(ch):
        print(ch.groupby(["table", "maps", "old_status", "new_status"]).size().to_string())


if __name__ == "__main__":
    main()
