"""streamlit run cs2model/dashboard/app.py"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` only adds the script's own dir

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cs2model.parlay import engine as E
from cs2model.pipeline import build_slips, diff_sig, hit_labels, line_sig, run_slate, stake_cap
from cs2model.scraper.lines import fetch_all_lines
from cs2model.tracking import store
from cs2model.training.train import MODEL_DIR

st.set_page_config(page_title="CS2 Props +EV", layout="wide", page_icon="🎯")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, .stApp, .stMarkdown, button, input {font-family: Inter, system-ui, sans-serif;}
.block-container {padding-top: 4.75rem; max-width: 1600px;}
header[data-testid="stHeader"] {background: rgba(11,15,20,0.94); backdrop-filter: blur(6px); border-bottom: 1px solid #1f2937;}
div[data-testid="stTabs"] [role="tablist"] {position: sticky; top: 3.75rem; z-index: 60; background: #0b0f14; padding-top: 6px;}
.age {display:inline-block;margin-left:8px;padding:2px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;}
.age.fresh {background:rgba(34,197,94,.15);color:#86efac;} .age.stale {background:rgba(250,204,21,.15);color:#fde047;}
.age.old {background:rgba(239,68,68,.18);color:#fca5a5;}
.summary {display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;background:#111827;border:1px solid #1f2937;
          border-radius:14px;padding:14px 18px;margin:6px 0 12px;}
.summary .k {color:#94a3b8;font-size:0.75rem;text-transform:uppercase;letter-spacing:.06em;}
.summary .v {font-size:1.5rem;font-weight:800;color:#f8fafc;} .summary .v.pos {color:#86efac;} .summary .v.neg {color:#fca5a5;}
.summary .t {grid-column:1/-1;font-weight:700;color:#e2e8f0;}
section[data-testid="stSidebar"] {background:#0d1218; border-right:1px solid #1f2937;}
div[data-testid="stMetric"] {background:#111827;border:1px solid #1f2937;border-radius:12px;padding:10px 14px;}
div[data-testid="stMetricValue"] {font-size:1.4rem;font-weight:800;}
button[data-baseweb="tab"] {font-weight:600;font-size:0.95rem;}
.hero {background:linear-gradient(120deg,#0f172a 0%,#10261b 55%,#0b0f14 100%);border:1px solid #1f2937;border-radius:16px;
       padding:18px 24px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;}
.hero h1 {font-size:1.7rem;margin:0;font-weight:800;letter-spacing:-0.02em;color:#f8fafc;}
.hero .sub {color:#94a3b8;font-size:0.85rem;margin-top:2px;}
.live {display:inline-flex;align-items:center;gap:8px;font-size:0.8rem;color:#86efac;background:rgba(34,197,94,.08);
       border:1px solid rgba(34,197,94,.25);padding:6px 12px;border-radius:999px;}
.live.off {color:#94a3b8;background:rgba(148,163,184,.08);border-color:#334155;}
.live .dot {width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 1.8s infinite;}
.live.off .dot {background:#64748b;animation:none;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(34,197,94,.6)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.cards {display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin:4px 0 18px;}
.card {background:#111827;border:1px solid #1f2937;border-left:4px solid var(--c);border-radius:12px;padding:12px 14px;}
.card .pl {font-weight:700;font-size:1.05rem;color:#f1f5f9;}
.card .mt {color:#94a3b8;font-size:0.75rem;margin-top:2px;}
.card .pk {font-size:1.3rem;font-weight:800;color:var(--c);margin:8px 0 6px;}
.badge {display:inline-block;padding:2px 8px;border-radius:999px;font-size:0.7rem;font-weight:700;margin:0 4px 2px 0;}
.b-edge {background:rgba(34,197,94,.15);color:#86efac;} .b-pp {background:rgba(168,85,247,.18);color:#d8b4fe;}
.b-ud {background:rgba(250,204,21,.14);color:#fde047;} .b-new {background:rgba(59,130,246,.2);color:#93c5fd;}
.b-tier {background:rgba(148,163,184,.15);color:#cbd5e1;}
.sect {font-size:0.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#64748b;margin:4px 0 6px;}
</style>""", unsafe_allow_html=True)

settings, payouts = E.load_yaml("settings.yaml"), E.load_yaml("payouts.yaml")

if not (MODEL_DIR / "kills_lgbm.txt").exists():
    st.title("CS2 Props +EV: first-time setup")
    st.write(f"No trained model yet. This downloads the last {settings['history_days']} days of tier 1-2 bo3.gg map stats "
             "(about 1-1.5 hours the first time, cached afterwards) and trains the models.")
    if st.button("Download data and train", type="primary"):
        from cs2model.scraper.bo3_history import crawl
        from cs2model.training.train import train
        with st.status("Setting up...", expanded=True) as stat:
            st.write("Crawling bo3.gg (progress is logged in the terminal / data folder)...")
            df = crawl(settings["history_days"], tiers=tuple(settings["tiers"]))
            st.write(f"{len(df):,} player-map rows. Training...")
            train(df)
            stat.update(label="Done", state="complete")
        st.rerun()
    st.stop()

# ---------------- sidebar
with st.sidebar:
    st.markdown("### CS2 Props +EV")
    auto = st.toggle("Auto-update on new lines", value=True)
    every = max(60, int(settings.get("auto_refresh_sec", 120)))
    forced = st.session_state.pop("force_refresh", False)
    if st.button("Refresh lines & odds", type="primary", width="stretch") or forced or "res" not in st.session_state:
        prev = st.session_state.get("res")
        with st.spinner("Pulling lines/odds and simulating matches..."):
            st.session_state.res = run_slate(prev["predictor"] if prev else None)
        st.session_state.changes = diff_sig(prev["sig"], st.session_state.res["sig"]) if prev else None
    res = st.session_state.res

    @st.fragment(run_every=every if auto else None)
    def watch():
        if not auto:
            return
        try:
            lines, _ = fetch_all_lines(("underdog", "prizepicks"))
        except Exception as e:  # noqa: BLE001  source hiccup: keep the current slate
            st.caption(f"Line check failed: {e}")
            return
        d = diff_sig(st.session_state.res["sig"], line_sig(lines))
        if d["added"] or d["moved"] or d["removed"]:
            st.toast(f"Board changed: {len(d['added'])} new · {len(d['moved'])} moved · {len(d['removed'])} pulled. Repricing...")
            st.session_state.force_refresh = True
            st.rerun(scope="app")
        st.caption(f"Checked {pd.Timestamp.now(tz='UTC'):%H:%M:%S} UTC · next check in {every // 60} min")
    watch()
    st.caption(f"Updated {res['ts']:%Y-%m-%d %H:%M} UTC · {res['n_lines']} lines · {res['n_books']} matches with odds")
    for e in res["errors"]:
        st.warning(f"Source unavailable: {e}")
    unverified = [p for p, c in payouts.items() if not c.get("verified")]
    st.subheader("Filters")
    platforms = st.multiselect("Platform", ["prizepicks", "underdog"], ["prizepicks", "underdog"])
    stats = st.multiselect("Stat", ["kills", "headshots"], ["kills", "headshots"])
    min_edge = st.slider("Min edge", 0.0, 0.15, float(settings["min_edge"]), 0.005, format="%.3f")
    only_ev = st.toggle("Only +EV (passes all rules)", value=False)
    tiers = st.multiselect("Confidence tier", ["A", "B", "C", "-"], ["A", "B", "C", "-"],
                           help="A/B/C = confidence in a +EV pick (sample size, tenure, edge, sharp odds). '-' = not a bet.")
    q = st.text_input("Search player / team")
    st.subheader("Bankroll")
    settings["bankroll"] = st.number_input("Bankroll", 10.0, 1e7, float(settings["bankroll"]), 50.0)
    settings["kelly_fraction"] = st.slider("Kelly fraction", 0.05, 1.0, float(settings["kelly_fraction"]), 0.05)
    settings["daily_exposure_pct"] = st.slider("Daily max exposure", 0.01, 0.5, float(settings["daily_exposure_pct"]), 0.01)

slate, details, pred = res["slate"], res["details"], res["predictor"]
if slate.empty:
    st.info("No CS2 kills/headshots lines matched to bo3 history right now.")
    st.stop()

view = slate[slate.source.isin(platforms) & slate.stat.isin(stats) & slate.tier.isin(tiers)]
if only_ev:
    view = view[view.plus_ev & (view.edge >= min_edge)]
if q:
    ql = q.lower()
    view = view[view.player.str.lower().str.contains(ql) | view.team.str.lower().str.contains(ql) |
                view.opponent.str.lower().str.contains(ql)]

ch = st.session_state.get("changes") or {"added": [], "moved": [], "removed": [], "old": {}}
added, moved = set(ch["added"]), set(ch["moved"])


def change_tag(r):
    k = (r.source, r.player, r.stat, tuple(r.maps), r.odds_type)
    if k in added:
        return "NEW"
    if k in moved:
        return f"{ch['old'][k]:g}→{r.line:g}"
    return ""


slate["chg"] = slate.apply(change_tag, axis=1) if len(slate) else ""
view = view.assign(chg=slate.loc[view.index, "chg"])
live_cls, live_txt = ("live", f"Live · auto-updates every {every // 60} min") if auto else ("live off", "Auto-update off")


@st.fragment(run_every=30)
def hero():
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(res["ts"])).total_seconds() / 60
    cls = "fresh" if age < 5 else "stale" if age < 15 else "old"
    age_txt = "just now" if age < 1 else f"{age:.0f} min old"
    st.markdown(f"""<div class="hero"><div><h1>CS2 Props +EV</h1>
<div class="sub">Updated {res['ts']:%b %d, %H:%M} UTC <span class="age {cls}" title="Time since lines/odds were last pulled and priced">
data {age_txt}</span></div>
<div class="sub">{res['n_lines']} lines on the board · {len(slate)} priced · {len(ch['added'])} new / {len(ch['moved'])} moved /
{len(ch['removed'])} pulled since last update</div></div>
<div class="{live_cls}"><span class="dot"></span>{live_txt}</div></div>""", unsafe_allow_html=True)


hero()
if unverified:
    st.error(f"Payouts not verified for {', '.join(unverified)}: EV uses placeholder payouts and Kelly stakes are hidden. "
             "Edit config/payouts.yaml to match your app and set verified: true.")
HELP = {"edge": "Model probability of the side minus the platform's breakeven probability for that leg.",
        "ev": "Expected return per $1 on this leg alone at the platform's implied single-leg price (P / breakeven − 1).",
        "tier": "Confidence in a +EV pick: A = large sample, settled roster, edge ≥ 6%, sharp odds; B = edge ≥ 4%; "
                "C = smaller edge or sample. '-' = not a bet.",
        "why": "One-line reason for the pick (or why it is skipped)."}

tab_slate, tab_player, tab_parlay, tab_track, tab_model = st.tabs(["Slate", "Player", "Parlays", "Tracking", "Model"])

# ---------------- slate
with tab_slate:
    ev = slate[slate.plus_ev]
    c = st.columns(5)
    c[0].metric("Lines priced", len(slate))
    c[1].metric("+EV picks", len(ev))
    c[2].metric("Confidence A", int((ev.tier == "A").sum()))
    c[3].metric("Best edge", f"{ev.edge.max():.1%}" if len(ev) else "–", help=HELP["edge"])
    c[4].metric("Skipped lines", len(res["unmatched"]), help="Pulled lines not priced (see 'Skipped lines' below for reasons)")
    top = view[view.plus_ev].sort_values("edge", ascending=False).head(8)
    if len(top):
        cards = []
        for _, r in top.iterrows():
            col = "#22c55e" if r.side == "over" else "#ef4444"
            plat = '<span class="badge b-pp">PrizePicks</span>' if r.source == "prizepicks" else '<span class="badge b-ud">Underdog</span>'
            new = f'<span class="badge b-new">{html.escape(r.chg)}</span>' if r.chg else ""
            cards.append(f"""<div class="card" style="--c:{col}"><div class="pl">{html.escape(r.player)}</div>
<div class="mt">{html.escape(r.team)} vs {html.escape(r.opponent)} · {r.maps_label}</div>
<div class="pk">{r.side.upper()} {r.line:g} {r.stat}</div>
<span class="badge b-edge" title="{HELP['edge']}">edge {r.edge:+.1%}</span><span class="badge b-tier">P {r.p_side:.0%} · mean {r.model_mean:.1f}</span>
{plat}<span class="badge b-tier" title="{HELP['tier']}">conf {r.tier}</span>{new}</div>""")
        st.markdown('<div class="sect">Top +EV plays</div><div class="cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    st.markdown('<div class="sect">Full board</div>', unsafe_allow_html=True)
    cols = ["chg", "player", "team", "opponent", "maps_label", "stat", "line", "model_mean", "p_over", "p_under", "side",
            "edge", "ev", "tier", "n_maps", "source", "why"]
    t = view.sort_values("edge", ascending=False)[cols].rename(columns={"maps_label": "maps", "model_mean": "mean"})

    def color(row):
        r = view.loc[row.name]
        if r.plus_ev:
            bg = "rgba(34,197,94,0.18)" if r.side == "over" else "rgba(239,68,68,0.18)"
        else:
            bg = "rgba(148,163,184,0.06)"
        return [f"background-color: {bg}"] * len(row)

    sty = t.style.apply(color, axis=1).format({"line": "{:.1f}", "mean": "{:.1f}", "p_over": "{:.1%}", "p_under": "{:.1%}",
                                              "edge": "{:+.1%}", "ev": "{:+.1%}"})
    st.dataframe(sty, width="stretch", height=620, hide_index=True,
                 column_config={"edge": st.column_config.Column("edge", help=HELP["edge"]),
                                "ev": st.column_config.Column("EV", help=HELP["ev"]),
                                "tier": st.column_config.Column("tier", help=HELP["tier"]),
                                "why": st.column_config.Column("why", help=HELP["why"], width="large"),
                                "chg": st.column_config.Column("chg", help="NEW = line added, a→b = line moved since last update")})
    st.caption("Green = +EV over · Red = +EV under · Gray = no qualifying edge. Edge = model P(side) − platform breakeven.")
    d1, d2 = st.columns(2)
    picks = slate[slate.plus_ev]
    d1.download_button("Export +EV picks (CSV)", picks.drop(columns=["maps"]).to_csv(index=False), "cs2_picks.csv")
    d2.download_button("Export +EV picks (JSON)", picks.drop(columns=["maps"]).to_json(orient="records", date_format="iso"),
                       "cs2_picks.json")
    if st.button("Log +EV picks to tracker"):
        store.log_picks(picks)
        st.success(f"Logged {len(picks)} picks")
    rc = res["recon"]
    with st.expander(f"Line accounting: {rc['lines']} pulled = {rc['priced']} priced + {rc['skipped']} skipped"
                     + ("" if rc["balanced"] else " (MISMATCH)")):
        a1, a2 = st.columns(2)
        a1.dataframe(pd.DataFrame({"skip reason": list(rc["by_reason"]), "lines": list(rc["by_reason"].values())}),
                     hide_index=True, width="stretch")
        a2.dataframe(pd.DataFrame(rc["per_platform"]).T.rename_axis("platform").reset_index(), hide_index=True, width="stretch")
        sk = res["unmatched"].drop_duplicates()
        st.dataframe(sk, hide_index=True, width="stretch")
        st.download_button("Export skipped lines (CSV)", sk.to_csv(index=False), "cs2_skipped.csv")

# ---------------- player
with tab_player:
    opts = view.apply(lambda r: f"{r.player} · {r.stat} {r.maps_label} {r.line} ({r.source})", axis=1)
    if len(opts):
        pick = st.selectbox("Line", opts.index, format_func=lambda i: opts[i])
        r = slate.loc[pick]
        tot = details[pick]["totals"]
        a, b = st.columns([3, 2])
        with a:
            vals, cnt = np.unique(tot, return_counts=True)
            fig = go.Figure(go.Bar(x=vals, y=cnt / cnt.sum(), marker_color=["#22c55e" if v > r.line else "#ef4444" for v in vals]))
            fig.add_vline(x=r.line, line_dash="dash", line_color="#e5e7eb", annotation_text=f"line {r.line}")
            fig.add_vline(x=r.model_mean, line_color="#60a5fa", annotation_text=f"mean {r.model_mean:.1f}")
            fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=30, b=10),
                              title=f"{r.player} {r.stat} {r.maps_label}: predictive distribution",
                              xaxis_title=r.stat, yaxis_title="probability", bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        with b:
            m = st.columns(2)
            m[0].metric("P(over)", f"{r.p_over:.1%}")
            m[1].metric("P(under)", f"{r.p_under:.1%}")
            m = st.columns(2)
            m[0].metric("Edge", f"{r.edge:+.1%}", r.side, help=HELP["edge"])
            m[1].metric("Confidence tier", r.tier, help=HELP["tier"])
            st.caption(r.reason)
            if r["flags"]:
                st.warning(r["flags"])
            st.caption(f"80% interval {r.q10:.0f}–{r.q90:.0f} · map win {r.p_map_win:.0%} · OT/map {r.p_ot:.0%} · "
                       f"map 3 {r.p_map3:.0%} · odds: {r.odds_source}")
        h = pred.hist[pred.hist.player_id == r.player_id].sort_values("date", ascending=False)
        h = h.assign(hs_pct=h.headshots / h.kills.replace(0, np.nan), kpr=h.kills / h.rounds)
        st.subheader("Last 20 maps")
        st.dataframe(h.head(20)[["date", "opponent", "map_name", "map_number", "rounds", "won", "kills", "headshots", "adr",
                                 "kast", "rating", "tier"]], hide_index=True, width="stretch")
        s1, s2 = st.columns(2)
        agg = dict(maps=("kills", "size"), kills=("kills", "mean"), hs=("headshots", "mean"), kpr=("kpr", "mean"),
                   hs_pct=("hs_pct", "mean"))
        s1.subheader("By map")
        s1.dataframe(h.groupby("map_name").agg(**agg).round(2).sort_values("maps", ascending=False), width="stretch")
        s2.subheader("By opponent")
        s2.dataframe(h.groupby("opponent").agg(**agg).round(2).sort_values("maps", ascending=False).head(15),
                     width="stretch")

# ---------------- parlays
def leg_cards(ids):
    cards = []
    for i in ids:
        r = slate.loc[i]
        col = "#22c55e" if r.side == "over" else "#ef4444"
        cards.append(f"""<div class="card" style="--c:{col}"><div class="pl">{html.escape(r.player)}</div>
<div class="mt">{html.escape(r.team)} vs {html.escape(r.opponent)} · {r.maps_label}</div>
<div class="pk">{r.side.upper()} {r.line:g} {r.stat}</div>
<span class="badge b-tier" title="Mean of the model's predictive distribution">mean {r.model_mean:.1f}</span>
<span class="badge b-tier" title="Calibrated model probability of this side (a push voids the leg)">P(side) {r.p_side:.1%}</span>
<span class="badge b-edge" title="{HELP['edge']}">edge {r.edge:+.1%}</span></div>""")
    st.markdown('<div class="sect">Legs</div><div class="cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def slip_summary(title, p_all, max_pay, ev, stake, verified):
    stake_txt = f"${stake:,.2f}" if verified else "hidden"
    pay_txt = f"{max_pay:.2f}x" + (f" (${stake * max_pay:,.2f})" if verified else "")
    st.markdown(f"""<div class="summary"><div class="t">{html.escape(title)}</div>
<div><div class="k" title="Probability every leg hits, from correlated simulations">Win prob</div><div class="v">{p_all:.1%}</div></div>
<div><div class="k" title="Top payout multiplier (all legs hit)">Payout</div><div class="v">{pay_txt}</div></div>
<div><div class="k" title="Expected return per $1 staked, including partial Flex payouts">EV</div>
<div class="v {'pos' if ev > 0 else 'neg'}">{ev:+.1%}</div></div>
<div><div class="k" title="Fractional Kelly, capped; hidden until payouts are verified">Stake</div><div class="v">{stake_txt}</div></div>
</div>""", unsafe_allow_html=True)


def slip_warnings(ids, ev=None, ev_indep=None):
    legs = slate.loc[list(ids)]
    dup = legs.player[legs.player_id.duplicated()].unique().tolist()
    if dup:
        st.error(f"Same-player legs: {', '.join(dup)}. PrizePicks/Underdog allow one pick per player per entry.")
    same = legs.groupby("match").player.apply(lambda x: sorted(set(x)))
    same = same[same.str.len() > 1]
    for m, pl in same.items():
        st.error(f"Same-match legs ({', '.join(pl)}): EV leans on the simulated correlation between them.")
    if ev is not None and ev > 0 and ev_indep < 0.5 * ev:
        st.error(f"Independent EV {ev_indep:+.1%} is less than half the correlated EV {ev:+.1%}: most of the edge comes from "
                 "the correlation estimate.")
    return bool(dup)


SLIP_COLS = {"play": "Play", "rank": "Rank", "legs": "Legs", "p_all": "Win prob", "ev": "EV", "ev_indep": "EV if independent",
             "hits": "Hits (0/1/2…)", "corr_ratio": "Correlation boost", "heavy_corr": "Heavy correlation",
             "sanity_ok": "Sanity check", "matches": "Matches", "kelly_full": "Kelly (full)", "stake": "Stake",
             "recommended": "Recommended"}
SLIP_FMT = {"Win prob": "{:.1%}", "EV": "{:+.1%}", "EV if independent": "{:+.1%}", "Kelly (full)": "{:.1%}", "Stake": "${:.2f}",
            "Correlation boost": lambda v: f"{v - 1:+.0%}", "Payout": "{:.2f}x"}
SLIP_HELP = {"Win prob": "Probability all legs hit (correlated simulation, calibrated legs).",
             "EV": "Expected return per $1 including Flex partial payouts.",
             "EV if independent": "EV if the legs were independent; compare with EV to see how much comes from correlation.",
             "Correlation boost": "P(all hit) relative to the product of leg probabilities, minus 1.",
             "Hits (0/1/2…)": "Probability of each number of legs hitting."}

with tab_parlay:
    plat = st.radio("Platform", ["prizepicks", "underdog"], horizontal=True)
    verified = bool(payouts.get(plat, {}).get("verified"))
    avail = E.plays(plat, payouts)
    mode = st.radio("Leg selection", ["Automatic (optimizer)", "Manual"], horizontal=True)
    if mode.startswith("Auto"):
        chosen = st.multiselect("Play types", list(avail), list(avail))
        with st.spinner("Searching slips on correlated simulations..."):
            slips = build_slips(slate, details, plat, settings, payouts, plays={k: avail[k] for k in chosen})
        if slips.empty:
            st.info("No +EV slips from the current +EV legs on this platform.")
        else:
            top = slips.iloc[0]
            slip_summary(f"Best slip: {top.play}", top.p_all, E.payout_vector(avail[top.play]).max(), top.ev, top.stake, verified)
            leg_cards(top.leg_ids)
            slip_warnings(top.leg_ids, top.ev, top.ev_indep)
            show = slips[list(SLIP_COLS)].rename(columns=SLIP_COLS)
            if not verified:
                show = show.drop(columns=["Kelly (full)", "Stake"])
            st.dataframe(show.style.format({k: v for k, v in SLIP_FMT.items() if k in show}), hide_index=True, width="stretch",
                         height=420, column_config={k: st.column_config.Column(k, help=v) for k, v in SLIP_HELP.items()})
            st.caption("6-leg slips must pass the sanity check (still +EV under independence with every leg 2pp worse).")
            st.subheader("EV by legs and Power vs Flex (best slip each)")
            st.bar_chart(slips.groupby("play").ev.max().sort_values())
            e1, e2, e3 = st.columns(3)
            e1.download_button("Slips CSV", slips.to_csv(index=False), "cs2_slips.csv")
            e2.download_button("Slips JSON", json.dumps(slips.to_dict("records"), default=str), "cs2_slips.json")
            if e3.button("Log recommended slips + picks"):
                p = slate[slate.plus_ev]
                store.log_picks(p)
                con = store.connect()
                idmap = {}
                for row in con.execute("SELECT id, source, player, stat, maps, side, line FROM picks"):
                    idmap[tuple(row[1:])] = row[0]
                for s in slips[slips.recommended].itertuples():
                    legs = [idmap.get((slate.loc[i, "source"], slate.loc[i, "player"], slate.loc[i, "stat"],
                                       "+".join(map(str, slate.loc[i, "maps"])), slate.loc[i, "side"], slate.loc[i, "line"]))
                            for i in s.leg_ids]
                    store.log_slip(plat, s.play, legs, s.p_all, s.ev, s.stake, s.hit_dist, con)
                st.success("Logged")
    else:
        cand = slate[slate.source == plat].sort_values("edge", ascending=False)
        labels = cand.apply(lambda r: f"{r.player} {r.side} {r.line} {r.stat} {r.maps_label} (edge {r.edge:+.1%})", axis=1)
        sel = st.multiselect("Legs (side = model's side)", cand.id.tolist(), format_func=lambda i: labels[i], max_selections=6)
        if sel:
            leg_cards(sel)
        rows = []
        if len(sel) >= 2 and not slate.loc[sel, "player_id"].duplicated().any():
            H = np.column_stack([details[i]["hit"] for i in sel])
            for name, play in avail.items():
                if play["legs"] != len(sel):
                    continue
                r = E.evaluate(H, range(len(sel)), play)
                f = E.kelly_fraction(r["hit_dist"], play)
                rows.append({"Play": name, "Win prob": r["p_all"], "Payout": E.payout_vector(play).max(), "EV": r["ev"],
                             "EV if independent": r["ev_indep"], "Correlation boost": r["corr_ratio"],
                             "Stake": min(settings["kelly_fraction"] * f, stake_cap(settings, payouts, plat)) * settings["bankroll"],
                             "Hits (0/1/2…)": hit_labels(r["hit_dist"])})
        best = max(rows, key=lambda x: x["EV"]) if rows else None
        if best:
            slip_summary(f"Best play for these legs: {best['Play']}", best["Win prob"], best["Payout"], best["EV"], best["Stake"],
                         verified)
        if sel:
            slip_warnings(sel, best["EV"] if best else None, best["EV if independent"] if best else None)
        if rows:
            t = pd.DataFrame(rows)
            if not verified:
                t = t.drop(columns=["Stake"])
            st.dataframe(t.style.format({k: v for k, v in SLIP_FMT.items() if k in t}), hide_index=True, width="stretch",
                         column_config={k: st.column_config.Column(k, help=v) for k, v in SLIP_HELP.items()})
        elif len(sel) >= 2 and not slate.loc[sel, "player_id"].duplicated().any():
            st.info(f"No {len(sel)}-leg play type configured for {plat} (4-leg plays are excluded).")

# ---------------- tracking
with tab_track:
    summ, p, s = store.performance()
    if not summ.get("picks"):
        st.info("No graded picks yet. Log picks, then run `python -m cs2model.cli grade` after matches finish.")
    else:
        c = st.columns(5)
        c[0].metric("Graded picks", summ["picks"])
        c[1].metric("Hit rate", f"{summ['hit_rate']:.1%}")
        c[2].metric("Flat ROI", f"{summ['flat_roi']:+.1%}")
        c[3].metric("Avg CLV (lines)", f"{summ['avg_clv']:+.2f}" if summ["avg_clv"] == summ["avg_clv"] else "–")
        c[4].metric("Slip profit", f"${summ['slip_profit']:.2f}")
        p["bucket"] = (p.p_model * 10).astype(int) / 10
        cal = p.groupby("bucket").agg(pred=("p_model", "mean"), obs=("win", "mean"), n=("win", "size")).reset_index()
        fig = go.Figure([go.Scatter(x=cal.pred, y=cal.obs, mode="markers+lines", name="picks"),
                         go.Scatter(x=[0.4, 0.9], y=[0.4, 0.9], mode="lines", line_dash="dash", name="perfect")])
        fig.update_layout(template="plotly_dark", height=320, title="Live calibration", xaxis_title="model P", yaxis_title="hit rate")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(p.sort_values("start_time", ascending=False), hide_index=True, width="stretch")
        if len(s):
            st.dataframe(s, hide_index=True, width="stretch")

# ---------------- model
with tab_model:
    bt = json.loads((MODEL_DIR / "backtest.json").read_text())
    st.caption(f"Time split: {bt['split_dates']}. Lines in the backtest are synthetic (naive mean rounded to .5) because "
               f"historical prop lines are not archived publicly, so ROI here is indicative only.")
    rows = []
    for model, r in bt["models"].items():
        for k, v in r.items():
            rows.append({"model": model, "target": k, "MAE": v["mae"], "pinball q50": v["pinball_0.5"],
                         "pinball q10": v["pinball_0.1"], "pinball q90": v["pinball_0.9"], "Brier": v["brier"],
                         "LogLoss": v["logloss"], "picks": v["betting"]["n_picks"], "hit": v["betting"]["hit_rate"],
                         "ROI": v["betting"]["roi"]})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    a, b = st.columns(2)
    for col, stat in ((a, "kills"), (b, "headshots")):
        cal = pd.DataFrame(bt[f"calibration_{stat}"])
        fig = go.Figure([go.Scatter(x=cal.pred, y=cal.obs, mode="markers+lines", text=cal.n, name="test"),
                         go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="perfect")])
        fig.update_layout(template="plotly_dark", height=320, title=f"Calibration ({stat}, test maps)")
        col.plotly_chart(fig, width="stretch")
    st.subheader("Edge buckets (LightGBM, test)")
    st.dataframe(pd.DataFrame([dict(target=k, **bk) for k, v in bt["models"]["lgbm"].items() for bk in v["betting"]["buckets"]]),
                 hide_index=True, width="stretch")
    st.subheader("Top features (kills, gain share)")
    st.bar_chart(pd.Series(bt["feature_importance_kills"]))
