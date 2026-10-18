"""streamlit run cs2model/dashboard/app.py"""
import base64
import html
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `streamlit run` only adds the script's own dir

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cs2model.inference import analytic as A
from cs2model.parlay import engine as E
from cs2model.pipeline import build_slips, diff_sig, hit_labels, line_sig, run_slate, stake_cap
from cs2model.inference.predict import Predictor
from cs2model.scraper.bo3_history import DATA as HDATA
from cs2model.scraper.bo3_history import HIST_PATH, player_photos, team_logos
from cs2model.scraper.lines import fetch_all_lines
from cs2model.tracking import store
from cs2model.training.train import MODEL_DIR

st.set_page_config(page_title="CS2 Props +EV", layout="wide", page_icon="◎")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root {--bg:#0a0d12;--panel:#11161d;--panel2:#161c25;--line:#232b36;--line2:#2e3846;--accent:#ff7a1a;--accent-soft:rgba(255,122,26,.12);
  --pos:#22c55e;--pos-soft:rgba(34,197,94,.13);--neg:#ef4444;--neg-soft:rgba(239,68,68,.13);--steel:#7aa2c8;--steel-soft:rgba(122,162,200,.12);
  --tx:#eef1f5;--tx2:#a3acb9;--mut:#6b7684;--r:10px;--s1:4px;--s2:8px;--s3:12px;--s4:16px;--s5:24px;
  --fd:'Barlow Condensed',Rajdhani,'Arial Narrow',sans-serif;--fb:Inter,system-ui,sans-serif;--fm:'JetBrains Mono',ui-monospace,monospace;}
html, body, .stApp, .stMarkdown, button, input, textarea, select {font-family:var(--fb);}
.stApp {background:radial-gradient(1200px 500px at 0% -10%,rgba(255,122,26,.07),transparent 60%),
  linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px) 0 0/40px 40px,
  linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px) 0 0/40px 40px,var(--bg);}
.block-container {padding-top:4.25rem;max-width:1640px;}
header[data-testid="stHeader"] {background:rgba(10,13,18,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);}
h1,h2,h3 {font-family:var(--fd)!important;letter-spacing:.02em;text-transform:uppercase;}
.logo {width:18px;height:18px;object-fit:contain;vertical-align:-4px;margin:0 4px 0 0;}
/* sidebar */
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1117,#0a0d12);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:1rem;}
section[data-testid="stSidebar"] label p {font-size:.78rem;color:var(--tx2);font-weight:500;}
.brand {display:flex;align-items:center;gap:10px;padding:2px 0 4px;}
.brand img {height:38px;width:auto;}
.brand .two {font-family:var(--fd);font-weight:700;font-size:2.1rem;line-height:1;color:var(--accent);margin-left:-2px;}
.brand-sub {font-family:var(--fd);font-weight:600;letter-spacing:.22em;font-size:.8rem;color:var(--mut);text-transform:uppercase;
  border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:6px;}
.brand-sub b {color:var(--accent);font-weight:700;}
.sb-h {display:flex;align-items:center;gap:8px;font-family:var(--fd);font-weight:700;font-size:.85rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--tx2);margin:14px 0 4px;}
.sb-h:before {content:"";width:3px;height:12px;background:var(--accent);}
.sb-h:after {content:"";flex:1;height:1px;background:var(--line);}
/* controls */
.stButton>button, .stDownloadButton>button {border-radius:6px;font-weight:600;letter-spacing:.02em;border:1px solid var(--line2);
  background:var(--panel2);color:var(--tx);transition:all .15s ease;}
.stButton>button:hover, .stDownloadButton>button:hover {border-color:var(--accent);color:var(--accent);background:var(--accent-soft);}
.stButton>button[kind="primary"] {background:var(--accent);border-color:var(--accent);color:#120800;font-family:var(--fd);
  font-size:1.02rem;letter-spacing:.1em;text-transform:uppercase;clip-path:polygon(8px 0,100% 0,100% calc(100% - 8px),calc(100% - 8px) 100%,0 100%,0 8px);}
.stButton>button[kind="primary"]:hover {background:#ff8f3d;color:#120800;}
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div, .stNumberInput input {background:var(--panel)!important;border-color:var(--line2)!important;border-radius:6px!important;}
span[data-baseweb="tag"] {background:var(--accent-soft)!important;border:1px solid rgba(255,122,26,.35);color:#ffb37a!important;border-radius:4px!important;}
div[data-testid="stAlert"] {border-radius:8px;border:1px solid var(--line2);}
div[data-testid="stExpander"] details {background:var(--panel);border:1px solid var(--line);border-radius:var(--r);}
/* tabs */
div[data-testid="stTabs"] [role="tablist"] {position:sticky;top:3.75rem;z-index:60;background:var(--bg);padding-top:6px;gap:4px;border-bottom:1px solid var(--line);}
[role="tab"] p, button[data-baseweb="tab"] p {font-family:var(--fd)!important;font-weight:600!important;font-size:1.05rem!important;letter-spacing:.12em;text-transform:uppercase;}
[role="tab"] {color:var(--mut);} [role="tab"][aria-selected="true"] {color:var(--tx)!important;}
button[data-baseweb="tab"] {font-family:var(--fd);font-weight:600;font-size:1.05rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);padding:6px 14px;}
button[data-baseweb="tab"][aria-selected="true"] {color:var(--tx);}
div[data-baseweb="tab-highlight"] {background:var(--accent)!important;height:3px;}
/* metrics (other tabs) */
div[data-testid="stMetric"] {background:var(--panel);border:1px solid var(--line);border-top:2px solid var(--accent);border-radius:8px;padding:10px 14px;}
div[data-testid="stMetricLabel"] p {font-family:var(--fd);letter-spacing:.12em;text-transform:uppercase;color:var(--tx2);font-size:.8rem;}
div[data-testid="stMetricValue"] {font-family:var(--fm);font-size:1.35rem;font-weight:700;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:var(--r);overflow:hidden;}
/* hero */
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#121821 0%,#0f141b 60%,#160f0a 100%);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:var(--r);padding:20px 26px;margin-bottom:14px;display:flex;justify-content:space-between;
  align-items:center;gap:18px;flex-wrap:wrap;clip-path:polygon(0 0,calc(100% - 22px) 0,100% 22px,100% 100%,0 100%);}
.hero:after {content:"";position:absolute;right:120px;top:-60px;width:220px;height:220px;border:1px solid rgba(255,122,26,.12);border-radius:50%;
  background:radial-gradient(circle,transparent 38%,rgba(255,122,26,.06) 39%,transparent 40%),linear-gradient(rgba(255,122,26,.1),rgba(255,122,26,.1)) center/1px 100% no-repeat,
  linear-gradient(90deg,rgba(255,122,26,.1),rgba(255,122,26,.1)) center/100% 1px no-repeat;pointer-events:none;}
.hero .eyebrow {font-family:var(--fd);font-weight:600;letter-spacing:.24em;font-size:.78rem;color:var(--accent);text-transform:uppercase;}
.hero h1 {font-family:var(--fd)!important;font-size:2.4rem;line-height:1;margin:2px 0 4px;padding:0;font-weight:700;letter-spacing:.03em;color:var(--tx);}
.hero h1 em {font-style:normal;color:var(--accent);}
.hero .sub {color:var(--tx2);font-size:.85rem;}
.hero .chips {display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;}
.chip {font-family:var(--fm);font-size:.72rem;padding:3px 9px;border-radius:4px;background:var(--panel2);border:1px solid var(--line2);color:var(--tx2);}
.chip b {color:var(--tx);font-weight:700;}
.hero .right {display:flex;flex-direction:column;align-items:flex-end;gap:8px;z-index:1;}
.live {display:inline-flex;align-items:center;gap:8px;font-family:var(--fd);font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  font-size:.8rem;color:#86efac;background:var(--pos-soft);border:1px solid rgba(34,197,94,.3);padding:5px 12px;border-radius:4px;}
.live.off {color:var(--tx2);background:var(--steel-soft);border-color:var(--line2);}
.live .dot {width:8px;height:8px;border-radius:50%;background:var(--pos);animation:pulse 1.8s infinite;}
.live.off .dot {background:var(--mut);animation:none;}
@keyframes pulse {0%{box-shadow:0 0 0 0 rgba(34,197,94,.6)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.age {display:inline-block;font-family:var(--fm);padding:3px 9px;border-radius:4px;font-size:.72rem;font-weight:700;}
.age.fresh {background:var(--pos-soft);color:#86efac;} .age.stale {background:rgba(250,204,21,.13);color:#fde047;} .age.old {background:var(--neg-soft);color:#fca5a5;}
/* kpi row */
.kpis {display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:10px;margin:2px 0 18px;}
.kpi {background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;position:relative;}
.kpi:before {content:"";position:absolute;left:0;top:10px;bottom:10px;width:2px;background:var(--line2);}
.kpi.hl:before {background:var(--accent);}
.kpi .k {font-family:var(--fd);font-weight:600;letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;color:var(--tx2);}
.kpi .v {font-family:var(--fm);font-weight:700;font-size:1.55rem;color:var(--tx);line-height:1.25;}
.kpi .v.pos {color:#4ade80;} .kpi .v.acc {color:var(--accent);}
@media (max-width:900px){.kpis{grid-template-columns:repeat(2,1fr);}}
/* section headings */
.sect {display:flex;align-items:center;gap:10px;font-family:var(--fd);font-weight:700;font-size:1.05rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--tx);margin:8px 0 10px;}
.sect:before {content:"";width:10px;height:10px;border:2px solid var(--accent);transform:rotate(45deg);}
.sect:after {content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line2),transparent);}
/* play cards */
.cards {display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px;margin:0 0 22px;}
.pc {position:relative;background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:var(--r);
  padding:14px 14px 12px;overflow:hidden;transition:border-color .15s, transform .15s;}
.pc:hover {border-color:var(--line2);transform:translateY(-1px);}
.pc:before {content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--c);}
.pc .tag {position:absolute;right:0;top:0;font-family:var(--fd);font-weight:700;letter-spacing:.14em;font-size:.78rem;padding:3px 10px 3px 16px;
  color:#0a0d12;background:var(--c);clip-path:polygon(10px 0,100% 0,100% 100%,0 100%);}
.pc .who {display:flex;align-items:center;gap:10px;}
.av {width:44px;height:44px;border-radius:8px;object-fit:cover;object-position:top;background:var(--panel2);border:1px solid var(--line2);flex:none;}
.av.ini {display:flex;align-items:center;justify-content:center;font-family:var(--fd);font-weight:700;font-size:1.2rem;color:var(--tx2);}
.pc .pl {font-family:var(--fd);font-weight:700;font-size:1.4rem;line-height:1;color:var(--tx);letter-spacing:.02em;}
.pc .mt {color:var(--tx2);font-size:.74rem;margin-top:4px;}
.pc .bet {display:flex;align-items:baseline;gap:8px;margin:12px 0 10px;}
.pc .bet .sd {font-family:var(--fd);font-weight:700;font-size:1.05rem;letter-spacing:.1em;color:var(--c);}
.pc .bet .ln {font-family:var(--fm);font-weight:700;font-size:1.7rem;color:var(--tx);line-height:1;}
.pc .bet .st {font-family:var(--fd);font-weight:600;font-size:1rem;letter-spacing:.08em;text-transform:uppercase;color:var(--tx2);}
.pc .edge {margin-left:auto;font-family:var(--fm);font-weight:700;font-size:.95rem;color:var(--accent);background:var(--accent-soft);
  border:1px solid rgba(255,122,26,.35);padding:2px 8px;border-radius:4px;}
.bars {display:grid;grid-template-columns:auto 1fr auto;gap:5px 8px;align-items:center;font-size:.7rem;color:var(--tx2);margin-bottom:10px;}
.bars .lbl {font-family:var(--fd);letter-spacing:.12em;text-transform:uppercase;font-weight:600;}
.bars .val {font-family:var(--fm);color:var(--tx);text-align:right;}
.bar {height:6px;background:#1d2530;border-radius:3px;position:relative;overflow:hidden;}
.bar i {position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--c);}
.bar u {position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--tx);opacity:.7;}
.foot {display:flex;flex-wrap:wrap;gap:5px;border-top:1px dashed var(--line);padding-top:9px;}
.badge {display:inline-block;padding:2px 8px;border-radius:4px;font-size:.68rem;font-weight:700;letter-spacing:.04em;margin:0 4px 2px 0;}
.foot .badge {margin:0;}
.b-edge {background:var(--pos-soft);color:#86efac;} .b-pp {background:rgba(168,85,247,.16);color:#d8b4fe;}
.b-ud {background:rgba(250,204,21,.12);color:#fde047;} .b-new {background:var(--steel-soft);color:#bcd3ea;}
.b-tier {background:rgba(148,163,184,.12);color:#cbd5e1;} .b-A {background:var(--accent);color:#120800;} .b-B {background:var(--accent-soft);color:#ffb37a;}
/* slip summary */
.summary {display:grid;grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;background:var(--panel);border:1px solid var(--line);
  border-top:2px solid var(--accent);border-radius:var(--r);padding:14px 18px;margin:6px 0 12px;}
.summary .k {font-family:var(--fd);color:var(--tx2);font-size:.8rem;text-transform:uppercase;letter-spacing:.14em;}
.summary .v {font-family:var(--fm);font-size:1.45rem;font-weight:700;color:var(--tx);} .summary .v.pos {color:#4ade80;} .summary .v.neg {color:#f87171;}
.summary .t {grid-column:1/-1;font-family:var(--fd);font-size:1.1rem;letter-spacing:.06em;font-weight:700;color:var(--tx);}
.phead {display:flex;align-items:center;gap:14px;margin:4px 0 10px;}
.phead .av {width:64px;height:64px;}
.phead .pl {font-family:var(--fd);font-weight:700;font-size:1.9rem;line-height:1;color:var(--tx);}
.phead .mt {color:var(--tx2);font-size:.8rem;margin-top:4px;}
/* refinement pass */
:root {--accent:#e8752a;--accent-soft:rgba(232,117,42,.10);--neu:#c3cad4;}
section[data-testid="stSidebar"] {width:272px!important;min-width:272px!important;}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:.75rem 1rem 1rem;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.55rem;}
.block-container {max-width:none!important;padding-left:2rem!important;padding-right:2rem!important;padding-top:3.75rem!important;}
span[data-baseweb="tag"] {background:var(--panel2)!important;border:1px solid var(--line2)!important;color:var(--tx)!important;}
.stButton>button[kind="primary"] {background:var(--accent);border-color:var(--accent);}
.hero {padding:16px 22px;} .hero h1 {font-size:2rem;}
.kpis {grid-template-columns:repeat(auto-fit,minmax(150px,1fr));}
.kpi .v.acc {color:var(--tx);}
.cards {grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;}
.pc {padding:14px 16px;border-radius:8px;transition:border-color .15s ease,box-shadow .15s ease;}
.pc:hover {transform:none;border-color:var(--line2);box-shadow:0 6px 18px rgba(0,0,0,.35);}
.pc:before {display:none;}
.pc .top {display:flex;align-items:center;gap:12px;}
.pc .pl {font-size:1.45rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pc .mt {white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pc .side {margin-left:auto;font-family:var(--fd);font-weight:700;letter-spacing:.1em;font-size:.85rem;padding:2px 8px;border-radius:4px;
  border:1px solid var(--line2);color:var(--neu);white-space:nowrap;}
.pc .side.o {color:#86efac;border-color:rgba(34,197,94,.35);} .pc .side.u {color:#fca5a5;border-color:rgba(239,68,68,.35);}
.pc .bet {margin:14px 0 12px;gap:10px;align-items:center;}
.pc .bet .ln {font-size:1.9rem;}
.pc .bet svg {width:20px;height:20px;color:var(--tx2);flex:none;}
.pc .edge {color:#4ade80;background:var(--pos-soft);border-color:rgba(34,197,94,.35);font-size:1.05rem;}
.pc .edge.neg {color:var(--tx2);background:var(--panel2);border-color:var(--line2);}
.prob {margin-bottom:10px;}
.prob .row {display:flex;justify-content:space-between;font-size:.72rem;color:var(--tx2);margin-bottom:5px;}
.prob .row b {font-family:var(--fm);color:var(--tx);font-weight:700;}
.prob .track {position:relative;height:6px;background:#1d2530;border-radius:3px;}
.prob .fill {position:absolute;left:0;top:0;bottom:0;border-radius:3px;}
.prob .be {position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--tx);}
.stats3 {display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}
.stats3>div {background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;}
.stats3 .k {font-family:var(--fd);letter-spacing:.12em;text-transform:uppercase;font-size:.68rem;color:var(--mut);}
.stats3 .v {font-family:var(--fm);font-weight:700;font-size:.95rem;color:var(--tx);}
.b-A {background:transparent;color:var(--accent);border:1px solid rgba(232,117,42,.5);}
.b-B, .b-tier {background:var(--panel2);color:var(--tx2);border:1px solid var(--line2);}
.b-pp, .b-ud {background:var(--panel2);border:1px solid var(--line2);}
.badge {transition:background .15s ease;}
.hrs {display:flex;gap:8px;flex-wrap:wrap;margin:0 0 4px;}
.hrc {background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 12px;display:grid;grid-template-columns:auto auto;
  gap:0 12px;font-family:var(--fm);font-size:.75rem;color:var(--tx2);min-width:120px;}
.hrc b {font-family:var(--fd);letter-spacing:.12em;color:var(--tx);grid-column:1/-1;font-size:.85rem;}
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

# ---------------- background pricing: the page renders immediately, the slate fills in when done
@st.cache_resource
def shared_predictor(hist_mtime):
    return Predictor()


@st.cache_data(ttl=3600)
def load_logos():
    f = HDATA / "team_logos.json"
    if not f.exists():
        team_logos()
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


def start_pricing(pred):
    job = {"done": False, "res": None, "err": None, "t0": time.time(), "lines": None, "shown": False}

    def work():
        try:
            job["res"] = run_slate(pred, on_lines=lambda ls: job.update(lines=ls))
        except Exception as e:  # noqa: BLE001  surfaced in the UI
            job["err"] = e
        finally:
            job["done"] = True
    threading.Thread(target=work, daemon=True).start()
    return job


@st.fragment(run_every=1.5)
def poll():
    j = st.session_state.get("job")
    if j is None or j["done"] or (j["lines"] is not None and not j["shown"]):
        if j is not None and j["lines"] is not None:
            j["shown"] = True
        st.rerun(scope="app")
    st.caption(f"Pricing… {time.time() - j['t0']:.0f}s")


LOGOS = load_logos()


@st.cache_data(ttl=3600)
def load_photos():
    f = HDATA / "player_photos.json"
    if not f.exists():
        player_photos()
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


PHOTOS = load_photos()
LOGO_SRC = "data:image/png;base64," + base64.b64encode((Path(__file__).parent / "assets" / "cs2_logo_light.png").read_bytes()).decode()
BRAND = (f'<div class="brand"><img src="{LOGO_SRC}" alt="Counter-Strike"/></div>'
         '<div class="brand-sub">Props <b>+EV</b> terminal</div>')


def avatar(pid, name):
    u = PHOTOS.get(int(pid)) if pid is not None and pid == pid else None
    if u:
        return f'<img class="av" src="{html.escape(u)}" loading="lazy"/>'
    return f'<div class="av ini">{html.escape((name or "?")[:2].upper())}</div>'


ICON = {
    "kills": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="7"/>'
             '<path d="M12 2v5M12 17v5M2 12h5M17 12h5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
    "headshots": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a6 6 0 0 0-6 6c0 2 .8 3.4 2 '
                 '4.5V17h8v-3.5c1.2-1.1 2-2.5 2-4.5a6 6 0 0 0-6-6z"/><path d="M9 21h6"/><circle cx="12" cy="9" r="2" fill="currentColor"/></svg>',
}


def play_card(r, help_edge="", help_tier=""):
    over = r.side == "over"
    plat = '<span class="badge b-pp">PrizePicks</span>' if r.source == "prizepicks" else '<span class="badge b-ud">Underdog</span>'
    tier = r.tier if r.tier in ("A", "B") else "tier"
    chg = getattr(r, "chg", "")
    new = f'<span class="badge b-new">{html.escape(chg)}</span>' if isinstance(chg, str) and chg else ""
    be = float(min(1.0, max(0.0, r.breakeven)))
    fill = "var(--pos)" if r.p_side > be else "var(--steel)"
    return f"""<div class="pc"><div class="top">{avatar(getattr(r, "player_id", None), r.player)}<div style="min-width:0">
<div class="pl">{html.escape(r.player)}</div>
<div class="mt">{logo(r.team_id)}{html.escape(r.team)} <span style="color:var(--mut)">vs</span> {logo(r.opp_id)}{html.escape(r.opponent)} · {r.maps_label}</div></div>
<span class="side {'o' if over else 'u'}">{'▲ OVER' if over else '▼ UNDER'}</span></div>
<div class="bet">{ICON.get(r.stat, "")}<span class="ln">{r.line:g}</span><span class="st">{r.stat}</span>
<span class="edge{'' if r.edge > 0 else ' neg'}" title="{help_edge}">{r.edge:+.1%}</span></div>
<div class="prob" title="Model probability of this side vs the platform breakeven probability (white tick); both on a 0–100% scale">
<div class="row"><span>Model P <b>{r.p_side:.1%}</b></span><span>Breakeven <b>{be:.1%}</b></span></div>
<div class="track"><div class="fill" style="width:{r.p_side:.1%};background:{fill}"></div><div class="be" style="left:{be:.1%}"></div></div></div>
<div class="stats3"><div title="Mean of the model's predictive distribution"><div class="k">Model mean</div><div class="v">{r.model_mean:.1f}</div></div>
<div title="Model mean minus the line"><div class="k">vs line</div><div class="v">{r.model_mean - r.line:+.1f}</div></div>
<div title="Prior maps in this player's history"><div class="k">Sample</div><div class="v">{int(r.n_maps)} maps</div></div></div>
<div class="foot">{plat}<span class="badge b-{tier}" title="{help_tier}">Conf {r.tier}</span>{new}</div></div>"""


def logo(team_id):
    u = LOGOS.get(int(team_id)) if team_id == team_id else None
    return f'<img class="logo" src="{html.escape(u)}"/>' if u else ""


# ---------------- sidebar
with st.sidebar:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.markdown('<div class="sb-h">Live feed</div>', unsafe_allow_html=True)
    auto = st.toggle("Auto-update on new lines", value=True)
    every = max(60, int(settings.get("auto_refresh_sec", 120)))
    forced = st.session_state.pop("force_refresh", False)
    job = st.session_state.get("job")
    clicked = st.button("Refresh lines & odds", type="primary", width="stretch")
    if (clicked or forced or ("res" not in st.session_state and job is None)) and not (job and not job["done"]):
        st.session_state.job = job = start_pricing(shared_predictor(HIST_PATH.stat().st_mtime))
    if job is not None and job["done"]:
        st.session_state.job = None
        if job["err"] is not None:
            st.error(f"Pricing failed: {job['err']}")
        else:
            prev = st.session_state.get("res")
            st.session_state.res = job["res"]
            st.session_state.changes = diff_sig(prev["sig"], job["res"]["sig"]) if prev else None
        job = None
    pricing = job is not None
    if pricing:
        st.info("Pricing in progress… stakes and slips are hidden until it finishes.")
        poll()
    res = st.session_state.get("res")

    @st.fragment(run_every=every if auto else None)
    def watch():
        if not auto:
            return
        try:
            lines, _ = fetch_all_lines(("underdog", "prizepicks"))
        except Exception as e:  # noqa: BLE001  source hiccup: keep the current slate
            st.caption(f"Line check failed: {e}")
            return
        if "res" not in st.session_state or st.session_state.get("job"):
            return
        d = diff_sig(st.session_state.res["sig"], line_sig(lines))
        if d["added"] or d["moved"] or d["removed"]:
            st.toast(f"Board changed: {len(d['added'])} new · {len(d['moved'])} moved · {len(d['removed'])} pulled. Repricing...")
            st.session_state.force_refresh = True
            st.rerun(scope="app")
        st.caption(f"Checked {pd.Timestamp.now(tz='UTC'):%H:%M:%S} UTC · next check in {every // 60} min")
    watch()
    if res is not None:
        st.caption(f"Updated {res['ts']:%Y-%m-%d %H:%M} UTC · {res['n_lines']} lines · {res['n_books']} matches with odds")
    for e in (res or {}).get("errors", []):
        st.warning(f"Source unavailable: {e}")
    unverified = [p for p, c in payouts.items() if not c.get("verified")]
    st.markdown('<div class="sb-h">Filters</div>', unsafe_allow_html=True)
    platforms = st.multiselect("Platform", ["prizepicks", "underdog"], ["prizepicks", "underdog"])
    stats = st.multiselect("Stat", ["kills", "headshots"], ["kills", "headshots"])
    min_edge = st.slider("Min edge", 0.0, 0.15, float(settings["min_edge"]), 0.005, format="%.3f")
    only_ev = st.toggle("Only +EV (passes all rules)", value=False)
    tiers = st.multiselect("Confidence tier", ["A", "B", "C", "-"], ["A", "B", "C", "-"],
                           help="A/B/C = confidence in a +EV pick (sample size, tenure, edge, sharp odds). '-' = not a bet.")
    q = st.text_input("Search player / team")
    st.markdown('<div class="sb-h">Bankroll</div>', unsafe_allow_html=True)
    settings["bankroll"] = st.number_input("Bankroll", 10.0, 1e7, float(settings["bankroll"]), 50.0)
    settings["kelly_fraction"] = st.slider("Kelly fraction", 0.05, 1.0, float(settings["kelly_fraction"]), 0.05)
    settings["daily_exposure_pct"] = st.slider("Daily max exposure", 0.01, 0.5, float(settings["daily_exposure_pct"]), 0.01)

if res is None:
    st.markdown('<div class="hero"><div><div class="eyebrow">Matchday board</div><h1>CS2 Props <em>+EV</em></h1>'
                '<div class="sub">Model-driven CS2 prop intelligence</div></div><div class="right"><div class="live off">'
                '<span class="dot"></span>Pricing in progress</div></div></div>', unsafe_allow_html=True)
    st.warning("Pricing in progress (first load). This is the raw board; edges, stakes and slips appear when pricing finishes.")
    raw = (st.session_state.get("job") or {}).get("lines")
    if raw is None:
        st.caption("Fetching lines…")
    else:
        st.dataframe(pd.DataFrame([{"platform": ln.source, "player": ln.player, "team": ln.team, "opponent": ln.opponent,
                                    "stat": ln.stat, "maps": "+".join(map(str, ln.maps)), "line": ln.line, "odds type": ln.odds_type}
                                   for ln in raw]), hide_index=True, width="stretch", height=600)
    st.stop()

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
    st.markdown(f"""<div class="hero"><div><div class="eyebrow">Matchday board</div><h1>CS2 Props <em>+EV</em></h1>
<div class="sub">Model-driven CS2 prop intelligence · updated {res['ts']:%b %d, %H:%M} UTC</div>
<div class="chips"><span class="chip"><b>{res['n_lines']}</b> lines on board</span><span class="chip"><b>{len(slate)}</b> priced</span>
<span class="chip"><b>{res['n_books']}</b> matches with odds</span><span class="chip" title="Since the previous pull">
<b>{len(ch['added'])}</b> new · <b>{len(ch['moved'])}</b> moved · <b>{len(ch['removed'])}</b> pulled</span></div></div>
<div class="right"><div class="{live_cls}"><span class="dot"></span>{live_txt}</div>
<span class="age {cls}" title="Time since lines/odds were last pulled and priced">DATA {age_txt.upper()}</span></div></div>""", unsafe_allow_html=True)


hero()
if unverified:
    st.error(f"Payouts not verified for {', '.join(unverified)}: EV uses placeholder payouts and Kelly stakes are hidden. "
             "Edit config/payouts.yaml to match your app and set verified: true.")
HELP = {"edge": "Model probability of the side minus the platform's breakeven probability for that leg.",
        "ev": "Expected return per $1 on this leg alone at the platform's implied single-leg price (P / breakeven − 1).",
        "tier": "Confidence in a +EV pick: A = large sample, settled roster, edge ≥ 6%, sharp odds; B = edge ≥ 4%; "
                "C = smaller edge or sample. '-' = not a bet.",
        "why": "One-line reason for the pick (or why it is skipped)."}


def nice_map(m):
    return m.replace("de_", "").title() if isinstance(m, str) else m


def series_chart(r, h):
    maps = [m for m in r.maps if m <= 3] or list(r.maps)
    g = h[h.map_number.isin(maps)]
    piv = g.pivot_table(index="match_id", columns="map_number", values=r.stat, aggfunc="sum")
    if not all(m in piv.columns for m in maps) or piv.dropna(subset=maps).empty:
        st.caption("No recent series with these maps.")
        return
    piv = piv.dropna(subset=maps).join(g.groupby("match_id").agg(date=("date", "max"), opp=("opponent", "first"))).sort_values("date")
    piv["total"] = piv[maps].sum(1)

    def chip(lbl, d):
        if not len(d):
            return f'<div class="hrc"><b>{lbl}</b><span>no maps</span></div>'
        return (f'<div class="hrc" title="{len(d)} series"><b>{lbl}</b><span>Over {(d.total > r.line).mean():.0%}</span>'
                f'<span>Avg {d.total.mean():.1f}</span></div>')
    chips = chip("L5", piv.tail(5)) + chip("L10", piv.tail(10)) + chip("L15", piv.tail(15)) + chip("H2H", piv[piv.opp == r.opponent])
    st.markdown(f'<div class="sect">{r.stat.title()} · {r.maps_label} · recent series vs line {r.line:g}</div>'
                f'<div class="hrs">{chips}</div>', unsafe_allow_html=True)
    v = piv.tail(12)
    x = [f"{pd.Timestamp(d):%m/%d}<br>{html.escape(str(o))[:12]}" for d, o in zip(v.date, v.opp)]
    over = (v.total > r.line).values
    fig = go.Figure()
    for k, m in enumerate(maps):
        cols = [("#15803d", "#22c55e", "#4ade80")[k] if o else ("#7f1d1d", "#b91c1c", "#dc2626")[k] for o in over]
        fig.add_bar(x=x, y=v[m], marker_color=cols, marker_line_width=0, text=[f"{int(val)}<br>M{m}" for val in v[m]],
                    textposition="inside", insidetextanchor="middle", textfont=dict(color="#f8fafc", size=12, family="JetBrains Mono"),
                    hovertemplate=f"Map {m}: %{{y}}<extra></extra>")
    if len(maps) > 1:
        fig.add_scatter(x=x, y=v.total, mode="text", text=[f"{int(t)}" for t in v.total], textposition="top center",
                        textfont=dict(color="#eef1f5", size=13, family="JetBrains Mono"), showlegend=False, hoverinfo="skip")
    fig.add_hline(y=r.line, line_dash="dash", line_color="#e5e7eb", annotation_text=f"line {r.line:g}", annotation_position="top left")
    fig.update_layout(barmode="stack", template="plotly_dark", height=400, margin=dict(l=10, r=10, t=24, b=10), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.18, uniformtext=dict(minsize=10, mode="hide"),
                      yaxis=dict(gridcolor="#1d2530", range=[0, max(v.total.max(), r.line) * 1.15]))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"series_{r.name}_{st.session_state.get('board_open')}")
    st.caption("Bottom segment = Map 1, next = Map 2 (Map 3 on top when included). Green = series went over the line, red = under.")


def player_detail(r, i):
    h = pred.hist[pred.hist.player_id == r.player_id].sort_values(["date", "map_number"], ascending=False)
    h = h.assign(hs_pct=h.headshots / h.kills.replace(0, np.nan), kpr=h.kills / h.rounds)
    series_chart(r, h)
    d = details.get(i, {})
    st.markdown(f'<div class="sect">Map predictions · vs {html.escape(r.opponent)}</div>', unsafe_allow_html=True)
    if "spec" in d:
        mp = pd.DataFrame(A.map_preds(d["spec"], d["pkey"]))
        bym = h.groupby("map_name").agg(maps_n=("kills", "size"), avg_k=("kills", "mean"), avg_hs=("headshots", "mean"),
                                        kpr_hist=("kpr", "mean"), hs_hist=("hs_pct", "mean"))
        mp = mp.merge(bym, left_on="map", right_index=True, how="left").sort_values("p_played", ascending=False)
        allm = {"map": "All maps", "maps_n": len(h), "avg_k": h.kills.mean(), "avg_hs": h.headshots.mean(),
                "kpr_hist": h.kpr.mean(), "hs_hist": h.hs_pct.mean()}
        w = mp.p_played / mp.p_played.sum()
        allm.update(kills=(w * mp.kills).sum(), headshots=(w * mp.headshots).sum(), kpr=(w * mp.kpr).sum())
        mp = pd.concat([mp, pd.DataFrame([allm])], ignore_index=True).assign(map=lambda x: x["map"].map(nice_map))
        mp = mp.assign(p_played=mp.p_played * 100, hs_hist=mp.hs_hist * 100, maps_n=mp.maps_n.fillna(0))
        fmt = {"p_played": "{:.0f}%", "kills": "{:.1f}", "headshots": "{:.1f}", "kpr": "{:.2f}", "maps_n": "{:.0f}",
               "avg_k": "{:.1f}", "avg_hs": "{:.1f}", "kpr_hist": "{:.2f}", "hs_hist": "{:.0f}%"}
        N = st.column_config.NumberColumn
        st.dataframe(mp[["map", "p_played", "kills", "headshots", "kpr", "maps_n", "avg_k", "avg_hs", "kpr_hist", "hs_hist"]]
                     .style.format(fmt, na_rep="–"),
                     hide_index=True, width="stretch", column_config={
                         "map": "Map", "p_played": st.column_config.ProgressColumn("P(played)", format="%.0f%%", min_value=0, max_value=100,
                                                                                    help="Chance this map is played in the series (map-pick model + P(map 3))"),
                         "kills": N("Pred kills", format="%.1f", help="Model expected kills on this map if it is played"),
                         "headshots": N("Pred HS", format="%.1f", help="Model expected headshots on this map if it is played"),
                         "kpr": N("Model KPR", format="%.2f"), "maps_n": N("Maps played", format="%d", help="Last 3 months, tier 1-2"),
                         "avg_k": N("Avg kills", format="%.1f"), "avg_hs": N("Avg HS", format="%.1f"),
                         "kpr_hist": N("KPR", format="%.2f"), "hs_hist": N("HS %", format="%.0f%%")})
        st.caption(f"Predictions use this match's odds ({r.odds_source}); win prob on map 1 {mp.p_win.dropna().iloc[0]:.0%}. "
                   "Right-hand columns are the player's actual averages on each map and across all maps.")
    else:
        st.caption("Map predictions are not available for this line.")
    st.markdown('<div class="sect">Last 5 matches</div>', unsafe_allow_html=True)
    lm = h[h.match_id.isin(h.match_id.drop_duplicates().head(5))]
    tr, orr = lm.team_rounds.astype("Int64").astype(str), lm.opp_rounds.astype("Int64").astype(str)
    lm = lm.assign(day=pd.to_datetime(lm.date).dt.strftime("%b %d"), score=tr + "–" + orr, res=np.where(lm.won == 1, "W", "L"),
                   mapn=lm.map_name.map(nice_map), hs_pct=(lm.hs_pct * 100).fillna(0))
    N = st.column_config.NumberColumn
    st.dataframe(lm[["day", "opponent", "map_number", "mapn", "score", "res", "kills", "headshots", "hs_pct", "adr", "rating"]],
                 hide_index=True, width="stretch", column_config={
                     "day": "Date", "opponent": "Opponent", "map_number": N("#", format="%d"), "mapn": "Map", "score": "Score",
                     "res": "W/L", "kills": N("K", format="%d"), "headshots": N("HS", format="%d"), "hs_pct": N("HS %", format="%.0f%%"),
                     "adr": N("ADR", format="%.0f"), "rating": N("Rating (bo3)", format="%.2f", help="bo3.gg player rating (its own scale)")})
    with st.expander("By opponent (last 3 months)"):
        st.dataframe(h.groupby("opponent").agg(maps=("kills", "size"), kills=("kills", "mean"), hs=("headshots", "mean"),
                                                kpr=("kpr", "mean"), hs_pct=("hs_pct", "mean")).round(2)
                     .sort_values("maps", ascending=False).head(15), width="stretch")


@st.dialog("Player panel", width="large")
def player_dialog(i):
    r = slate.loc[i]
    st.markdown(f'<div class="phead">{avatar(getattr(r, "player_id", None), r.player)}<div><div class="pl">{html.escape(r.player)}</div>'
                f'<div class="mt">{logo(r.team_id)}{html.escape(r.team)} vs {logo(r.opp_id)}{html.escape(r.opponent)} · '
                f'{r.stat} {r.maps_label} · line {r.line:g} · {r.side} · edge {r.edge:+.1%} · mean {r.model_mean:.1f}</div></div></div>',
                unsafe_allow_html=True)
    player_detail(r, i)


tab_slate, tab_player, tab_parlay, tab_track, tab_model = st.tabs(["Slate", "Player", "Parlays", "Tracking", "Model"])

# ---------------- slate
with tab_slate:
    ev = slate[slate.plus_ev]
    best = f"{ev.edge.max():+.1%}" if len(ev) else "–"
    kp = [("Lines priced", len(slate), "", ""), ("+EV picks", len(ev), "pos", ""), ("Confidence A", int((ev.tier == "A").sum()), "acc", HELP["tier"]),
          ("Best edge", best, "acc", HELP["edge"]), ("Skipped lines", len(res["unmatched"]), "", "Pulled lines not priced (see Line accounting)")]
    st.markdown('<div class="kpis">' + "".join(f'<div class="kpi{" hl" if c_ else ""}" title="{h}"><div class="k">{k}</div><div class="v {c_}">{v}</div></div>'
                                               for k, v, c_, h in kp) + "</div>", unsafe_allow_html=True)
    top = view[view.plus_ev].sort_values("edge", ascending=False).head(8)
    if len(top):
        st.markdown('<div class="sect">Top +EV plays</div><div class="cards">' + "".join(play_card(r, HELP["edge"], HELP["tier"]) for _, r in top.iterrows())
                    + "</div>", unsafe_allow_html=True)
    st.markdown('<div class="sect">Full board</div>', unsafe_allow_html=True)
    if pricing:
        st.warning("Repricing in progress: showing the previous slate until the new one is ready.")
    view = view.assign(logo=view.team_id.map(lambda t: LOGOS.get(int(t))))
    cols = ["chg", "logo", "player", "team", "opponent", "maps_label", "stat", "line", "model_mean", "p_over", "p_under", "side",
            "edge", "ev", "tier", "n_maps", "source", "why"]
    t = view.sort_values("edge", ascending=False)[cols].rename(columns={"maps_label": "maps", "model_mean": "mean"})
    t["side"] = t.side.map({"over": "▲ over", "under": "▼ under"}).fillna(t.side)

    def color(row):
        r = view.loc[row.name]
        if r.plus_ev:
            bg = "rgba(34,197,94,0.10)"
        else:
            bg = "rgba(0,0,0,0)"
        return [f"background-color: {bg}"] * len(row)

    sty = t.style.apply(color, axis=1).format({"line": "{:.1f}", "mean": "{:.1f}", "p_over": "{:.1%}", "p_under": "{:.1%}",
                                              "edge": "{:+.1%}", "ev": "{:+.1%}"}, na_rep="")
    sel = st.dataframe(sty, width="stretch", height=720, hide_index=True, on_select="rerun", selection_mode="single-row", key="board",
                 column_config={
                     "chg": st.column_config.Column("Chg", width=70, help="NEW = line added, a→b = line moved since last update"),
                     "logo": st.column_config.ImageColumn("", width=36),
                     "player": st.column_config.Column("Player", width=110), "team": st.column_config.Column("Team", width=140),
                     "opponent": st.column_config.Column("Opponent", width=140), "maps": st.column_config.Column("Maps", width=80),
                     "stat": st.column_config.Column("Stat", width=85), "line": st.column_config.Column("Line", width=60),
                     "mean": st.column_config.Column("Mean", width=60, help="Mean of the model's predictive distribution"),
                     "p_over": st.column_config.Column("P(over)", width=75), "p_under": st.column_config.Column("P(under)", width=80),
                     "side": st.column_config.Column("Side", width=80), "edge": st.column_config.Column("Edge", width=70, help=HELP["edge"]),
                     "ev": st.column_config.Column("EV", width=70, help=HELP["ev"]), "tier": st.column_config.Column("Conf", width=50, help=HELP["tier"]),
                     "n_maps": st.column_config.Column("Maps n", width=65, help="Prior maps in the player's history"),
                     "source": st.column_config.Column("Platform", width=90), "why": st.column_config.Column("Why", help=HELP["why"], width=420)})
    rows = sel.selection.rows if sel is not None else []
    k = t.index[rows[0]] if rows else None
    if k is None:
        st.session_state.board_open = None
    elif st.session_state.get("board_open") != k:
        st.session_state.board_open = k
        player_dialog(k)
    st.markdown('<div class="sect">Actions</div>', unsafe_allow_html=True)
    st.caption("Green rows = +EV (passes all rules) · ▲ over / ▼ under · click a header to sort, click a row to open the player panel. Edge = model P(side) − platform breakeven.")
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
        pmf = details[pick]["pmf"]
        st.markdown(f'<div class="phead">{avatar(getattr(r, "player_id", None), r.player)}<div><div class="pl">{html.escape(r.player)}</div>'
                    f'<div class="mt">{logo(r.team_id)}{html.escape(r.team)} vs {logo(r.opp_id)}{html.escape(r.opponent)} · '
                    f'{r.stat} {r.maps_label} · line {r.line:g}</div></div></div>', unsafe_allow_html=True)
        a, b = st.columns([3, 2])
        with a:
            vals = np.flatnonzero(pmf > 5e-4)
            fig = go.Figure(go.Bar(x=vals, y=pmf[vals], marker_color=["#22c55e" if v > r.line else "#ef4444" for v in vals]))
            fig.add_vline(x=r.line, line_dash="dash", line_color="#e5e7eb", annotation_text=f"line {r.line}")
            fig.add_vline(x=r.model_mean, line_color="#ff7a1a", annotation_text=f"mean {r.model_mean:.1f}")
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
        player_detail(r, pick)

# ---------------- parlays
def leg_cards(ids):
    st.markdown('<div class="sect">Legs</div><div class="cards">' + "".join(play_card(slate.loc[i], HELP["edge"], HELP["tier"]) for i in ids)
                + "</div>", unsafe_allow_html=True)


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
    if pricing:
        st.info("Slips and stakes are hidden while pricing is in progress.")
    else:
        plat = st.radio("Platform", ["prizepicks", "underdog"], horizontal=True)
        verified = bool(payouts.get(plat, {}).get("verified"))
        avail = E.plays(plat, payouts)
        mode = st.radio("Leg selection", ["Automatic (optimizer)", "Manual"], index=1, horizontal=True)
        if mode.startswith("Auto"):
            chosen = st.multiselect("Play types", list(avail), list(avail))
            with st.spinner("Searching slips on correlated simulations..."):
                # every Streamlit interaction reruns the whole script; only re-optimise when its inputs change
                key = (str(res["ts"]), plat, tuple(chosen), json.dumps([settings, payouts], sort_keys=True, default=str))
                cache = st.session_state.setdefault("slip_cache", {})
                if key not in cache:
                    cache.clear()
                    cache[key] = build_slips(slate, details, plat, settings, payouts, plays={k: avail[k] for k in chosen})
                slips = cache[key]
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
            cand = slate[(slate.source == plat) & (slate.n_sims > 0)].sort_values("edge", ascending=False)
            st.caption("Legs come from matches with at least one +EV leg (the matches simulated for slips).")
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
