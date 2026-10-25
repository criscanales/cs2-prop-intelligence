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
import streamlit.components.v1 as components

from cs2model.inference import analytic as A
from cs2model.parlay import engine as E
from cs2model.pipeline import build_slips, diff_sig, hit_labels, hit_matrix, line_sig, ml_support, prematch_odds_ok, run_slate, stake_cap
from cs2model.features.build import AWP_MIN_MAPS, AWP_PATH, attach_awp, awp_role
from cs2model.inference.predict import Predictor, line_probs
from cs2model.odds import decimal_to_american
from cs2model.dashboard import history_filters as HF
from cs2model.scraper.bo3_history import DATA as HDATA
from cs2model.scraper.bo3_history import HIST_PATH, player_photos, team_logos
from cs2model.scraper.portraits import DEFAULT as PIC_DEFAULT
from cs2model.scraper.portraits import frames as portrait_frames
from cs2model.scraper.lines import fetch_all_lines
from cs2model.tracking import store
from cs2model.training import bootstrap as BS
from cs2model.training.train import MODEL_DIR

# Narrow windows start with the sidebar collapsed, unless the user has already chosen a state (Streamlit stores it).
# The automatic collapse is not stored, so a later wide window still opens with the sidebar.
SIDEBAR_JS = """<script>(() => {
  const w = window.parent || window, ls = w.localStorage, KEY = 'stSidebarCollapsed-';
  if (w.innerWidth > 900 || w.__kprSidebarAuto || Object.keys(ls).some(k => k.startsWith(KEY))) return;
  w.__kprSidebarAuto = true;
  const go = n => {
    const d = w.document, sb = d.querySelector('section[data-testid="stSidebar"]');
    const b = d.querySelector('[data-testid="stSidebarCollapseButton"] button');
    if (!(sb && b)) { if (n > 0) setTimeout(() => go(n - 1), 250); return; }
    if (sb.getAttribute('aria-expanded') !== 'true') return;
    b.click();
    setTimeout(() => Object.keys(ls).filter(k => k.startsWith(KEY)).forEach(k => ls.removeItem(k)), 400);
  };
  go(40);
})();</script>"""
st.set_page_config(page_title="KPR Lab · CS2 Props +EV", layout="wide",
                   page_icon=str(Path(__file__).parent / "assets" / "kprlab_icon_512.png"))
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
header[data-testid="stHeader"] {background:var(--bg);border-bottom:1px solid var(--line);}
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
.av.pic {position:relative;overflow:hidden;border-color:rgba(255,122,26,.14);background-color:#121820;background-repeat:no-repeat;
  background-image:var(--pic),radial-gradient(70% 55% at 50% 30%,rgba(214,222,232,.10),transparent 70%),radial-gradient(120% 85% at 50% 28%,rgba(255,122,26,.12),transparent 62%),linear-gradient(180deg,#1f2631,#0f141b);
  background-size:var(--bs,175%) auto,100% 100%,100% 100%,100% 100%;background-position:var(--bp,50% 4%),0 0,0 0,0 0;}
.av.pic::after {content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;background:linear-gradient(180deg,transparent 66%,rgba(10,13,18,.35));}
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
.foot .when {margin-left:auto;align-self:center;min-width:0;overflow:hidden;text-overflow:ellipsis;font-family:var(--fm);font-size:.72rem;color:var(--tx2);white-space:nowrap;}
.foot .when b {color:var(--tx);font-weight:600;} .foot .when.live b, .ph-meta .live {color:var(--accent);}
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
section[data-testid="stSidebar"][aria-expanded="true"] {width:272px!important;min-width:272px!important;}
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
.ph {display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;align-items:center;padding:18px 22px;background:var(--panel);
  border:1px solid var(--line);border-radius:var(--r);margin:0 0 14px;}
.ph-id {display:flex;gap:18px;align-items:center;min-width:0;}
.ph-av .av {width:96px;height:96px;border-radius:10px;object-fit:cover;object-position:50% 12%;font-size:1.6rem;}
.ph-name {font-family:var(--fd);font-size:2.25rem;font-weight:700;line-height:1;color:var(--tx);letter-spacing:.01em;}
.ph-team {display:flex;align-items:center;gap:6px;margin-top:10px;font-size:.92rem;color:var(--tx2);flex-wrap:wrap;}
.ph-team img {width:20px;height:20px;object-fit:contain;} .ph-team b {color:var(--tx);font-weight:600;}
.ph-team .vs {color:var(--mut);margin:0 6px;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;}
.ph-meta {margin-top:8px;font-size:.75rem;color:var(--mut);font-family:var(--fm);}
.ph-meta .awp {color:var(--tx2);} .ph-meta .awp.awper {color:var(--accent);font-weight:700;} .ph-meta .awp.hyb {color:var(--tx);}
.ph-bet {border-left:1px solid var(--line);padding-left:24px;min-width:430px;}
.ph-k {font-family:var(--fd);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);}
.ph-line {display:flex;align-items:center;gap:10px;margin:6px 0 12px;}
.ph-side {font-family:var(--fd);font-weight:700;letter-spacing:.08em;font-size:.82rem;padding:2px 8px;border-radius:4px;border:1px solid var(--line2);}
.ph-side.over {color:#86efac;border-color:rgba(34,197,94,.4);} .ph-side.under {color:#fca5a5;border-color:rgba(239,68,68,.4);}
.ph-line .ln {font-family:var(--fm);font-size:2rem;font-weight:700;color:var(--tx);line-height:1;}
.ph-line .ic svg {width:18px;height:18px;color:var(--tx2);display:block;}
.ph-line .st {font-family:var(--fd);text-transform:uppercase;letter-spacing:.1em;color:var(--tx2);}
.ph-line .scope {font-family:var(--fm);font-size:.75rem;padding:2px 9px;border-radius:999px;background:rgba(232,117,42,.1);color:#e8752a;
  border:1px solid rgba(232,117,42,.35);}
.ph-nums {display:grid;grid-template-columns:repeat(4,auto);gap:26px;} .ph-nums .v {font-family:var(--fm);font-size:1.15rem;color:var(--tx);margin-top:3px;}
.ph-nums .v.pos {color:#4ade80;}
.ss-top {font-size:.8rem;color:var(--tx2);margin:-2px 0 10px;}
.ssr {display:grid;grid-template-columns:repeat(4,minmax(0,1fr)) minmax(0,1.1fr);gap:10px;margin-bottom:12px;}
.ss {background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;transition:border-color .15s;}
.ss:hover {border-color:var(--line2);}
.ss-h {display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;}
.ss-h b {font-family:var(--fd);letter-spacing:.14em;font-size:.85rem;color:var(--tx);} .ss-h span {font-family:var(--fm);font-size:.7rem;color:var(--mut);}
.ss-r {display:grid;grid-template-columns:62px auto 1fr;gap:8px;align-items:baseline;font-family:var(--fm);}
.ss-r .k {font-size:.7rem;color:var(--mut);font-family:Inter,sans-serif;} .ss-r .v {font-size:1.05rem;color:var(--tx);}
.ss-r .f {font-size:.72rem;color:var(--tx2);text-align:right;white-space:nowrap;} .ss-r .f.pos {color:#4ade80;} .ss-r .f.neg {color:#a3acb9;}
.ss-m {border-color:rgba(232,117,42,.45);background:linear-gradient(180deg,rgba(232,117,42,.07),var(--panel));}
.nr {border-bottom:1px dotted var(--mut);cursor:help;}
.ss-na {font-size:.78rem;color:var(--mut);padding:6px 0;}
.ss.ss-low {border-style:dashed;} .ss.ss-low .ss-r .v {color:var(--tx2);} .ss.ss-low .ss-r .f.pos {color:var(--tx2);}
.ss-w {font-size:.68rem;color:#fbbf24;margin:-2px 0 6px;letter-spacing:.02em;}
.lg {display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:.75rem;color:var(--tx2);margin:2px 0 -20px;position:relative;z-index:1;}
.lg span {display:inline-flex;align-items:center;gap:5px;} .lg i {display:inline-block;width:10px;height:10px;border-radius:2px;}
.lg i.dash {width:18px;height:0;border-top:2px dashed #cbd5e1;border-radius:0;} .lg .mut {color:var(--mut);margin-left:auto;}
.wi {font-size:.8rem;color:var(--tx2);border:1px dashed rgba(245,165,36,.6);border-radius:8px;padding:6px 10px;} .wi b {color:#fbbf24;}
.ml {background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:14px;margin:10px 0 14px;}
.ml-h {display:flex;justify-content:space-between;align-items:flex-start;gap:8px;} .ml-h b {font-size:.95rem;} .ml-h .mut {color:var(--mut);font-size:.75rem;}
.ml-bo {font-size:.7rem;color:var(--tx2);border:1px solid var(--line2);border-radius:999px;padding:2px 9px;white-space:nowrap;}
.ml-b {display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;}
.ml-t {display:grid;grid-template-columns:1fr auto 1fr;align-items:center;text-align:center;margin:14px 0 10px;}
.ml-t .logo {width:44px;height:44px;margin:0 auto 6px;display:block;} .ml-t b {display:block;font-size:.85rem;} .ml-t .vs {color:var(--mut);font-weight:700;}
.ml-o {border:1px solid var(--line2);border-radius:8px;padding:10px 12px;} .ml-na {color:var(--tx2);text-align:center;padding:10px 0 4px;font-size:.85rem;}
.ml-k {display:flex;justify-content:space-between;align-items:center;font-size:.7rem;letter-spacing:.06em;color:var(--tx2);text-transform:uppercase;}
.ml-p {display:flex;justify-content:space-between;margin-top:8px;font-family:'JetBrains Mono',monospace;}
.ml-p div:last-child {text-align:right;} .ml-p b {font-size:1.15rem;} .ml-p span {display:block;color:var(--tx2);font-size:.8rem;}
.ml-bar {display:flex;position:relative;height:7px;border-radius:999px;overflow:hidden;margin:8px 0 4px;gap:2px;} .ml-bar u {position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.45);} .ml-f {font-size:.7rem;color:var(--mut);text-align:center;}
@media (max-width:1180px) {.ph {grid-template-columns:1fr;} .ph-bet {border-left:0;padding-left:0;min-width:0;} .ssr {grid-template-columns:repeat(2,1fr);}}
.ph-q {margin-top:10px;font-size:.75rem;color:var(--tx2);border-left:2px solid var(--line2);padding-left:8px;max-width:520px;}
.ph-q.ok {color:#86efac;border-left-color:rgba(34,197,94,.6);}
.ph-q.t3x {color:var(--tx);border-left-color:var(--steel);background:var(--steel-soft);padding:6px 8px;}
.pp-tw {overflow-x:auto;border:1px solid var(--line);border-radius:var(--r);background:var(--panel);}
.pp-tbl {width:100%;border-collapse:collapse;font-size:.85rem;min-width:860px;}
.pp-tbl.sm {min-width:620px;font-size:.82rem;}
.pp-tbl th {font-family:var(--fd);font-weight:600;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--tx2);
  text-align:right;padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--panel);}
.pp-tbl td {font-family:var(--fm);text-align:right;padding:8px 12px;border-bottom:1px solid rgba(35,43,54,.7);color:var(--tx);
  white-space:nowrap;font-variant-numeric:tabular-nums;}
.pp-tbl tbody tr:last-child td {border-bottom:0;}
.pp-tbl .l {text-align:left;} .pp-tbl .c {text-align:center;}
.pp-tbl .g-m, .pp-tbl .g-h {border-left:1px solid var(--line2);}
.pp-tbl tr.grp th {font-size:.68rem;letter-spacing:.16em;padding:9px 12px 5px;text-align:center;border-bottom:2px solid transparent;}
.pp-tbl tr.grp th.g-m {color:#e8752a;border-bottom-color:rgba(232,117,42,.55);}
.pp-tbl tr.grp th.g-h {color:var(--steel);border-bottom-color:rgba(122,162,200,.5);}
.pp-tbl tbody tr {transition:background .12s;} .pp-tbl tbody tr:hover td {background:rgba(255,255,255,.025);}
.pp-tbl .mapc {font-family:Inter,sans-serif;font-weight:600;}
.pp-tbl tr.hl td.mapc {box-shadow:inset 3px 0 0 #e8752a;}
.pp-tbl tr.tot td {background:var(--panel2);border-top:1px solid var(--line2);font-weight:600;}
.pp-lk {font-family:var(--fd);font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:#e8752a;margin-left:8px;
  border:1px solid rgba(232,117,42,.4);border-radius:3px;padding:0 4px;font-weight:600;vertical-align:1px;}
.pp-pb {display:flex;align-items:center;gap:8px;} .pp-pbt {width:120px;height:6px;border-radius:3px;background:#1d2530;overflow:hidden;}
.pp-pbt i {display:block;height:100%;background:#5f86ad;border-radius:3px;} .pp-pb.hl .pp-pbt i {background:#e8752a;}
.pp-pb span {min-width:36px;text-align:right;}
.pp-tip {border-bottom:1px dotted var(--mut);cursor:help;} .pp-tip:focus-visible {outline:2px solid #e8752a;outline-offset:2px;}
.pp-na {color:var(--mut);cursor:help;}
.pp-dl {font-style:normal;font-size:.6rem;margin-left:4px;vertical-align:1px;} .pp-dl.up {color:rgba(74,222,128,.8);} .pp-dl.dn {color:rgba(248,113,113,.8);}
.pp-wl {display:inline-block;min-width:20px;text-align:center;font-family:var(--fd);font-weight:700;font-size:.72rem;padding:1px 5px;border-radius:3px;}
.pp-wl.w {color:#86efac;background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);}
.pp-wl.l {color:#fca5a5;background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);}
.pp-wl.t {color:var(--tx2);border:1px solid var(--line2);}
.pp-note {font-size:.75rem;color:var(--mut);margin:6px 0 10px;}
.pp-empty {border:1px dashed var(--line2);border-radius:8px;padding:14px 16px;color:var(--tx2);font-size:.85rem;background:var(--panel);}
.pp-mhh, .pp-mh > summary {display:grid;grid-template-columns:14px 62px minmax(120px,1fr) 78px 52px 56px 56px 64px;gap:12px;align-items:center;}
.pp-mhh {padding:0 14px 6px;font-family:var(--fd);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);}
.pp-mhh .r, .pp-mh .r {text-align:right;}
.pp-mh {border:1px solid var(--line);border-radius:8px;background:var(--panel);margin-bottom:8px;overflow:hidden;}
.pp-mh > summary {list-style:none;cursor:pointer;padding:10px 14px;font-family:var(--fm);font-size:.82rem;color:var(--tx2);transition:background .12s;}
.pp-mh > summary::-webkit-details-marker {display:none;}
.pp-mh > summary::before {content:"▸";color:var(--mut);transition:transform .15s;display:inline-block;}
.pp-mh[open] > summary::before {transform:rotate(90deg);}
.pp-mh > summary:hover {background:rgba(255,255,255,.025);}
.pp-mh > summary:focus-visible {outline:2px solid #e8752a;outline-offset:-2px;}
.pp-mh > summary.one {grid-template-columns:14px 1fr auto;}
.pp-mh .so {font-family:Inter,sans-serif;font-weight:600;color:var(--tx);font-size:.9rem;} .pp-mh summary b {color:var(--tx);font-weight:600;}
.pp-mh .pp-tw {border:0;border-top:1px solid var(--line);border-radius:0;}
div[role="dialog"] {overflow-x:hidden;}
@media (max-width:900px) {.pp-mhh, .pp-mh > summary {grid-template-columns:14px 56px 1fr 74px;} .pp-mhh .hm, .pp-mh .hm {display:none;}
  .ph-nums {grid-template-columns:repeat(2,auto);gap:12px 22px;} .ph-name {font-size:1.7rem;} .ph-av .av {width:72px;height:72px;}}
.pc .av {width:58px;height:58px;border-radius:9px;object-position:50% 12%;}
.pc .av.ini {font-size:1.45rem;}
.pc .top {gap:14px;} .pc .top > div, .pc .who > div {min-width:0;}
.pc .bet {margin:10px 0 10px;}
@media (max-width:640px) {.pc .av {width:50px;height:50px;} .ssr {grid-template-columns:1fr;} .ss {min-width:0;}}
.pp-df {color:var(--tx2);}
.pp-mcs {display:flex;flex-wrap:wrap;gap:10px 28px;align-items:center;margin:4px 0 10px;font-family:var(--fm);font-size:.85rem;color:var(--tx);}
.pp-mcs > div {display:flex;align-items:center;gap:10px;} .pp-mcs .k {font-family:var(--fd);font-size:.68rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--mut);} .pp-mcs b {font-weight:600;}
/* narrow windows: tighter padding, stacked distribution cards, rows that wrap instead of squeezing labels */
@media (max-width:900px) {.block-container {padding-left:1.25rem!important;padding-right:1.25rem!important;}}
.st-key-pdist {container-type:inline-size;}
@container (max-width:860px) {
  .st-key-pdist > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
  .st-key-pdist > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100%!important;width:100%!important;min-width:100%!important;}
  .st-key-pcards {flex-direction:row!important;flex-wrap:wrap;}
  .st-key-pcards > [data-testid="stLayoutWrapper"] {flex:1 1 300px;min-width:0;}}
.st-key-pcards [data-testid="stMetricLabel"] *, .st-key-pcards [data-testid="stMetricValue"] * {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
[class*="st-key-fitrow"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
[class*="st-key-fitrow"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {min-width:max-content!important;}
[class*="st-key-fitrow"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {min-width:min(240px,100%)!important;}
.st-key-kprjs, [data-testid="stLayoutWrapper"]:has(> .st-key-kprjs) {display:none;}
</style>""", unsafe_allow_html=True)
with st.container(key="kprjs"):
    st.html(SIDEBAR_JS, unsafe_allow_javascript=True)

settings, payouts = E.load_yaml("settings.yaml"), E.load_yaml("payouts.yaml")

def bundle_import_ui(key):
    from cs2model import bundle as BU
    st.caption("Import a bundle you exported from CS2 Props on another computer (Model tab, Data bundle). Its history is merged "
               "with yours without duplicates, then only missing or newer matches are fetched from bo3.gg. "
               "Only import bundles you made yourself.")
    up = st.file_uploader("Bundle (.zip)", type=["zip"], key=f"bu_up_{key}")
    have = (MODEL_DIR / "artifacts.json").exists()
    rep_ = st.checkbox("Replace my models with the bundle's", value=False, key=f"bu_rep_{key}", disabled=not have,
                       help="Off: your current models are kept and only the history is merged. On: your models are backed up, "
                            "then replaced. With no local models the bundle's are always installed.")
    if up is not None and st.button("Import bundle", key=f"bu_go_{key}", type="primary"):
        try:
            with st.spinner("Checking and importing..."):
                r_ = BU.import_bundle(up.getvalue(), replace_models=rep_)
        except BU.BundleError as e:
            st.error(f"Import failed, nothing was changed: {e}")
            return
        except Exception as e:  # noqa: BLE001
            st.error(f"Import failed: {e}")
            return
        m_ = (r_["bundle"].get("model") or {})
        st.success(f"Imported {r_['rows_in_bundle']:,} player-map rows: {r_['rows_added']:,} new, "
                   f"{r_['duplicates_skipped']:,} already here. History now {r_['history_from'][:10]} to {r_['history_to'][:10]}"
                   f"{'' if r_['history_complete'] else ' (partial: the rest downloads in the background)'}.")
        if r_["models_installed"]:
            st.success(f"Models installed (trained through {str(m_.get('trained_through'))[:10]}"
                       + (f"; your previous models were backed up to {r_['model_backup']}" if r_["model_backup"] else "") + ").")
        else:
            st.info(f"Models not installed: {r_['models_skipped_reason']}.")
        st.session_state.pull_after_import = True
        if not have and r_["models_installed"]:
            st.button("Open the dashboard", key=f"bu_open_{key}")


if not (MODEL_DIR / "artifacts.json").exists():
    st.title("CS2 Props +EV: first-time setup")
    qd = int(settings.get("quickstart_days", 45))
    st.write(f"No trained model yet. **Quick start** downloads the last {qd} days of tier 1-3 bo3.gg map stats (about 1.5-2 hours), "
             f"trains the models and opens the dashboard. The rest of the {settings['history_days']}-day history then downloads in "
             "the background (about 4 more hours) at the same request rate while you use the app, and the app says "
             "\"partial history\" until it finishes. If anything is interrupted (app closed, computer asleep), click the button "
             "again or reopen the app: everything already downloaded is reused.")
    if st.button("Quick start: download and train", type="primary"):
        with st.status("Setting up...", expanded=True) as stat:
            st.write(f"Downloading the last {qd} days of bo3.gg map stats...")
            bar = st.progress(0.0, text="Starting...")
            df = BS.quick_start(settings, progress=lambda d, n, r: bar.progress(
                d / n, text=(f"Day {d} of {n} · {r:,} player-map rows" if d < n else f"Downloaded {r:,} player-map rows. Training...")))
            st.write(f"{len(df):,} player-map rows. Models trained.")
            stat.update(label="Done", state="complete")
        st.rerun()
    with st.expander("Moving from another computer? Import a data bundle instead"):
        bundle_import_ui("setup")
    st.stop()

# ---------------- background pricing: the page renders immediately, the slate fills in when done
@st.cache_resource
def shared_predictor(hist_mtime, model_mtime=0):
    return Predictor()


def model_mtime():
    f = MODEL_DIR / "artifacts.json"
    return f.stat().st_mtime if f.exists() else 0


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


@st.cache_resource
def results_updater():
    return {"running": False, "err": None, "added": None, "t": None, "training": False}


def start_results_update(u):
    u.update(running=True, err=None)

    def work():
        try:
            from cs2model.scraper.bo3_history import crawl, read_history, save_history
            old_df = read_history() if HIST_PATH.exists() else None
            old = 0 if old_df is None else len(old_df)
            df = crawl(settings["history_days"], tiers=tuple(settings.get("history_tiers", settings["tiers"])), write=False,
                       known=old_df)
            if len(df) < 0.9 * old:
                raise RuntimeError(f"only {len(df):,} rows pulled vs {old:,} on disk; kept the current history")
            save_history(df)
            u["added"] = len(df) - old
            pr = Predictor()
            store.grade(pr.hist, pr.resolve_player)
            if BS.model_history().get("quick_start"):
                last = ((BS.read_coverage() or {}).get("promotion") or {}).get("t") or 0
                if time.time() - last >= 86400:  # full-history candidate vs quick-start model, at most daily
                    u["training"] = True
                    BS.gated_retrain(df, settings)
            elif time.time() - model_mtime() >= float(settings.get("retrain_days", 7)) * 86400:
                from cs2model.training.train import train
                u["training"] = True
                train()
        except Exception as e:  # noqa: BLE001  surfaced in the sidebar
            u["err"] = str(e)
        finally:
            u.update(running=False, training=False, t=time.time())
    threading.Thread(target=work, daemon=True).start()


@st.cache_resource
def backfill_job():
    return {"running": False, "err": None, "t_err": 0.0, "phase": None, "day": 0, "n": 0, "rows": 0, "result": None}


def start_backfill(b):
    b.update(running=True, err=None, phase="downloading", day=0)

    def work():
        try:
            b["result"] = BS.backfill(settings, b)
        except Exception as e:  # noqa: BLE001  surfaced in the sidebar, retried in 10 min
            b.update(err=str(e), t_err=time.time())
        finally:
            b.update(running=False, phase=None)
    threading.Thread(target=work, daemon=True).start()


@st.cache_data(show_spinner=False)
def hist_span(_mtime):
    return BS.history_span()


def backfill_status(u):
    """Starts/resumes the background backfill while history is partial; returns True while it is partial."""
    b = backfill_job()
    if BS.is_complete():
        if b["running"]:
            st.caption("Full history downloaded. Training and checking the full-history model…")
            return True
        pr = (BS.read_coverage() or {}).get("promotion") or {}
        if BS.model_history().get("quick_start") and pr and not pr.get("promote"):
            st.caption(f"Still using the quick-start model: {pr.get('reason')}. Rechecked daily.")
        return False
    if not b["running"] and not u["running"] and time.time() - b["t_err"] >= 600:
        start_backfill(b)
    sp = hist_span(HIST_PATH.stat().st_mtime if HIST_PATH.exists() else 0)
    st.caption(f"Partial history: {sp['days']} of {settings['history_days']} days")
    if b["phase"] == "downloading" and b["n"]:
        st.progress(b["day"] / b["n"], text=f"Backfill day {b['day']} of {b['n']} · {b['rows']:,} rows")
    elif b["phase"] == "training":
        st.caption("Full history downloaded. Training and checking the full-history model…")
    if b["err"]:
        st.caption(f"Backfill paused: {b['err']} (retrying in 10 min)")
    return True


@st.fragment(run_every=60)
def results_watch(on, rmin):
    u = results_updater()
    partial = backfill_status(u)
    if st.session_state.pop("pull_after_import", False) and not partial and not u["running"]:
        start_results_update(u)
    mt = HIST_PATH.stat().st_mtime if HIST_PATH.exists() else 0
    mm = model_mtime()
    seen = st.session_state.setdefault("hist_seen", mt)
    seen_m = st.session_state.setdefault("model_seen", mm)
    if (mt != seen or mm != seen_m) and not st.session_state.get("job") and not u["running"]:
        st.session_state.hist_seen, st.session_state.model_seen = mt, mm
        st.toast("Retrained models loaded. Repricing..." if mm != seen_m else "New match results loaded. Repricing...")
        st.session_state.force_refresh = True
        st.rerun(scope="app")
    if on and not partial and not u["running"] and time.time() - max(mt, u["t"] or 0) >= rmin * 60:
        start_results_update(u)
    if u["training"]:
        st.caption("Weekly retrain running in the background…")
    elif u["running"]:
        st.caption("Pulling new results from bo3.gg…")
    elif partial:
        pass
    else:
        nxt = max(0, rmin * 60 - (time.time() - max(mt, u["t"] or 0))) / 60
        st.caption(f"Results updated {pd.Timestamp(mt, unit='s', tz='UTC'):%H:%M} UTC"
                   + (f" · {u['added']:+,} rows" if u["added"] is not None else "")
                   + (f" · next pull in {nxt:.0f} min" if on else " · auto-pull off"))
    if mm:
        due = max(0.0, float(settings.get("retrain_days", 7)) - (time.time() - mm) / 86400)
        st.caption(f"Models trained {pd.Timestamp(mm, unit='s', tz='UTC'):%d %b}"
                   + (f" · retrain in {due:.0f}d" if due >= 1 else " · retrain due with the next results pull" if on else " · retrain due (turn on auto-pull)"))
    if u["err"]:
        st.caption(f"Results pull failed: {u['err']}")


LOGOS = load_logos()


@st.cache_data(ttl=3600)
def load_photos():
    f = HDATA / "player_photos.json"
    if not f.exists() or f.stat().st_size < 3:
        player_photos()
    return {int(k): v for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


PHOTOS = load_photos()


@st.cache_resource
def _frames_box():
    return {"f": {}, "busy": False, "n": -1}


def load_frames():
    """Per-photo face crops, computed once in the background (until then the default crop is used)."""
    b = _frames_box()
    if not b["busy"] and b["n"] != len(PHOTOS):
        b["busy"], b["n"] = True, len(PHOTOS)
        photos = dict(PHOTOS)

        def work():
            try:
                b["f"] = portrait_frames(photos)
            except Exception:  # noqa: BLE001  portraits just keep the default crop
                pass
            finally:
                b["busy"] = False
        threading.Thread(target=work, daemon=True).start()
    return b["f"]


FRAMES = load_frames()
LOGO_SRC = "data:image/png;base64," + base64.b64encode((Path(__file__).parent / "assets" / "kprlab_wordmark.png").read_bytes()).decode()
BRAND = (f'<div class="brand"><img src="{LOGO_SRC}" alt="KPR Lab"/></div>'
         '<div class="brand-sub">Props <b>+EV</b> terminal</div>')


def avatar(pid, name):
    u = PHOTOS.get(int(pid)) if pid is not None and pid == pid else None
    if u:
        bs, bx, by = FRAMES.get(int(pid), PIC_DEFAULT)
        return (f'<div class="av pic" role="img" aria-label="{html.escape(name or "")}" '
                f'style="--pic:url(&quot;{html.escape(u)}&quot;);--bs:{bs}%;--bp:{bx}% {by}%"></div>')
    return f'<div class="av ini">{html.escape((name or "?")[:2].upper())}</div>'


ICON = {
    "kills": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="7"/>'
             '<path d="M12 2v5M12 17v5M2 12h5M17 12h5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
    "headshots": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3a6 6 0 0 0-6 6c0 2 .8 3.4 2 '
                 '4.5V17h8v-3.5c1.2-1.1 2-2.5 2-4.5a6 6 0 0 0-6-6z"/><path d="M9 21h6"/><circle cx="12" cy="9" r="2" fill="currentColor"/></svg>',
}


TZ, TZ_LABEL = "America/New_York", "ET"


def start_info(t):
    """('Today', '6:45 AM ET', 'in 2h 05m' | 'started' | '') for a match start time; empty strings when unknown."""
    t = pd.to_datetime(t, utc=True, errors="coerce")
    if pd.isna(t):
        return "", "", ""
    t, now = t.tz_convert(TZ), pd.Timestamp.now(tz=TZ)
    d = (t.normalize() - now.normalize()).days
    day = "Today" if d == 0 else "Tomorrow" if d == 1 else "Yesterday" if d == -1 else f"{t:%a %b} {t.day}"
    m = int((t - now).total_seconds() // 60)
    rel = "started" if m <= 0 else f"in {m}m" if m < 60 else f"in {m // 60}h {m % 60:02d}m" if m < 24 * 60 else ""
    return day, f"{t.strftime('%I:%M %p').lstrip('0')} {TZ_LABEL}", rel


def start_meta(r):
    day, clock, rel = start_info(getattr(r, "start_time", None))
    if not day:
        return ""
    return f'<br>Starts {day} {clock}' + (f' · <span class="{"live" if rel == "started" else ""}">{rel}</span>' if rel else "")


T3X_HELP = ("Model probability minus breakeven on a tier-3 line. The tier-3 calibration for this market is not validated, "
            "so a positive difference does not qualify the pick (no confidence rating, never +EV).")


def t3_experimental(r):
    return bool(getattr(r, "t3", False)) and not bool(getattr(r, "t3_validated", False))


def play_card(r, help_edge="", help_tier=""):
    over = r.side == "over"
    t3x = t3_experimental(r)
    plat = '<span class="badge b-pp">PrizePicks</span>' if r.source == "prizepicks" else '<span class="badge b-ud">Underdog</span>'
    ot = getattr(r, "odds_type", "standard")
    if isinstance(ot, str) and ot != "standard":
        plat += (f'<span class="badge b-new" title="PrizePicks {ot} line (over only). Priced at the standard payout unless '
                 f'you set one in payouts.yaml; demons pay more, so the edge shown is a lower bound.">{ot.title()}</span>')
    tier = r.tier if r.tier in ("A", "B") else "tier"
    if bool(getattr(r, "t3", False)):
        plat += ('<span class="badge b-new" title="Tier-3 match: priced with the tier-3 calibration, needs a larger edge, '
                 'confidence capped">T3</span>' if getattr(r, "t3_validated", False) else
                 '<span class="badge b-new" title="Tier-3 match: experimental model-only estimate. The tier-3 calibration '
                 'for this market is not validated, so it is never a +EV pick.">T3 experimental</span>')
    chg = getattr(r, "chg", "")
    new = f'<span class="badge b-new">{html.escape(chg)}</span>' if isinstance(chg, str) and chg else ""
    be = float(min(1.0, max(0.0, r.breakeven)))
    fill = "var(--pos)" if r.p_side > be and not t3x else "var(--steel)"
    day, clock, rel = start_info(getattr(r, "start_time", None))
    short = clock if rel and rel != "started" else f"{day} {clock}"
    rel_s = f" · <b>{rel.removeprefix('in ')}</b>" if rel else ""
    live = " live" if rel == "started" else ""
    when = f'<span class="when{live}" title="Match start: {day} {clock}">{short}{rel_s}</span>' if day else ""
    return f"""<div class="pc"><div class="top">{avatar(getattr(r, "player_id", None), r.player)}<div style="min-width:0">
<div class="pl">{html.escape(r.player)}</div>
<div class="mt">{logo(r.team_id)}{html.escape(r.team)} <span style="color:var(--mut)">vs</span> {logo(r.opp_id)}{html.escape(r.opponent)} · {r.maps_label}</div></div>
<span class="side {'o' if over else 'u'}">{'▲ OVER' if over else '▼ UNDER'}</span></div>
<div class="bet">{ICON.get(r.stat, "")}<span class="ln">{r.line:g}</span><span class="st">{r.stat}</span>
{f'<span class="edge neg" title="{T3X_HELP}">Unvalidated model difference {r.edge:+.1%}</span>' if t3x else
   f'<span class="edge{"" if r.edge > 0 else " neg"}" title="{help_edge}">{r.edge:+.1%}</span>'}</div>
<div class="prob" title="Model probability of this side vs the platform breakeven probability (white tick); both on a 0–100% scale">
<div class="row"><span>Model P <b>{r.p_side:.1%}</b></span><span>Breakeven <b>{be:.1%}</b></span></div>
<div class="track"><div class="fill" style="width:{r.p_side:.1%};background:{fill}"></div><div class="be" style="left:{be:.1%}"></div></div></div>
<div class="stats3"><div title="Mean of the model's predictive distribution"><div class="k">Model mean</div><div class="v">{r.model_mean:.1f}</div></div>
<div title="Model mean minus the line"><div class="k">vs line</div><div class="v">{r.model_mean - r.line:+.1f}</div></div>
<div title="Prior maps in this player's history"><div class="k">Sample</div><div class="v">{int(r.n_maps)} maps</div></div></div>
<div class="foot">{plat}<span class="badge b-{tier}" title="{help_tier}">Conf {r.tier}</span>{new}{when}</div></div>"""


def logo(team_id):
    u = LOGOS.get(int(team_id)) if team_id == team_id else None
    return f'<img class="logo" src="{html.escape(u)}"/>' if u else ""


# ---------------- sidebar
with st.sidebar:
    st.markdown(BRAND, unsafe_allow_html=True)
    st.markdown('<div class="sb-h">Live feed</div>', unsafe_allow_html=True)
    auto = st.toggle("Auto-update on new lines", value=True)
    low_dem = st.toggle("Include lower-tier demons", value=bool(settings.get("demon_lower_tier", False)),
                        help=f"Also price PrizePicks demons when a team has {settings.get('demon_min_team_maps', 4)}+ tier-1/2 maps "
                             f"(normally {settings['min_team_maps']}). These are capped at confidence C.")
    if st.session_state.setdefault("low_dem", low_dem) != low_dem:
        st.session_state.low_dem, st.session_state.force_refresh = low_dem, True
    t3_on = st.toggle("Price tier-3 lines (T3)", value=bool(settings.get("price_tier3", False)),
                      help=f"Also price teams short on tier-1/2 maps if they have {settings['min_team_maps']}+ tier 1-3 maps. "
                           f"Uses a tier-3 calibration, needs a {float(settings.get('tier3_min_edge', 0.06)):.0%} edge and is "
                           f"capped at confidence {settings.get('tier3_max_conf', 'B')}. Markets whose tier-3 calibration "
                           "isn't validated show as experimental estimates only.")
    if st.session_state.setdefault("t3_on", t3_on) != t3_on:
        st.session_state.t3_on, st.session_state.force_refresh = t3_on, True
    every = max(60, int(settings.get("auto_refresh_sec", 120)))
    forced = st.session_state.pop("force_refresh", False)
    job = st.session_state.get("job")
    clicked = st.button("Refresh lines & odds", type="primary", width="stretch")
    if (clicked or forced or ("res" not in st.session_state and job is None)) and not (job and not job["done"]):
        pred = shared_predictor(HIST_PATH.stat().st_mtime, model_mtime())
        pred.settings["demon_lower_tier"] = low_dem
        pred.settings["price_tier3"] = t3_on
        st.session_state.job = job = start_pricing(pred)
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
    rmin = max(15, int(settings.get("results_refresh_min", 60)))
    auto_res = st.toggle("Auto-pull new results", value=True,
                         help=f"Every {rmin} min: pull newly finished bo3.gg matches into player history, grade logged picks and re-price.")
    results_watch(auto_res, rmin)
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


@st.cache_data(ttl=3600)
def awp_profiles(_v):
    """AWP and total kills over each player's last 20 maps with bo3.gg weapon data: {player_id: (awp, kills, maps)}."""
    h = pd.read_csv(HIST_PATH, usecols=lambda c: c in {"game_id", "player_id", "date", "weapon_kills", "awp_kills"})
    h = attach_awp(h)
    h = h[h.weapon_kills > 0]
    if h.empty:
        return {}
    h["date"] = pd.to_datetime(h.date, utc=True, format="ISO8601")
    g = h.sort_values("date").groupby("player_id").tail(20).groupby("player_id")
    a = g[["awp_kills", "weapon_kills"]].sum().assign(n=g.size())
    return {int(i): (int(x.awp_kills), int(x.weapon_kills), int(x.n)) for i, x in a.iterrows()}


def awp_meta(r):
    pid = getattr(r, "player_id", None)
    v = [f.stat().st_mtime if f.exists() else 0 for f in (HIST_PATH, AWP_PATH)]
    a = awp_profiles(tuple(v)).get(int(pid)) if pid == pid and pid is not None else None
    if not a:
        return ""
    awp, kills, n = a
    role, share = awp_role(awp, kills, n)
    tip = (f"Last {n} maps with bo3.gg weapon data: {awp} AWP kills of {kills} kills ({share:.0%}). "
           f"AWPer = 30%+, Hybrid = 10-30%, Rifler = under 10%. A role needs {AWP_MIN_MAPS}+ maps.")
    if role is None:
        return f' · <span class="awp" title="{tip}">AWP {share:.0%} of kills · {n} maps</span>'
    cls = {"AWPer": "awper", "Hybrid": "hyb"}.get(role, "")
    return f' · <span class="awp {cls}" title="{tip}">{role} · AWP {share:.0%} of kills</span>'


def profile_header(r):
    over = r.side == "over"
    why = html.escape(str(getattr(r, "why", "") or ""), quote=True)
    t3x = t3_experimental(r)
    qual = ('<div class="ph-q t3x">T3 experimental · not a pick. A positive model difference does not qualify this line: the '
            'tier-3 calibration for this market is not validated, so it gets no confidence rating and is never +EV.</div>' if t3x else
            f'<div class="ph-q ok">✓ Qualified +EV pick · confidence {r.tier}</div>' if r.tier != "-" else
            f'<div class="ph-q" title="{why}">Model edge only — not a qualified pick{" · " + why if why else ""}</div>')
    conf = (f"Conf {r.tier}" if r.tier != "-" else
            '<span class="nr" title="Confidence tiers (A/B/C) are only given to lines that pass every +EV rule '
            f'(min edge, sample size, data checks). This line does not{": " + why if why else ""}.">Not rated</span>')
    return (f'<div class="ph"><div class="ph-id"><div class="ph-av">{avatar(getattr(r, "player_id", None), r.player)}</div><div>'
            f'<div class="ph-name">{html.escape(r.player)}</div>'
            f'<div class="ph-team">{logo(r.team_id)}<b>{html.escape(r.team)}</b><span class="vs">vs</span>{logo(r.opp_id)}{html.escape(r.opponent)}</div>'
            f'<div class="ph-meta">{html.escape(str(r.source).title())} · {conf} · {int(r.n_maps)} prior maps{start_meta(r)}{awp_meta(r)}</div></div></div>'
            f'<div class="ph-bet"><div class="ph-k">Selected prop</div><div class="ph-line">'
            f'<span class="ph-side {"over" if over else "under"}">{"▲ OVER" if over else "▼ UNDER"}</span>'
            f'<span class="ln">{r.line:g}</span><span class="ic">{ICON.get(r.stat, "")}</span><span class="st">{r.stat}</span>'
            f'<span class="scope">{r.maps_label}</span></div><div class="ph-nums">'
            f'<div><div class="ph-k">Model P({r.side})</div><div class="v">{r.p_side:.1%}</div></div>'
            f'<div><div class="ph-k">Breakeven</div><div class="v">{r.breakeven:.1%}</div></div>'
            + (f'<div title="{T3X_HELP}"><div class="ph-k">Unvalidated model difference</div><div class="v">{r.edge:+.1%}</div></div>' if t3x else
             f'<div><div class="ph-k">Edge</div><div class="v {"pos" if r.edge > 0 and r.tier != "-" else ""}">{r.edge:+.1%}</div></div>')
            + f'<div><div class="ph-k">Model mean</div><div class="v">{r.model_mean:.1f}</div></div></div>{qual}</div></div>')


OVERC = ("#22c55e", "#15803d", "#14532d")
UNDERC = ("#ef4444", "#b91c1c", "#7f1d1d")


def american(p):
    p = min(max(float(p), 0.01), 0.99)
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


ODDS_SRC = {"pinnacle": "Pinnacle", "polymarket": "Polymarket", "thunderpick": "Thunderpick", "elo": "model Elo (no odds listed)"}


def match_card(r):
    """Informational only: teams, format, tier and the pre-match moneyline. Nothing here feeds the model."""
    day, clock, _ = start_info(getattr(r, "start_time", None))
    ev, tier, et, at = r.get("event"), r.get("match_tier"), r.get("event_type"), r.get("odds_at")
    ev = ev if isinstance(ev, str) and ev else "Event unknown"
    tier = f"Tier {tier.upper()}" if isinstance(tier, str) and tier else "Tier unknown"
    et = {"lan": "LAN", "online": "Online"}.get(et, "LAN/online unknown")
    p = r.get("p_series_win", np.nan)
    kind = r.get("series_src")
    ok = p == p and kind in ("direct", "maps") and prematch_odds_ok(r.odds_source, at, getattr(r, "start_time", None))
    if ok:
        ca, cb = ("#22c55e", "#f5a524") if p >= 0.5 else ("#f5a524", "#22c55e")
        src = ODDS_SRC.get(r.odds_source, str(r.odds_source))
        pa = round(100 * p)
        lab = "Series moneyline" if kind == "direct" else "Estimated series probability · derived from map odds"
        bd, bo = r.get("book_dec", np.nan), r.get("book_dec_opp", np.nan)
        book = kind == "direct" and bd == bd and bo == bo and bd > 1 and bo > 1

        def side(q, d):
            if kind != "direct":
                return "<span>estimate</span>"
            return (f"<span>Fair odds (no-vig) {american(q)}</span>"
                    + (f"<span>Book odds {decimal_to_american(d):+d}</span>" if book else ""))
        odds = (f'<div class="ml-p"><div><b style="color:{ca}">{pa}%</b>{side(p, bd)}</div>'
                f'<div><b style="color:{cb}">{100 - pa}%</b>{side(1 - p, bo)}</div></div>'
                f'<div class="ml-bar"><i style="width:{p:.1%};background:{ca}"></i><i style="flex:1;background:{cb}"></i>'
                '<u style="left:30%"></u><u style="left:70%"></u></div>'
                f'<div class="ml-f">{"Implied win % (no-vig)" if kind == "direct" else "Estimate from no-vig map prices, not a quoted series price"} · {html.escape(src)} · pre-match snapshot '
                f'{pd.Timestamp(at).tz_convert("UTC"):%b %d %H:%M} UTC'
                + (f' · book odds = {html.escape(src)} quoted price incl. vig, same snapshot' if book else
                   ' · fair odds only, no quoted book price captured' if kind == "direct" else '')
                + ' · ticks at 30% / 70%</div>')
    else:
        odds = '<div class="ml-na">Pre-match odds unavailable</div>'
    return (f'<div class="ml"><div class="ml-h"><div><b>{html.escape(ev)}</b><br><span class="mut">{html.escape(f"{day} {clock}".strip()) or "Upcoming"}</span></div>'
            f'<div class="ml-b"><span class="ml-bo">BO{int(r.get("best_of", 3))}</span><span class="ml-bo">{tier}</span>'
            f'<span class="ml-bo">{et}</span></div></div>'
            f'<div class="ml-t"><div>{logo(r.team_id)}<b>{html.escape(r.team)}</b></div><span class="vs">VS</span>'
            f'<div>{logo(r.opp_id)}<b>{html.escape(r.opponent)}</b></div></div>'
            f'<div class="ml-o"><div class="ml-k">{lab if ok else "Moneyline"}</div>{odds}</div>'
            '<div class="ml-f" style="margin-top:6px">Information only: does not change projections, edges, confidence or picks.</div></div>')


def round_score(won, a, b):
    if a != a or b != b:
        return "–"
    return f"{'W' if a > b else 'L' if a < b else 'D'} {int(a)}–{int(b)}"


def series_score(w, n):
    if w != w or n != n:
        return "–"
    w, n = int(w), int(n)
    return f"{'W' if w > n - w else 'L' if w < n - w else 'D'} {w}–{n - w}"


def wl_color(txt):
    return "#4ade80" if txt.startswith("W") else "#f87171" if txt.startswith("L") else "#a3acb9"


SMALL_N = 5  # fewer past series than this: hit rates are flagged as a small sample


def series_chart(r, h, scope):
    maps = [m for m in r.maps if m <= 3] or list(r.maps)
    over_side = r.side == "over"
    g = h[h.map_number.isin(maps)]
    g = g.assign(_ot=g.rounds > 24, _bl=(g.team_rounds - g.opp_rounds).abs() >= 8)
    piv = g.pivot_table(index="match_id", columns="map_number", values=r.stat, aggfunc="sum")
    piv = piv.dropna(subset=maps) if all(m in piv.columns for m in maps) else piv.iloc[0:0]
    if len(piv):
        piv = piv.join(g.groupby("match_id").agg(date=("date", "max"), opp=("opponent", "first"), ot=("_ot", "max"), bl=("_bl", "max"))).sort_values("date")
        piv["total"] = piv[maps].sum(1)
        mp = g.groupby(["match_id", "map_number"]).agg(won=("won", "max"), tr=("team_rounds", "max"), orr=("opp_rounds", "max"),
                                                      mn=("map_name", "first"))
        for m in maps:
            x_ = mp.xs(m, level="map_number").reindex(piv.index)
            piv[f"wn{m}"] = x_.won.values
            piv[f"sc{m}"] = [round_score(w, a, b) for w, a, b in zip(x_.won, x_.tr, x_.orr)]
            piv[f"mn{m}"] = [nice_map(n) if isinstance(n, str) else "?" for n in x_.mn]
        ser = h[h.match_id.isin(piv.index)].groupby(["match_id", "map_number"]).won.max().groupby("match_id").agg(["sum", "size"])
        ser = ser.reindex(piv.index)
        piv["series"] = [series_score(w, n) for w, n in zip(ser["sum"], ser["size"])]

    def cell(lbl, d, empty="No series in the last 6 months"):
        n = len(d)
        if n == 0:
            return f'<div class="ss"><div class="ss-h"><b>{lbl}</b><span>n=0</span></div><div class="ss-na">{empty}</div></div>'
        hit = (d.total > r.line) if over_side else (d.total < r.line)
        push, avg = int((d.total == r.line).sum()), d.total.mean()
        diff = avg - r.line
        low = n < SMALL_N
        tags = f" · {push} push" if push else ""
        warn = f'<div class="ss-w">Small sample: {n} series, not reliable</div>' if low else ""
        return (f'<div class="ss{" ss-low" if low else ""}"><div class="ss-h"><b>{lbl}</b><span>n={n} series{tags}</span></div>{warn}'
                f'<div class="ss-r"><span class="k">{"Over" if over_side else "Under"} hit</span><span class="v">{hit.mean():.0%}</span>'
                f'<span class="f">{int(hit.sum())} of {n}</span></div>'
                f'<div class="ss-r"><span class="k">Avg</span><span class="v">{avg:.1f}</span>'
                f'<span class="f {"pos" if (diff > 0) == over_side and diff != 0 else "neg"}">{diff:+.1f} vs line</span></div></div>')
    st.markdown(f'<div class="sect">Recent series · {r.stat} {r.maps_label}</div>'
                f'<div class="ss-top"><b>Historical series vs the line:</b> hit rates and averages come from past series only and are not a prediction (the Model card is the forecast for this match). Side: <span class="ph-side {"over" if over_side else "under"}">'
                f'{"▲ OVER" if over_side else "▼ UNDER"} {r.line:g}{" (hypothetical)" if r.get("what_if") else ""}</span> · counted on series where all of {r.maps_label} was played</div>', unsafe_allow_html=True)
    total, nf = len(piv), 0
    if total:
        piv, nf = history_filters(r, h, maps, piv, scope)
    h2h = piv[piv.opp == r.opponent] if len(piv) else piv
    model_cell = (f'<div class="ss ss-m"><div class="ss-h"><b>MODEL</b><span>{"forecast · hypothetical line" if r.get("what_if") else "forecast · this match"}</span></div>'
                  f'<div class="ss-r"><span class="k">P({r.side})</span><span class="v">{r.p_side:.1%}</span><span class="f">{"hyp. edge" if r.get("what_if") else "edge"} {r.edge:+.1%}</span></div>'
                  f'<div class="ss-r"><span class="k">Mean</span><span class="v">{r.model_mean:.1f}</span>'
                  f'<span class="f">{r.model_mean - r.line:+.1f} vs line</span></div></div>')
    nomatch = "No series match the current filters"
    st.markdown(f'<div class="ssr">{cell("L5", piv.tail(5), *([nomatch] if nf else []))}{cell("L10", piv.tail(10), *([nomatch] if nf else []))}'
                f'{cell("L15", piv.tail(15), *([nomatch] if nf else []))}'
                f'{cell("H2H", h2h, nomatch if nf else "No series vs " + html.escape(r.opponent) + " in the last 6 months")}{model_cell}</div>', unsafe_allow_html=True)
    if piv.empty:
        st.info(f"No series match the current filters (0 of {total}). Use Reset filters to see them all." if nf
                else f"No recent series with {r.maps_label} played; nothing to chart.")
        return
    win = st.segmented_control("Window", ["L5", "L10", "L15", "H2H", "All"], default="L10", key=f"win_{scope}_{r.name}",
                               label_visibility="collapsed") or "L10"
    if win == "H2H" and h2h.empty:
        st.info(f"No series vs {r.opponent} with {r.maps_label} played in the last 6 months.")
        return
    v = h2h if win == "H2H" else piv.tail({"L5": 5, "L10": 10, "L15": 15}.get(win, len(piv)))
    with st.container(key=f"fitrow_tog_{scope}_{r.name}"):
        t = st.columns([1, 1, 3])
    show_ot = t[0].toggle("High round maps", True, key=f"ot_{scope}_{r.name}", help="Mark series where a map went to overtime (25+ rounds).")
    show_bl = t[1].toggle("Blowout maps", True, key=f"bl_{scope}_{r.name}", help="Mark series where a map was won by 8+ rounds (e.g. 13-5).")
    ot_f = v.ot.astype(bool).values & show_ot
    bl_f = v.bl.astype(bool).values & show_bl
    tags = [", ".join([x for x, f in (("high-round map (OT)", o), ("blowout map", b_)) if f]) or "–" for o, b_ in zip(v.ot.astype(bool), v.bl.astype(bool))]
    hit = (v.total >= r.line).values
    x = list(range(len(v)))
    labels = [f"{pd.Timestamp(d):%b %d}<br>{html.escape(str(o))[:13]}<br><span style='color:{wl_color(sr)}'><b>{sr}</b></span>"
              for d, o, sr in zip(v.date, v.opp, v.series)]
    res = np.where(v.total > r.line, "OVER", np.where(v.total < r.line, "UNDER", "PUSH (= line)"))
    maprows = [[f"Map {m} · {n} · <span style='color:{wl_color(sc)}'><b>{sc}</b></span> · {int(val)} {r.stat}"
                for n, sc, val in zip(v[f"mn{m}"], v[f"sc{m}"], v[m])] for m in maps]
    cd = np.column_stack([[f"{pd.Timestamp(d):%b %d}" for d in v.date], v.opp.astype(str)] + maprows
                         + [v.total.astype(int), res, tags, v.series])
    ht = ("<b>%{customdata[0]} · vs %{customdata[1]}</b> · series <b>" + f"%{{customdata[{5 + len(maps)}]}}</b><br>"
          + "".join(f"%{{customdata[{2 + k}]}}<br>" for k in range(len(maps)))
          + f"Combined total: <b>%{{customdata[{2 + len(maps)}]}}</b><br>Line: {r.line:g}<br>Result: <b>%{{customdata[{3 + len(maps)}]}}</b><br>Maps: %{{customdata[{4 + len(maps)}]}}<extra></extra>")
    fig = go.Figure()
    for k, m in enumerate(maps):
        fig.add_bar(x=x, y=v[m], name=f"Map {m}", customdata=cd, hovertemplate=ht,
                    marker=dict(color=[(OVERC if h_ else UNDERC)[k] for h_ in hit], line=dict(color="rgba(10,13,18,.85)", width=1)),
                    text=[f"<b>{int(val)}</b><br><span style='font-size:9px'>M{m}<br>{html.escape(sc)}</span>" for val, sc in zip(v[m], v[f"sc{m}"])],
                    textposition="inside",
                    insidetextanchor="middle", textangle=0, constraintext="inside",
                    textfont=dict(color="#ffffff", size=13, family="JetBrains Mono"))
    fig.add_scatter(x=x, y=v.total, mode="text", text=[f"<b>{int(t_)}</b>" + (" <span style='color:#fbbf24'>⚠</span>" if o else "") + (" <span style='color:#93c5fd'>✓</span>" if b_ else "")
                                              for t_, o, b_ in zip(v.total, ot_f, bl_f)], textposition="top center",
                    textfont=dict(color=["#4ade80" if h_ else "#f87171" for h_ in hit], size=17, family="JetBrains Mono"),
                    hoverinfo="skip", showlegend=False, cliponaxis=False)
    fig.add_hline(y=r.line, line_dash="dash", line_color="rgba(229,231,235,.8)", line_width=2, layer="above")
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", y=r.line, text=f"<b>{r.line:g}</b>", showarrow=False, bgcolor="#1b222c",
                       bordercolor="rgba(229,231,235,.6)", borderwidth=1, borderpad=3,
                       font=dict(color="#eef1f5", size=12, family="JetBrains Mono"))
    fig.update_layout(barmode="stack", template="plotly_dark", height=410, margin=dict(l=8, r=52, t=22, b=8), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.45 if len(v) < 6 else 0.16,
                      uniformtext=dict(minsize=10, mode="hide"),
                      hoverlabel=dict(bgcolor="#11161d", bordercolor="#2e3846", font=dict(family="Inter", size=12, color="#eef1f5")),
                      xaxis=dict(tickvals=x, ticktext=labels, tickfont=dict(size=11, color="#a3acb9"), range=[-0.6, len(v) - 0.4]),
                      yaxis=dict(gridcolor="#1a212b", zeroline=False, title=dict(text=r.stat, font=dict(size=11, color="#6b7684")),
                                 range=[0, max(v.total.max(), r.line) * 1.14]))
    try:
        fig.update_layout(barcornerradius=6)
    except ValueError:
        pass
    cl = st.container()
    seg = " · ".join(f"M{m} {'bottom' if k == 0 else 'top' if k == len(maps) - 1 else 'middle'}" for k, m in enumerate(maps))
    cl.markdown(f'<div class="lg"><span><i style="background:{OVERC[0]}"></i><i style="background:{OVERC[1]}"></i>Total ≥ line</span>'
                f'<span><i style="background:{UNDERC[0]}"></i><i style="background:{UNDERC[1]}"></i>Total below line</span>'
                f'<span>{seg}</span><span><i class="dash"></i>Line {r.line:g}</span>'
                '<span><b style="color:#4ade80">W</b>/<b style="color:#f87171">L</b> 13–9 = team rounds–opponent rounds</span>'
                + ('<span><b style="color:#fbbf24">⚠</b> high-round map</span>' if show_ot else "")
                + ('<span><b style="color:#93c5fd">✓</b> blowout map</span>' if show_bl else "") +
                f'<span class="mut">{len(v)} series shown{f" (filtered, {len(piv)} of {total} match)" if nf else ""} · hover a bar for details</span></div>', unsafe_allow_html=True)
    cl.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"series_{scope}_{r.name}")


@st.cache_data(show_spinner=False)
def event_names(ids):
    from cs2model.scraper.bo3_history import tournament
    return {t: (tournament(t) or {}).get("name") or f"Event {t}" for t in ids}


def history_filters(r, h, maps, piv, scope):
    """Filters for the historical chart and the L5/L10/L15/H2H cells only; nothing here reaches the model."""
    sm = HF.series_table(h, maps).reindex(piv.index)
    pre = f"hf_{scope}_{r.name}_"

    wk = HF.widget_prefix(st.session_state, pre)

    def reset():
        HF.reset_state(st.session_state, pre)
    ev = sm.assign(d=piv.date).dropna(subset=["event"]).groupby("event").d.max().sort_values(ascending=False)
    names = event_names(tuple(int(e) for e in ev.index))
    lo, hi = int(sm.rounds.min()), int(sm.rounds.max())
    with st.container(key=f"fitrow_{pre}"):
        c = st.columns([1.1, 1.1, 4], vertical_alignment="center")
    with c[0].popover("Filters", icon=":material/tune:", width="stretch"):
        a, b = st.columns(2)
        f = {"opp": a.multiselect("Opponent", sorted(sm.opp.dropna().unique()), key=wk + "opp", placeholder="All"),
             "team": b.multiselect("Team", sorted(sm.team.dropna().unique()), key=wk + "team", placeholder="All",
                                   help="Includes the player's previous teams.")}
        f["event"] = st.multiselect("Event", [int(e) for e in ev.index], format_func=lambda t: names.get(t, f"Event {t}"),
                                    key=wk + "event", placeholder="All")
        a, b, d = st.columns(3)
        f["result"] = a.selectbox("Series result", ["Any", "Won", "Lost"], key=wk + "result")
        f["lan"] = b.selectbox("LAN / online", ["Any", "LAN", "Online"], key=wk + "lan")
        f["bo"] = d.multiselect("Series length", sorted(int(x) for x in sm.bo.dropna().unique()), format_func=lambda x: f"BO{x}",
                                key=wk + "bo", placeholder="Any")
        f["tier"] = st.multiselect("Tier", sorted(sm.tier.dropna().unique(), key="sabcd".find), format_func=lambda t: f"Tier {t.upper()}",
                                   key=wk + "tier", placeholder="Any")
        f["rounds"] = (st.slider(f"Total rounds ({r.maps_label})", lo, hi, (lo, hi), key=wk + "rounds",
                                 help="Rounds summed over the counted maps. A filter only: the blowout marker stays an 8+ round margin.")
                       if hi > lo else None)
        f["maps"] = st.multiselect("Includes maps", sorted({m for t in sm.map_names for m in t}), format_func=nice_map,
                                   key=wk + "maps", placeholder="Select maps", help="Keeps series where any counted map is one of these.")
    n = HF.n_active(f, (lo, hi))
    c[1].button("Reset filters", on_click=reset, disabled=not n, key=pre + "reset", width="stretch")
    out = piv.loc[HF.apply(sm, f)]
    c[2].markdown(f'<span class="mut">{f"{n} filter" + ("s" if n > 1 else "") + " active" if n else "No filters"} · '
                  f'{len(out)} of {len(piv)} series · history only, never changes the model</span>', unsafe_allow_html=True)
    return out, n


def line_stepper(r, i, scope):
    """- / + line adjuster (e.g. a bumped line): the Recent Series chart, hit rates and Model card re-price at it."""
    c1, c2 = st.columns([1, 5], vertical_alignment="bottom")
    ln = c1.number_input("Line", min_value=0.5, max_value=200.0, value=float(r.line), step=1.0, format="%.1f",
                         key=f"line_{scope}_{r.source}_{r.player_id}_{r.stat}_{r.maps_label}_{r.odds_type}_{r.line:g}", help="Try another line, e.g. a bumped one. Same model, calibration and payout as the slate.")
    pmf = details.get(i, {}).get("pmf")
    if ln == r.line or pmf is None:
        c2.caption(f"Posted line {r.line:g} ({r.source}). Use − / + to see the chart, hit rates and model edge at another line.")
        return r
    be_o = r.get("be_over", r.breakeven)
    be_u = r.get("be_under", r.breakeven)
    over_only = be_u != be_u
    a = line_probs(pmf, ln, lambda p: pred.m.calibrate(r.stat, len(r.maps), p), be_o, 0.5 if over_only else be_u)
    side = "over" if over_only or a["edge_over"] >= a["edge_under"] else "under"
    live = max(1e-9, 1 - a["p_push"])
    r2 = r.copy()
    r2["line"], r2["side"], r2["edge"] = ln, side, a[f"edge_{side}"]
    r2["p_side"] = min(1.0, a[f"p_{side}"] / live)
    r2["what_if"] = True
    push = f" · push {a['p_push']:.1%} (voids the leg)" if a["p_push"] > 0 else ""
    c2.markdown(f'<div class="wi"><b>HYPOTHETICAL LINE {ln:g}</b> · not an offered line (posted: {r.line:g} on {html.escape(str(r.source).title())}) · '
                f'model P(over) {a["p_over"]:.1%} · P(under) {a["p_under"]:.1%}{push} · '
                f'hypothetical edge {side} {r2.edge:+.1%} at the current price vs {r.edge:+.1%} at the posted line</div>', unsafe_allow_html=True)
    return r2


@st.cache_data(ttl=86400, show_spinner=False)
def career_totals(slug):
    from cs2model.scraper.bo3_history import career_stats
    return career_stats(slug)  # failures raise and are not cached


def career_panel(r):
    pid = getattr(r, "player_id", None)
    pl = shared_predictor(HIST_PATH.stat().st_mtime, model_mtime()).players
    slug = pl.loc[pl.player_id == pid, "player_slug"]
    slug = slug.iloc[0] if len(slug) and isinstance(slug.iloc[0], str) else None
    with st.expander("Career statistics · bo3.gg career totals"):
        try:
            c = career_totals(slug) if slug else None
        except Exception:  # noqa: BLE001
            c = None
        if not c:
            st.caption("bo3.gg career totals aren't available for this player right now.")
            return

        def n(k):
            return int(c.get(k) or 0)

        def wr(w, l_):
            return f"{w / (w + l_):.0%} won" if w + l_ else "–"
        rnd = n("rounds_count")
        items = [("Matches", f"{n('matches_count'):,}", f"{n('matches_won_count'):,}–{n('matches_lost_count'):,} · "
                  f"{wr(n('matches_won_count'), n('matches_lost_count'))}"),
                 ("Maps", f"{n('games_count'):,}", f"{n('games_won_count'):,}–{n('games_lost_count'):,} · "
                  f"{wr(n('games_won_count'), n('games_lost_count'))}"),
                 ("Events", f"{n('events_count'):,}", f"{n('events_won_count'):,} won"),
                 ("K / D / A", f"{n('kills_sum'):,} / {n('deaths_sum'):,} / {n('assists_sum'):,}", ""),
                 ("Career KPR", f"{n('kills_sum') / rnd:.2f}" if rnd else "–", f"{n('kills_sum'):,} kills ÷ {rnd:,} rounds"),
                 ("Career ADR", f"{n('damage_sum') / rnd:.1f}" if rnd and c.get("damage_sum") else "–",
                  f"{n('damage_sum'):,} damage ÷ {rnd:,} rounds" if c.get("damage_sum") else "")]
        st.markdown('<div class="kpis">' + "".join(
            f'<div class="kpi"><div class="k">{k}</div><div class="v" style="font-size:{".95" if len(v) > 12 else "1.15"}rem;'
            f'white-space:normal">{HF.kpi_value_html(v)}</div>'
            f'<div style="font-size:.72rem;color:var(--tx2)">{html.escape(sub)}</div></div>' for k, v, sub in items) + "</div>",
            unsafe_allow_html=True)
        st.caption("Career totals from bo3.gg for every match it has on record: all tiers and all dates, not only tier 1-2 "
                   "or the last 6 months. Context only; the model doesn't use them. bo3.gg publishes no career headshot "
                   "totals, so none are shown. Win rate = wins ÷ (wins + losses).")


def player_detail(r, i, scope):
    r = line_stepper(r, i, scope)
    h = pred.hist[pred.hist.player_id == r.player_id].sort_values(["date", "map_number"], ascending=False)
    h = h.assign(hs_pct=h.headshots / h.kills.replace(0, np.nan), kpr=h.kills / h.rounds)
    series_chart(r, h, scope)
    st.markdown(match_card(r), unsafe_allow_html=True)
    d = details.get(i, {})
    st.markdown(f'<div class="sect">Map predictions · vs {html.escape(r.opponent)}</div>', unsafe_allow_html=True)
    if "spec" in d:
        mp = pd.DataFrame(A.map_preds(d["spec"], d["pkey"]))
        bym = h.groupby("map_name").agg(maps_n=("kills", "size"), avg_k=("kills", "mean"), avg_hs=("headshots", "mean"),
                                        kpr_hist=("kpr", "mean"), hs_hist=("hs_pct", "mean"))
        mp = mp.merge(bym, left_on="map", right_index=True, how="left").sort_values("p_played", ascending=False)
        w = mp.p_played / mp.p_played.sum()
        allm = dict(kills=(w * mp.kills).sum(), headshots=(w * mp.headshots).sum(), kpr=(w * mp.kpr).sum(), maps_n=len(h),
                    avg_k=h.kills.mean(), avg_hs=h.headshots.mean(), kpr_hist=h.kpr.mean(), hs_hist=h.hs_pct.mean())
        st.markdown(map_table(mp, allm, d["spec"].best_of), unsafe_allow_html=True)
        pw = mp.p_win.dropna()
        wtxt = (f"map win prob {pw.min():.0%}–{pw.max():.0%} depending on the map" if len(pw) and pw.max() - pw.min() > 0.005
                else f"map win prob {pw.iloc[0]:.0%} on every map" if len(pw) else "map win prob unavailable")
        st.caption(f"Model projections use this match's odds ({r.odds_source}); {wtxt}. "
                   "P(played) is per map and is not normalised: a series plays 2-3 maps, so the column sums to the expected map count.")
        map_compare(r, mp, allm, h, scope)
    else:
        st.markdown(pp_empty("Map predictions are unavailable for this line (no match specification was priced)."), unsafe_allow_html=True)
    match_history(r, h, scope)
    opponent_table(h)


def map_compare(r, mp, allm, h, scope):
    st.markdown('<div class="sect">Map comparison</div>', unsafe_allow_html=True)
    opts = list(mp["map"])
    sel = st.segmented_control("Map", opts, default=opts[0], format_func=nice_map, key=f"mc_{scope}_{r.name}") or opts[0]
    x, hm = mp[mp["map"] == sel].iloc[0], h[h.map_name == sel]
    n, nm = len(hm), nice_map(sel)
    miss = f"No {nm} maps for this player in the last 6 months"

    def diff(m, v, f):
        if pd.isna(m) or pd.isna(v):
            return pp_na("Needs both a projection and history on this map")
        return f'<span class="pp-df">{"▲" if m > v else "▼" if m < v else "="} {f.format(m - v)}</span>'

    proj = [("Kills", x.kills, hm.kills.mean(), allm["avg_k"], "{:.1f}", "{:+.1f}"),
            ("Headshots", x.headshots, hm.headshots.mean(), allm["avg_hs"], "{:.1f}", "{:+.1f}"),
            (pp_tip("KPR", TIPS["kpr"]), x.kpr, hm.kpr.mean(), allm["kpr_hist"], "{:.2f}", "{:+.2f}")]
    rows = [f'<tr><td class="l mapc">{lab}</td><td class="g-m">{pp_num(m, f)}</td><td class="g-h">{pp_num(v, f, miss)}</td>'
            f'<td>{pp_num(b, f)}</td><td class="g-m">{diff(m, v, fd)}</td></tr>' for lab, m, v, b, f, fd in proj]
    nop = pp_na("History only: the model does not project this")
    for lab, v, b in [(pp_tip("HS %", TIPS["hs"]), hm.hs_pct.mean(), h.hs_pct.mean()), ("Map win %", hm.won.mean(), h.won.mean())]:
        rows.append(f'<tr><td class="l mapc">{lab}</td><td class="g-m">{nop}</td><td class="g-h">{pp_num(v, "{:.0%}", miss)}</td>'
                    f'<td>{pp_num(b, "{:.0%}")}</td><td class="g-m">{nop}</td></tr>')
    head = pp_head([("Metric", "l"), (pp_tip("If played", TIPS["pk"]), "g-m"), (f"On {nm}", "g-h"), ("All maps", ""),
                    (pp_tip("Model − map avg", f"Model projection minus his historical {nm} average"), "g-m")],
                   [("", 1, ""), ("Model projection", 1, "g-m"), ("Historical · last 6 months", 2, "g-h"), ("Difference", 1, "g-m")])
    small = ' <span class="pp-lk">small sample</span>' if 0 < n < 3 else ""
    wins = int(hm.won.sum())
    strip = (f'<div class="pp-mcs"><div><span class="k">P(played)</span>{pp_pbar(x.p_played, False)}</div>'
             f'<div><span class="k">Maps on {nm}</span><b>{n}</b>{small}</div>'
             f'<div><span class="k">Map record</span><b>{wins}–{n - wins}</b></div></div>')
    out = [strip, f'<div class="pp-tw"><table class="pp-tbl sm">{head}<tbody>{"".join(rows)}</tbody></table></div>']
    if n:
        rec = "".join(f'<tr><td class="l">{pd.Timestamp(y.date):%b %d}</td><td class="l">{html.escape(str(y.opponent))}</td>'
                      f'<td>{pp_num(y.team_rounds, "{:.0f}")}–{pp_num(y.opp_rounds, "{:.0f}")}</td><td class="c">{pp_wl(y.won == 1)}</td>'
                      f'<td>{int(y.kills)}</td><td>{int(y.headshots)}</td><td>{pp_num(y.hs_pct, "{:.0%}")}</td>'
                      f'<td>{pp_num(y.rating, "{:.2f}")}</td></tr>' for y in hm.head(5).itertuples())
        rh = pp_head([("Date", "l"), ("Opponent", "l"), ("Score", ""), ("W/L", "c"), ("K", ""), ("HS", ""), ("HS %", ""),
                      (pp_tip("Rating", "bo3.gg player rating (its own scale)"), "")])
        out.append(f'<div class="pp-note">Last {min(n, 5)} of his {n} {nm} maps</div>'
                   f'<div class="pp-tw"><table class="pp-tbl sm">{rh}<tbody>{rec}</tbody></table></div>')
    else:
        out.append(f'<div class="pp-note"></div>{pp_empty(miss + ", so only the model projection is available.")}')
    st.markdown("".join(out), unsafe_allow_html=True)


def pp_tip(label, text):
    return f'<span class="pp-tip" title="{html.escape(text, quote=True)}" tabindex="0">{label}</span>'


def pp_na(text="Unavailable"):
    return f'<span class="pp-na" title="{html.escape(text, quote=True)}">–</span>'


def pp_num(x, f, missing="Unavailable"):
    return pp_na(missing) if x is None or pd.isna(x) else f.format(x)


def pp_empty(msg):
    return f'<div class="pp-empty">{html.escape(msg)}</div>'


def pp_pbar(p, hl):
    return (f'<div class="pp-pb{" hl" if hl else ""}" role="img" aria-label="{p:.0%} chance this map is played">'
            f'<div class="pp-pbt"><i style="width:{min(p, 1) * 100:.0f}%"></i></div><span>{p:.0%}</span></div>')


def pp_wl(won):
    if won is None:
        return '<span class="pp-wl t" title="Series level">–</span>'
    return f'<span class="pp-wl {"w" if won else "l"}" title="{"Win" if won else "Loss"}">{"W" if won else "L"}</span>'


def pp_delta(pred, hist, n, what):
    if pd.isna(hist) or pd.isna(pred) or n < 3 or not hist or abs(pred - hist) / hist < 0.1:
        return ""
    up = pred > hist
    return (f'<i class="pp-dl {"up" if up else "dn"}" title="Model projects {abs(pred - hist):.1f} {"more" if up else "fewer"} {what} '
            f'than his {n}-map average on this map">{"▲" if up else "▼"}</i>')


def pp_head(cols, groups=None):
    g = "".join(f'<th colspan="{n}" class="{c}">{t}</th>' for t, n, c in groups) if groups else ""
    return (f'<thead>{"<tr class=grp>" + g + "</tr>" if g else ""}<tr>'
            + "".join(f'<th class="{c}">{t}</th>' for t, c in cols) + "</tr></thead>")


TIPS = {"pp": "Chance this map is played in the series: map-pick model plus P(map 3). Per map, so several maps can be likely at once.",
        "pk": "Model expected kills on this map if it is played, given the match odds.",
        "ph": "Model expected headshots on this map if it is played.",
        "mk": "Model kills per round on this map.",
        "n": "Player maps on this map in the last 6 months (tier 1-2).",
        "kpr": "Kills per round: kills / rounds played, averaged over his maps here.",
        "hs": "Headshot %: headshot kills / kills, averaged over his maps here."}


def map_table(mp, allm, best_of):
    top = set(mp.head(1 if best_of == 1 else 2)["map"])
    head = pp_head([("Map", "l"), (pp_tip("P(played)", TIPS["pp"]), "l"), (pp_tip("Kills", TIPS["pk"]), "g-m"), (pp_tip("HS", TIPS["ph"]), ""),
                    (pp_tip("KPR", TIPS["mk"]), ""), (pp_tip("Maps", TIPS["n"]), "g-h"), ("Avg kills", ""), ("Avg HS", ""),
                    (pp_tip("KPR", TIPS["kpr"]), ""), (pp_tip("HS %", TIPS["hs"]), "")],
                   [("", 2, ""), ("Model projections", 3, "g-m"), ("Historical performance · last 6 months", 5, "g-h")])
    rows = []
    for x in mp.itertuples():
        n, hl = int(x.maps_n) if pd.notna(x.maps_n) else 0, x.map in top
        miss = f"No {nice_map(x.map)} maps for this player in the last 6 months"
        tag = '<b class="pp-lk">likely</b>' if hl else ""
        rows.append(f'<tr class="{"hl" if hl else ""}"><td class="l mapc">{nice_map(x.map)}{tag}</td><td class="l">{pp_pbar(x.p_played, hl)}</td>'
                    f'<td class="g-m">{x.kills:.1f}{pp_delta(x.kills, x.avg_k, n, "kills")}</td>'
                    f'<td>{x.headshots:.1f}{pp_delta(x.headshots, x.avg_hs, n, "headshots")}</td><td>{pp_num(x.kpr, "{:.2f}")}</td>'
                    f'<td class="g-h">{n}</td><td>{pp_num(x.avg_k, "{:.1f}", miss)}</td><td>{pp_num(x.avg_hs, "{:.1f}", miss)}</td>'
                    f'<td>{pp_num(x.kpr_hist, "{:.2f}", miss)}</td><td>{pp_num(x.hs_hist, "{:.0%}", miss)}</td></tr>')
    a = allm
    exp = pp_tip(f"exp. {mp.p_played.sum():.2f} maps", "Expected number of maps played = sum of P(played)")
    rows.append(f'<tr class="tot"><td class="l mapc">All maps</td><td class="l">{exp}</td>'
                f'<td class="g-m">{a["kills"]:.1f}</td><td>{a["headshots"]:.1f}</td><td>{pp_num(a["kpr"], "{:.2f}")}</td>'
                f'<td class="g-h">{a["maps_n"]}</td><td>{pp_num(a["avg_k"], "{:.1f}")}</td><td>{pp_num(a["avg_hs"], "{:.1f}")}</td>'
                f'<td>{pp_num(a["kpr_hist"], "{:.2f}")}</td><td>{pp_num(a["hs_hist"], "{:.0%}")}</td></tr>')
    legend = ('<div class="pp-note"><b class="pp-lk">likely</b> most probable maps · <i class="pp-dl up">▲</i><i class="pp-dl dn">▼</i> '
              'model differs from his average on that map by 10%+ (3+ maps) · hover headers for definitions</div>')
    return f'<div class="pp-tw"><table class="pp-tbl">{head}<tbody>{"".join(rows)}</tbody></table></div>{legend}'


def match_history(r, h, scope):
    st.markdown('<div class="sect">Match history</div>', unsafe_allow_html=True)
    if h.empty:
        st.markdown(pp_empty("No tier 1-2 matches for this player in the last 6 months."), unsafe_allow_html=True)
        return
    c1, c2 = st.columns([2, 3])
    nsel = c1.segmented_control("Series", ["5", "10", "20", "All"], default="5", key=f"mh_n_{scope}_{r.name}") or "5"
    opps = ["All opponents"] + list(h.opponent.drop_duplicates())
    osel = c2.selectbox("Opponent", opps, key=f"mh_o_{scope}_{r.name}")
    g0 = h if osel == "All opponents" else h[h.opponent == osel]
    ids = g0.match_id.drop_duplicates()
    shown = ids if nsel == "All" else ids.head(int(nsel))
    cols = [("#", ""), ("Map", "l"), ("Score", ""), ("W/L", "c"), ("K", ""), ("HS", ""), ("HS %", ""), ("ADR", ""),
            (pp_tip("Rating", "bo3.gg player rating (its own scale)"), "")]
    out = [f'<div class="pp-note">Showing {len(shown)} of {len(ids)} series{"" if osel == "All opponents" else " vs " + html.escape(osel)}'
           ' · newest first · click a series to show its maps</div>',
           '<div class="pp-mhh"><span></span><span>Date</span><span>Opponent</span><span>Series</span><span class="r hm">Maps</span>'
           '<span class="r hm">K</span><span class="r hm">HS</span><span class="r hm">Rating</span></div>']
    for k, mid in enumerate(shown):
        g = g0[g0.match_id == mid].sort_values("map_number")
        mw = int(g.won.sum())
        ml = len(g) - mw
        rows = "".join(
            f'<tr><td>{int(x.map_number)}</td><td class="l">{nice_map(x.map_name)}</td>'
            f'<td>{pp_num(x.team_rounds, "{:.0f}")}–{pp_num(x.opp_rounds, "{:.0f}")}</td><td class="c">{pp_wl(x.won == 1)}</td>'
            f'<td>{int(x.kills)}</td><td>{int(x.headshots)}</td><td>{pp_num(x.hs_pct, "{:.0%}")}</td><td>{pp_num(x.adr, "{:.0f}")}</td>'
            f'<td>{pp_num(x.rating, "{:.2f}")}</td></tr>' for x in g.itertuples())
        out.append(f'<details class="pp-mh"{" open" if k == 0 else ""}><summary><span class="sd">{pd.Timestamp(g.date.max()):%b %d}</span>'
                   f'<span class="so">vs {html.escape(str(g.opponent.iloc[0]))}</span>'
                   f'<span>{pp_wl(None if mw == ml else mw > ml)} {mw}–{ml}</span><span class="r hm">{len(g)}</span>'
                   f'<span class="r hm"><b>{int(g.kills.sum())}</b></span><span class="r hm"><b>{int(g.headshots.sum())}</b></span>'
                   f'<span class="r hm">{pp_num(g.rating.mean(), "{:.2f}")}</span></summary>'
                   f'<div class="pp-tw"><table class="pp-tbl sm">{pp_head(cols)}<tbody>{rows}</tbody></table></div></details>')
    st.markdown("".join(out), unsafe_allow_html=True)


def opponent_table(h):
    if h.empty:
        return
    t = h.groupby("opponent").agg(maps=("kills", "size"), kills=("kills", "mean"), hs=("headshots", "mean"),
                                   kpr=("kpr", "mean"), hs_pct=("hs_pct", "mean")).sort_values("maps", ascending=False).head(15)
    rows = "".join(f'<tr><td class="l">{html.escape(str(o))}</td><td>{x.maps}</td><td>{x.kills:.1f}</td><td>{x.hs:.1f}</td>'
                   f'<td>{pp_num(x.kpr, "{:.2f}")}</td><td>{pp_num(x.hs_pct, "{:.0%}")}</td></tr>' for o, x in zip(t.index, t.itertuples()))
    head = pp_head([("Opponent", "l"), ("Maps", ""), ("Avg kills", ""), ("Avg HS", ""), (pp_tip("KPR", TIPS["kpr"]), ""), (pp_tip("HS %", TIPS["hs"]), "")])
    st.markdown('<div class="sect">Opponent breakdown</div>'
                f'<details class="pp-mh"><summary class="one"><span class="so">By opponent · last 6 months</span>'
                f'<span class="r">{len(t)} opponents · most maps first</span></summary>'
                f'<div class="pp-tw"><table class="pp-tbl sm">{head}<tbody>{rows}</tbody></table></div></details>', unsafe_allow_html=True)


@st.dialog("Player panel", width="large")
def player_dialog(i):
    r = slate.loc[i]
    st.markdown(profile_header(r), unsafe_allow_html=True)
    player_detail(r, i, "dlg")


if not BS.is_complete():
    _sp = hist_span(HIST_PATH.stat().st_mtime if HIST_PATH.exists() else 0)
    st.info(f"Partial history: the models and player stats use {_sp['days']} days of bo3.gg maps while the full "
            f"{settings['history_days']}-day history downloads in the background (progress in the sidebar). Players need "
            f"{settings['min_maps']} prior maps to be priced, so fewer lines qualify until it finishes.")
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
    view = view.assign(logo=view.team_id.map(lambda t: LOGOS.get(int(t)) or ""), start=pd.to_datetime(view.start_time, utc=True, errors="coerce"))
    _pw = view.get("p_series_win", pd.Series(np.nan, index=view.index))
    _ok = [pw == pw and src in ("direct", "maps") and prematch_odds_ok(os_, at, st_) for pw, src, os_, at, st_ in
           zip(_pw, view.get("series_src", pd.Series(None, index=view.index)), view.odds_source,
               view.get("odds_at", pd.Series(None, index=view.index)), view.start_time)]
    _pw = _pw.where(_ok)
    view = view.assign(ml=[f"{p:.0%} · {american(p)}" if p == p else "" for p in _pw],
                       ml_fit=[ml_support(sd, p) for sd, p in zip(view.side, _pw)])
    view = view.assign(matchup=view.team.astype(str) + " vs " + view.opponent.astype(str),
                       prop=view.stat.astype(str).str.capitalize() + " · " + view.maps_label.astype(str))
    cols = ["logo", "player", "matchup", "prop", "line", "side", "edge", "ml", "ev", "tier", "model_mean", "p_over", "p_under",
            "start", "n_maps", "source", "why"]
    t = view.sort_values("edge", ascending=False)[cols].rename(columns={"model_mean": "mean"})
    t["side"] = t.side.map({"over": "▲ over", "under": "▼ under"}).fillna(t.side)
    t["ml"] = np.where(view.loc[t.index, "ml_fit"].eq("supports") & t.ml.ne(""), "✓ " + t.ml, t.ml)
    if "t3" in view:
        t3v = view.loc[t.index]
        t.loc[t3v.t3.astype(bool), "player"] += np.where(t3v.t3_validated[t3v.t3.astype(bool)].astype(bool), " · T3", " · T3 exp.")

    pos = {k: i for i, k in enumerate(t.index)}

    def color(row):
        r = view.loc[row.name]
        bg = f"background-color: {'#151b23' if pos[row.name] % 2 else '#11161d'};"
        out = []
        for c in row.index:
            x = bg
            if c == "player":
                x += "color: #f5f7fa; font-weight: 700;"
            elif c == "side":
                x += f"color: {'#22c55e' if r.side == 'over' else '#ef4444'};"
            elif c == "edge" and row[c] == row[c]:
                x += f"color: {'#86efac' if row[c] > 0 else '#7b8794'};"
            elif c == "ev" and row[c] == row[c]:
                x += f"color: {'#7fa68e' if row[c] > 0 else '#7b8794'};"
            elif c == "tier":
                x += {"A": "color: #ff7a1a;", "B": "color: #bcd3ea;"}.get(r.tier, "color: #7b8794;")
            elif c == "ml":
                x += "color: #86efac;" if r.ml_fit == "supports" else "color: #9aa7b6;"
            elif c in ("start", "n_maps", "source", "why"):
                x += "color: #9aa7b6;"
            out.append(x)
        return out

    sty = t.style.apply(color, axis=1).format({"line": "{:.1f}", "mean": "{:.1f}", "p_over": "{:.1%}", "p_under": "{:.1%}",
                                              "edge": "{:+.1%}", "ev": "{:+.1%}"}, na_rep="")
    sel = st.dataframe(sty, width="stretch", height=720, row_height=38, hide_index=True, on_select="rerun", selection_mode="single-row", key="board",
                 column_config={
                     "logo": st.column_config.ImageColumn("", width=40),
                     "player": st.column_config.Column("Player", width=130),
                     "matchup": st.column_config.Column("Matchup", width=230, help="Player's team vs opponent"),
                     "prop": st.column_config.Column("Prop", width=150), "line": st.column_config.Column("Line", width=60),
                     "side": st.column_config.Column("Side", width=85),
                     "edge": st.column_config.Column("Edge", width=75, help=HELP["edge"]),
                     "ml": st.column_config.Column("Team ML", width=125, help="Pre-match series win probability of the player's team "
                                                   "(no-vig book or market price) and its fair American odds. ✓ = the moneyline "
                                                   "leans the same way as the pick (favourite for over, underdog for under; 45 to "
                                                   "55% counts as neutral). Blank when no pre-start book snapshot exists. "
                                                   "Informational only, not used by the model."),
                     "ev": st.column_config.Column("EV", width=70, help=HELP["ev"]), "tier": st.column_config.Column("Conf", width=55, help=HELP["tier"]),
                     "mean": st.column_config.Column("Mean", width=60, help="Mean of the model's predictive distribution"),
                     "p_over": st.column_config.Column("P(over)", width=75), "p_under": st.column_config.Column("P(under)", width=80),
                     "start": st.column_config.DatetimeColumn("Start (ET)", width=150, format="ddd D MMM, h:mm A", timezone=TZ, help="Match start time, US Eastern (EST/EDT)"),
                     "n_maps": st.column_config.Column("Maps n", width=65, help="Prior maps in the player's history"),
                     "source": st.column_config.Column("Platform", width=90), "why": st.column_config.Column("Why", help=HELP["why"], width=420)})
    rows = sel.selection.rows if sel is not None else []
    k = t.index[rows[0]] if rows and rows[0] < len(t) else None
    if k is None:
        st.session_state.board_open = None
    elif st.session_state.get("board_open") != k:
        st.session_state.board_open = k
        player_dialog(k)
    st.markdown('<div class="sect">Actions</div>', unsafe_allow_html=True)
    st.caption("+EV rows (pass all rules) show a confidence tier in Conf · ▲ over / ▼ under · click a header to sort, click a row to open the player panel. Edge = model P(side) − platform breakeven.")
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
    opts = view.apply(lambda r: f"{r.player} · {r.stat} {r.maps_label} {r.line} ({r.source})"
                      + (" · T3" if bool(r.get("t3", False)) else ""), axis=1)
    if len(opts):
        pick = st.selectbox("Line", opts.index, format_func=lambda i: opts[i])
        r = slate.loc[pick]
        pmf = details[pick]["pmf"]
        st.markdown(profile_header(r), unsafe_allow_html=True)
        career_panel(r)
        with st.container(key="pdist"):
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
            with st.container(key="pcards"):
                m = st.columns(2)
                m[0].metric("P(over)", f"{r.p_over:.1%}")
                m[1].metric("P(under)", f"{r.p_under:.1%}")
                m = st.columns(2)
                if t3_experimental(r):
                    m[0].metric("Unvalidated model difference", f"{r.edge:+.1%}", help=T3X_HELP)
                else:
                    m[0].metric("Edge", f"{r.edge:+.1%}", r.side, help=HELP["edge"])
                m[1].metric("Confidence tier", r.tier, help=HELP["tier"])
            st.caption(r.reason)
            if r["flags"]:
                st.warning(r["flags"])
            st.caption(f"80% interval {r.q10:.0f}–{r.q90:.0f} · map win {r.p_map_win:.0%} · OT/map {r.p_ot:.0%} · "
                       f"map 3 {r.p_map3:.0%} · odds: {r.odds_source}")
        player_detail(r, pick, "tab")

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
                H = hit_matrix(details, sel)
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

APP_LINKS = {"prizepicks": ("Open PrizePicks", "https://app.prizepicks.com/"),
             "underdog": ("Open Underdog", "https://underdogfantasy.com/pick-em")}
PLATFORM_NAME = {"prizepicks": "PrizePicks", "underdog": "Underdog"}
XS_CSS = """<style>
.xs-card {background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:14px 16px;margin:4px 0 6px;}
.xs-top {display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:10px;}
.xs-top .t {font-family:var(--fd);font-size:1.3rem;font-weight:700;letter-spacing:.04em;color:var(--tx);margin-right:4px;}
.xs-chip {font-family:var(--fd);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;padding:2px 8px;border-radius:4px;
  border:1px solid var(--line2);color:var(--tx2);background:var(--panel2);}
.xs-chip.a {color:#ffb37a;border-color:rgba(255,122,26,.35);background:var(--accent-soft);}
.xs-m {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px;}
.xs-m > div {background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px 10px;}
.xs-m .k {font-family:var(--fd);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--tx2);}
.xs-m .v {font-family:var(--fm);font-size:1.25rem;font-weight:700;color:var(--tx);} .xs-m .v.pos {color:#4ade80;}
.xs-m .s {font-size:.72rem;color:var(--mut);}
.xs-leg {display:grid;grid-template-columns:minmax(0,1fr) 74px 56px 88px 76px;gap:8px;align-items:center;padding:7px 0;border-top:1px solid var(--line);}
.xs-leg .p b {color:var(--tx);} .xs-leg .p span {display:block;font-size:.74rem;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.xs-leg .sd {font-family:var(--fd);font-weight:700;font-size:.8rem;letter-spacing:.1em;text-align:center;padding:2px 6px;border-radius:4px;}
.xs-leg .sd.o {color:#4ade80;background:var(--pos-soft);} .xs-leg .sd.u {color:#f87171;background:var(--neg-soft);}
.xs-leg .ln {font-family:var(--fm);font-weight:700;color:var(--tx);text-align:right;} .xs-leg .st, .xs-leg .mp {color:var(--tx2);font-size:.85rem;}
.xs-note {font-size:.76rem;color:var(--mut);margin-top:8px;}
@media (max-width:640px) {.xs-m {grid-template-columns:repeat(2,minmax(0,1fr));}
  .xs-leg {grid-template-columns:auto auto 1fr auto;grid-template-areas:"p p p p" "sd ln st mp";row-gap:4px;}
  .xs-leg .p {grid-area:p;} .xs-leg .sd {grid-area:sd;} .xs-leg .ln {grid-area:ln;text-align:left;} .xs-leg .st {grid-area:st;} .xs-leg .mp {grid-area:mp;}}
</style>"""


def play_label(play):
    return play.replace("_", " ").title()


def payout_text(play):
    if play["type"] == "power":
        return f"{play['legs']}/{play['legs']} pays {play['payout']:g}x"
    return " · ".join(f"{h}/{play['legs']} {m:g}x" for h, m in sorted(play["payouts"].items(), reverse=True))


def slip_text(ids, plat, play_name):
    """Plain-text slip to enter by hand in the app."""
    legs = [f"{r.player} ({r.team}) {r.side.upper()} {r.line:g} {r.stat.title()} · {r.maps_label}"
            for r in slate.loc[list(ids)].itertuples()]
    return f"{PLATFORM_NAME[plat]} {play_label(play_name)} (hypothetical suggestion)\n" + "\n".join(legs)


def copy_button(text):
    components.html(f"""<button id="b" style="width:100%;height:38px;border-radius:6px;border:1px solid #2e3846;background:#161c25;
color:#eef1f5;font:600 14px Inter,system-ui,sans-serif;cursor:pointer">Copy slip</button><script>
const t={json.dumps(text)}, b=document.getElementById("b");
b.onclick=async()=>{{try{{await navigator.clipboard.writeText(t);}}catch(e){{const a=document.createElement("textarea");a.value=t;
document.body.appendChild(a);a.select();document.execCommand("copy");a.remove();}}b.textContent="Copied";
setTimeout(()=>b.textContent="Copy slip",1500);}};</script>""", height=44)


def slip_card(s_, plat, play, verified):
    legs = slate.loc[list(s_.leg_ids)]
    per_match = legs.match.value_counts()
    stake = f"${s_.stake:,.2f}" if verified else "hidden"
    rows = "".join(f'<div class="xs-leg"><div class="p"><b>{html.escape(r.player)}</b>'
                   f'<span>{html.escape(str(r.team))} vs {html.escape(str(r.opponent))}</span></div>'
                   f'<span class="sd {r.side[0]}">{r.side.upper()}</span><span class="ln">{r.line:g}</span>'
                   f'<span class="st">{r.stat.title()}</span><span class="mp">{html.escape(r.maps_label)}</span></div>'
                   for r in legs.itertuples())
    shared = int((per_match > 1).sum())
    corr = (f"{shared} match{'es' if shared != 1 else ''} with 2 legs: those legs are simulated together, so their dependence is "
            f"modeled (all-hit is {s_.corr_ratio - 1:+.0%} vs treating every leg as independent). " if shared else
            "Every leg is from a different match. ")
    st.markdown(f"""<div class="xs-card"><div class="xs-top"><span class="t">{play_label(s_.play)}</span>
<span class="xs-chip a">{PLATFORM_NAME[plat]}</span><span class="xs-chip">{play['type'].title()}</span>
<span class="xs-chip">{play['legs']} legs</span><span class="xs-chip">{len(per_match)} matches</span><span class="xs-chip">Hypothetical</span></div>
<div class="xs-m"><div><div class="k">All hit</div><div class="v">{s_.p_all:.1%}</div><div class="s">simulated</div></div>
<div><div class="k">Est. EV</div><div class="v {'pos' if s_.ev > 0 else ''}">{s_.ev:+.0%}</div><div class="s">{s_.ev_indep:+.0%} if independent</div></div>
<div><div class="k">Stake</div><div class="v">{stake}</div><div class="s">fractional Kelly, capped</div></div>
<div><div class="k">Payout</div><div class="v">{E.payout_vector(play).max():g}x</div><div class="s">{html.escape(payout_text(play))}</div></div></div>
{rows}<div class="xs-note">{corr}Different matches are treated as independent.</div></div>""", unsafe_allow_html=True)


@st.dialog("Example slips", width="large")
def example_slips_dialog():
    st.markdown(XS_CSS, unsafe_allow_html=True)
    ex_plat = st.radio("Platform", ["prizepicks", "underdog"], horizontal=True, key="ex_plat", format_func=PLATFORM_NAME.get)
    key = (str(res["ts"]), ex_plat, json.dumps([settings, payouts], sort_keys=True, default=str))
    cache = st.session_state.setdefault("ex_slip_cache", {})
    if key not in cache:
        cache.clear()
        with st.spinner("Simulating slips..."):
            cache[key] = build_slips(slate, details, ex_plat, settings, payouts)
    ex = cache[key]
    avail = E.plays(ex_plat, payouts)
    verified = bool(payouts.get(ex_plat, {}).get("verified"))
    st.caption("Hypothetical suggestions from the optimizer, not bets and not placed anywhere. Built from live +EV legs: one leg per "
               f"player, at most {settings['max_legs_per_match']} legs per match. Best slip of each play type by EV, repeats of "
               "the same legs skipped.")
    if not ex.empty:
        ex = ex.groupby("play").head(1)
        ex = ex[~ex.leg_ids.map(lambda x: tuple(sorted(x))).duplicated()].head(3)
    if ex.empty:
        st.info(f"No +EV slips on {PLATFORM_NAME[ex_plat]} right now.")
    for s_ in ex.itertuples():
        slip_card(s_, ex_plat, avail[s_.play], verified)
        copy_button(slip_text(s_.leg_ids, ex_plat, s_.play))
    label, url = APP_LINKS[ex_plat]
    st.link_button(label, url, width="stretch")
    st.caption(f"Opens {PLATFORM_NAME[ex_plat]}'s own pick'em page. Slips are not filled in for you: enter the legs by hand.")
    st.markdown(f"""<div class="xs-note"><b>How these numbers are estimated</b><br>
Payouts come from config/payouts.yaml ({PLATFORM_NAME[ex_plat]} {'marked verified' if verified else 'not verified'}). Your app's
multipliers can differ by state, promo or line, and lines with their own price are still paid at the table here.<br>
All-hit and EV come from the model's leg probabilities and simulated games. Legs in the same match share simulations, so their
dependence is modeled; legs in different matches are assumed independent. EV picks the best of many combinations and leg
errors compound across legs, so treat it as optimistic: slip results haven't been validated on graded slips yet.</div>""",
                unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="sb-h">Example slips</div>', unsafe_allow_html=True)
    if pricing:
        st.caption("Available when pricing finishes.")
    elif st.button("Example slips", width="stretch", help="Up to 3 hypothetical slips from the current +EV legs"):
        example_slips_dialog()

# ---------------- tracking
with tab_track:
    g1, g2, _ = st.columns([1, 1, 2])
    if g1.button("Grade logged picks", type="primary", width="stretch",
                 help="Grade logged picks whose matches are already in the player history"):
        with st.spinner("Grading..."):
            pr_ = shared_predictor(HIST_PATH.stat().st_mtime if HIST_PATH.exists() else 0, model_mtime())
            n_graded = store.grade(pr_.hist, pr_.resolve_player)
        st.success(f"Graded {n_graded} pick{'s' if n_graded != 1 else ''}." if n_graded else
                   "Nothing new to grade yet. Picks are graded once their maps are in the results.")
    ru = results_updater()
    if g2.button("Pull new results", width="stretch", disabled=ru["running"],
                 help="Fetch newly finished bo3.gg matches, then grade (also runs automatically with Auto-pull)"):
        start_results_update(ru)
        st.toast("Pulling new results from bo3.gg. Picks are graded when it finishes.")
    summ, p, s = store.performance()
    pend = store.pending()
    if len(pend):
        n_up = int((pend.status == "upcoming").sum())
        st.caption(f"{len(pend)} logged pick{'s' if len(pend) != 1 else ''} not graded yet: {n_up} upcoming, "
                   f"{len(pend) - n_up} awaiting results.")
        st.dataframe(pend[["status", "start_time", "player", "team", "opponent", "stat", "maps", "side", "line", "p_model",
                           "edge", "source"]], hide_index=True, width="stretch")
    if not summ.get("picks"):
        st.info("No graded picks yet. Log picks from the Slate or Parlays tab, then grade them here after their matches finish.")
    else:
        c = st.columns(5)
        c[0].metric("Graded picks", summ["picks"])
        c[1].metric("Hit rate", f"{summ['hit_rate']:.1%}")
        c[2].metric("Flat ROI", f"{summ['flat_roi']:+.1%}")
        c[3].metric("Avg CLV (lines)", f"{summ['avg_clv']:+.2f}" if summ["avg_clv"] == summ["avg_clv"] else "–")
        c[4].metric("Slip profit", f"${summ['slip_profit']:.2f}")
        if any(g["group"] == "Tier 3" for g in summ.get("by_group", [])):
            st.dataframe(pd.DataFrame(summ["by_group"]), hide_index=True, width="content",
                         column_config={"hit_rate": st.column_config.NumberColumn("Hit rate", format="percent"),
                                        "flat_roi": st.column_config.NumberColumn("Flat ROI", format="percent"),
                                        "avg_edge": st.column_config.NumberColumn("Avg edge", format="percent"),
                                        "avg_clv": st.column_config.NumberColumn("Avg CLV", format="%.2f")})
        p["bucket"] = (p.p_model * 10).astype(int) / 10
        cal = p.groupby("bucket").agg(pred=("p_model", "mean"), obs=("win", "mean"), n=("win", "size")).reset_index()
        fig = go.Figure([go.Scatter(x=cal.pred, y=cal.obs, mode="markers+lines", name="picks"),
                         go.Scatter(x=[0.4, 0.9], y=[0.4, 0.9], mode="lines", line_dash="dash", name="perfect")])
        fig.update_layout(template="plotly_dark", height=320, title="Live calibration", xaxis_title="model P", yaxis_title="hit rate")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(p.sort_values("start_time", ascending=False), hide_index=True, width="stretch")
        if len(s):
            st.dataframe(s, hide_index=True, width="stretch")
    st.subheader("Tier-3 reliability (graded tier-3 lines)")
    rel = store.t3_reliability()
    if rel.empty:
        st.caption("No graded tier-3 lines yet. Tier-3 lines are priced and logged in the background (even with the T3 toggle off) "
                   "and graded when results come in.")
    else:
        st.caption("Every graded tier-3 line, not just picks: predicted probability of the side the model favours vs how often it hit. "
                   "Gap > 0 = overconfident. 60–75% is where tier-3 picks would sit. Pushes are left out.")
        st.dataframe(rel, hide_index=True, width="stretch",
                     column_config={"avg_p": st.column_config.NumberColumn("Avg predicted P", format="percent"),
                                    "hit_rate": st.column_config.NumberColumn("Hit rate", format="percent"),
                                    "gap": st.column_config.NumberColumn("Gap (pred - hit)", format="percent")})

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
    fi = pd.Series(bt["feature_importance_kills"]).sort_values(ascending=False).head(12).iloc[::-1]
    fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#e8833a",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(tickformat=".0%", title=None), yaxis=dict(tickfont=dict(size=13, family="JetBrains Mono")),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Does the model run high?")
    sh = bt.get("shrink")
    if sh:
        st.dataframe(pd.DataFrame([{"stat": k, "pull toward recent form": f"{1 - v['lambda']:.0%}",
                                    "test bias before": v["test_bias_raw"], "test bias after": v["test_bias_shrunk"],
                                    "test MAE before": v["test_mae_raw"], "test MAE after": v["test_mae_shrunk"]}
                                   for k, v in sh.items()]).round(3), hide_index=True, width="stretch")
        st.caption("The raw model over-extends when it projects far from a player's recent average, so every projection is pulled "
                   "part of the way back toward his shrunk last-20-map rate (fit on validation maps, scored on test maps). Bias = model minus actual per map.")
    br = store.bias_report(store.connect())
    if br.empty:
        st.caption("Real lines: no graded lines yet. Every priced line is logged with its last pre-match projection and graded when results come in.")
    else:
        st.caption("Real lines, graded: model minus actual > 0 means the model ran high; line minus actual shows the same for the books.")
        st.dataframe(br.round(3), hide_index=True, width="stretch")
    mm = bt.get("map_model")
    st.subheader("Map pool model (test period)")
    if not mm:
        st.caption("Not trained yet: run `python -m cs2model.cli train` to fit and backtest it.")
    else:
        art_mm = json.loads((MODEL_DIR / "artifacts.json").read_text()).get("map_model", {})
        pk, wn = mm["picks"], mm["wins"]
        st.dataframe(pd.DataFrame([
            {"part": "Which maps get played", "metric": "log loss of pick order / series", "previous": pk["nll_per_series"]["baseline"],
             "map pool": pk["nll_per_series"]["map_pool"], "used in pricing": art_mm.get("use_picks", False)},
            {"part": "Which maps get played", "metric": "Brier, played in maps 1-2", "previous": pk["brier_played_maps12"]["baseline"],
             "map pool": pk["brier_played_maps12"]["map_pool"], "used in pricing": art_mm.get("use_picks", False)},
            {"part": "Map winner", "metric": "log loss per map", "previous": wn["logloss"]["same_every_map"],
             "map pool": wn["logloss"]["per_map"], "used in pricing": art_mm.get("use_wins", False)},
        ]).round(4), hide_index=True, width="stretch")
        st.caption(f"{mm['n_series']} series, {wn['maps']} maps, {mm['test_period'][0]} to {mm['test_period'][1]}; fit only on earlier series. "
                   "Each part is used in pricing only if it beat the previous approach here. \"Previous\" for map winners is one "
                   "Elo win chance on every map.")

    st.subheader("History and data bundle")
    _mh, _sp = BS.model_history(), hist_span(HIST_PATH.stat().st_mtime if HIST_PATH.exists() else 0)
    _pr = (BS.read_coverage() or {}).get("promotion") or {}
    st.caption(f"Player history: {_sp['rows']:,} player-map rows, {str(_sp['from'])[:10]} to {str(_sp['to'])[:10]} "
               f"({'full' if BS.is_complete() else 'partial'}, target {settings['history_days']} days). "
               + (f"Models trained on {str(_mh.get('from'))[:10]} to {str(_mh.get('to'))[:10]} ({_mh.get('days')} days"
                  f"{', quick start' if _mh.get('quick_start') else ''}). " if _mh else "")
               + (f"Last full-history check: {_pr.get('reason')} ({_pr.get('held_out_rows', 0):,} held-out player-maps)." if _pr else ""))
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Export**")
        st.caption("Saves your bo3.gg history and trained models to one zip for moving to another computer. Not included: "
                   "your tracking log, settings, API keys or tokens, and the HTTP cache. For your own use only; don't share "
                   "it publicly.")
        if st.button("Create bundle", key="bu_export"):
            from cs2model import bundle as BU
            try:
                with st.spinner("Writing bundle..."):
                    st.session_state.bundle_path = str(BU.export_bundle())
            except Exception as e:  # noqa: BLE001
                st.error(f"Export failed: {e}")
        bp = st.session_state.get("bundle_path")
        if bp and Path(bp).exists():
            st.caption(f"Saved to {bp} ({Path(bp).stat().st_size / 1e6:.1f} MB)")
            st.download_button("Download bundle", Path(bp).read_bytes(), file_name=Path(bp).name, mime="application/zip")
    with e2:
        st.markdown("**Import**")
        bundle_import_ui("model")
