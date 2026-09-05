from __future__ import annotations

import html
import math
from typing import Any

import plotly.graph_objects as go
import streamlit as st

_METRIC_COLORS = ["#66e6ff", "#a78bfa", "#f6c85f", "#72e5a1"]
_METRIC_GLOWS  = ["rgba(102,230,255,.15)", "rgba(167,139,250,.15)", "rgba(246,200,95,.13)", "rgba(114,229,161,.12)"]
_METRIC_HOVER  = ["rgba(102,230,255,.30)", "rgba(167,139,250,.30)", "rgba(246,200,95,.28)", "rgba(114,229,161,.28)"]
_metric_counter = [0]

def css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root{--bg:#07090d;--card:rgba(16,21,29,.70);--line:rgba(132,157,190,.18);--text:#f4f7fb;--muted:#8c9aab;--cyan:#66e6ff;--violet:#a78bfa;--amber:#f6c85f;--red:#ff6577;--green:#72e5a1}
    .stApp{background:radial-gradient(circle at 8% 0%,rgba(74,135,255,.10),transparent 28%),radial-gradient(circle at 90% 15%,rgba(157,89,255,.08),transparent 25%),var(--bg);color:var(--text);font-family:Manrope,sans-serif}
    .stApp:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:48px 48px;mask-image:linear-gradient(to bottom,black,transparent 75%);z-index:0}
    [data-testid="stSidebar"]{background:rgba(7,10,15,.84);backdrop-filter:blur(22px);border-right:1px solid var(--line)}
    .block-container{max-width:1500px;padding-top:1.4rem;position:relative;z-index:1}
    h1,h2,h3{font-family:Manrope,sans-serif;letter-spacing:-.045em}
    .hero{padding:.4rem 0 1.6rem}
    .hero-title{font-size:3.35rem;font-weight:800;line-height:1.0;margin:.35rem 0 .65rem;background:linear-gradient(100deg,#fff 20%,#c8ddff 55%,#86e7ff 85%);-webkit-background-clip:text;background-clip:text;color:transparent}
    .hero-sub{max-width:920px;color:var(--muted);font-size:1rem;line-height:1.6}
    .eyebrow{font-family:'DM Mono',monospace;font-size:.64rem;letter-spacing:.14em;color:#7e9ab8;text-transform:uppercase}
    .status{display:inline-flex;gap:.45rem;align-items:center;border:1px solid rgba(114,229,161,.22);background:rgba(114,229,161,.07);padding:.38rem .66rem;border-radius:999px;font:500 .65rem 'DM Mono';color:#b9f3d0}
    .dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 16px rgba(114,229,161,.9);animation:pulse 1.8s infinite}
    @keyframes pulse{0%,100%{opacity:.55;transform:scale(.9)}50%{opacity:1;transform:scale(1.2)}}
    .metric{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(19,27,38,.85),rgba(10,14,20,.78));border:1px solid var(--line);border-radius:18px;padding:1.2rem 1.1rem 1rem;min-height:128px;box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 18px 50px rgba(0,0,0,.22);backdrop-filter:blur(16px);margin-bottom:.75rem}
    .metric .label{color:#7890aa;font:500 .61rem 'DM Mono';letter-spacing:.11em;text-transform:uppercase}
    .metric .value{font-size:2.05rem;font-weight:800;margin-top:.42rem;letter-spacing:-.05em}
    .metric .note{color:#687b91;font-size:.69rem;margin-top:.35rem}
    .metric .glow{position:absolute;width:120px;height:120px;right:-52px;top:-52px;border-radius:50%;pointer-events:none}
    .panel{background:linear-gradient(145deg,rgba(15,21,29,.82),rgba(9,13,19,.70));border:1px solid var(--line);border-radius:20px;padding:1.1rem 1.2rem .5rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.035),0 20px 70px rgba(0,0,0,.16);backdrop-filter:blur(18px);margin-bottom:.5rem}
    .panel-title{font-weight:800;font-size:1rem;margin-bottom:.18rem}
    .panel-note{font-size:.72rem;color:#708198;margin-bottom:.85rem}
    .terminal{background:rgba(3,6,10,.82);border:1px solid rgba(126,151,180,.16);border-radius:14px;padding:1rem 1.1rem;font:400 .72rem/2.0 'DM Mono';color:#b9cadb;margin:.6rem 0}
    .terminal .accent{color:var(--cyan)}
    .big-action{position:relative;overflow:hidden;border:1px solid rgba(255,101,119,.42);background:radial-gradient(circle at 50% 0%,rgba(255,101,119,.14),transparent 52%),linear-gradient(145deg,rgba(42,14,21,.84),rgba(14,12,17,.78));border-radius:20px;padding:1.1rem;text-align:center}
    .big-action.review{border-color:rgba(246,200,95,.40);background:radial-gradient(circle at 50% 0%,rgba(246,200,95,.12),transparent 52%),linear-gradient(145deg,rgba(39,31,15,.80),rgba(14,13,16,.78))}
    .big-action.allow{border-color:rgba(114,229,161,.32);background:radial-gradient(circle at 50% 0%,rgba(114,229,161,.11),transparent 52%),linear-gradient(145deg,rgba(12,34,25,.76),rgba(10,14,17,.78))}
    .big-action .action{font-size:2.3rem;font-weight:850;letter-spacing:.07em}
    .big-action.block .action{color:var(--red)}
    .big-action.review .action{color:var(--amber)}
    .big-action.allow .action{color:var(--green)}
    .muted{color:#78889b}
    .action-pill{display:inline-flex;padding:.25rem .58rem;border-radius:999px;font:600 .64rem 'DM Mono';letter-spacing:.06em}
    .action-pill.block{background:rgba(255,101,119,.13);color:#ff9ba8;border:1px solid rgba(255,101,119,.25)}
    .action-pill.review{background:rgba(246,200,95,.11);color:#f8d985;border:1px solid rgba(246,200,95,.22)}
    .action-pill.allow{background:rgba(114,229,161,.11);color:#a7f0c8;border:1px solid rgba(114,229,161,.20)}
    .evidence{position:relative;border:1px solid rgba(127,151,181,.13);border-left:3px solid #657b95;background:rgba(16,22,30,.74);padding:.75rem .85rem;border-radius:0 12px 12px 0;margin:.5rem 0;transition:transform .18s ease}
    .evidence:hover{transform:translateX(4px)}
    .evidence strong{display:block;font-size:.76rem;margin-bottom:.18rem}
    .evidence span{color:#8998a9;font-size:.7rem;line-height:1.5}
    .evidence.strong{border-left-color:var(--red);background:rgba(255,101,119,.04)}
    .evidence.moderate{border-left-color:var(--amber);background:rgba(246,200,95,.03)}
    .evidence.weak{border-left-color:#71839a}
    .alert-card{border:1px solid rgba(127,151,181,.14);background:rgba(14,19,26,.72);border-radius:16px;padding:.9rem 1rem;margin:.45rem 0;transition:transform .18s ease,border-color .18s ease}
    .alert-card:hover{transform:translateY(-2px);border-color:rgba(102,230,255,.22)}
    .flow{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;padding:.5rem 0}
    .flow-step{padding:.55rem .75rem;border-radius:12px;background:rgba(102,230,255,.055);border:1px solid rgba(102,230,255,.14);font:500 .65rem 'DM Mono';color:#b8cfe1}
    .flow-arrow{color:#5f7892;font-family:'DM Mono'}
    .hero-badge{display:inline-flex;align-items:center;gap:.45rem;padding:.35rem .65rem;border-radius:999px;background:rgba(102,230,255,.07);border:1px solid rgba(102,230,255,.18);font:500 .63rem 'DM Mono';color:#9fd9e8}
    div[data-testid="stButton"] button{border-radius:12px;border:1px solid rgba(127,151,181,.18);background:rgba(17,23,31,.80);color:#dce6f0;font-weight:700;transition:all .18s ease}
    div[data-testid="stButton"] button:hover{border-color:rgba(102,230,255,.38);color:#fff;background:rgba(24,34,47,.92);transform:translateY(-2px)}
    div[data-baseweb="select"]>div{background:rgba(15,21,29,.84);border-color:rgba(127,151,181,.18)}
    [data-testid="stDataFrame"]{border:1px solid rgba(127,151,181,.14);border-radius:14px;overflow:hidden}
    .section-kicker{font:500 .64rem 'DM Mono';letter-spacing:.13em;color:#7592ad;text-transform:uppercase;margin:.9rem 0 .4rem}
    .big-number{font-size:3.3rem;font-weight:850;letter-spacing:-.06em;color:var(--cyan)}
    .delta-up{color:var(--green)}.delta-neutral{color:#9fb2c7}
    .divider{height:1px;background:linear-gradient(90deg,transparent,rgba(127,151,181,.22),transparent);margin:1.4rem 0}
    div[data-testid="column"]{padding-left:.4rem !important;padding-right:.4rem !important}
    </style>
    """, unsafe_allow_html=True)
    # reset counter each page render
    _metric_counter[0] = 0


def metric(label: str, value: str, note: str = "") -> None:
    i = _metric_counter[0] % 4
    _metric_counter[0] += 1
    color = _METRIC_COLORS[i]
    glow  = _METRIC_GLOWS[i]
    st.markdown(
        f'<div class="metric">'
        f'<div class="glow" style="background:radial-gradient(circle,{glow},transparent 70%)"></div>'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def panel(title: str, note: str = "") -> None:
    st.markdown(f'<div class="panel"><div class="panel-title">{html.escape(title)}</div><div class="panel-note">{html.escape(note)}</div>', unsafe_allow_html=True)


def end_panel() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def action_card(action: str, event_id: str, verification: float, evidence_count: int) -> None:
    a = action.lower()
    cls = a if a in {"block", "review", "allow"} else "review"
    st.markdown(f'<div class="big-action {cls}"><div class="eyebrow">CURRENT POLICY ACTION</div><div class="action">{html.escape(a.upper())}</div><div class="muted">{html.escape(event_id)} · verification {verification:.3f} · {evidence_count} evidence items</div></div>', unsafe_allow_html=True)


def network_figure(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], focus: str | None = None) -> go.Figure:
    n = len(nodes)
    if not n:
        return go.Figure()
    pos = {node["id"]: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i, node in enumerate(nodes)}
    ex, ey = [], []
    for edge in edges:
        a, b = edge.get("source"), edge.get("target")
        if a in pos and b in pos:
            ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(102,160,205,.38)", width=1.8), hoverinfo="none"))
    xs=[pos[nd["id"]][0] for nd in nodes]; ys=[pos[nd["id"]][1] for nd in nodes]
    sizes=[32 if nd["id"]==focus else 18 for nd in nodes]
    labels=[nd["label"] for nd in nodes]
    roles=[nd.get("role","peer") for nd in nodes]
    texts=["ALERT ACCOUNT" if r=="alert" else "BEHAVIORAL PEER" for r in roles]
    colors=["#ff6577" if r=="alert" else "#66e6ff" for r in roles]
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=labels,textposition="bottom center",textfont=dict(size=9,color="#a8bacd"),customdata=texts,hovertemplate="%{text}<br>%{customdata}<extra></extra>",marker=dict(size=sizes,color=colors,line=dict(width=1.8,color="#0b1119"),opacity=.93)))
    fig.update_layout(height=470,margin=dict(l=0,r=0,t=5,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig


def gauge(value: float, title: str) -> go.Figure:
    value = max(0, min(1, float(value)))
    bar_color = "#72e5a1" if value >= 0.6 else "#f6c85f" if value >= 0.35 else "#ff6577"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value*100,
        title={"text": title, "font": {"size": 12, "color": "#8fa2b8"}},
        number={"suffix": "%", "font": {"size": 28, "color": "#f4f7fb"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#46596d"},
               "bar": {"color": bar_color},
               "bgcolor": "rgba(16,23,31,.75)",
               "bordercolor": "rgba(127,151,181,.18)",
               "steps": [{"range": [0,35], "color": "rgba(255,101,119,.07)"},
                         {"range": [35,60], "color": "rgba(246,200,95,.05)"},
                         {"range": [60,100], "color": "rgba(114,229,161,.05)"}]}))
    fig.update_layout(height=195, margin=dict(l=10,r=10,t=25,b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig