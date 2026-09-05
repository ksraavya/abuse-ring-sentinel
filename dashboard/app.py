from __future__ import annotations

import math
import pandas as pd
import streamlit as st

from dashboard.data import (
    Paths, ring_for_account, demo_records, load_freeze, load_manifest, load_snapshot,
    load_world_b, load_world_d, money, neo4j_neighbors, num, pct, manifest_graph,
)
from dashboard.ui import css, end_panel, gauge, metric, network_figure, panel, action_card

st.set_page_config(page_title="Risk Manager · Coordinated Abuse", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
css()

paths = Paths()
world_b = load_world_b()
world_d = load_world_d()
snapshot = load_snapshot()
manifest = load_manifest(paths.world_c_manifest)
freeze = load_freeze(paths.verifier_freeze)

if "records" not in st.session_state:
    st.session_state.records = demo_records(120)
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = 0
if "executed" not in st.session_state:
    st.session_state.executed = set()
if "replay_filter" not in st.session_state:
    st.session_state.replay_filter = "All"


def header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def val(data: dict, key: str, fmt, fallback: str = "—") -> str:
    x = data.get(key)
    return fmt(x) if x is not None else fallback


def action_for(r: dict) -> str:
    return str(r.get("policy", {}).get("action", "review")).lower()


def filtered_records() -> list[dict]:
    rows = st.session_state.records
    f = st.session_state.replay_filter.lower()
    if f == "all":
        return rows
    if f == "high evidence":
        return [r for r in rows if len(r.get("evidence", []) or []) >= 5]
    return [r for r in rows if action_for(r) == f]


with st.sidebar:
    st.markdown('<div class="eyebrow">◈ RISK / OPS CONSOLE</div>', unsafe_allow_html=True)
    st.markdown("## Coordinated Abuse")
    st.caption("Event-time detection · evidence · response")
    page = st.radio("Navigate", ["Command Center", "Live Replay", "Ring Explorer", "Evidence & Response", "Evaluation"], label_visibility="collapsed")
    st.divider()
    st.markdown('<span class="status"><span class="dot"></span> ENGINE READY</span>', unsafe_allow_html=True)
    st.caption("Frozen detector + verifier policy. Dashboard actions are local prototype operations.")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">SYSTEM MODE</div>', unsafe_allow_html=True)
    st.markdown("**HELD-OUT / FROZEN**")
    st.caption("World D results are read-only. No experiment is rerun by the UI.")


if page == "Command Center":
    header(
        "COORDINATED ABUSE / RISK ENGINE",
        "See the ring before the loss.",
        "Temporal behavior finds coordination early. Independent evidence verifies the alert. Deterministic policy turns corroboration into an auditable intervention.",
    )
    st.markdown('<span class="hero-badge">● FINAL WORLD D · FROZEN END-TO-END RESULT</span>', unsafe_allow_html=True)
    st.markdown("###")

    cols = st.columns(4)
    with cols[0]: metric("Pre-abuse ring detection", val(world_d, "pre_abuse", pct), "39 / 40 rings · detector only")
    with cols[1]: metric("Fraud transaction recall", val(world_d, "recall", pct), "Temporal detector · World D")
    with cols[2]: metric("BLOCK precision", val(world_d, "block_precision", pct), "Full system · intervention-positive")
    with cols[3]: metric("Economic cost reduction", val(world_d, "cost_reduction_pct", pct), "vs temporal detector-only")

    st.markdown("###")
    left, right = st.columns([1.12, .88])
    with left:
        panel("The operating loop", "The detector is the opening move — not the final decision.")
        st.markdown('''<div class="flow"><span class="flow-step">TRANSACTION</span><span class="flow-arrow">→</span><span class="flow-step">TEMPORAL ENGINE</span><span class="flow-arrow">→</span><span class="flow-step">GRAPH CONTEXT</span><span class="flow-arrow">→</span><span class="flow-step">EVIDENCE</span><span class="flow-arrow">→</span><span class="flow-step">POLICY</span><span class="flow-arrow">→</span><span class="flow-step">ACTION + AUDIT</span></div><div class="terminal" style="margin-top:.8rem">state(<span class="accent">&lt; T</span>)  →  features  →  score  →  verify  →  decide  →  update(T)<br><br><span class="accent">CAUSAL CONTRACT</span> · current event never enriches its own pre-event score.</div>''', unsafe_allow_html=True)
        end_panel()
    with right:
        panel("Final held-out pulse", "Fresh World D · model and policy frozen before evaluation.")
        st.markdown(f'<div class="big-number">{val(world_d,"pre_abuse",pct)}</div><div class="muted">abuse rings detected before first abuse</div><div style="height:.65rem"></div><div class="section-kicker">EVENTUAL COVERAGE</div><strong style="font-size:1.3rem">{val(world_d,"ring_recall",pct)}</strong><span class="muted"> · 40 / 40 rings</span><div style="height:.65rem"></div><div class="section-kicker">MEDIAN DETECTOR LEAD</div><strong style="font-size:1.3rem">{val(world_d,"median_lead_days",lambda x:f'{float(x):.2f} days')}</strong><div style="height:.65rem"></div><div class="muted">Observed blocked fraudulent amount: <strong>{money(world_d.get("blocked_exposure"))}</strong> · {val(world_d,"blocked_exposure_pct",pct)} of observed fraud exposure.</div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    st.markdown('<div class="section-kicker">GENERALIZATION CHECK</div>', unsafe_allow_html=True)
    b, d, delta = st.columns(3)
    b_pre = float(world_b.get("pre_abuse", 0) or 0)
    d_pre = float(world_d.get("pre_abuse", 0) or 0)
    with b: metric("WORLD B · PRE-ABUSE", pct(b_pre), "fresh detector hold-out")
    with d: metric("WORLD D · PRE-ABUSE", pct(d_pre), "fresh final hold-out")
    with delta: metric("GENERALIZATION DELTA", f"{(d_pre-b_pre)*100:+.1f} pp", "World B → World D")

    st.markdown("###")
    a, b = st.columns(2)
    with a:
        panel("Detector-only operating point", "Broad candidate discovery.")
        st.markdown(f'''<div class="alert-card"><div class="eyebrow">TEMPORAL DETECTOR</div><h3>{val(world_d,"recall",pct)} recall · {val(world_d,"precision",pct)} precision</h3><span class="muted">FPR {val(world_d,"fpr",pct,)} · PR-AUC {val(world_d,"pr_auc",lambda x:f'{float(x):.3f}')} · median lead {val(world_d,"median_lead_days",lambda x:f'{float(x):.2f}d')}</span></div>''', unsafe_allow_html=True)
        end_panel()
    with b:
        panel("Intervention operating point", "Evidence-gated action quality.")
        st.markdown(f'''<div class="alert-card"><div class="eyebrow">FULL SYSTEM</div><h3>{val(world_d,"block_precision",pct)} BLOCK precision</h3><span class="muted">{num(world_d.get("blocks"))} blocks · {num(world_d.get("reviews"))} reviews · cost {money(world_d.get("cost"))}</span></div>''', unsafe_allow_html=True)
        end_panel()

    st.info("The dashboard is a replayable research prototype. Saved artifacts override the bundled frozen snapshot; the UI never retrains or reruns World D.")


elif page == "Live Replay":
    header("REPLAY / OPERATOR MODE", "Watch the engine make a decision.", "Recorded World C alerts make the workflow demonstrable without pretending this console is connected to live payment rails.")
    rows_all = st.session_state.records
    if not rows_all:
        st.warning("No World C verification records found. The replay needs the recorded verifier JSONL.")
        st.stop()

    fcols = st.columns([1, 1, 1, 1.5])
    with fcols[0]:
        if st.button("All", use_container_width=True): st.session_state.replay_filter = "All"; st.session_state.replay_idx = 0; st.rerun()
    with fcols[1]:
        if st.button("Review", use_container_width=True): st.session_state.replay_filter = "Review"; st.session_state.replay_idx = 0; st.rerun()
    with fcols[2]:
        if st.button("Block", use_container_width=True): st.session_state.replay_filter = "Block"; st.session_state.replay_idx = 0; st.rerun()
    with fcols[3]:
        if st.button("High evidence", use_container_width=True): st.session_state.replay_filter = "High evidence"; st.session_state.replay_idx = 0; st.rerun()
    st.caption(f"Replay filter: **{st.session_state.replay_filter}** · the stream is intentionally mixed so the operator sees different outcomes.")

    rows = filtered_records()
    if not rows:
        st.info("No alerts match this filter.")
        st.stop()
    idx = min(st.session_state.replay_idx, len(rows)-1)
    current = rows[idx]; policy = current.get("policy", {}); action = action_for(current)
    det = float(current.get("detector_probability", 0) or 0)
    ver = float(current.get("verification_confidence", 0) or 0)
    evidence = current.get("evidence", []) or []

    c = st.columns([1, 1, 1, 1.65])
    with c[0]: metric("Replay alert", f"{idx+1}/{len(rows)}", "recorded verifier alert")
    with c[1]: metric("Detector score", f"{det:.3f}", f"threshold {float(current.get('detector_threshold',0)):.2f}")
    with c[2]: metric("Verification", f"{ver:.3f}", f"{len(evidence)} evidence items")
    with c[3]: action_card(action, str(current.get("event_id", "")), ver, len(evidence))

    st.markdown("###")
    left, right = st.columns([1.0, 1.0])
    with left:
        panel("Alert timeline", "The event-time story that an operator can follow.")
        ts = str(current.get("timestamp", "—"))
        timeline = [
            ("01", "CURRENT EVENT", ts, "Transaction arrives"),
            ("02", "TEMPORAL DETECTOR", f"score {det:.3f}", "Candidate surfaced"),
            ("03", "INVESTIGATORS", f"{len(evidence)} evidence items", "Independent corroboration"),
            ("04", "POLICY", action.upper(), "Intervention gate evaluated"),
        ]
        for n, title, value, note in timeline:
            st.markdown(f'<div class="alert-card"><span class="eyebrow">{n} · {title}</span><h3 style="margin:.2rem 0">{value}</h3><span class="muted">{note}</span></div>', unsafe_allow_html=True)
        end_panel()
    with right:
        panel("Replay controls", "Move through the recorded alert stream.")
        x, y, z = st.columns(3)
        if x.button("↶ Reset", use_container_width=True): st.session_state.replay_idx = 0; st.rerun()
        if y.button("Next →", use_container_width=True): st.session_state.replay_idx = (idx+1) % len(rows); st.rerun()
        if z.button("+10", use_container_width=True): st.session_state.replay_idx = min(idx+10, len(rows)-1); st.rerun()
        new_idx = st.slider("Alert position", 0, len(rows)-1, idx)
        if new_idx != idx: st.session_state.replay_idx = new_idx; st.rerun()
        st.markdown(f'<div class="terminal"><span class="accent">EVENT</span> {current.get("event_id","—")}<br><span class="accent">ACCOUNT</span> {current.get("account_id","—")}<br><span class="accent">TIMESTAMP</span> {ts}<br><span class="accent">AMOUNT</span> {money(current.get("amount"))}<br><span class="accent">MODEL</span> {current.get("detector_model","temporal-world-a-frozen")}</div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    panel("Decision comparison", "Same alert — first as a detector signal, then as an evidence-backed action.")
    x, y = st.columns(2)
    with x: st.markdown(f'<div class="alert-card"><div class="eyebrow">DETECTOR ONLY</div><h2 style="margin:.25rem 0">ALERT</h2><span class="muted">Risk score {det:.3f} ≥ {float(current.get("detector_threshold",0)):.2f}</span></div>', unsafe_allow_html=True)
    with y: st.markdown(f'<div class="alert-card"><div class="eyebrow">FULL SYSTEM</div><h2 style="margin:.25rem 0">{action.upper()}</h2><span class="muted">Verifier {ver:.3f} · {len(evidence)} evidence items · {str(policy.get("risk_tier","—")).upper()}</span></div>', unsafe_allow_html=True)
    end_panel()

    st.markdown("###")
    panel("Alert queue", "A quick operator view of the mixed replay stream.")
    queue = [{"#": i+1, "event": r.get("event_id"), "account": r.get("account_id"), "detector": round(float(r.get("detector_probability",0) or 0),3), "verification": round(float(r.get("verification_confidence",0) or 0),3), "action": action_for(r).upper()} for i,r in enumerate(rows)]
    st.dataframe(pd.DataFrame(queue).iloc[max(0,idx-4):min(len(queue),idx+9)], hide_index=True, use_container_width=True)
    end_panel()


elif page == "Ring Explorer":
    header("NETWORK INTELLIGENCE", "Find the ring behind the alert.", "A single transaction becomes much more informative when its surrounding relationships and topology are visible.")
    rows = st.session_state.records
    if not rows:
        st.info("No alert records available."); st.stop()
    accounts = sorted({str(r.get("account_id")) for r in rows if r.get("account_id")})
    selected = st.selectbox("Alert account", accounts)
    ring = ring_for_account(selected, manifest)
    nodes, edges, source = [], [], "manifest-backed visual approximation"
    if ring: nodes, edges = manifest_graph(ring)
    live_nodes, live_edges = neo4j_neighbors(selected)
    if live_nodes: nodes, edges, source = live_nodes, live_edges, "Neo4j persisted behavioral graph"

    left, right = st.columns([1.35, .65])
    with left:
        st.markdown('<div class="section-kicker">RELATIONSHIP MAP</div>', unsafe_allow_html=True)
        st.plotly_chart(network_figure(nodes, edges, selected), use_container_width=True, config={"displayModeBar":False})
        st.caption(f"Graph source: **{source}**. Neo4j is optional. The manifest fallback is visual and deterministic; event-time behavioral state remains authoritative for the detector.")
    with right:
        panel("Ring profile", "Structural context around the selected account.")
        if ring:
            metric("Members", num(len(ring.get("account_ids", []))), "accounts")
            metric("Topology", str(ring.get("topology", "—")).upper(), str(ring.get("kind", "—")))
            metric("Strength", str(ring.get("strength", "—")).upper(), "generated ring descriptor")
            metric("Activation", f'{float(ring.get("activation_day",0)):.1f}', "days from world start")
            st.markdown(f'<div class="terminal"><span class="accent">RING</span> {ring.get("ring_id","—")}<br><span class="accent">ACCOUNTS</span> {len(ring.get("account_ids",[]))}<br><span class="accent">EDGES</span> {len(edges)}<br><span class="accent">MODE</span> pre-event investigation</div>', unsafe_allow_html=True)
        else:
            st.info("No manifest ring maps to this account. Try another alert account or connect Neo4j.")
        end_panel()

    st.markdown("###")
    panel("Why the graph matters", "Coordination is a relational and temporal phenomenon.")
    a,b,c = st.columns(3)
    with a: metric("RELATIONSHIPS", num(len(edges)), "visible edges in this view")
    with b: metric("PEERS", num(max(0,len(nodes)-1)), "accounts around alert")
    with c: metric("SIGNAL", "NETWORK + TIME", "not transaction-local only")
    end_panel()


elif page == "Evidence & Response":
    header("INVESTIGATION / RESPONSE", "From suspicion to an auditable action.", "Independent investigators provide structured evidence; deterministic fusion summarizes it; policy decides whether corroboration is sufficient for intervention.")
    rows = st.session_state.records
    if not rows:
        st.info("No verification records available."); st.stop()
    options = [f'{i+1:04d} · {r.get("event_id")} · {action_for(r).upper()}' for i,r in enumerate(rows)]
    choice = st.selectbox("Investigation", options, index=0)
    i = options.index(choice); r = rows[i]
    evidence = r.get("evidence", []) or []; policy = r.get("policy", {}) or {}; fusion = r.get("fusion", {}) or {}
    det = float(r.get("detector_probability",0) or 0); ver = float(r.get("verification_confidence",0) or 0); action = action_for(r)

    a,b,c = st.columns(3)
    with a: metric("Detector", f"{det:.3f}", "frozen temporal score")
    with b: metric("Verification", f"{ver:.3f}", "evidence-fused score")
    with c: action_card(action, str(r.get("event_id","")), ver, len(evidence))

    st.markdown("###")
    left, right = st.columns([1.12, .88])
    with left:
        panel("Evidence ledger", f'{len(evidence)} items · {len(fusion.get("contributing_agent_names", []))} contributing investigators')
        if not evidence:
            st.info("No structured evidence attached to this record.")
        for item in evidence:
            typ = str(item.get("evidence_type", item.get("type", "evidence"))).replace("_", " ").upper()
            strength = str(item.get("strength", "moderate")).lower()
            conf = float(item.get("confidence",0) or 0)
            st.markdown(f'<div class="evidence {strength}"><strong>{typ} · {strength.upper()} · {conf:.2f}</strong><span>{item.get("summary","")}</span></div>', unsafe_allow_html=True)
        end_panel()
    with right:
        panel("Evidence fusion", "Deterministic corroboration — not a calibrated replacement probability.")
        st.plotly_chart(gauge(ver, "Verification confidence"), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div class="terminal">detector          {det:.4f}<br>evidence support  {float(fusion.get("evidence_support",0) or 0):.4f}<br>agent coverage    {float(fusion.get("agent_coverage",0) or 0):.4f}<br>coverage bonus    {float(fusion.get("coverage_bonus",0) or 0):.4f}<br>evidence score    {float(fusion.get("evidence_score",0) or 0):.4f}<br><span class="accent">fused confidence  {ver:.4f}</span></div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    left,right = st.columns([1.0,1.0])
    with left:
        panel("Policy gates", "The model does not get to BLOCK by itself.")
        reasons = policy.get("reason_codes", []) or []
        for reason in reasons:
            st.markdown(f'<div class="alert-card"><span class="action-pill {"block" if action=="block" else "review"}">✓ GATE</span> <strong>{str(reason).replace("_"," ")}</strong></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="terminal"><span class="accent">FINAL ACTION</span>  {action.upper()}<br>risk tier          {str(policy.get("risk_tier","—")).upper()}<br>policy version     {policy.get("policy_version","12b-world-c-frozen-v1")}</div>', unsafe_allow_html=True)
        end_panel()
    with right:
        panel("Responder sandbox", "Local in-memory execution only. No real payment API is called.")
        if st.button("Execute local prototype action", use_container_width=True):
            st.session_state.executed.add(r.get("event_id")); st.rerun()
        executed = r.get("event_id") in st.session_state.executed
        st.markdown(f'<div class="alert-card"><div class="eyebrow">ACTION EXECUTION</div><h3>{"EXECUTED" if executed else "READY"}</h3><span class="muted">idempotency key = event + policy version + action</span></div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    panel("Audit trail", "Operator-facing representation of the immutable/hash-chained audit interface.")
    st.markdown(f'<div class="terminal">POLICY_DECISION   {action.upper()}<br>EVIDENCE_FUSED     {ver:.4f}<br>ACTION_EXECUTED    {"YES" if r.get("event_id") in st.session_state.executed else "PENDING"}<br>IDEMPOTENCY         READY<br>AUDIT_HASH          HASH-CHAIN RECORD</div>', unsafe_allow_html=True)
    end_panel()


else:
    header("EVALUATION / EVIDENCE", "Does it generalize?", "The final result is a frozen measurement, not a dashboard-generated experiment. World D was evaluated after detector, verifier and policy freeze.")
    st.markdown('<span class="hero-badge">◈ WORLD D · FINAL HELD-OUT · NO TUNING AFTER FREEZE</span>', unsafe_allow_html=True)
    st.markdown("###")
    cards = [("Pre-abuse rings", "pre_abuse", pct), ("Recall", "recall", pct), ("Precision", "precision", pct), ("FPR", "fpr", pct), ("PR-AUC", "pr_auc", lambda x:f'{float(x):.3f}'), ("ROC-AUC", "roc_auc", lambda x:f'{float(x):.3f}')]
    c = st.columns(6)
    for col,(label,key,fmt) in zip(c,cards):
        with col: metric(label, val(world_d,key,fmt), "World D")

    st.markdown("###")
    a,b = st.columns(2)
    with a:
        panel("Ring outcomes", "Early detection and eventual detection answer different questions.")
        st.markdown(f'<div class="alert-card"><div class="eyebrow">PRE-ABUSE</div><h2>{val(world_d,"pre_abuse",pct)}</h2><span class="muted">39 / 40 rings before first abuse</span></div><div class="alert-card"><div class="eyebrow">EVENTUAL</div><h2>{val(world_d,"ring_recall",pct)}</h2><span class="muted">40 / 40 rings eventually detected</span></div><div class="alert-card"><div class="eyebrow">LEAD TIME</div><h2>{val(world_d,"median_lead_days",lambda x:f'{float(x):.2f} days')}</h2><span class="muted">median detector lead</span></div>', unsafe_allow_html=True)
        end_panel()
    with b:
        panel("Verifier + responder", "BLOCK is the intervention-positive class; its recall is not detector recall.")
        for label,key,fmt in [("BLOCK precision","block_precision",pct),("BLOCK recall","block_recall",pct),("BLOCK FPR","block_fpr",pct),("Observed exposure blocked","blocked_exposure_pct",pct),("Blocks","blocks",num),("Reviews","reviews",num)]:
            st.markdown(f'<div class="alert-card"><strong>{label}</strong><span class="muted">{val(world_d,key,fmt)}</span></div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    a,b = st.columns(2)
    with a:
        panel("Economic comparison", "Same World D stream · detector-only versus evidence-gated intervention.")
        detector_cost = float(world_d.get("detector_cost") or world_d.get("cost") or 0)
        system_cost = float(world_d.get("system_cost") or 0)
        # Commit 13 stores both costs; fallback snapshot preserves the final values.
        st.markdown(f'<div class="alert-card"><div class="eyebrow">DETECTOR ONLY</div><h2>{money(detector_cost)}</h2></div><div style="text-align:center;font-size:1.6rem;color:#66e6ff">↓</div><div class="alert-card"><div class="eyebrow">FULL SYSTEM</div><h2>{money(system_cost)}</h2><span class="muted">{val(world_d,"cost_reduction_pct",pct)} lower modeled economic cost</span></div>', unsafe_allow_html=True)
        end_panel()
    with b:
        panel("Freeze / integrity", "Critical artifacts are fingerprinted so the result can be reproduced and defended.")
        model_sha = world_d.get("model_sha") or "487b225a80266ecf8f5232a86d26c0b90b4fedc837f827512d387b368b77234f"
        verifier_sha = world_d.get("verifier_sha") or "1e15806514d588777842692afe3e57e25826c0704692556997dd1989b40be529"
        st.markdown(f'<div class="terminal">TEMPORAL MODEL SHA256<br>{model_sha}<br><br>VERIFIER CONFIG SHA256<br>{verifier_sha}<br><br>WORLD D TUNING AFTER FREEZE<br><span class="accent">FALSE</span></div>', unsafe_allow_html=True)
        end_panel()

    st.markdown("###")
    panel("Four-world evaluation design", "Each world has a distinct role; the final held-out world is not a tuning set.")
    st.markdown('''<div class="flow"><span class="flow-step">WORLD A · TRAIN / FREEZE DETECTOR</span><span class="flow-arrow">→</span><span class="flow-step">WORLD B · DETECTOR HOLD-OUT</span><span class="flow-arrow">→</span><span class="flow-step">WORLD C · VERIFIER / POLICY DEVELOPMENT</span><span class="flow-arrow">→</span><span class="flow-step">WORLD D · FINAL END-TO-END HOLD-OUT</span></div>''', unsafe_allow_html=True)
    st.markdown("###")
    cols = st.columns(4)
    world_cards = [("A", "Detector development", "Train + validate + freeze"), ("B", "Detector generalization", "Fresh ecosystem / held-out"), ("C", "Verifier development", "Evidence + policy tuning"), ("D", "Final evaluation", "Everything frozen")]
    for col,(w,title,note) in zip(cols,world_cards):
        with col:
            st.markdown(f'<div class="alert-card"><div class="eyebrow">WORLD {w}</div><h3>{title}</h3><span class="muted">{note}</span></div>', unsafe_allow_html=True)
    end_panel()
