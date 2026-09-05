# Coordinated Abuse Risk Manager

> **Temporal behavior finds the ring. Independent evidence verifies it. Policy decides. The audit trail remembers.**

A defense-first, event-time risk system for detecting coordinated abuse rings that look legitimate at the individual-transaction level — then verifying the alert with independent evidence and turning corroboration into an auditable intervention.

---

## The 30-second version

Most transaction-risk systems ask:

> *"Does this transaction look suspicious?"*

Coordinated abuse asks a harder question:

> *"Do these accounts look suspicious together — and can we detect that coordination before the first abusive transaction occurs?"*

This project is built around that second question. It combines transaction-local signals → historical infrastructure → evolving behavioral graph → temporal features → independent investigators → deterministic evidence fusion → policy gates → idempotent action execution → audit trail.

The final detector was frozen on World A, tested on held-out World B, and the complete verifier/responder stack was tuned only on World C — before a final end-to-end evaluation on completely fresh World D.

> **This is a controlled research prototype, not a production payment connection.** The worlds are synthetic, the dashboard is an interactive replay, and observed blocked amounts are not automatically equivalent to realized monetary savings. A defensible prototype should be clear about where its evidence ends.

---

## Final held-out World D results

| Outcome | Result |
|---|---|
| Pre-abuse ring detection | **39 / 40 · 97.50%** |
| Eventual ring detection | **40 / 40 · 100%** |
| Fraud transaction recall | 87.34% |
| Fraud transaction precision | 22.94% |
| Transaction FPR | 0.338% |
| PR-AUC | 0.763 |
| ROC-AUC | **0.988** |
| Median detector lead time | **5.73 days** |
| BLOCK precision | **77.40%** |
| BLOCK recall | 67.90% |
| BLOCK FPR | 0.0229% |
| Eventual rings blocked | 40 / 40 · 100% |
| Rings blocked before abuse | 19 / 40 · 47.50% |
| Observed fraudulent exposure blocked | ₹3.209M · 69.20% of observed fraud exposure |
| Economic cost | ₹7.151M → ₹5.803M |
| Cost reduction | **18.86%** |

The headline is not "an ML model has a high AUC."

The headline is: on a fresh held-out world, the temporal detector identified **39 of 40 abuse rings before their first abusive transaction**. The verifier/responder then moved the operating point toward higher-confidence intervention — **77.4% BLOCK precision** and **18.86% lower modeled economic cost** than detector-only operation.

---

## Why this problem is interesting

Coordinated abuse is fundamentally different from isolated fraud.

A single account may have an ordinary transaction amount, a normal payment channel, a familiar merchant, no obviously malicious device, and no individually extreme behavior.

But the network around it may reveal:

- several accounts transacting with the same counterparties
- bursts of newly created relationships
- synchronized activity across peers
- convergence on the same merchants
- shared device or IP infrastructure
- behavioral acceleration over time
- a sequence that becomes suspicious only when historical context is considered

That creates the central hypothesis:

> **Coordination is often visible in relationships and temporal behavior before it is visible in any individual transaction.**

The system therefore does not jump straight from "ML score" to "BLOCK." Instead, it builds an evidence ladder:

```
                     CURRENT TRANSACTION
                            │
                            ▼
                 ┌──────────────────────┐
                 │  TEMPORAL DETECTOR   │
                 │  26 frozen features  │
                 └──────────┬───────────┘
                            │
                     alert / no alert
                            │
                            ▼
              ┌───────────────────────────┐
              │    INDEPENDENT EVIDENCE   │
              │  Ring Investigator        │
              │  Infrastructure Invest.   │
              │  Context Investigator     │
              └─────────────┬─────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  EVIDENCE FUSION     │
                 │  deterministic       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  POLICY ENGINE       │
                 │  ALLOW/REVIEW/BLOCK  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ ACTION + AUDIT TRAIL │
                 │ idempotent execution │
                 └──────────────────────┘
```

---

## What makes this more than a classifier

### 1. The detector sees time

At transaction time T, the system reads infrastructure and behavioral state strictly before T, computes features, scores the transaction, runs the verifier, executes the policy decision, and only then updates graph/state with the current event.

```
state(< T)
   │
   ├──► feature extraction
   ├──► detector score
   ├──► verifier evidence
   ├──► policy decision
   └──► state update(T)
```

This prevents the classic temporal leakage mistake. Neo4j is not the anti-leakage mechanism — event-time state discipline is.

### 2. The detector and verifier have different jobs

| | Detector only | Full system |
|---|---|---|
| Precision | 22.94% | **77.40% BLOCK** |
| Recall | 87.34% | 67.90% BLOCK |
| Economic cost | ₹7.151M | **₹5.803M** |

The full system is not claimed to "improve recall." It changes the operating point from broad high-recall alerting toward more selective, evidence-backed intervention.

---

## The evidence ladder

### Baseline A — transaction-local

How far can we get without knowing anything about the account's network? Baseline A uses transaction-local information only — the control experiment for the hypothesis that coordinated abuse cannot be adequately captured by isolated transaction characteristics.

**World B results:**

| Metric | Value |
|---|---|
| Precision | 0% |
| Recall | 0% |
| PR-AUC | 0.0026 |
| ROC-AUC | 0.656 |
| Rings detected | 0 / 40 |
| Economic cost | ₹15.447M |

This is not presented as "ML doesn't work." It is presented as evidence that transaction-local information alone is poorly matched to this coordinated-abuse problem.

### Baseline B — static infrastructure

Adds static infrastructure relationships such as shared device and IP prefixes.

**World B results:**

| Metric | Value |
|---|---|
| Precision | 0% |
| Recall | 0% |
| PR-AUC | 0.0069 |
| ROC-AUC | 0.832 |
| Rings detected | 0 / 40 |
| Economic cost | ₹15.441M |

The ROC-AUC improvement is meaningful — static relationships contain signal — but at the frozen operating threshold, that signal was still insufficient to produce useful ring detection.

### Temporal detector — evolving behavior

Combines 10 transaction-local features and 16 locked temporal/behavioral features. Exactly **26 features** enter the classifier. The seven raw Baseline-B infrastructure columns are deliberately not appended — this keeps the evidence ladder interpretable.

```
Baseline A       transaction-local only
    │
    ▼
Baseline B       + static infrastructure
    │
    ▼
Temporal         + historical context
                 + evolving behavior
                 + temporal dynamics
```

Representative temporal signals include transaction history over rolling windows, P2P relationship novelty, edge-creation velocity, cluster-level synchronization, merchant overlap, behavioral acceleration, and contextual deviation from recent behavior.

---

## The verifier

### Ring Investigator

Looks for relationship-level evidence including ring structure, behavioral acceleration, peer synchrony, and merchant convergence. Bounded and deterministic — not scanning an unbounded graph.

### Infrastructure Investigator

Looks for shared current device and IP infrastructure and overlapping infrastructure among accounts. Intentionally conservative — shared infrastructure is evidence of coordination, not proof of abuse.

### Context Investigator

Adds account and transaction context including merchant novelty, recent P2P vs merchant composition, unusual amounts relative to recent behavior, and weak contextual P2P history.

### Evidence fusion

Deterministic and auditable. No LLM decides whether money should be blocked.

| Evidence type | Weight |
|---|---|
| Ring structure | 0.24 |
| Behavioral acceleration | 0.18 |
| Peer synchrony | 0.14 |
| Merchant convergence | 0.12 |
| Infrastructure sharing | 0.12 |
| Temporal context | 0.08 |
| Account context | 0.07 |
| Infrastructure churn | 0.05 |

Strength multipliers: `weak = 0.35` · `moderate = 0.65` · `strong = 1.00`

The fused value is an auditable verification score, not a calibrated probability.

### Policy and response

A BLOCK requires multiple gates: detector score above the frozen block threshold, verifier confidence above the frozen verifier threshold, sufficient independent-agent coverage, and strong evidence present. Detector evidence alone cannot trigger a block.

```
model says:     "this deserves attention"
verifier says:  "here is corroborating evidence"
policy says:    "the evidence crosses the intervention gates"
executor says:  "the action was executed once"
audit says:     "here is exactly what happened"
```

### Action execution and auditability

Every execution is tied to `event_id + policy_version + action`. Contradictory actions are rejected. Repeated executions are recognized as idempotent replays. The audit layer records all decision and execution events using immutable hash-chained records.

---

## The four-world evaluation design

| World | Purpose | Can tune detector? | Can tune verifier/policy? |
|---|---|---|---|
| A | Detector development | Yes | No |
| B | Detector held-out evaluation | No | No |
| C | Verifier/responder development | No | Yes |
| D | Final end-to-end held-out evaluation | No | No |

### World A

Used for temporal detector training, validation, threshold selection, and freeze.

Frozen temporal model SHA256: `487b225a80266ecf8f5232a86d26c0b90b4fedc837f827512d387b368b77234f`

### World B

Fresh ecosystem — new legitimate entities, new hard negatives, new rings, new infrastructure.

| Metric | World B |
|---|---|
| Pre-abuse ring detection | 37 / 40 · 92.50% |
| Transaction recall | 85.78% |
| Eventual ring detection | 40 / 40 |
| ROC-AUC | 0.9825 |

### World C

Used only to develop the verifier and response policy. The frozen World C configuration achieved 97.5% eventual ring recall and 74.45% BLOCK precision on the World C replay.

### World D

Fresh final held-out ecosystem. No detector retraining. No verifier tuning. No policy tuning after evaluation begins.

---

## Final World D results — detail

### Temporal detector

| Metric | Value |
|---|---|
| Transactions | 2,956,587 |
| Fraud transactions | 3,405 (0.1152%) |
| Threshold | 0.010 |
| Precision | 22.94% |
| Recall | 87.34% |
| FPR | 0.338% |
| PR-AUC | 0.763 |
| ROC-AUC | 0.988 |
| TP / FP / FN | 2,974 / 9,992 / 431 |
| Economic cost | ₹7,151,000 |
| Eventual rings detected | 40 / 40 |
| Pre-abuse rings detected | 39 / 40 · 97.50% |
| Median detector lead time | 5.73 days |
| Alerted fraud exposure | ₹4.113M · 88.69% of observed fraud exposure |

### Temporal + verifier + responder

12,966 alerts → 9,979 reviews · 2,987 blocks

| Metric | Value |
|---|---|
| BLOCK precision | 77.40% |
| BLOCK recall | 67.90% |
| BLOCK FPR | 0.0229% |
| BLOCK TP / FP / FN | 2,312 / 675 / 1,093 |
| Economic cost | ₹5,802,500 |
| Eventual rings blocked | 40 / 40 |
| Pre-abuse rings blocked | 19 / 40 · 47.50% |
| Median block lead time | 0.77 days |
| Blocked fraud exposure | ₹3.209M · 69.20% of total observed fraud exposure |

### Generalization: World B → World D

| Metric | World B | World D |
|---|---|---|
| Pre-abuse ring detection | 92.50% | **97.50%** |
| Transaction recall | 85.78% | 87.34% |
| Eventual ring detection | 40/40 | 40/40 |
| ROC-AUC | 0.9825 | **0.9877** |

The behavioral coordination signature of rings is stable across independently generated world realizations.

---

## Reproducibility and freeze integrity

| Artifact | SHA256 |
|---|---|
| Temporal detector | `487b225a80266ecf8f5232a86d26c0b90b4fedc837f827512d387b368b77234f` |
| World D verifier configuration | `1e15806514d588777842692afe3e57e25826c0704692556997dd1989b40be529` |

World D tuning after freeze: **FALSE**

---

## Technology stack

- **Python** — core implementation
- **Apache Kafka** — event streaming
- **Neo4j** — persisted graph representation and investigation
- **LightGBM** — temporal detector
- **scikit-learn** — evaluation metrics
- **pandas / NumPy** — data and feature processing
- **Pydantic** — event and evidence contracts
- **Streamlit + Plotly** — operator-style dashboard

---

## Running the project

```bash
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

The dashboard operates from bundled final-result snapshots. If generated artifacts are present, they take precedence over the snapshot.

---

## Known limitations and next steps

**Synthetic-to-real transfer** — real coordinated abuse has messier distributions, adversarial adaptation, and incomplete labels. A production deployment would require real-world backtesting, drift monitoring, analyst feedback loops, calibration, and staged intervention rollout.

**Graph provenance** — the dashboard's manifest-backed graph is a visualization fallback, not a reconstruction of every observed event edge.

**Intervention policy** — the current policy is intentionally simple. A production system would introduce escalation queues, account-level holds, merchant controls, human review workflows, and policy version management.

**Explanation layer** — an LLM can be useful for turning structured evidence into analyst explanations, but should remain downstream of the deterministic evidence and policy layers.

---

## What the system does not claim

- Does not connect to production payment rails
- Does not use real customer data
- Worlds are synthetic; dashboard is an interactive replay prototype
- Neo4j is optional for the demonstration layer
- Observed blocked transaction amount is not automatically equivalent to realized monetary savings
- Evidence fusion score is not a calibrated probability
- Detector recall and BLOCK recall have different denominators

---

## The core result

Can coordinated abuse be detected while the individual transactions still look legitimate?

On the final held-out World D:

- **39 of 40** abuse rings detected before their first abusive transaction
- **87.34%** transaction-level recall at **0.338%** FPR
- **77.40%** BLOCK precision with **18.86%** lower modeled economic cost vs detector-only operation

Don't just ask whether a transaction looks bad. Look for the behavior of the network around it, verify that suspicion with independent evidence, and only then decide what action is justified.

---

*Built for the Razorpay AI Buildathon · Loss class: Coordinated abuse / abuse rings*