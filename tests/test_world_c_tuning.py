from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from verifier.evidence_fusion import EvidenceFusionConfig
from verifier.policy import AutoResponderPolicyConfig
from verifier.world_c_tuning import TuningCase, evaluate_candidate


def case(*, fraud: bool, ring: str | None, probability: float, confidence: float, account: str, amount: float = 100.0):
    ts = datetime(2026, 1, 2, tzinfo=timezone.utc)
    evidence = [{
        "evidence_type": "ring_structure", "strength": "strong", "confidence": 0.9,
        "source_agent": "ring-investigator",
    }, {
        "evidence_type": "infrastructure_sharing", "strength": "strong", "confidence": 0.9,
        "source_agent": "infrastructure-investigator",
    }]
    return TuningCase("evt-"+account, ts, account, amount, probability, confidence, 2, 2, tuple(evidence), fraud, ring)


def test_candidate_blocks_only_when_gate_is_met():
    rows = [case(fraud=True, ring="r1", probability=.60, confidence=.60, account="a1"),
            case(fraud=False, ring=None, probability=.15, confidence=.20, account="a2")]
    result = evaluate_candidate(rows, fusion=EvidenceFusionConfig(), policy=AutoResponderPolicyConfig(), ring_members={"r1":{"a1"}})
    assert result.blocks == 1
    assert result.blocked_fraud_transactions == 1


def test_pre_abuse_requires_block_before_first_fraud():
    rows = [case(fraud=False, ring=None, probability=.60, confidence=.60, account="a1"),
            case(fraud=True, ring="r1", probability=.60, confidence=.60, account="a1")]
    rows[1] = TuningCase(rows[1].event_id + "-fraud", rows[0].timestamp.replace(hour=1), rows[1].account_id, rows[1].amount, rows[1].detector_probability, rows[1].verification_confidence, rows[1].agent_coverage, rows[1].strong_evidence_count, rows[1].evidence, rows[1].is_fraud, rows[1].ring_id)
    result = evaluate_candidate(rows, fusion=EvidenceFusionConfig(), policy=AutoResponderPolicyConfig(), ring_members={"r1":{"a1"}})
    assert result.pre_abuse_recall == 1.0


def test_world_c_tuning_does_not_modify_detector_threshold():
    cfg = AutoResponderPolicyConfig(block_detector_threshold=.3, policy_version="x")
    assert cfg.detector_alert_threshold == .01
