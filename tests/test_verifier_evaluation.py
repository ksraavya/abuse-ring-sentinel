import pytest

from verifier.evaluation import VerificationEvaluationCase, VerifierEvaluationHarness


def cases():
    return [
        VerificationEvaluationCase("f1", 0.80, 0.90, True, "r1"),
        VerificationEvaluationCase("f2", 0.70, 0.40, True, "r1"),
        VerificationEvaluationCase("f3", 0.60, 0.85, True, "r2", True),
        VerificationEvaluationCase("l1", 0.90, 0.20, False),
        VerificationEvaluationCase("l2", 0.20, 0.80, False),
        VerificationEvaluationCase("l3", 0.05, 0.90, False),
    ]


def test_detector_metrics_are_computed_from_labels_only():
    result = VerifierEvaluationHarness(cases()).evaluate_detector(0.50)
    assert (result.tp, result.fp, result.fn, result.tn) == (3, 1, 0, 2)
    assert result.precision == pytest.approx(0.75)
    assert result.recall == pytest.approx(1.0)


def test_gated_metrics_require_detector_and_verifier():
    result = VerifierEvaluationHarness(cases()).evaluate_gated(0.50, 0.50)
    assert (result.tp, result.fp, result.fn, result.tn) == (2, 0, 1, 3)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(2 / 3)


def test_compare_reports_fp_reduction_and_recall_change():
    result = VerifierEvaluationHarness(cases()).compare(0.50, 0.50)
    assert result.fp_reduction == 1
    assert result.fp_reduction_rate == pytest.approx(1.0)
    assert result.recall_change == pytest.approx(-1 / 3)
    assert result.ring_recall_detector == pytest.approx(1.0)
    assert result.ring_recall_gated == pytest.approx(1.0)
    assert result.pre_abuse_recall_detector == pytest.approx(1.0)
    assert result.pre_abuse_recall_gated == pytest.approx(1.0)


def test_sweep_does_not_choose_a_threshold():
    result = VerifierEvaluationHarness(cases()).sweep(0.50, [0.2, 0.5, 0.8])
    assert [row.verifier_threshold for row in result] == [0.2, 0.5, 0.8]


def test_duplicate_event_ids_are_rejected():
    with pytest.raises(ValueError, match="event_id"):
        VerifierEvaluationHarness(
            [
                VerificationEvaluationCase("same", 0.5, 0.5, True),
                VerificationEvaluationCase("same", 0.2, 0.2, False),
            ]
        )


def test_pre_abuse_requires_ring_id():
    with pytest.raises(ValueError, match="ring_id"):
        VerificationEvaluationCase("x", 0.5, 0.5, True, is_pre_abuse=True)


def test_thresholds_are_validated():
    harness = VerifierEvaluationHarness(cases())
    with pytest.raises(ValueError):
        harness.evaluate_detector(1.1)
    with pytest.raises(ValueError):
        harness.evaluate_gated(0.5, -0.1)

def test_sweep_preserves_threshold_order():
    result = VerifierEvaluationHarness(cases()).sweep(0.50, [0.8, 0.2, 0.5])
    assert [row.verifier_threshold for row in result] == [0.8, 0.2, 0.5]
