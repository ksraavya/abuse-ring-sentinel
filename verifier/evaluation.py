from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VerificationEvaluationCase:
    """One alert/event evaluated by the verifier harness.

    Ground truth is evaluation-only metadata. It is never passed to the
    verifier or evidence investigators during normal runtime.
    """

    event_id: str
    detector_probability: float
    verification_confidence: float
    is_fraud: bool
    ring_id: str | None = None
    is_pre_abuse: bool = False

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not 0.0 <= self.detector_probability <= 1.0:
            raise ValueError("detector_probability must be in [0, 1]")
        if not 0.0 <= self.verification_confidence <= 1.0:
            raise ValueError("verification_confidence must be in [0, 1]")
        if self.is_pre_abuse and not self.ring_id:
            raise ValueError("pre-abuse cases must have ring_id")


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float


@dataclass(frozen=True)
class VerificationComparison:
    """Detector-only versus detector+verifier gating at one policy threshold."""

    detector_threshold: float
    verifier_threshold: float
    detector: BinaryMetrics
    gated: BinaryMetrics
    fp_reduction: int
    fp_reduction_rate: float
    recall_change: float
    ring_recall_detector: float
    ring_recall_gated: float
    pre_abuse_recall_detector: float
    pre_abuse_recall_gated: float


class VerifierEvaluationHarness:
    """Offline, label-aware evaluation harness for World C development.

    The harness deliberately sits *outside* the verifier decision path. It
    receives detector scores and already-produced fusion scores, then applies
    explicit evaluation thresholds. Ground truth may therefore be used here
    without contaminating event-time evidence collection.

    World C may use this harness to compare detector-only behavior against a
    detector-triggered verifier gate. World D must use a frozen threshold and
    configuration selected before evaluation begins.
    """

    def __init__(self, cases: Iterable[VerificationEvaluationCase]) -> None:
        self.cases = tuple(cases)
        ids = [case.event_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique")

    def evaluate_detector(self, detector_threshold: float) -> BinaryMetrics:
        self._validate_threshold(detector_threshold)
        return self._metrics(
            scores=[case.detector_probability for case in self.cases],
            threshold=detector_threshold,
        )

    def evaluate_gated(
        self,
        detector_threshold: float,
        verifier_threshold: float,
    ) -> BinaryMetrics:
        """Evaluate alerts requiring both detector and verifier thresholds."""
        self._validate_threshold(detector_threshold)
        self._validate_threshold(verifier_threshold)
        y_pred = [
            case.detector_probability >= detector_threshold
            and case.verification_confidence >= verifier_threshold
            for case in self.cases
        ]
        return self._metrics_from_predictions(y_pred)

    def compare(
        self,
        detector_threshold: float,
        verifier_threshold: float,
    ) -> VerificationComparison:
        detector = self.evaluate_detector(detector_threshold)
        gated = self.evaluate_gated(detector_threshold, verifier_threshold)

        detector_ring_ids = self._detected_ring_ids(
            detector_threshold=detector_threshold,
            verifier_threshold=None,
        )
        gated_ring_ids = self._detected_ring_ids(
            detector_threshold=detector_threshold,
            verifier_threshold=verifier_threshold,
        )
        all_ring_ids = {case.ring_id for case in self.cases if case.ring_id}
        detector_pre = self._detected_pre_abuse_ids(
            detector_threshold=detector_threshold,
            verifier_threshold=None,
        )
        gated_pre = self._detected_pre_abuse_ids(
            detector_threshold=detector_threshold,
            verifier_threshold=verifier_threshold,
        )
        all_pre = {
            case.ring_id for case in self.cases if case.ring_id and case.is_pre_abuse
        }

        fp_reduction = detector.fp - gated.fp
        fp_reduction_rate = fp_reduction / detector.fp if detector.fp else 0.0
        recall_change = gated.recall - detector.recall

        return VerificationComparison(
            detector_threshold=detector_threshold,
            verifier_threshold=verifier_threshold,
            detector=detector,
            gated=gated,
            fp_reduction=fp_reduction,
            fp_reduction_rate=fp_reduction_rate,
            recall_change=recall_change,
            ring_recall_detector=(
                len(detector_ring_ids) / len(all_ring_ids) if all_ring_ids else 0.0
            ),
            ring_recall_gated=(
                len(gated_ring_ids) / len(all_ring_ids) if all_ring_ids else 0.0
            ),
            pre_abuse_recall_detector=(
                len(detector_pre) / len(all_pre) if all_pre else 0.0
            ),
            pre_abuse_recall_gated=(
                len(gated_pre) / len(all_pre) if all_pre else 0.0
            ),
        )

    def sweep(
        self,
        detector_threshold: float,
        verifier_thresholds: Iterable[float],
    ) -> tuple[VerificationComparison, ...]:
        """Return a threshold sweep without selecting a winner automatically."""
        return tuple(
            self.compare(detector_threshold, threshold)
            for threshold in verifier_thresholds
        )

    def _detected_ring_ids(
        self,
        *,
        detector_threshold: float,
        verifier_threshold: float | None,
    ) -> set[str]:
        detected: set[str] = set()
        for case in self.cases:
            if not case.ring_id or case.detector_probability < detector_threshold:
                continue
            if verifier_threshold is not None and case.verification_confidence < verifier_threshold:
                continue
            detected.add(case.ring_id)
        return detected

    def _detected_pre_abuse_ids(
        self,
        *,
        detector_threshold: float,
        verifier_threshold: float | None,
    ) -> set[str]:
        detected: set[str] = set()
        for case in self.cases:
            if not case.is_pre_abuse or not case.ring_id:
                continue
            if case.detector_probability < detector_threshold:
                continue
            if verifier_threshold is not None and case.verification_confidence < verifier_threshold:
                continue
            detected.add(case.ring_id)
        return detected

    def _metrics(self, *, scores: list[float], threshold: float) -> BinaryMetrics:
        predictions = [score >= threshold for score in scores]
        return self._metrics_from_predictions(predictions, threshold=threshold)

    def _metrics_from_predictions(
        self,
        predictions: list[bool],
        *,
        threshold: float = 0.0,
    ) -> BinaryMetrics:
        labels = [case.is_fraud for case in self.cases]
        tp = sum(pred and label for pred, label in zip(predictions, labels))
        fp = sum(pred and not label for pred, label in zip(predictions, labels))
        fn = sum((not pred) and label for pred, label in zip(predictions, labels))
        tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return BinaryMetrics(
            threshold=threshold,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
        )

    @staticmethod
    def _validate_threshold(value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")


__all__ = [
    "BinaryMetrics",
    "VerificationComparison",
    "VerificationEvaluationCase",
    "VerifierEvaluationHarness",
]
