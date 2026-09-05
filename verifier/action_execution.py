from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from math import isfinite
from typing import Protocol

from verifier.contracts import VerificationRequest
from verifier.policy import PolicyAction, PolicyDecision


class ActionExecutionStatus(str, Enum):
    """Outcome of an action-execution attempt."""

    EXECUTED = "executed"
    IDEMPOTENT_REPLAY = "idempotent_replay"


@dataclass(frozen=True)
class ExecutionCommand:
    """Normalized command sent from policy to an execution backend."""

    execution_id: str
    idempotency_key: str
    event_id: str
    account_id: str
    action: PolicyAction
    amount: float
    issued_at: datetime
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ActionExecutionReceipt:
    """Immutable result returned by an execution backend."""

    execution_id: str
    idempotency_key: str
    event_id: str
    action: PolicyAction
    status: ActionExecutionStatus
    executed_at: datetime
    message: str


class ActionExecutionError(RuntimeError):
    """Base error for a rejected or failed execution request."""


class ActionConflictError(ActionExecutionError):
    """Raised when the same transaction receives a contradictory action."""


class ActionExecutionBackend(Protocol):
    """Backend contract for applying one normalized action command.

    A production implementation can translate these commands into the payment
    platform's authorization, review-queue, or transaction-blocking APIs. The
    reference implementation below is deliberately in-memory and deterministic
    so 11B can be tested without external side effects.
    """

    def execute(self, command: ExecutionCommand) -> ActionExecutionReceipt:
        """Apply a command exactly once, or return an idempotent replay."""
        ...


class InMemoryActionExecutionBackend:
    """Safe reference backend that records transaction-level action state.

    The backend models the three policy outcomes without touching an external
    payment system. It also prevents contradictory commands for one event:
    retries of the same action are idempotent, while a different action for an
    already-actioned transaction is rejected instead of silently overwriting
    the prior decision.
    """

    def __init__(self) -> None:
        self.allowed_event_ids: set[str] = set()
        self.review_event_ids: set[str] = set()
        self.blocked_event_ids: set[str] = set()
        self._receipts_by_key: dict[str, ActionExecutionReceipt] = {}
        self._action_by_event: dict[str, PolicyAction] = {}

    @property
    def receipts(self) -> tuple[ActionExecutionReceipt, ...]:
        """Return receipts in deterministic insertion order."""
        return tuple(self._receipts_by_key.values())

    def execute(self, command: ExecutionCommand) -> ActionExecutionReceipt:
        previous = self._receipts_by_key.get(command.idempotency_key)
        if previous is not None:
            if previous.event_id != command.event_id or previous.action is not command.action:
                raise ActionConflictError(
                    "idempotency key is already bound to a different action"
                )
            return ActionExecutionReceipt(
                execution_id=previous.execution_id,
                idempotency_key=previous.idempotency_key,
                event_id=previous.event_id,
                action=previous.action,
                status=ActionExecutionStatus.IDEMPOTENT_REPLAY,
                executed_at=previous.executed_at,
                message="Existing execution receipt returned for an idempotent replay.",
            )

        prior_action = self._action_by_event.get(command.event_id)
        if prior_action is not None and prior_action is not command.action:
            raise ActionConflictError(
                f"event {command.event_id!r} already executed as {prior_action.value}"
            )

        if command.action is PolicyAction.ALLOW:
            self.allowed_event_ids.add(command.event_id)
        elif command.action is PolicyAction.REVIEW:
            self.review_event_ids.add(command.event_id)
        elif command.action is PolicyAction.BLOCK:
            self.blocked_event_ids.add(command.event_id)
        else:  # pragma: no cover - defensive guard for future enum additions
            raise ActionExecutionError(f"unsupported policy action: {command.action!r}")

        executed_at = command.issued_at
        receipt = ActionExecutionReceipt(
            execution_id=command.execution_id,
            idempotency_key=command.idempotency_key,
            event_id=command.event_id,
            action=command.action,
            status=ActionExecutionStatus.EXECUTED,
            executed_at=executed_at,
            message=f"Action {command.action.value} executed by in-memory backend.",
        )
        self._receipts_by_key[command.idempotency_key] = receipt
        self._action_by_event[command.event_id] = command.action
        return receipt


class ActionExecutor:
    """Execute an already-issued policy decision through a backend.

    This layer deliberately does not recalculate risk, inspect ground truth,
    collect evidence, or choose ALLOW/REVIEW/BLOCK. Its sole responsibility is
    to turn the immutable 11A policy decision into an explicit execution
    command and hand that command to a backend.
    """

    name = "action-executor"
    version = "11b-initial-v1"

    def __init__(self, backend: ActionExecutionBackend) -> None:
        self.backend = backend

    def execute(
        self,
        request: VerificationRequest,
        decision: PolicyDecision,
        *,
        executed_at: datetime | None = None,
    ) -> ActionExecutionReceipt:
        self._validate(request, decision)
        timestamp = executed_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            raise ValueError("executed_at must be timezone-aware")
        timestamp = timestamp.astimezone(timezone.utc)

        command = ExecutionCommand(
            execution_id=self._execution_id(request, decision),
            idempotency_key=self._idempotency_key(request, decision),
            event_id=request.event_id,
            account_id=request.alert_event.account_id,
            action=decision.action,
            amount=request.alert_event.amount,
            issued_at=timestamp,
            reason_codes=decision.reason_codes,
        )
        return self.backend.execute(command)

    @staticmethod
    def _validate(request: VerificationRequest, decision: PolicyDecision) -> None:
        if decision.policy_version.strip() == "":
            raise ValueError("policy_version must be non-empty")
        if decision.detector_probability != request.detector_probability:
            raise ValueError("decision detector probability must match request")
        if not 0.0 <= decision.detector_probability <= 1.0 or not isfinite(
            decision.detector_probability
        ):
            raise ValueError("decision detector probability must be finite and in [0, 1]")
        if not 0.0 <= decision.verification_confidence <= 1.0 or not isfinite(
            decision.verification_confidence
        ):
            raise ValueError("decision verification confidence must be finite and in [0, 1]")
        if decision.action not in (PolicyAction.ALLOW, PolicyAction.REVIEW, PolicyAction.BLOCK):
            raise ValueError("unsupported policy action")

    @staticmethod
    def _idempotency_key(
        request: VerificationRequest, decision: PolicyDecision
    ) -> str:
        material = "|".join(
            (
                request.event_id,
                decision.policy_version,
                decision.action.value,
            )
        )
        return sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _execution_id(
        request: VerificationRequest, decision: PolicyDecision
    ) -> str:
        material = "|".join(
            (
                "11b",
                request.event_id,
                decision.policy_version,
                decision.action.value,
            )
        )
        return f"exec-{sha256(material.encode('utf-8')).hexdigest()[:24]}"


__all__ = [
    "ActionConflictError",
    "ActionExecutionBackend",
    "ActionExecutionError",
    "ActionExecutionReceipt",
    "ActionExecutionStatus",
    "ActionExecutor",
    "ExecutionCommand",
    "InMemoryActionExecutionBackend",
]
