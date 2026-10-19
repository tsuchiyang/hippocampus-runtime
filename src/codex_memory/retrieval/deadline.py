"""Monotonic, total deadline budgets for optional provider work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .errors import RetrievalValidationError


_NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class DeadlineBudget:
    started_ns: int
    deadline_ns: int

    @classmethod
    def start(cls, total_ms: int, *, now_ns: Callable[[], int]) -> "DeadlineBudget":
        if isinstance(total_ms, bool) or not isinstance(total_ms, int) or total_ms < 1:
            raise RetrievalValidationError("total_ms must be a positive integer")
        started_ns = now_ns()
        if isinstance(started_ns, bool) or not isinstance(started_ns, int) or started_ns < 0:
            raise RetrievalValidationError("now_ns must return a non-negative integer")
        return cls(started_ns, started_ns + total_ms * _NS_PER_MS)

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.started_ns, self.deadline_ns)):
            raise RetrievalValidationError("deadline values must be non-negative integers")
        if self.deadline_ns < self.started_ns:
            raise RetrievalValidationError("deadline cannot precede start")

    def remaining_ms(
        self,
        *,
        now_ns: Callable[[], int],
        cap_ms: int | None = None,
        reserve_ms: int = 0,
    ) -> int:
        if cap_ms is not None and (isinstance(cap_ms, bool) or not isinstance(cap_ms, int) or cap_ms < 1):
            raise RetrievalValidationError("cap_ms must be a positive integer or null")
        if isinstance(reserve_ms, bool) or not isinstance(reserve_ms, int) or reserve_ms < 0:
            raise RetrievalValidationError("reserve_ms must be a non-negative integer")
        current_ns = now_ns()
        if isinstance(current_ns, bool) or not isinstance(current_ns, int) or current_ns < 0:
            raise RetrievalValidationError("now_ns must return a non-negative integer")
        available_ns = self.deadline_ns - current_ns - reserve_ms * _NS_PER_MS
        remaining = max(0, available_ns // _NS_PER_MS)
        return min(remaining, cap_ms) if cap_ms is not None else remaining


class AbsoluteDeadlineMilliseconds(int):
    """Positive millisecond view that retains its originating absolute budget.

    It remains an ``int`` for the provider protocol's compatibility surface,
    while deadline-aware adapters can keep the original monotonic cutoff
    instead of starting a fresh relative budget at method entry.
    """

    absolute_deadline: DeadlineBudget

    def __new__(
        cls, remaining_ms: int, absolute_deadline: DeadlineBudget
    ) -> "AbsoluteDeadlineMilliseconds":
        if (
            isinstance(remaining_ms, bool)
            or not isinstance(remaining_ms, int)
            or remaining_ms < 1
        ):
            raise RetrievalValidationError(
                "remaining_ms must be a positive integer"
            )
        if not isinstance(absolute_deadline, DeadlineBudget):
            raise RetrievalValidationError(
                "absolute_deadline must be DeadlineBudget"
            )
        value = int.__new__(cls, remaining_ms)
        value.absolute_deadline = absolute_deadline
        return value
