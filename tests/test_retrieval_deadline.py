from __future__ import annotations

import pytest

from codex_memory.retrieval.deadline import DeadlineBudget
from codex_memory.retrieval.errors import RetrievalValidationError


def test_deadline_budget_uses_one_monotonic_total_budget() -> None:
    clock = [1_000_000_000]
    budget = DeadlineBudget.start(8_000, now_ns=lambda: clock[0])

    assert (budget.started_ns, budget.deadline_ns) == (1_000_000_000, 9_000_000_000)
    assert budget.remaining_ms(now_ns=lambda: clock[0]) == 8_000
    clock[0] += 1_250_000_000
    assert budget.remaining_ms(now_ns=lambda: clock[0], cap_ms=500) == 500
    assert budget.remaining_ms(now_ns=lambda: clock[0], reserve_ms=6_750) == 0


def test_deadline_budget_rejects_invalid_arithmetic_and_expires_at_deadline() -> None:
    budget = DeadlineBudget.start(100, now_ns=lambda: 0)
    assert budget.remaining_ms(now_ns=lambda: 100_000_000) == 0
    assert budget.remaining_ms(now_ns=lambda: 101_000_000) == 0

    with pytest.raises(RetrievalValidationError):
        DeadlineBudget.start(0, now_ns=lambda: 0)
    with pytest.raises(RetrievalValidationError):
        budget.remaining_ms(now_ns=lambda: 0, cap_ms=0)
    with pytest.raises(RetrievalValidationError):
        budget.remaining_ms(now_ns=lambda: 0, reserve_ms=-1)
