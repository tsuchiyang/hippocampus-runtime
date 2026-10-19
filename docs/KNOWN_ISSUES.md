# Known issues before publication

This is a locally prepared Alpha, not a published or production-certified release.

## Current local Windows verification

The complete public Windows suite at implementation commit
`5b697c4916b51f39941f5846823b9a489393ce59` finished with **1763 passed,
0 failed, 0 skipped**. Start/end HEAD matched and the checkout was clean.
The original rerank test retained its 180 ms internal budget, 250 ms outer
assertion, provider/completion checks and exact Legacy comparison.

The confirmed fix carries the request's absolute deadline into the provider-health
lock context and anchors the existing physical wait before path preparation.
Admission and settlement cannot restart their sampled allowance after preparation.
The physical wait cap remains effective when a test injects a frozen budget clock.
Path/descriptor validation, health settlement and provider reserve constants remain.

Two new real-OS-lock/controlled-clock regressions failed before the fix: admission
reached 200 ms and settlement 190 ms against the original 180 ms total. They now
respect their original 120 ms and 170 ms cutoffs. These are deterministic scheduling
tests, not measurements of Windows wall-clock latency or reenactments of history.

This local regression gate has passed. Public push, hosted CI, platform-wide
certification and production support are separate decisions and are not claimed.

## GitHub-hosted private staging: publication is still on hold

The hosted snapshot `f00e2b3b8ed0560a208e7801ee98e251b75ead33` has the same
193-file Git tree as local candidate `a63249cd91f75cbfff045814e6e70ca81ceb7783`.
Its actual hosted results are:

- Ubuntu local-core profile: **747 passed, 1 skipped**. The skip is
  `test_windows_lock_contention_is_bounded_and_lock_is_released`, explicitly
  marked `Windows advisory lock contract`. The installed-wheel regressions
  separately passed all 44 cases; do not add overlapping cases to the source total.
- Windows full public suite: **1754 passed, 9 failed, 0 skipped**. Later build
  and installed-wheel steps did not run after the source-test failure.

The Windows failures cover selected-host delivery, optional compact enhancement,
semantic benchmark/shadow deadline handling and provider status/dispatch
preconditions under short budgets. Earlier private snapshots also failed Windows
tests (1753 passed/10 failed and 1755 passed/8 failed). All failed logs remain
preserved. The previously repaired rerank test was not among those failures.
Do not call the new failures harmless machine jitter or claim that later passing
nodes close them. The exact causes and any required fixture versus Runtime fixes
still require evidence; no budget or numerical assertion has been relaxed.

Linux fixture corrections preserve native case-sensitive scope isolation instead
of assuming Windows case-folding on every OS. Runtime source and dependency pins
were not changed to obtain that Linux result. Neither Linux success nor the prior
local Windows full suite establishes a passing Windows-hosted gate. This remains
an unpublished Alpha candidate, not a production or public-release approval.

### Follow-up Windows hardening candidate

A deterministic regression proved that repeatedly parsing source-frozen expected
DDL can consume a short provider budget even after the same oracle has already
been established. Only the immutable expected definitions are now reused;
current schema/DDL, columns, event history and projections remain revalidated.
New tests reject actual DDL/projection tampering after warming and reject
mutation of the shared expected mapping. Original deadlines, reservations and
assertions are unchanged.

The connection-unavailable test now explicitly constructs a connection refusal
at the socket boundary instead of assuming a closed Windows port will refuse
before its 100 ms cutoff. This is an identified fixture precondition correction,
not a production status reclassification. Real loopback read-timeout and count
validation remain. These changes do not prove the unique causes of the previous
nine hosted failures, nor do focused local passes establish a hosted full-suite
PASS. A new matching hosted run is still required.

## Preserved historical failure

The complete Windows public suite at implementation commit
`fb36cb50e0971b4c439a2e8bdf7b68a54fe3b734` finished with **1760 passed,
1 failed, 0 skipped**. The failure is:

```text
tests/test_retrieval_query.py::test_optional_rerank_orchestration_returns_legacy_result_inside_deadline
```

The test uses an internal 180 ms budget and an unchanged 250 ms outer assertion.
The observed call took 266 ms. Its unique historical cause is not established. One subsequent
fixed-boundary instrumented diagnostic completed in 140 ms, but that does not
cancel the original failure, prove a cause, or establish an unmodified full-suite
PASS. That diagnostic did not change the implementation. The later budget fix
above establishes a reachable defect, not proof of the historical event's unique
cause. Optional rerank is not enabled by default.

An earlier preparation snapshot passed 1760 tests; it predates the additional
historical-path audit regression and must not be reported as the final suite.
All original failed logs and intermediate regression results are preserved locally;
the new successful result does not overwrite them.

## Other evidence boundaries

- Installed-wheel checks are recorded separately from the source full suite.
  Their overlapping test cases are not added to its unique test total.
- Windows and Linux CI results must be checked in the matching GitHub Actions run;
  a workflow definition alone is not evidence of successful hosted execution.
  The Linux job uses an explicitly limited local-core profile, not Windows-only
  injection certification.
- No new real-host/model acceptance was run for Alpha version `0.1.0a1`.
  Prior host results keep their own Runtime and host version bindings.
- The privacy checker is a bounded heuristic, not proof of exhaustive secret
  absence, legal ownership or universal safety.
- Cooperative deadlines do not prove hard interruption of arbitrary operating
  system/filesystem I/O, or a universal latency guarantee under machine stalls.
- SPDX MIT metadata and contributor attribution are prepared. The owner confirmed
  independent personal development without known restricted external source;
  final public publication remains a separate owner decision.

Documentation added after the tested implementation does not constitute another
full-suite execution. See [testing scope](TESTING.md) and [provenance](PROVENANCE.md).
