# Known issues before publication

This is a locally prepared Alpha, not a published or production-certified release.

## Current candidate: local verification, Windows hosted gate failed

Implementation `83d1236f1072a2425d1ec6af103111f6c4ef553b` passed the complete
public Windows suite: **1770 passed, 0 failed, 0 errors, 0 skipped**, 263.27
seconds. Start/end HEAD and tree matched; the checkout/index were clean. The
affected six-file regression separately passed 169 cases and the independent
targeted code review was CLEAN. These overlapping counts are not added together.
Reporting-only commits after this implementation do not constitute a rerun.

The new deterministic counterexample proves that shadow rerank could create a
fresh query budget after the original deadline. The original deadline is now
forwarded; no inner budget or physical latency assertion was enlarged. The test
proves the already-expired case, not the unique cause of a past hosted failure.
The existing controller-reserve edge still maps a pre-dispatch exhaustion to
rerank TIMEOUT; this change does not claim to resolve every such status mapping.
Optional rerank remains disabled by default.

State-machine fixtures previously froze monotonic time while durable lease wall
time advanced through a 100 ms expiry. Their clocks are now explicitly controlled,
with real 99/100 ms fencing/expired-unknown coverage. Fake semantic functionality
uses an opt-in logical clock and checks actual dispatch, not an injected ranking.
Real Hook, stall, lock and numerical wall-clock tests keep their original clocks
and limits. Bounded ingress measurements improve future failure evidence without
changing Runtime policy; they do not establish cold-start performance.

The matching private commit is `aeda99894a8dbc4f38ef4f4c48e40accb3134457`,
with the same Git tree as documented candidate `8466dbc40c4b45a14e3d879f54fa2fc39924708a`.
Actual run [39293185327](https://github.com/tsuchiyang/hippocampus-runtime/actions/runs/39293185327)
completed with **Windows 1757 passed, 13 failed, 0 skipped**, 977.99 seconds.
Later Windows build/privacy/wheel steps were skipped, not passed. Ubuntu passed
747 cases with one explicit Windows-only skip (17.09 seconds); its actual wheel
passed all 44 repair/version cases (3.42 seconds).

Current Windows failures include two selected ingress paths, two real benchmark
paths, one real Hook request-arrival check, provider dispatch/settlement and two
unchanged 200 ms assertions (about 203/204 ms). No unique common root cause is
established. A bounded local schema measurement found only about 1–3 ms per
verification in tokenizer cost; it cannot explain all hosted exhaustion cases.
Publication stays on hold. Prior failures below remain preserved; these are not
converted to PASS by local success or a shorter subsequent diagnostic profile.

Private diagnostic run39298125252 used unchanged Runtime source and the 13
failed nodes only. Both fresh layouts passed: system-temp 33.38 seconds and
runner-temp 22.88 seconds. This does not reproduce or close the earlier full
failure. The next full CI uses an explicit isolated RUNNER_TEMP child and
preserves bounded failure-stage evidence; no budget, assertion, clock or Runtime
behavior is changed to obtain a result. Its full outcome is not yet known.

That full run, [39299483666](https://github.com/tsuchiyang/hippocampus-runtime/actions/runs/39299483666),
is now preserved as **Windows 1767 passed, 3 failed**, 341.57 seconds. All three
failures are variants of the evidence-sidecar regression: the new pytest request
argument was not forwarded by its direct test caller, causing TypeError before
the exercised Hook helper started. The original 13 failed nodes passed in this
full run; their historical unique cause is not inferred from this result.
Ubuntu again passed 747 cases/one explicit Windows-only skip and 44 actual-wheel
cases. The two-line caller fix passed its original three cases and the related
real Hook subprocess cases (7 passed, 12.93 seconds), with independent targeted
review CLEAN. This failed full run is not converted to PASS and the corrected
candidate still needs its own full CI result.

The corrected-caller run [39300953035](https://github.com/tsuchiyang/hippocampus-runtime/actions/runs/39300953035)
is preserved as **Windows 1769 passed, 1 failed**, 276.68 seconds. The three
caller cases passed. The sole failure was the real HTTP trickle test receiving
TIMEOUT at 187 ms instead of DEADLINE_EXHAUSTED under a 200 ms budget. That test
did not establish that the complete body returned before testing the post-read
cutoff. A transport timeout can legitimately occur first; the log does not prove
its unique socket phase or operating-system cause. No production classification
is changed or historical failure relabeled. The fixture correction now proves
complete real-body delivery before advancing its local logical clock, and tests
the earlier read-timeout branch separately; neither is physical latency evidence.
The affected adapter file passed 40 cases (17.25 seconds), and the final two
classification cases passed separately (1.26 seconds). An isolated in-memory
mutation bypassing only the post-read cutoff was rejected by the exact original
state assertion. Independent targeted review was CLEAN. Corrected hosted results
remain pending; these focused checks are not a full-suite PASS.

## Previous local Windows verification (preserved)

At implementation commit `fdebc8a63a2d86c3dac68ead78f8ac15df7c9940`, the complete
public Windows suite passed **1767 tests, 0 failed, 0 errors, 0 skipped** in
283.03 seconds. Start/end HEAD matched and the checkout/index were clean.
The separate actual-wheel recovery/version regressions passed all 44 cases;
their overlapping cases are not added to the source-suite count. Targeted
independent review of the seven-file hardening change was CLEAN.

The matching private hosted candidate is
`d96638b859661a2baa88cbac9552099b70918a9d`, with identical Git tree
`a5699ae96dfc26515ad20db8329b979573d17f03`. Its new Actions run is
[39285048474](https://github.com/tsuchiyang/hippocampus-runtime/actions/runs/39285048474).
The actual hosted outcome was **failure**:

- Ubuntu local-core profile: **747 passed, 1 skipped**, 14.74 seconds; the
  separately installed wheel passed 44 cases in 3.04 seconds. Privacy/build and
  actual-wheel import checks succeeded. The skip remains the explicitly
  Windows-only advisory-lock contract.
- Windows full public suite: **1742 passed, 25 failed, 0 skipped**, 1050.24
  seconds. Following build/privacy/wheel steps were skipped after that failure.
  Failures span prompt/compact setup, semantic benchmark/shadow expiry,
  provider admission/start/settlement and lease/control expectations. Two
  existing 200 ms assertions observed 313 ms and 281 ms. These cannot all be
  classified as harmless scheduling noise or as one confirmed Runtime defect.

Earlier status API snapshots reported queued/in-progress states that lagged
the raw execution logs. They are retained as API observations, not evidence
that execution had not started. The raw Ubuntu log proves its work completed
before some queued snapshots. No cause of that status discrepancy is claimed.

All prior failed logs remain unchanged; no identical rerun was started to
replace this failed result. The local success and targeted CLEAN review do not
establish a passing Windows-hosted/public gate. Publication remains on hold.

## Earlier local Windows verification (preserved)

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
