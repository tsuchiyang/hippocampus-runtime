# Hippocampus Runtime: Alpha source provenance

The public brand is Hippocampus Runtime（海马体）, previously Codex Memory
Runtime. This does not rename the `codex-memory-runtime` distribution,
`codex_memory` import, `codex-memory-hook` entry point or existing data paths.
Historical evidence and contributor attribution keep their original bindings.

Runtime base: `ee2b0a3e445e9b226f3e9d843817d0d76ec16971`.
Software export reference: `548d06edd481d1d7d08bfe1509238695dd81da1a`.

These are provenance identifiers, not a promise that private engineering Git
objects, reports or test artifacts are publicly retrievable. The public repository
starts a new history and does not share an original Git object database, worktree,
remote, refs, hooks or authentication. The original project/history is preserved.

The original export contained all 72 Runtime Python source modules, with only
`__init__.py` version metadata changed from 0.1.0 to 0.1.0a1. Alpha hardening
subsequently changed `retrieval/provider_health.py` and `retrieval/service.py`
to carry the original absolute deadline through health-store lock preparation
and retries, rather than restarting a sampled relative allowance. The other
69 modules retain the base implementation bytes. Package metadata, user-facing
documentation, test selection and CI are prepared specifically for this Alpha.
The new wheel/hash must not be confused with the earlier 0.1.0 blind candidate.
Fourteen synthetic security-test literals are expressed as constant concatenation
so automated credential scans do not mistake deliberate negative fixtures for
live credentials. Their decoded values and runtime assertions are retained.
The old portable installer/selftest, which require a separate manifest-driven
bundle, are not shipped as incomplete installers here; standard Python wheel/
sdist installation is the supported Alpha path. `portable/verify.py` remains as
the inventory-validation implementation used by its safety regression tests.

Before the original Alpha export, the base had independent Linux
synthetic repair retesting and local Windows installed-entry verification,
including a single synthetic-session real-host/compact cycle. Those results
remain bound to that base and the exact host/environment, not retroactively
certification of the Alpha build or every host. A test workflow in this tree is
not a claim that hosted CI has executed.

No historical score, private memory, credentials, raw transcripts, local Hook
configuration, formal acceptance database or model weights are part of this
distribution. Omitted historical acceptance tests retain their original outcomes;
see [test selection](TEST_SELECTION.json). Nothing in this export converts a
prior failed or uncertified gate into PASS.

The owner approved preparing an MIT-licensed public Alpha. Proposed attribution
is “Codex Memory Runtime contributors”; public push still requires the owner's
final publication approval and confirmation of attribution/rights. Automated
source/notice scanning cannot establish legal title or an exhaustive license audit.
