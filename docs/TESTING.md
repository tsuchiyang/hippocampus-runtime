# Public Alpha tests and evidence boundaries

## Run the default suite

```sh
uv sync --locked
uv run --no-sync pytest -q -rs
```

The command above is the full public Windows suite. Linux's explicitly supported
local-core CI profile is enumerated in `tests/linux-core.txt` and runs as:

```sh
uv run --no-sync pytest -q -rs $(cat tests/linux-core.txt)
```

That Linux job does not claim the Windows-only injection/audit tests passed.
The same list can be collected on any platform, but a Windows run of it is not
a Linux execution. The current preparation machine cannot execute Linux; no
Linux hosted CI success is claimed before publication.

Use Python 3.12, Git and a writable local temporary directory. Do not run the suite
with real project memory, a private host HOME or model credentials. Tests create
synthetic repositories/databases, disposable child processes and loopback HTTP
servers. Loopback transport fault tests are not calls to a real Ollama or model.

The suite retains canonical schema/DDL/invariant, append-only, provenance, scope,
idempotency, projection, deterministic retrieval, provider-control, timeout and
recovery-transport regressions. Existing numerical budget and negative-path
assertions are not relaxed for the public release.

## What was separated, and why

The original engineering repository/history is preserved privately. It contains
machine-specific Hook checks, formally frozen workspace/source adoptions,
authentication/successor acceptance workflows and live-host diagnostic runners.
Those are not reproducible product dependencies for an unrelated contributor.

The public export does not run that historical certification program. Excluded
files and individual nodes are enumerated in [TEST_SELECTION.json](TEST_SELECTION.json).
They are NOT_RUN here, not passed, repaired, xfailed or silently counted in this
suite's totals. Existing failures remain failures in the original records.
Synthetic benchmark fixtures with historical phase names are included only for
the real offline benchmark and assembler tests, not to replay a formal workflow.

Eight mixed-file nodes tied to those runners/frozen files or the obsolete
developer README are left in the original archive; the other runtime tests in
their files are retained. The public canonical/cache-isolation test instead
asserts that no `.codex` registration ships, while retaining the assertion that
retrieval tables never enter canonical SQLite. The version smoke test checks the
new Alpha version. This is a documented distribution change, not a claim that
the original full suite turned green.

## Platform coverage

Windows-specific handle, injection and process tests must remain platform-specific.
Existing capability skips are reported with `-rs` and never counted as PASS.
Linux runs validate the supported local deterministic subset and applicable
fault cases; they do not certify Windows APIs or real host Hooks.
If a CI runner cannot create a required symlink/junction, preserve the skip or
failure and explain that coverage limit; do not enable privileges or pretend the
attack was blocked merely because creating it failed.

The workflow has a 30-minute outer job cap, read-only contents permissions,
fixed Python/uv versions and commit-pinned Actions. Tests do not acquire fresh
inner deadlines to satisfy this outer budget. CI has no login, model, private
HOME, active Hook, real compact or publication step.
Python 3.12.13 is provisioned by uv on both runners, rather than relying on a
Windows installer absent from the setup-python release manifest. Action pins
identify actual commits, not annotated tag objects. See [CI operation](CI.md).
After the source tests and build, it replaces the editable install with the
actual wheel offline, verifies site-packages import origin and runs version plus
repair regressions. These are separately labeled checks, not extra independent
tests to add to the full-suite total.

## Package and privacy checks

```sh
uv run --no-sync python scripts/check_release.py --history
uv build --no-sources
```

The scanner checks tracked files plus reachable Git blobs with redacted output.
Private paths/files, recognized credential shapes, redirected/hardlinked files,
read errors and size-limit failures cannot produce PASS. This heuristic does
not prove exhaustive secret absence, legal ownership or product safety. Do not
publish raw scans of a private repository.

A source build or local Windows test is not a Linux CI run. Until the workflow
actually executes after owner-approved publication, GitHub-hosted CI is NOT_RUN.
Wheel/sdist install checks do not demonstrate real-host context delivery. Prior
same-code host evidence must retain its own precise Runtime/host versions.
