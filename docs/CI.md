# Hippocampus Runtime: CI operation and evidence

## What the workflow does

The workflow targets GitHub-hosted Windows and Ubuntu, with a 30-minute cap for
each job, read-only contents permissions, no saved checkout credentials and no
publication step. It uses synthetic fixtures and loopback services, not a Codex
HOME, model login, real Hook or existing memory database.

The Windows job runs the full public suite. Ubuntu uses the 37-file local-core
profile in `tests/linux-core.txt`; that does not certify Windows-only injection.
Both jobs scan the tracked tree and reachable Git history, build wheel/sdist,
install the actual wheel offline and run version/recovery repair regressions.

Windows uses a new run-specific pytest `--basetemp` child under GitHub's
`RUNNER_TEMP`, outside the checkout and user-profile temp tree. The workflow
validates the numeric run/attempt identity, exact parent and target absence
before pytest owns that disposable directory. It does not change global TEMP,
HOME, permissions, security settings or any inner deadline. Ordinary local tests
can retain their own writable temporary layout.

A preceding private diagnostic pair passed the same 13 fixed nodes in both
temporary layouts (system-temp 33.38 s; runner-temp 22.88 s). This is evidence
about that small pair, not a proof that the drive caused previous full-suite
failures. The explicit CI layout requires its own actual full-suite result;
the original failures remain preserved. Failed tests include bounded safe phase
sections and the synthetic installed-Hook sidecar; passing focused nodes are not
counted as a full/release gate.

## Reproducible bootstrap

CI pins uv 0.12.1 and installs CPython 3.12.13 with uv on both operating systems:

```sh
uv python install 3.12.13 --no-bin --no-registry
uv sync --locked --python 3.12.13 --link-mode=copy
uv run --no-sync python -c "import sys; assert sys.version_info[:3] == (3, 12, 13)"
```

This avoids depending on a Windows build that the setup-python 3.12.13 manifest
does not provide. The workflow prints the actual Git commit, OS and interpreter,
without dumping environment variables or credentials.

Action references must use commits from their official repositories. An
annotated tag's own object SHA is not its target commit SHA; peel it and verify
the action definition at the target before pinning. Existing read permissions,
inner deadlines, tests and numerical assertions must not be relaxed to get CI green.

Expected schema definitions are parsed from frozen source tuples into a bounded,
read-only mapping. This avoids charging the same immutable DDL compilation to
every short provider admission and settlement. It does not cache observed
database validity: schema text, columns, append-only history and projections are
still read and checked on every validation. Warm-cache tampering tests retain
that distinction.

The adapter's connection-refused classification test injects an actual
`ConnectionRefusedError` only at `socket.create_connection`. The real urllib
request, budgeted HTTP connection and adapter classifier remain in the path.
A closed Windows port does not guarantee refusal before the connect deadline;
it is therefore not used as an implicit timing fixture. Read-timeout and response
count cases still exercise a real synthetic loopback HTTP server. Neither case
is evidence about a real model service.

## Safe prepublication execution

Before public publication, use an owner-approved private project or private CI
staging repository. Do not upload into an unrelated repository. Grant the existing
GitHub connection access only to the selected project and use the current Actions
quota; enabling paid capacity or changing visibility is a separate owner decision.
No new login, PAT or copied credential is required by this workflow itself.

For each actual hosted run, retain the repository/ref, exact commit, run/attempt
and job IDs, runner image/version, command logs, exit/conclusion, all skip reasons
and built-file hashes. A manual dispatch requires the workflow on the repository's
default branch. The first push can also trigger the existing push event.

Successful local bootstrap, actionlint, source tests or archive checks are not
GitHub-hosted CI evidence. A missing or inaccessible repository, denied Actions
access or exhausted quota stays BLOCKED/NOT_RUN, not PASS. Preserve failed runs;
only retry for a documented change or diagnosed transient failure.

Do not add credentials, private memory, old engineering history, a real Hook or
model invocation to make the job start. Public push, a release/tag, switching
visibility and deployment remain explicit maintainer operations.
