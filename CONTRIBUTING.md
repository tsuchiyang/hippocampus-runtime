# Contributing to Hippocampus Runtime

This is an experimental Alpha. Discuss user-visible changes before implementing
them; keep patches small and avoid unrelated framework or dependency additions.
Contributions are intended to use the project's MIT license. Do not submit code
or data you are not entitled to contribute; retain required third-party notices.

Use Python 3.12 and the lockfile:

```sh
uv sync --locked
uv run --no-sync pytest -q -rs
uv run --no-sync python scripts/check_release.py --history
uv build --no-sources
```

Run new bug reproducers before fixing the bug, preserve failure evidence locally,
then run the affected tests and full public suite. Keep real negative-path,
timeout, source-integrity and append-only assertions. Do not replace failures
with broad xfail/skip/fallback success or claim skipped Windows cases passed.

Tests use synthetic, disposable data and loopback servers. Do not add real
authentication/model calls or machine-specific HOME/configuration fixtures to
default CI. Keep platform-required skips explicit and separate from PASS.
Never upload existing memory state or original private engineering history.

CI has read-only repository permissions and must not use pull_request_target to
run untrusted changes, inherit credentials into test fixtures or publish releases
from pull requests. Release/publication is a separate maintainer operation.
