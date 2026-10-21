# Hippocampus Runtime 0.1.0a1 — experimental Alpha

Maintainer-ready notes for the first GitHub **prerelease**. This file is not
evidence that a tag, GitHub Release, binary download or PyPI publication exists.
Use the exact published source commit when creating `v0.1.0a1`; do not move any
existing tag. Record the commit and SHA-256 of the actual uploaded assets.

## Included

- Local append-only structured memory with explicit lifecycle and review.
- Compaction checkpoints and retryable recovery after output-transport failure.
- Deterministic exact/BM25/symbol/state retrieval in a disposable SQLite cache.
- Python 3.12, no third-party runtime dependencies, opt-in integration.
- English/Chinese entry documentation and an isolated no-Hook quickstart.

Compatibility names remain `codex-memory-runtime`, `codex_memory` and
`codex-memory-hook`. Installation registers no Hook. Dense and active remain
default-off. This is independent community software, not an OpenAI product.

## Evidence and limits

The last complete hosted code check is
[run 39303948377](https://github.com/tsuchiyang/hippocampus-runtime/actions/runs/39303948377)
at `6b32f2a3a8acddb6f8b2ef658520fff9a9e2a67a`:

- Windows source suite: 1771 passed, zero failures/errors/skips.
- Ubuntu limited local-core: 747 passed, one explicit Windows-only skip.
- Actual installed-wheel repair/version checks: 44 passed on each platform,
  separately recorded and not added to overlapping source totals.

Later documentation/media-only commits do not imply another full-suite run.
Runtime source, tests and dependency metadata must match that checked code before
this evidence is inherited. Rebuilt binary files require their own exact hashes;
old binary hashes are not interchangeable merely because the version is equal.

No new real-host/model acceptance is claimed for this Alpha. Windows-specific
guarded injection is not certified on Linux. Live sparse/reranker, global active,
universal deadline guarantees, exactly-once recovery delivery and production
certification are outside this release. Historical failures and unknown causes
remain documented in [known issues](KNOWN_ISSUES.md), not rewritten as PASS.

## Asset and publication checklist

- Exact-commit `codex_memory_runtime-0.1.0a1-py3-none-any.whl`.
- Exact-commit `codex_memory_runtime-0.1.0a1.tar.gz`.
- `SHA256SUMS.txt` covering those exact files and any separately uploaded film.
- Verify licensed contents, privacy/history checks and actual wheel import.
- Keep private databases, credentials, Hook configs and developer artifacts out.
- If uploading the promotional film, keep it a separate media asset; do not
  present the story as evidence that a model consumed the memory packet.
- Mark the GitHub Release **pre-release**; do not label it production-stable.

To test the software safely, start with [QUICKSTART](QUICKSTART.md). Report
issues using the repository template, without raw prompts, databases or auth.
