# Hippocampus Runtime（海马体）

Local memory and context recovery for coding agents, currently targeting Codex.

[![Alpha checks](https://github.com/tsuchiyang/hippocampus-runtime/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tsuchiyang/hippocampus-runtime/actions/workflows/ci.yml)

[简体中文](README.zh-CN.md) · [Quickstart without Hooks](docs/QUICKSTART.md) · [Installation and operations](docs/OPERATIONS.md) · [Testing](docs/TESTING.md)

**Experimental Alpha — local-first memory, not a promise of production reliability.**
Hippocampus Runtime is now open source on GitHub under the MIT license.
The Windows full-suite and limited Ubuntu local-core CI checks have passed. See
[known issues and current evidence](docs/KNOWN_ISSUES.md).
No Hook is registered by installing this package. Dense retrieval and active
injection are not enabled by default. This is an independent community project,
not an official OpenAI product.

<a href="docs/DEMO.md"><img src="docs/media/night-shift-preview.gif" alt="Night Shift: an original pixel-art story about carrying project memory between context windows" width="270"></a>

[Night Shift — animated preview and film notes](docs/DEMO.md). Fictional product
story, **not a recording of a real Codex session or evidence of model recall**.

## Name and compatibility

Hippocampus Runtime is the public name of the project previously called Codex
Memory Runtime. This is a branding change, not a package or interface migration:

- Python distribution: `codex-memory-runtime`
- Python import: `codex_memory`
- Installed command: `codex-memory-hook`

Existing commands, `.codex-memory/` data paths and installation identifiers remain
unchanged. Historical source and copyright notices may retain the earlier name.

## What it does

- Stores observable prompt/tool events and compaction checkpoints locally.
- Keeps structured permanent memory and lifecycle events append-only in SQLite.
- Separates unverified, same-session temporary memory from reviewed permanent memory.
- Supports deterministic exact, BM25, symbol and state retrieval in a disposable cache.
- Makes recovery retryable when its output transport fails; checkpoint text is retained.

Memory is evidence, not authority. The runtime does not automatically extract
natural-language facts or approve them. A model/operator must explicitly capture
and review candidates. It does not provide unlimited context or guaranteed recall.

## Requirements and installation

CPython **3.12.x**, Git, a writable local filesystem. Windows is the primary
tested platform. Local deterministic Linux functionality has independent test
evidence; Windows-only guarded injection/audit features are not advertised as
cross-platform. No third-party runtime dependencies are required.

From a source checkout, choose a new virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -c "from importlib.metadata import version; print(version('codex-memory-runtime'))"
```

On Windows PowerShell (no activation or ExecutionPolicy change needed):

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install .
& .\.venv\Scripts\python.exe -c "from importlib.metadata import version; print(version('codex-memory-runtime'))"
```

Building from source may download Hatchling. Installing a prepared wheel with
`python -m pip install --no-index --no-deps path/to/package.whl` does not require
network access when Python/pip are already installed. No PyPI publication is
claimed; do not assume a package of the same name on an index is this release.

## Try it without enabling Hooks

Use a disposable Git project, run the installed entry by its full path, and read
the [copy-and-run quickstart](docs/QUICKSTART.md) to create, activate and retrieve
one synthetic memory without registering Hooks or calling a model. See also the
[command reference](docs/CLI.md). This Alpha preserves
the existing strict CLI: `--help`/`--version` are not implemented. Command families include:

```text
record propose | activate | show | list
ledger verify
memory capture | evidence | pending | recall | review | expire
retrieval init | status | rebuild | verify | query
```

All local state is project-local under `.codex-memory/`; keep it out of Git and
back it up before upgrades. See [operations](docs/OPERATIONS.md) for boundaries,
backup, recovery, removal and a synthetic event example.

## Host integration is explicitly opt-in

The repository ships **no active `.codex` directory or user configuration**.
Do not blindly register every command or copy another machine's Hook settings.
Host Hook APIs, trust rules and supported fields are version-dependent. Review
the [integration boundary](docs/INTEGRATION.md) and validate one synthetic
project/session before any real project. The selected-host experimental route
requires its complete root/session/entry/source binding; its guards must not be
removed for convenience.

The default Legacy path does not call an external model, embedding service or
MCP. Optional local provider code is present but disabled by default; real host
model turns are external to the Runtime and use the host's own authentication
and network. Sparse/reranker live integration is not certified.

## Development

With uv 0.12.1 and Python 3.12:

```sh
uv sync --locked
uv run --no-sync pytest -q -rs
uv run --no-sync python scripts/check_release.py --history
uv build --no-sources
```

The full-suite command above targets Windows. Linux uses the explicit local-core
command in [testing](docs/TESTING.md); Windows-only injection is not cross-platform.
The public tests use synthetic data and loopback test servers, not
your Codex credentials, a real model or an existing memory database. Platform
limitations and historical test separation are recorded in [testing](docs/TESTING.md).
CI is configured for Windows and Linux; a workflow definition is not proof
that GitHub has executed it.

## Security and limitations

Memory can contain prompts, tool output, repository details and confidential
material. Do not attach databases, transcripts, auth files or raw Hook payloads
to public issues. See [SECURITY.md](SECURITY.md).

This Alpha has bounded validation evidence, not universal host compatibility,
long-running reliability certification, exactly-once delivery or proof of the
model's final request bytes. A successful stdout write/flush is not a host ACK.
Fallback, prepared artifacts and host reception must be reported separately.

MIT licensed; see [LICENSE](LICENSE), [NOTICE](NOTICE.md) and
[source provenance](docs/PROVENANCE.md). Contributions: [CONTRIBUTING](CONTRIBUTING.md).
