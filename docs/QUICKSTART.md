# Try one memory without enabling Hooks

This Windows PowerShell example uses synthetic data in a **new disposable Git
project**, outside the source checkout. It does not register Hooks, start Codex,
read credentials, call a model or enable dense retrieval. It demonstrates local
storage and deterministic CLI retrieval, not automatic model memory.

## 1. Install from this repository

Install CPython 3.12.x and Git first. Clone this repository (or use your existing
checkout), then follow the source installation in [README](../README.md).
Run the example below **from the checkout**, with that `.venv` already installed.
No PowerShell activation or ExecutionPolicy change is required.

The source installation may download build tools. If you already have the exact
wheel and Python/pip, `python -m pip install --no-index --no-deps <wheel>` installs
it offline. No PyPI publication or unrelated same-name package is implied.

## 2. Create, activate and retrieve one synthetic memory

Copy the entire block into PowerShell. Keep the final printed lab directory for
inspection; the example does not delete it. A new run creates a new lab.

```powershell
$ErrorActionPreference = 'Stop'
$hook = (Resolve-Path -LiteralPath '.\.venv\Scripts\codex-memory-hook.exe').Path
$lab = Join-Path ([System.IO.Path]::GetTempPath()) ('hippocampus-demo-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $lab -ErrorAction Stop | Out-Null
Push-Location -LiteralPath $lab
try {
    git init -q
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the disposable Git project' }
    $scope = (Get-Location).Path

    $proposed = & $hook record propose --type decision --scope project --scope-key $scope --topic-key demo.storage --statement 'Use SQLite for this synthetic demo.' --idempotency-key demo-propose-1
    if ($LASTEXITCODE -ne 0) { throw 'Propose failed; stop and keep the lab for inspection' }
    $record = $proposed | ConvertFrom-Json
    if (-not $record.record_id) { throw 'Propose returned no record ID' }
    $proposed

    & $hook record activate $record.record_id --idempotency-key demo-activate-1
    if ($LASTEXITCODE -ne 0) { throw 'Activate failed' }
    & $hook record list --scope project --scope-key $scope --limit 10
    if ($LASTEXITCODE -ne 0) { throw 'List failed' }

    & $hook retrieval rebuild
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval rebuild failed' }
    & $hook retrieval query --session-id synthetic-demo --turn-id turn-1 --context-id window-1 --query SQLite --explain
    if ($LASTEXITCODE -ne 0) { throw 'Retrieval query failed' }
    & $hook ledger verify
    if ($LASTEXITCODE -ne 0) { throw 'Ledger verification failed' }

    Write-Output ('Synthetic lab retained at: ' + $lab)
} finally {
    Pop-Location
}
```

Expected results:

- `propose` returns a generated `record_id` and `state: proposed`.
- `activate` returns `state: active`; `record list` contains the same record.
- `retrieval rebuild` returns `status: current`; query candidates contain that
  record, with an explanation trace. Identifiers and timestamps vary per lab.
- `ledger verify` reports one permanent record and two lifecycle events without
  a verification error. The retrieval cache is derived state, not the ledger.

The CLI's retrieval `mode: off` is expected here: this is an explicit offline
query, **not an enabled Hook, shadow session or active injection**. Reading a
record from CLI output does not prove a model received it.

Stop at any failed command; do not disable guards or paste private state into
an issue. Use the bounded error, platform, Python version and package version.
This Alpha does not implement `codex-memory-hook --help` or `--version`; obtain
the version with Python's `importlib.metadata` as shown in the README.

## 3. Inspect or stop

Only the new lab receives `.git/` and `.codex-memory/` state. There is no `.codex`
configuration, user-HOME edit, login, background service or Hook to disable.
Do not commit or upload the lab's memory database; even synthetic examples should
establish that habit. After inspection and after writers have stopped, you may
remove **only that printed disposable lab** with your normal file manager.

For other commands, Linux installation, backup and real-host boundaries, see
[CLI](CLI.md), [operations](OPERATIONS.md) and [integration](INTEGRATION.md).
The copy-and-run block above is Windows-specific; it is not Linux host validation.

中文：本例只验证本地账本与确定性检索。不会启用 Hook、读取登录或调用模型；
试验目录保留供检查，不要把其中的记忆数据库提交到 Git 或上传到公开问题。
