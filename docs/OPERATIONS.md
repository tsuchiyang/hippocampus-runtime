# Installation, data handling and removal

## Safe first use

Use Python 3.12.x, Git, a fresh virtual environment and a disposable project.
Installation does not create a Codex Hook or read your Codex credentials.
Do not run this experimental package against your only copy of important memory.

After installing from the repository as described in the README, create an empty
directory outside the source checkout and run `git init` there. Invoke the installed
`codex-memory-hook` by absolute path; this avoids selecting a different checkout's
executable. Consult [the command reference](CLI.md) for required scope, provenance
and idempotency arguments; the strict inherited CLI has no `--help`. Scope keys are not filesystem
permissions and must not be used to bypass session or project guards.

For a no-host synthetic lifecycle check, send JSON to the installed command on
stdin, with the event name matching the command, a synthetic session/turn and the
actual disposable project's cwd. The commands are `user-prompt-submit`,
`post-tool-use`, `pre-compact`, `post-compact` and `session-start`.
For example, UserPromptSubmit needs:

```json
{"hook_event_name":"UserPromptSubmit","cwd":"<absolute disposable project>","session_id":"synthetic-demo","turn_id":"turn-1","prompt":"Synthetic local test only."}
```

This is a synthetic protocol check, not proof that a real host performed compaction.
Do not fabricate transcript paths or evidence for real tasks.

## Data and privacy

The project-local `.codex-memory/` contains canonical SQLite data, event/checkpoint
history, raw blobs, recovery state, manifests and derived retrieval state.
Canonical append-only tables are the truth source. Retrieval is derived state;
rebuilding it must not rewrite canonical memory.

Runtime writes are local. An explicitly integrated host can include memory in
model context, so its provider/network policy also applies. Disabled local providers
do not make a real model turn offline. Do not store passwords, tokens or personal
data merely because the database is local; built-in secret screening is not a
complete DLP or encryption system.

Input limits include 1 MiB of raw Hook bytes, a 64 KiB UTF-8 prompt limit and
256 KiB bounds on relevant tool JSON fields. Oversize input is rejected, not
silently truncated into a different event. New unsafe executable evidence URI
schemes are rejected. Existing historical data is not silently rewritten.

## Backup and restore

1. Disable only the explicitly configured project Hook and stop that project's
   writers. Confirm no related test/host process is still running.
2. Copy the **entire** `.codex-memory/` directory to a protected backup location.
   Do not copy just memory.sqlite3 while active WAL writers are running.
3. Record the Runtime version, package hash and project scope identity privately.
4. Restore into an isolated copy of the same logical project with writers stopped.
   Do not merge databases or rewrite stored scope keys to pretend a different
   project/session is the original one.
5. Validate with `ledger verify`; rebuild disposable retrieval state if needed.

For a live database, use an appropriate SQLite-aware backup procedure plus the
corresponding blobs/checkpoints snapshot. A database-only live backup is not a
complete Memory Runtime backup. This Alpha does not provide cross-device private
memory migration or automatic rollback of schema/data versions.

## Disable, uninstall and deletion

- Disable the exact project Hook entry you added; do not remove unrelated Hooks or
  change global trust, user history or authentication. Confirm the host no longer
  discovers this project's Runtime Hook.
- Uninstall from the selected environment with that environment's
  `python -m pip uninstall codex-memory-runtime`; do not uninstall system Python.
- Uninstallation intentionally leaves `.codex-memory/` intact. Delete or move that
  exact project directory only after backup and explicit confirmation; doing so
  loses its stored memory. Do not issue recursive deletion against a workspace or
  home root. No cleanup command is automatically run by this package.

## Recovery delivery semantics

SessionStart prepares and validates recovery, writes and flushes the complete
response, then marks it locally delivered. A failed write/flush keeps recovery
retryable. This does not implement a host acknowledgement or exactly-once delivery:
if the process fails after output but before settlement, a retry can repeat content.
Never edit checkpoint flags by hand to conceal an incomplete delivery.
