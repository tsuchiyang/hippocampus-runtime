# Alpha command reference

Run the installed `codex-memory-hook` (Windows: `codex-memory-hook.exe`) by its
full path **from the intended Git project**. The inherited strict parser has no
`--help` or `--version`; unsupported/invalid arguments return nonzero rather than
starting a Runtime. Check the installed version with Python importlib.metadata.

Angle-bracket values below are placeholders, not literal data. Never invent real
evidence or use private memory in a demo.

## Structured records

```text
record propose --type decision --scope project --scope-key <absolute-project-root> --topic-key demo.storage --statement "Use SQLite for this synthetic demo." --idempotency-key demo-propose-1
record activate <record-id> --idempotency-key demo-activate-1
record show <record-id>
record list --scope project --scope-key <absolute-project-root> --limit 20
ledger verify
```

Propose optionally accepts `--value-json <strict-JSON>` and repeated
`--evidence-uri <URI>`. Types are constitution, decision, assumption,
rejected_approach and failure; a proposed constitution is not automatically
certified. Scope is project or workspace. Use the appropriate actual scope key.
List also supports `--topic-key`, `--type` and `--state`.

Record IDs come from actual JSON output. Reusing an idempotency key with a
different operation/content is a conflict, not an update. The lifecycle API
supports more actions than this small CLI: `record invalidate/close/reopen/
supersede` are not exposed commands in this Alpha.

## Temporary memory and review

```text
memory capture --session-id <actual-session> --turn-id <actual-turn> --context-id <actual-window> --type decision --scope project --scope-key <actual-project-root> --topic-key demo.choice --statement "Synthetic candidate." --idempotency-key demo-capture-1
memory evidence --session-id <actual-session> --limit 20
memory pending --session-id <actual-session> --limit 20
memory recall --session-id <actual-session> --query "demo" --max-chars 4000
memory review <candidate-id> --outcome promote --reason "Verified synthetic decision." --idempotency-key demo-review-1 --evidence-uri <actual-evidence-URI>
memory expire <candidate-id> --reason "No longer applicable." --idempotency-key demo-expire-1
```

Capture optionally takes `--value-json` and repeated `--evidence-uri`. Evidence
supports `--turn-id`; review outcomes are promote/reject/defer/expire, with
optional repeated `--evidence-uri` and `--supersedes <record-id>`.
Review/authority/provenance requirements still apply; copying this example is not
authority to promote an actual candidate. Temporary memory is session-limited.

## Disposable deterministic retrieval

```text
retrieval status
retrieval init
retrieval rebuild
retrieval verify
retrieval query --session-id <session> --turn-id <turn> --context-id <window> --query "SQLite" --explain
```

Query also accepts `--prompt-event-uri`, `--max-chars` (1–4000), `--legacy`,
`--disable-dense`, `--disable-sparse` and `--disable-reranker`. Maintenance does not
make missing/invalid canonical data valid. A fresh empty project can report a
non-current state until canonical state exists; that is not permission to bypass
validation. Do not turn on semantic configuration to hide a deterministic miss.

## Hook protocol

Commands `user-prompt-submit`, `post-tool-use`, `pre-compact`, `post-compact` and
`session-start` consume bounded JSON on stdin; event names must match the command.
Use `trigger=manual` in synthetic Pre/PostCompact events and `source=compact`
for compact SessionStart. See [operations](OPERATIONS.md) and
[integration boundaries](INTEGRATION.md). A synthetic event is not a real-host
notification. Selected/isolated experimental flags require their explicit marker
contracts and must not be used to bypass default-off behavior.

Diagnostics are intentionally bounded. A nonzero command is a failure/refusal,
not an invitation to ignore the error or to change canonical data by hand.
