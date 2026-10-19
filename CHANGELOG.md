# Hippocampus Runtime changelog

## 0.1.0a1 — Alpha preparation

- Adopt Hippocampus Runtime（海马体）as the public name; keep the distribution
  `codex-memory-runtime`, import `codex_memory` and command `codex-memory-hook`.
- Prepare a history-free, privacy-reviewed public source tree and portable tests.
- Add MIT licensing, user/security/operations documentation and read-only CI.
- Inherit append-only memory, deterministic retrieval and bounded recovery from
  the verified implementation, including retryable recovery after output failure,
  bounded raw/prompt/tool input and transient SQLite sidecar handling.
- Carry the shared absolute deadline through provider-health lock preparation
  and retries; preserve admission, settlement, safety checks and original budgets.
- Reuse only immutable expected schema definitions, never observed database
  validity; pass the original shadow-query deadline into rerank orchestration.
- Set Alpha version metadata to 0.1.0a1; this is not a new claim of production
  certification or a guarantee of cross-platform host integration.
- Dense remains optional/default-off. No automatic Hook registration, real model
  deployment, sparse/reranker live certification or global active rollout.
