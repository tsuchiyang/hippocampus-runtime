# Hippocampus Runtime security policy

Only the current Alpha candidate is considered for fixes; no production support
or response-time guarantee is provided. Do not use this Alpha as the sole durable
record of critical work.

## Report privately

Do not publish exploitable details, raw databases, transcript/blobs, auth files,
tokens, cookies, API keys, private repository paths or full environment dumps in
an Issue or CI artifact. Use GitHub private vulnerability reporting **if the
maintainer has enabled it**. If that option is absent, first open a minimal public
issue asking for a private reporting channel without vulnerability details or
private data. No private mailbox or enabled GitHub feature is claimed here.

Include the package/version/hash, OS/Python/host version, a minimal synthetic
reproducer and a redacted error category. Do not run an exploit on another user's
project or change permissions to reproduce it.

## Boundaries

- Local data is not automatically encrypted. Protect backups and filesystem access.
- Memory text is untrusted evidence, not system instructions or guaranteed truth.
- Retrieved relevance cannot bypass project/session/lifecycle/source controls.
- Runtime input and URI screening are bounded heuristics, not complete DLP.
- Installing the package grants no Hook trust and creates no credentials.
- Optional provider calls are off by default; no automatic cloud fallback exists.
- Release/privacy checks flag known shapes and unsafe paths; PASS is not a proof
  that all secrets, malicious content or licensing issues have been found.

No public push, artifact signature or stable support is implied by a local build.
