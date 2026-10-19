# Host integration boundary

The package is an Alpha library/CLI. There is deliberately no active project or
user Hook registration in this checkout, no automatic login and no global setup.

Start with one synthetic project and one session. Verify the host's version,
effective configuration, source and trust inventory, installed Runtime entry and
the applicable event schema before enabling any selected Hook. Use a separate
prepared HOME only with its owner's authentication approval; never copy auth.json
or alter a user's default HOME to make an acceptance test pass.

Core lifecycle commands map to UserPromptSubmit, PostToolUse, PreCompact,
PostCompact and SessionStart(source=compact). The optional guarded experimental
selected-host/compact paths additionally bind a synthetic marker to the physical
project, session, installed executable, HOME and exact Hook definitions. They are
not shortcuts for arbitrary real-project active injection. Do not remove those
guards or reuse a marker across sessions.

PreCompact/PostCompact context-output options differ from prompt/session events.
Do not copy event fields merely because another event supports them. A discovered
Hook is not necessarily trusted, an RPC acknowledgement is not a completed compact,
and successful Runtime stdout is not proof of model consumption.

Check real lifecycle notifications, saved checkpoint/highwater, exact returned
context and available host-native evidence. A native developer-context match is
stronger than a correct answer to a canary question, but still not an attestation
of final server-side model input bytes. Prepared manifests must not be relabeled
as emitted or host-acknowledged based on an inference.

Runtime must fall back or fail closed according to the existing canonical/source
boundary. Do not make malformed canonical data acceptable to get a packet. Do not
extend the 1000/1500 ms experimental enhancement/Hook contract to hide a timeout.
Hook APIs and execution cleanup are host-version-specific; repeat the selected
scope validation when the relevant host binary/configuration changes.

Official upstream reference: [Codex Hooks](https://developers.openai.com/codex/hooks)
and [App Server](https://developers.openai.com/codex/app-server). These references
describe upstream APIs, not certification of this Alpha on every upstream version.
