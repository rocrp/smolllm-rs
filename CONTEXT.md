# SmolLLM (Rust)

Rust port of smolllm: a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface, API-key/endpoint balancing, model fallback. Shares its ubiquitous language with the Python lib (`../smolllm/CONTEXT.md` is the canonical copy; keep in sync).

## Language

**Provider**:
A named OpenAI-compatible endpoint (e.g. `openai`, `groq`); credentials and base URL resolve from env by name. Bare model specs have no provider — identity surfaces as empty string.
_Avoid_: vendor, backend.

**Model spec**:
The user-facing model string `provider/model`, or bare `model` (no `/`) — bare form has no provider and resolves base URL/API key from explicit builder options only, never env. Comma-separated specs form a fallback chain; may mix both forms. Explicit base URL applies to every leg. A leg may carry a `!effort` suffix (`proxy/gpt-5!high`) that overrides the call-level reasoning effort for that leg alone and never reaches the wire; a suffix with nothing after the `!` is rejected as a typo rather than ignored.

**Fallback chain**:
Ordered or weighted candidate models; on failure the call advances to the next candidate. A leg counts as failed until it delivers its first chunk — an HTTP error, an error frame after a 200, a dropped connection, or a stream that ends empty all advance the chain. Once content has reached the consumer the chain is committed: a later failure surfaces as a stream error carrying the Partial output, because splicing a second model onto half an answer is worse than failing.
_Avoid_: confusing with retry.

**Partial output**:
What a stream delivered to the consumer before it failed; carried on the error so the caller keeps what arrived.

**Retry**:
Re-attempt of the *same* model after a transient failure, with backoff. Distinct from fallback (which switches models), and yields to it: while another leg remains, a transient failure advances the Fallback chain immediately instead of sleeping. The backoff is for the last leg, where there is nothing else to try.

**Balancer pair**:
One (API key, base URL) combination for a provider; the least-used pair is chosen per call.

**Reported usage**:
Token counts the provider itself sent, requested with `stream_options.include_usage` and merged field by field across the frames that carry them. An endpoint that rejects the field with a 400 is retried once without it.

**Estimated usage**:
Token counts derived by heuristic (chars/4), used for whichever side the provider did not report. `Usage.estimated` is true when any count came from the heuristic, and the metrics log marks those totals with `~`.

**Reasoning**:
Model thinking text, kept in a channel separate from content.
_Avoid_: mixing reasoning into content.

**ResolvedModel**:
The model the server reports as having produced a response, read from the wire's top-level `model` field; differs from the requested Model spec behind aliases and proxies, and is absent when the backend reports none. The `keepalive` sentinel omlx emits is transport, not identity, and never counts.
_Avoid_: actual model (that is the resolved-or-requested identity below), real model.

**Actual model**:
Best available identity of the model that produced a response: the ResolvedModel when the server named one, otherwise the requested Model spec.

**FinishReason**:
Verbatim provider string explaining why generation ended; never normalized.

**Truncation**:
An answer cut short rather than ended: the FinishReason is `length`, or a stream carried content and then ended with no FinishReason at all, having lost its terminal frame. A response with no content is the empty case, not a truncated one. `ask` treats a truncated leg as failed and advances the Fallback chain; a stream reports it, since its output has already been delivered.

**Request hook**:
Per-attempt observation callback receiving usage or error; the library's only telemetry surface.

**Escape hatch**:
A pass-through (`extra_body`) letting callers set raw request fields the library does not model, merged last so the caller wins. The fields the library machinery reads back (`stream`, `stream_options`, `messages`, `model`) are rejected.
_Avoid_: raw options, extra params.

**Tool call**:
A provider-issued request to run a named function, surfaced verbatim as a `ToolCall` carrying the wire fields (unknown provider keys preserved) with the argument JSON as an opaque string. The caller executes it and replays the assistant and `tool` messages; the library runs no agentic loop and never inspects or repairs the argument JSON.
_Avoid_: function call, tool use, implying smolllm executes anything.
