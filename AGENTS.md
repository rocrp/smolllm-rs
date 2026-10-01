# AGENTS.md

> **Deprecated (2026-10-01):** this crate is being retired; no further development is planned. The maintained clients are `smolllm` (Python, `../smolllm`) and `smolllm-go` (`../smolllm-go`). hntui still pins a tag of this crate.

Guidance for coding agents working in this repository.

## Project Overview

smolllm-rs (crate `smolllm`) is the Rust port of smolllm (Python, sibling repo `../smolllm`): a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface (`ask`/`stream`/`validate` builders), API-key/endpoint load balancing, model fallback chains. Consumed by hntui (pins a git tag) — cut a tag after every release-worthy change.

## Design Philosophy

**Extreme minimalism.** Scope is frozen at chat over the OpenAI-compat wire: no new modalities, no native provider transports, and no new API surface without a consumer to exercise it. Tool calling and the `extra_body` escape hatch follow the design recorded in the Python repo; everything else stays deferred.

Canonical doctrine lives in the Python repo — read before proposing features:
- `../smolllm/docs/adr/0001-extreme-minimalism.md` — scope freeze
- `../smolllm/docs/adr/0002-token-only-accounting.md` — usage stops at tokens; cost is the caller's concern
- `../smolllm/docs/DEFERRED.md` — recorded designs and their status per port

Domain glossary: [CONTEXT.md](CONTEXT.md).

## Development

- `cargo test` — offline: resolver/parser tests plus byte-stream SSE tests; there is no mock HTTP harness (keep it that way — test the request-body builder and the SSE parser directly)
- `cargo clippy --all-targets` — keep clean
- `cargo run --example e2e` — manual live checks against real providers (`~/.env.smolllm`)
- Provider map is hand-maintained in `src/provider.rs` (the Python repo's `providers.json` is generated; sync manually)
