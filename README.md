# smolllm-rs (ARCHIVED)

Archived 2026-11-23. The Rust port of smolllm is retired; no further development.

To call LLMs, use the **smolllm-server front door**: OpenAI-compatible HTTP at
`http://<mac>:11435/v1` with an access key. It owns aliases, fallback, key balancing, the usage
Ledger and alerts. Maintained client libraries: smolllm (Python) and smolllm-go.

The last release, `v0.9.0`, stays on GitHub. hntui pins `v0.8.0`:

```toml
smolllm = { git = "https://github.com/rocrp/smolllm-rs.git", tag = "v0.8.0" }
```

Domain glossary: [CONTEXT.md](CONTEXT.md).
