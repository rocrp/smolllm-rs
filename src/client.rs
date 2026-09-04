use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use crate::balancer;
use crate::endpoint::resolve_model;
use crate::error::Error;
use crate::image::image_to_data_url;
use crate::provider::{resolve_api_key, ParsedModel};
use crate::request::{EventHook, RequestConfig};
use crate::think::ThinkTagFilter;
use crate::toolcall::ToolCall;
use crate::types::*;
use crate::utils::*;

const MAX_RETRIES: usize = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_secs(2);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(30);
const RETRY_BACKOFF_SCALE: u32 = 3;
const SSE_CHANNEL_CAPACITY: usize = 32;

// --- Public entry points ----------------------------------------------------

pub(crate) async fn execute_ask(config: RequestConfig) -> Result<LLMResponse, Error> {
    run_with_fallback(&config, |dispatch, response, started, config| {
        Box::pin(ask_finalize(dispatch, response, started, config))
    })
    .await
}

pub(crate) async fn execute_stream(config: RequestConfig) -> Result<StreamResponse, Error> {
    run_with_fallback(&config, |dispatch, response, started, config| {
        Box::pin(stream_finalize(dispatch, response, started, config))
    })
    .await
}

// --- Fallback driver -------------------------------------------------------

type FinalizeFut<'a, R> = Pin<Box<dyn Future<Output = Result<R, Error>> + Send + 'a>>;

/// A failed attempt, with the identity of the leg that failed so the hook can
/// name it.
struct AttemptFailure {
    error: Error,
    usage: Usage,
}

async fn run_with_fallback<R, F>(config: &RequestConfig, finalize: F) -> Result<R, Error>
where
    R: Finalized,
    F: for<'a> Fn(Dispatch, reqwest::Response, Instant, &'a RequestConfig) -> FinalizeFut<'a, R>,
{
    let model_input = config.resolve_model_input().ok_or(Error::NoValidModels)?;
    let mut selector = model_input.into_selector();
    let client = config.http_client.clone().unwrap_or_default();

    let mut last_err: Option<Error> = None;
    while let Some(model_str) = selector.next_model() {
        let attempt = dispatch_one(&model_str, config, &client, &finalize).await;
        match attempt {
            Ok(response) => {
                // A stream reports only once exhausted, so it fires its own.
                if let Some(usage) = response.success_usage() {
                    fire_hook(config, usage, None);
                }
                return Ok(response);
            }
            Err(failure) => {
                let AttemptFailure { error, usage } = *failure;
                if selector.has_more() {
                    log::warn!("Model {model_str} failed, trying fallback: {error}");
                } else {
                    log::warn!("Model {model_str} failed: {error}");
                }
                fire_hook(config, usage, Some(error.to_string()));
                last_err = Some(error);
            }
        }
    }
    Err(last_err.unwrap_or(Error::NoValidModels))
}

async fn dispatch_one<R, F>(
    model_str: &str,
    config: &RequestConfig,
    client: &reqwest::Client,
    finalize: &F,
) -> Result<R, Box<AttemptFailure>>
where
    F: for<'a> Fn(Dispatch, reqwest::Response, Instant, &'a RequestConfig) -> FinalizeFut<'a, R>,
{
    let mut dispatch = match Dispatch::prepare(model_str, config) {
        Ok(dispatch) => dispatch,
        // Nothing was sent, so the only identity we have is what was asked for.
        Err(error) => {
            return Err(Box::new(AttemptFailure {
                error,
                usage: Usage {
                    model: model_str.to_string(),
                    ..Usage::default()
                },
            }))
        }
    };
    log::info!(
        "Sending request: url={} model={} key={} approx_tokens={}",
        dispatch.request_url,
        dispatch.parsed.model_name,
        preview_api_key(&dispatch.chosen_key),
        dispatch.input_tokens,
    );
    let started = Instant::now();
    let identity = dispatch.usage_identity();
    let fail = |error: Error| {
        Box::new(AttemptFailure {
            error,
            usage: Usage {
                duration: started.elapsed(),
                ..identity.clone()
            },
        })
    };
    let response = match send_with_retry(client, &mut dispatch, config.timeout).await {
        Ok(response) => response,
        Err(error) => return Err(fail(error)),
    };
    finalize(dispatch, response, started, config).await.map_err(fail)
}

fn fire_hook(config: &RequestConfig, usage: Usage, error: Option<String>) {
    if let Some(ref hook) = config.hook {
        hook(RequestEvent { usage, error });
    }
}

// --- Per-model dispatch ----------------------------------------------------

pub(crate) struct Dispatch {
    pub model_str: String,
    pub parsed: ParsedModel,
    pub request_url: String,
    pub chosen_key: String,
    pub body: serde_json::Value,
    pub input_tokens: usize,
}

impl Dispatch {
    fn prepare(model_str: &str, config: &RequestConfig) -> Result<Self, Error> {
        let resolved = resolve_model(model_str, config.base_url.as_deref())?;
        let api_keys = parse_api_key_list(&resolve_api_key(
            &resolved.parsed,
            config.api_key.as_deref(),
        )?)?;
        let (chosen_key, chosen_url) = balancer::choose_pair(&api_keys, &resolved.base_urls)?;
        let request_url = resolved.endpoint_for(&chosen_url).url;
        let parsed = resolved.parsed;
        let body = build_request_body(&parsed.model_name, config)?;
        let input_tokens = estimate_tokens(&serde_json::to_string(&body).unwrap_or_default());
        Ok(Self {
            model_str: model_str.to_string(),
            parsed,
            request_url,
            chosen_key,
            body,
            input_tokens,
        })
    }

    /// Who this attempt was, for a hook that has to report a failure.
    fn usage_identity(&self) -> Usage {
        Usage {
            provider: self.parsed.provider_name.clone(),
            model: self.model_str.clone(),
            model_name: self.parsed.model_name.clone(),
            api_key_hint: preview_api_key(&self.chosen_key),
            input_tokens: self.input_tokens,
            ..Usage::default()
        }
    }
}

// --- HTTP send with retry --------------------------------------------------

async fn send_with_retry(
    client: &reqwest::Client,
    dispatch: &mut Dispatch,
    timeout: Duration,
) -> Result<reqwest::Response, Error> {
    let mut last_err: Option<Error> = None;
    let mut may_drop_stream_options = true;
    let mut attempt = 0usize;
    while attempt < MAX_RETRIES {
        if attempt > 0 {
            let delay = retry_delay(attempt);
            log::warn!(
                "Retrying after transient error: model={} attempt={} delay={:?} err={}",
                dispatch.model_str,
                attempt + 1,
                delay,
                last_err.as_ref().map(|e| e.to_string()).unwrap_or_default(),
            );
            tokio::time::sleep(delay).await;
        }

        let send_result = client
            .post(&dispatch.request_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", dispatch.chosen_key))
            .timeout(timeout)
            .json(&dispatch.body)
            .send()
            .await;

        let resp = match send_result {
            Ok(r) => r,
            Err(e) => {
                let err = Error::Request(e);
                if err.is_retryable() && attempt < MAX_RETRIES - 1 {
                    last_err = Some(err);
                    attempt += 1;
                    continue;
                }
                return Err(err);
            }
        };

        let status = resp.status().as_u16();
        if status >= 400 {
            let body_text = resp.text().await.unwrap_or_default();
            // Some OpenAI-compatible endpoints reject `stream_options` outright.
            // Retry the same leg once without it — no backoff, no retry budget,
            // since nothing here is transient.
            if status == 400 && may_drop_stream_options && drop_stream_options(&mut dispatch.body) {
                may_drop_stream_options = false;
                log::warn!(
                    "Endpoint rejected stream_options (400), retrying without it: model={}",
                    dispatch.model_str,
                );
                continue;
            }
            let err = Error::Http {
                status,
                body: body_text,
            };
            if err.is_retryable() && attempt < MAX_RETRIES - 1 {
                last_err = Some(err);
                attempt += 1;
                continue;
            }
            return Err(err);
        }

        return Ok(resp);
    }
    Err(last_err.unwrap_or(Error::Other("max retries exhausted".into())))
}

fn retry_delay(attempt: usize) -> Duration {
    let mut d = RETRY_BASE_DELAY;
    for _ in 0..attempt {
        d = d.saturating_mul(RETRY_BACKOFF_SCALE);
    }
    d.min(RETRY_MAX_DELAY)
}

// --- Finalizers ------------------------------------------------------------

async fn ask_finalize(
    dispatch: Dispatch,
    response: reqwest::Response,
    started: Instant,
    config: &RequestConfig,
) -> Result<LLMResponse, Error> {
    let byte_stream = response.bytes_stream();
    let outcome = consume_sse_stream(byte_stream, config.handler.as_deref(), started).await?;
    let SseOutcome {
        content,
        reasoning,
        ttft,
        finish_reason,
        resolved_model,
        reported_usage,
        tool_calls,
    } = outcome;

    let content = if config.remove_backticks {
        strip_backticks(&content)
    } else {
        content
    };

    // An assistant turn that only requests tool calls carries no text, but it is
    // a complete, useful response.
    if content.trim().is_empty() && reasoning.trim().is_empty() && tool_calls.is_empty() {
        return Err(Error::EmptyResponse {
            model: dispatch.model_str,
        });
    }

    let total = started.elapsed();
    let (input_tokens, output_tokens, estimated) = resolve_usage_tokens(
        reported_usage,
        dispatch.input_tokens,
        estimate_tokens(&format!("{content}{reasoning}")),
    );

    log::info!(
        "{}",
        format_metrics(
            &dispatch.parsed.model_name,
            input_tokens,
            output_tokens,
            total,
            ttft,
            estimated,
        )
    );

    Ok(LLMResponse {
        text: content,
        reasoning,
        finish_reason,
        resolved_model,
        tool_calls,
        model: dispatch.model_str.clone(),
        model_name: dispatch.parsed.model_name.clone(),
        provider: dispatch.parsed.provider_name.clone(),
        usage: Usage {
            provider: dispatch.parsed.provider_name,
            model: dispatch.model_str,
            model_name: dispatch.parsed.model_name,
            api_key_hint: preview_api_key(&dispatch.chosen_key),
            input_tokens,
            output_tokens,
            estimated,
            duration: total,
            ttft,
        },
    })
}

async fn stream_finalize(
    dispatch: Dispatch,
    response: reqwest::Response,
    started: Instant,
    config: &RequestConfig,
) -> Result<StreamResponse, Error> {
    let byte_stream = response.bytes_stream();
    let (tx, mut rx) = mpsc::channel::<Result<StreamChunk, Error>>(SSE_CHANNEL_CAPACITY);
    let tail: SharedTail = std::sync::Arc::new(std::sync::Mutex::new(StreamTail::default()));

    let task_tail = std::sync::Arc::clone(&tail);
    tokio::spawn(async move {
        process_sse_task(byte_stream, tx, task_tail).await;
    });

    // A 200 proves nothing: a proxy can still answer with an error frame, drop
    // the connection, or stream nothing at all. Waiting here for the first chunk
    // keeps all of that inside the fallback chain — past this point output has
    // been delivered and switching models would splice two answers together.
    let first = take_first_chunk(&mut rx, &dispatch.model_str).await?;

    Ok(StreamResponse::new(StreamInit {
        rx,
        tail,
        model: dispatch.model_str,
        model_name: dispatch.parsed.model_name,
        provider: dispatch.parsed.provider_name,
        api_key_hint: preview_api_key(&dispatch.chosen_key),
        input_tokens: dispatch.input_tokens,
        start: started,
        first: Some(first),
        hook: config.hook.clone(),
    }))
}

/// Awaits the first item of a stream: a chunk, the failure that came instead, or
/// `EmptyResponse` for a stream that ended without producing anything.
async fn take_first_chunk(
    rx: &mut mpsc::Receiver<Result<StreamChunk, Error>>,
    model: &str,
) -> Result<StreamChunk, Error> {
    match rx.recv().await {
        Some(Ok(chunk)) => Ok(chunk),
        Some(Err(error)) => Err(error),
        None => Err(Error::EmptyResponse {
            model: model.to_string(),
        }),
    }
}

// --- Request body construction ---------------------------------------------

fn build_request_body(
    model_name: &str,
    config: &RequestConfig,
) -> Result<serde_json::Value, Error> {
    let mut messages = Vec::new();

    if let Some(ref sys) = config.system_prompt {
        if !sys.trim().is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": sys,
            }));
        }
    }

    match &config.prompt {
        Prompt::Text(text) => {
            if config.image_paths.is_empty() {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": text,
                }));
            } else {
                let mut parts = vec![serde_json::json!({ "type": "text", "text": text })];
                for path in &config.image_paths {
                    let data_url = image_to_data_url(path)?;
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": data_url, "detail": "auto" },
                    }));
                }
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": parts,
                }));
            }
        }
        Prompt::Messages(msgs) => {
            if !config.image_paths.is_empty() {
                return Err(Error::Image(
                    "image_paths cannot be used with message-list prompts".into(),
                ));
            }
            for msg in msgs {
                let mut value = serde_json::json!({
                    "role": msg.role.as_str(),
                    "content": msg.content,
                });
                let object = value
                    .as_object_mut()
                    .expect("json! object literal is always an object");
                if !msg.tool_calls.is_empty() {
                    object.insert("tool_calls".into(), serde_json::to_value(&msg.tool_calls)?);
                }
                if let Some(ref id) = msg.tool_call_id {
                    object.insert("tool_call_id".into(), serde_json::Value::String(id.clone()));
                }
                if let Some(ref name) = msg.name {
                    object.insert("name".into(), serde_json::Value::String(name.clone()));
                }
                messages.push(value);
            }
        }
    }

    let request = ChatCompletionRequest {
        model: model_name.to_string(),
        messages,
        stream: true,
        // Every request streams on the wire, `ask` included, so both paths can
        // be handed the provider's own counts instead of a chars/4 guess.
        stream_options: StreamOptions { include_usage: true },
        temperature: config.temperature,
        top_p: config.top_p,
        reasoning_effort: config.reasoning_effort.clone(),
    };
    let mut body = serde_json::to_value(request)?;

    // Merged last: the caller wins over library defaults. Reserved keys were
    // already rejected by the builder setter.
    if let (Some(target), Some(extra)) = (
        body.as_object_mut(),
        config.extra_body.as_ref().and_then(|v| v.as_object()),
    ) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
    Ok(body)
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    pub stream: bool,
    pub stream_options: StreamOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct StreamOptions {
    pub include_usage: bool,
}

/// Strips `stream_options` from a prepared body, reporting whether it was there.
/// Some OpenAI-compatible endpoints reject the field with a 400.
pub(crate) fn drop_stream_options(body: &mut serde_json::Value) -> bool {
    body.as_object_mut()
        .and_then(|object| object.remove("stream_options"))
        .is_some()
}

// --- SSE parsing -----------------------------------------------------------

#[derive(Default, Deserialize)]
struct SseFrame {
    #[serde(default)]
    error: Option<SseError>,
    #[serde(default)]
    choices: Vec<SseChoice>,
    /// The model the server says is answering; differs from the requested spec
    /// behind aliases and proxies. Absent on providers that omit it.
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<SseUsage>,
}

#[derive(Default, Deserialize)]
struct SseUsage {
    #[serde(default)]
    prompt_tokens: Option<usize>,
    #[serde(default)]
    completion_tokens: Option<usize>,
}

/// Token counts as the provider reported them; a field is None when it said
/// nothing about that side.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ReportedUsage {
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
}

/// Reported counts win field by field; whatever the provider omitted falls back
/// to the estimate. The third element says whether any field was estimated.
pub(crate) fn resolve_usage_tokens(
    reported: ReportedUsage,
    estimated_input: usize,
    estimated_output: usize,
) -> (usize, usize, bool) {
    (
        reported.prompt_tokens.unwrap_or(estimated_input),
        reported.completion_tokens.unwrap_or(estimated_output),
        reported.prompt_tokens.is_none() || reported.completion_tokens.is_none(),
    )
}

#[derive(Default, Deserialize)]
struct SseError {
    #[serde(default)]
    message: String,
}

#[derive(Default, Deserialize)]
struct SseChoice {
    #[serde(default)]
    delta: SseDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct SseDelta {
    #[serde(default)]
    tool_calls: Vec<crate::toolcall::ToolCallDelta>,
    // Option, not String: a tool-call delta carries an explicit `content: null`,
    // which `#[serde(default)]` alone does not accept.
    #[serde(default)]
    content: Option<String>,
    #[serde(default, alias = "reasoning")]
    reasoning_content: Option<String>,
}

struct SseParser {
    buffer: Vec<u8>,
    think_filter: ThinkTagFilter,
    finish_reason: Option<String>,
    resolved_model: Option<String>,
    reported_usage: ReportedUsage,
    tools: crate::toolcall::ToolCallAccumulator,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            think_filter: ThinkTagFilter::new(),
            finish_reason: None,
            resolved_model: None,
            reported_usage: ReportedUsage::default(),
            tools: crate::toolcall::ToolCallAccumulator::default(),
        }
    }

    /// The assembled tool calls; complete only once the stream has ended.
    fn tool_calls(&self) -> Vec<ToolCall> {
        self.tools.result()
    }

    /// The provider's finish reason, verbatim; None until a frame carries one.
    fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// The server-reported model, from the first frame that names one.
    fn resolved_model(&self) -> Option<&str> {
        self.resolved_model.as_deref()
    }

    /// Token counts the provider reported, merged across every frame that
    /// carried a `usage` object.
    fn reported_usage(&self) -> ReportedUsage {
        self.reported_usage
    }

    fn feed(&mut self, bytes: &[u8]) -> Vec<Result<StreamChunk, Error>> {
        self.buffer.extend_from_slice(bytes);
        let mut chunks = Vec::new();

        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buffer.drain(..=pos).collect();
            let mut line = std::str::from_utf8(&line_bytes[..line_bytes.len() - 1])
                .map(str::to_string)
                .unwrap_or_else(|_| {
                    String::from_utf8_lossy(&line_bytes[..line_bytes.len() - 1]).into_owned()
                });
            if line.ends_with('\r') {
                line.pop();
            }

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed == "data: [DONE]" {
                continue;
            }
            let payload = match trimmed.strip_prefix("data:") {
                Some(p) => p.trim(),
                None => continue,
            };
            if payload.is_empty() {
                continue;
            }

            match self.parse_sse_data(payload) {
                Ok(Some(chunk)) => chunks.push(Ok(chunk)),
                Ok(None) => {}
                Err(e) => chunks.push(Err(e)),
            }
        }

        chunks
    }

    fn flush(&mut self) -> Option<StreamChunk> {
        let chunk = self.think_filter.flush();
        (!chunk.is_empty()).then_some(chunk)
    }

    fn parse_sse_data(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        let frame: SseFrame = serde_json::from_str(data)?;

        if self.resolved_model.is_none() {
            // `keepalive` is omlx's transport sentinel, not a model identity.
            self.resolved_model = frame
                .model
                .filter(|model| !model.is_empty() && model != "keepalive");
        }

        // Providers may split the two counts across frames, so fields merge
        // individually rather than replacing the pair.
        if let Some(usage) = frame.usage {
            if usage.prompt_tokens.is_some() {
                self.reported_usage.prompt_tokens = usage.prompt_tokens;
            }
            if usage.completion_tokens.is_some() {
                self.reported_usage.completion_tokens = usage.completion_tokens;
            }
        }

        if let Some(err) = frame.error {
            let message = if err.message.is_empty() {
                "unknown error".to_string()
            } else {
                err.message
            };
            return Err(Error::Stream {
                message,
                partial: String::new(),
            });
        }

        let choice = match frame.choices.into_iter().next() {
            Some(choice) => choice,
            None => return Ok(None),
        };
        if let Some(reason) = choice.finish_reason {
            self.finish_reason = Some(reason);
        }
        let mut delta = choice.delta;

        // Tool-call fragments feed the accumulator; they are never forwarded as
        // chunks, so a caller only ever sees complete calls after the stream ends.
        if !delta.tool_calls.is_empty() {
            self.tools.feed(std::mem::take(&mut delta.tool_calls));
        }

        let content = delta.content.unwrap_or_default();
        let reasoning = delta.reasoning_content.unwrap_or_default();
        if content.is_empty() && reasoning.is_empty() {
            return Ok(None);
        }

        let raw = StreamChunk { content, reasoning };
        let filtered = self.think_filter.feed(raw);
        Ok((!filtered.is_empty()).then_some(filtered))
    }
}

/// Everything a consumed (non-streaming) response yielded.
pub(crate) struct SseOutcome {
    pub content: String,
    pub reasoning: String,
    pub ttft: Option<Duration>,
    pub finish_reason: Option<String>,
    pub resolved_model: Option<String>,
    pub reported_usage: ReportedUsage,
    pub tool_calls: Vec<ToolCall>,
}

async fn consume_sse_stream(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
    handler: Option<&(dyn Fn(&StreamChunk) + Send + Sync)>,
    start: Instant,
) -> Result<SseOutcome, Error> {
    let mut parser = SseParser::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut first_token_time: Option<Instant> = None;

    tokio::pin!(byte_stream);

    while let Some(result) = byte_stream.next().await {
        let bytes = result?;
        for chunk_result in parser.feed(&bytes) {
            let chunk = chunk_result?;
            if first_token_time.is_none() && !chunk.content.is_empty() {
                first_token_time = Some(Instant::now());
            }
            content.push_str(&chunk.content);
            reasoning.push_str(&chunk.reasoning);
            if let Some(handler) = handler {
                handler(&chunk);
            }
        }
    }

    if let Some(chunk) = parser.flush() {
        content.push_str(&chunk.content);
        reasoning.push_str(&chunk.reasoning);
        if let Some(handler) = handler {
            handler(&chunk);
        }
    }

    let ttft = first_token_time.map(|t| t.duration_since(start));
    Ok(SseOutcome {
        content: content.trim().to_string(),
        reasoning: reasoning.trim().to_string(),
        ttft,
        finish_reason: parser.finish_reason().map(str::to_string),
        resolved_model: parser.resolved_model().map(str::to_string),
        reported_usage: parser.reported_usage(),
        tool_calls: parser.tool_calls(),
    })
}

async fn process_sse_task(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
    tx: mpsc::Sender<Result<StreamChunk, Error>>,
    tail: SharedTail,
) {
    let mut parser = SseParser::new();
    // Record what the stream yielded besides chunks, however it ends.
    let record = |parser: &SseParser| {
        if let Ok(mut tail) = tail.lock() {
            tail.finish_reason = parser.finish_reason().map(str::to_string);
            tail.resolved_model = parser.resolved_model().map(str::to_string);
            tail.reported_usage = parser.reported_usage();
            tail.tool_calls = parser.tool_calls();
        }
    };
    // The resolved model is published as soon as a frame names it, so a consumer
    // can show it mid-stream rather than only after exhaustion.
    let publish_model = |parser: &SseParser| {
        let Some(model) = parser.resolved_model() else {
            return;
        };
        if let Ok(mut tail) = tail.lock() {
            if tail.resolved_model.is_none() {
                tail.resolved_model = Some(model.to_string());
            }
        }
    };
    tokio::pin!(byte_stream);

    loop {
        let result = tokio::select! {
            biased;
            _ = tx.closed() => { record(&parser); return },
            result = byte_stream.next() => result,
        };
        let Some(result) = result else {
            break;
        };

        match result {
            Ok(bytes) => {
                let parsed = parser.feed(&bytes);
                publish_model(&parser);
                for chunk_result in parsed {
                    match chunk_result {
                        Ok(chunk) => {
                            if tx.send(Ok(chunk)).await.is_err() {
                                record(&parser);
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            record(&parser);
                            return;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(Error::Request(e))).await;
                record(&parser);
                return;
            }
        }
    }

    if let Some(chunk) = parser.flush() {
        let _ = tx.send(Ok(chunk)).await;
    }
    record(&parser);
}

// --- StreamResponse --------------------------------------------------------

/// Everything a consumed stream yields besides the chunks themselves. Written by
/// the SSE task once the stream ends, so accessors read empty until then.
#[derive(Default)]
pub(crate) struct StreamTail {
    finish_reason: Option<String>,
    resolved_model: Option<String>,
    reported_usage: ReportedUsage,
    tool_calls: Vec<ToolCall>,
}

pub(crate) type SharedTail = std::sync::Arc<std::sync::Mutex<StreamTail>>;

/// Everything `StreamResponse::new` needs from the dispatch that produced it.
pub(crate) struct StreamInit {
    pub rx: mpsc::Receiver<Result<StreamChunk, Error>>,
    pub tail: SharedTail,
    pub model: String,
    pub model_name: String,
    pub provider: String,
    pub api_key_hint: String,
    pub input_tokens: usize,
    /// When the request was sent, so duration and ttft include connect time.
    pub start: Instant,
    /// The chunk already taken off the channel to prove the leg works.
    pub first: Option<StreamChunk>,
    pub hook: Option<std::sync::Arc<EventHook>>,
}

pub struct StreamResponse {
    tail: SharedTail,
    rx: tokio_stream::wrappers::ReceiverStream<Result<StreamChunk, Error>>,
    model: String,
    model_name: String,
    provider: String,
    api_key_hint: String,
    reasoning: String,
    /// Everything handed to the consumer so far, so a mid-stream failure can
    /// report what did arrive.
    delivered: String,
    first: Option<StreamChunk>,
    hook: Option<std::sync::Arc<EventHook>>,
    input_chars: usize,
    output_chars: usize,
    start: Instant,
    first_token_time: Option<Instant>,
    finished: bool,
}

impl StreamResponse {
    pub(crate) fn new(init: StreamInit) -> Self {
        Self {
            tail: init.tail,
            rx: tokio_stream::wrappers::ReceiverStream::new(init.rx),
            model: init.model,
            model_name: init.model_name,
            provider: init.provider,
            api_key_hint: init.api_key_hint,
            reasoning: String::new(),
            delivered: String::new(),
            first: init.first,
            hook: init.hook,
            input_chars: init.input_tokens.saturating_mul(4),
            output_chars: 0,
            start: init.start,
            first_token_time: None,
            finished: false,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn reasoning(&self) -> &str {
        &self.reasoning
    }

    /// The provider's finish reason, verbatim and never normalized. Populated
    /// once the stream is exhausted, like `reasoning` and `usage`.
    pub fn finish_reason(&self) -> Option<String> {
        self.tail
            .lock()
            .map(|tail| tail.finish_reason.clone())
            .unwrap_or_default()
    }

    /// The model the server reported as answering, available as soon as a frame
    /// names one. None when the backend reports no model.
    pub fn resolved_model(&self) -> Option<String> {
        self.tail
            .lock()
            .map(|tail| tail.resolved_model.clone())
            .unwrap_or_default()
    }

    /// Best available identity of the model that produced this response: the
    /// ResolvedModel when the server named one, else the requested spec.
    pub fn actual_model(&self) -> String {
        self.resolved_model().unwrap_or_else(|| self.model.clone())
    }

    /// Tool calls the model requested, assembled from streamed deltas. Empty
    /// until the stream is exhausted: partial argument JSON is never exposed.
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.tail
            .lock()
            .map(|tail| tail.tool_calls.clone())
            .unwrap_or_default()
    }

    pub fn usage(&self) -> Usage {
        let (input_tokens, output_tokens, estimated) = self.resolved_tokens();
        Usage {
            provider: self.provider.clone(),
            model: self.model.clone(),
            model_name: self.model_name.clone(),
            api_key_hint: self.api_key_hint.clone(),
            input_tokens,
            output_tokens,
            estimated,
            duration: self.start.elapsed(),
            ttft: self.first_token_time.map(|t| t.duration_since(self.start)),
        }
    }

    /// Token counts as they stand now: exact where the provider reported them,
    /// estimated otherwise. Reported counts only arrive with the final frames.
    fn resolved_tokens(&self) -> (usize, usize, bool) {
        let reported = self
            .tail
            .lock()
            .map(|tail| tail.reported_usage)
            .unwrap_or_default();
        resolve_usage_tokens(reported, self.input_chars / 4, self.output_chars / 4)
    }
}

impl StreamResponse {
    /// Records a chunk on its way to the consumer.
    fn account(&mut self, chunk: &StreamChunk) {
        if self.first_token_time.is_none() && !chunk.content.is_empty() {
            self.first_token_time = Some(Instant::now());
        }
        self.output_chars += chunk.content.len() + chunk.reasoning.len();
        self.reasoning.push_str(&chunk.reasoning);
        self.delivered.push_str(&chunk.content);
    }

    /// Ends the attempt exactly once: logs its metrics and fires the hook.
    fn finish(&mut self, error: Option<String>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let usage = self.usage();
        log::info!(
            "{}",
            format_metrics(
                &self.model_name,
                usage.input_tokens,
                usage.output_tokens,
                usage.duration,
                usage.ttft,
                usage.estimated,
            )
        );
        if let Some(hook) = self.hook.clone() {
            hook(RequestEvent { usage, error });
        }
    }
}

impl Stream for StreamResponse {
    type Item = Result<StreamChunk, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.first.take() {
            self.account(&chunk);
            return Poll::Ready(Some(Ok(chunk)));
        }
        match Pin::new(&mut self.rx).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                self.account(&chunk);
                Poll::Ready(Some(Ok(chunk)))
            }
            // Only reachable once content has been delivered: a failure before
            // that was consumed by `take_first_chunk` and fell back instead. The
            // consumer keeps what arrived.
            Poll::Ready(Some(Err(error))) => {
                let message = error.to_string();
                self.finish(Some(message.clone()));
                Poll::Ready(Some(Err(Error::Stream {
                    message,
                    partial: self.delivered.clone(),
                })))
            }
            Poll::Ready(None) => {
                self.finish(None);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// What the fallback driver needs from a finished attempt.
trait Finalized {
    /// Usage to report for a successful attempt, or None when the value fires
    /// its own hook later.
    fn success_usage(&self) -> Option<Usage>;
}

impl Finalized for LLMResponse {
    fn success_usage(&self) -> Option<Usage> {
        Some(self.usage.clone())
    }
}

impl Finalized for StreamResponse {
    /// None: a stream has produced nothing yet, so its hook fires at exhaustion
    /// with real counts rather than here with zeros.
    fn success_usage(&self) -> Option<Usage> {
        None
    }
}

#[cfg(test)]
mod tests;
