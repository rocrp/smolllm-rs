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
use crate::error::Error;
use crate::image::image_to_data_url;
use crate::provider::{
    build_request_url, parse_model_string, resolve_api_key, resolve_base_url, ParsedModel,
};
use crate::request::RequestConfig;
use crate::think::ThinkTagFilter;
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
    run_with_fallback(&config, |dispatch, response, started, _config| {
        Box::pin(stream_finalize(dispatch, response, started))
    })
    .await
}

// --- Fallback driver -------------------------------------------------------

type FinalizeFut<'a, R> = Pin<Box<dyn Future<Output = Result<R, Error>> + Send + 'a>>;

async fn run_with_fallback<R, F>(config: &RequestConfig, finalize: F) -> Result<R, Error>
where
    R: HasUsage,
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
                fire_hook(config, response.usage(), None);
                return Ok(response);
            }
            Err(err) => {
                if selector.has_more() {
                    log::warn!("Model {model_str} failed, trying fallback: {err}");
                } else {
                    log::warn!("Model {model_str} failed: {err}");
                }
                fire_hook(config, Usage::default(), Some(err.to_string()));
                last_err = Some(err);
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
) -> Result<R, Error>
where
    F: for<'a> Fn(Dispatch, reqwest::Response, Instant, &'a RequestConfig) -> FinalizeFut<'a, R>,
{
    let dispatch = Dispatch::prepare(model_str, config)?;
    log::info!(
        "Sending request: url={} model={} key={} approx_tokens={}",
        dispatch.request_url,
        dispatch.parsed.model_name,
        preview_api_key(&dispatch.chosen_key),
        dispatch.input_tokens,
    );
    let started = Instant::now();
    let response = send_with_retry(client, &dispatch, config.timeout).await?;
    finalize(dispatch, response, started, config).await
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
    pub body: ChatCompletionRequest,
    pub input_tokens: usize,
}

impl Dispatch {
    fn prepare(model_str: &str, config: &RequestConfig) -> Result<Self, Error> {
        let parsed = parse_model_string(model_str)?;
        let base_url = resolve_base_url(&parsed, config.base_url.as_deref())?;
        let api_key = resolve_api_key(&parsed, config.api_key.as_deref())?;
        let (chosen_key, chosen_url) = balancer::choose_pair(&api_key, &base_url)?;
        let request_url = build_request_url(&chosen_url, &parsed.provider_name);
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
}

// --- HTTP send with retry --------------------------------------------------

async fn send_with_retry(
    client: &reqwest::Client,
    dispatch: &Dispatch,
    timeout: Duration,
) -> Result<reqwest::Response, Error> {
    let mut last_err: Option<Error> = None;
    for attempt in 0..MAX_RETRIES {
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
                    continue;
                }
                return Err(err);
            }
        };

        let status = resp.status().as_u16();
        if status >= 400 {
            let body_text = resp.text().await.unwrap_or_default();
            let err = Error::Http {
                status,
                body: body_text,
            };
            if err.is_retryable() && attempt < MAX_RETRIES - 1 {
                last_err = Some(err);
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
    let (content, reasoning, ttft) =
        consume_sse_stream(byte_stream, config.handler.as_deref(), started).await?;

    let content = if config.remove_backticks {
        strip_backticks(&content)
    } else {
        content
    };

    if content.trim().is_empty() && reasoning.trim().is_empty() {
        return Err(Error::EmptyResponse {
            model: dispatch.model_str,
        });
    }

    let total = started.elapsed();
    let output_tokens = estimate_tokens(&format!("{content}{reasoning}"));

    log::info!(
        "{}",
        format_metrics(
            &dispatch.parsed.model_name,
            dispatch.input_tokens,
            output_tokens,
            total,
            ttft
        )
    );

    Ok(LLMResponse {
        text: content,
        reasoning,
        model: dispatch.model_str.clone(),
        model_name: dispatch.parsed.model_name.clone(),
        provider: dispatch.parsed.provider_name.clone(),
        usage: Usage {
            provider: dispatch.parsed.provider_name,
            model: dispatch.model_str,
            model_name: dispatch.parsed.model_name,
            api_key_hint: preview_api_key(&dispatch.chosen_key),
            input_tokens: dispatch.input_tokens,
            output_tokens,
            duration: total,
            ttft,
        },
    })
}

async fn stream_finalize(
    dispatch: Dispatch,
    response: reqwest::Response,
    _started: Instant,
) -> Result<StreamResponse, Error> {
    let byte_stream = response.bytes_stream();
    let (tx, rx) = mpsc::channel::<Result<StreamChunk, Error>>(SSE_CHANNEL_CAPACITY);

    tokio::spawn(async move {
        process_sse_task(byte_stream, tx).await;
    });

    Ok(StreamResponse::new(
        rx,
        dispatch.model_str,
        dispatch.parsed.model_name,
        dispatch.parsed.provider_name,
        dispatch.input_tokens,
    ))
}

// --- Request body construction ---------------------------------------------

fn build_request_body(
    model_name: &str,
    config: &RequestConfig,
) -> Result<ChatCompletionRequest, Error> {
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
                messages.push(serde_json::json!({
                    "role": msg.role.as_str(),
                    "content": msg.content,
                }));
            }
        }
    }

    Ok(ChatCompletionRequest {
        model: model_name.to_string(),
        messages,
        stream: true,
        temperature: config.temperature,
        top_p: config.top_p,
        reasoning_effort: config.reasoning_effort.clone(),
    })
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

// --- SSE parsing -----------------------------------------------------------

#[derive(Default, Deserialize)]
struct SseFrame {
    #[serde(default)]
    error: Option<SseError>,
    #[serde(default)]
    choices: Vec<SseChoice>,
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
}

#[derive(Default, Deserialize)]
struct SseDelta {
    #[serde(default)]
    content: String,
    #[serde(default, alias = "reasoning")]
    reasoning_content: String,
}

struct SseParser {
    buffer: Vec<u8>,
    think_filter: ThinkTagFilter,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            think_filter: ThinkTagFilter::new(),
        }
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

        let delta = match frame.choices.into_iter().next() {
            Some(choice) => choice.delta,
            None => return Ok(None),
        };

        if delta.content.is_empty() && delta.reasoning_content.is_empty() {
            return Ok(None);
        }

        let raw = StreamChunk {
            content: delta.content,
            reasoning: delta.reasoning_content,
        };
        let filtered = self.think_filter.feed(raw);
        Ok((!filtered.is_empty()).then_some(filtered))
    }
}

async fn consume_sse_stream(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
    handler: Option<&(dyn Fn(&StreamChunk) + Send + Sync)>,
    start: Instant,
) -> Result<(String, String, Option<Duration>), Error> {
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
    Ok((
        content.trim().to_string(),
        reasoning.trim().to_string(),
        ttft,
    ))
}

async fn process_sse_task(
    byte_stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send,
    tx: mpsc::Sender<Result<StreamChunk, Error>>,
) {
    let mut parser = SseParser::new();
    tokio::pin!(byte_stream);

    while let Some(result) = byte_stream.next().await {
        match result {
            Ok(bytes) => {
                for chunk_result in parser.feed(&bytes) {
                    match chunk_result {
                        Ok(chunk) => {
                            if tx.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(Err(Error::Request(e))).await;
                return;
            }
        }
    }

    if let Some(chunk) = parser.flush() {
        let _ = tx.send(Ok(chunk)).await;
    }
}

// --- StreamResponse --------------------------------------------------------

pub struct StreamResponse {
    rx: tokio_stream::wrappers::ReceiverStream<Result<StreamChunk, Error>>,
    model: String,
    model_name: String,
    provider: String,
    reasoning: String,
    input_chars: usize,
    output_chars: usize,
    start: Instant,
    first_token_time: Option<Instant>,
    metrics_logged: bool,
}

impl StreamResponse {
    pub(crate) fn new(
        rx: mpsc::Receiver<Result<StreamChunk, Error>>,
        model: String,
        model_name: String,
        provider: String,
        input_tokens: usize,
    ) -> Self {
        Self {
            rx: tokio_stream::wrappers::ReceiverStream::new(rx),
            model,
            model_name,
            provider,
            reasoning: String::new(),
            input_chars: input_tokens.saturating_mul(4),
            output_chars: 0,
            start: Instant::now(),
            first_token_time: None,
            metrics_logged: false,
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

    pub fn usage(&self) -> Usage {
        Usage {
            provider: self.provider.clone(),
            model: self.model.clone(),
            model_name: self.model_name.clone(),
            api_key_hint: String::new(),
            input_tokens: self.input_chars / 4,
            output_tokens: self.output_chars / 4,
            duration: self.start.elapsed(),
            ttft: self.first_token_time.map(|t| t.duration_since(self.start)),
        }
    }
}

impl Stream for StreamResponse {
    type Item = Result<StreamChunk, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.rx).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if self.first_token_time.is_none() && !chunk.content.is_empty() {
                    self.first_token_time = Some(Instant::now());
                }
                self.output_chars += chunk.content.len() + chunk.reasoning.len();
                self.reasoning.push_str(&chunk.reasoning);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => {
                if !self.metrics_logged {
                    self.metrics_logged = true;
                    let total = self.start.elapsed();
                    let ttft = self.first_token_time.map(|t| t.duration_since(self.start));
                    log::info!(
                        "{}",
                        format_metrics(
                            &self.model_name,
                            self.input_chars / 4,
                            self.output_chars / 4,
                            total,
                            ttft,
                        )
                    );
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

trait HasUsage {
    fn usage(&self) -> Usage;
}

impl HasUsage for LLMResponse {
    fn usage(&self) -> Usage {
        self.usage.clone()
    }
}

impl HasUsage for StreamResponse {
    fn usage(&self) -> Usage {
        StreamResponse::usage(self)
    }
}
