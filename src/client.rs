use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_core::Stream;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use crate::balancer;
use crate::error::Error;
use crate::image::image_to_data_url;
use crate::provider::{build_request_url, parse_model_string, resolve_api_key, resolve_base_url};
use crate::think::ThinkTagFilter;
use crate::types::*;
use crate::utils::*;

const MAX_RETRIES: usize = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_secs(2);
const RETRY_MAX_DELAY: Duration = Duration::from_secs(30);
const RETRY_BACKOFF_SCALE: u32 = 3;

pub(crate) struct RequestConfig {
    pub prompt: Prompt,
    pub system_prompt: Option<String>,
    pub model_input: Option<crate::selector::ModelInput>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Duration,
    pub remove_backticks: bool,
    pub image_paths: Vec<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub reasoning_effort: Option<String>,
    pub handler: Option<Box<dyn Fn(&StreamChunk) + Send + Sync>>,
    pub hook: Option<Box<dyn Fn(RequestEvent) + Send + Sync>>,
    pub http_client: Option<reqwest::Client>,
}

pub(crate) async fn execute_ask(config: RequestConfig) -> Result<LLMResponse, Error> {
    let model_env = config
        .model_input
        .clone()
        .or_else(|| {
            std::env::var("SMOLLLM_MODEL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|s| crate::selector::ModelInput::from(s.as_str()))
        })
        .ok_or(Error::NoValidModels)?;

    let mut selector = model_env.into_selector();
    let mut last_err: Option<Error> = None;

    while let Some(model_str) = selector.next_model() {
        match try_ask_model(&model_str, &config).await {
            Ok(response) => {
                if let Some(ref hook) = config.hook {
                    hook(RequestEvent {
                        usage: response.usage.clone(),
                        error: None,
                    });
                }
                return Ok(response);
            }
            Err(e) => {
                if selector.has_more() {
                    log::warn!("Model {} failed, trying fallback: {}", model_str, e);
                } else {
                    log::warn!("Model {} failed: {}", model_str, e);
                }
                if let Some(ref hook) = config.hook {
                    hook(RequestEvent {
                        usage: Usage::default(),
                        error: Some(e.to_string()),
                    });
                }
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or(Error::NoValidModels))
}

async fn try_ask_model(model_str: &str, config: &RequestConfig) -> Result<LLMResponse, Error> {
    let parsed = parse_model_string(model_str)?;
    let base_url = resolve_base_url(&parsed, config.base_url.as_deref())?;
    let api_key = resolve_api_key(&parsed, config.api_key.as_deref())?;
    let (chosen_key, chosen_url) = balancer::choose_pair(&api_key, &base_url)?;
    let request_url = build_request_url(&chosen_url, &parsed.provider_name);
    let body = build_request_body(&parsed.model_name, config)?;
    let input_tokens = estimate_tokens(&serde_json::to_string(&body).unwrap_or_default());

    log::info!(
        "Sending request: url={} model={} key={} approx_tokens={}",
        request_url,
        parsed.model_name,
        preview_api_key(&chosen_key),
        input_tokens,
    );

    let mut last_err: Option<Error> = None;
    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let delay = retry_delay(attempt);
            log::warn!(
                "Retrying after transient error: model={} attempt={} delay={:?} err={}",
                model_str,
                attempt + 1,
                delay,
                last_err.as_ref().map(|e| e.to_string()).unwrap_or_default(),
            );
            tokio::time::sleep(delay).await;
        }

        let client = config
            .http_client
            .as_ref()
            .cloned()
            .unwrap_or_else(reqwest::Client::new);

        let start = Instant::now();

        let resp = client
            .post(&request_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {chosen_key}"))
            .timeout(config.timeout)
            .json(&body)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(Error::Request(e));
                if last_err.as_ref().is_some_and(|e| e.is_retryable()) && attempt < MAX_RETRIES - 1
                {
                    continue;
                }
                return Err(last_err.unwrap());
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

        let byte_stream = resp.bytes_stream();
        let (content, reasoning, ttft) =
            consume_sse_stream(byte_stream, config.handler.as_deref(), start).await?;

        let content = if config.remove_backticks {
            strip_backticks(&content)
        } else {
            content
        };

        if content.trim().is_empty() && reasoning.trim().is_empty() {
            return Err(Error::EmptyResponse {
                model: model_str.to_string(),
            });
        }

        let total = start.elapsed();
        let output_tokens = estimate_tokens(&format!("{content}{reasoning}"));

        log::info!(
            "{}",
            format_metrics(&parsed.model_name, input_tokens, output_tokens, total, ttft)
        );

        return Ok(LLMResponse {
            text: content,
            reasoning,
            model: model_str.to_string(),
            model_name: parsed.model_name.clone(),
            provider: parsed.provider_name.clone(),
            usage: Usage {
                provider: parsed.provider_name.clone(),
                model: model_str.to_string(),
                model_name: parsed.model_name.clone(),
                api_key_hint: preview_api_key(&chosen_key),
                input_tokens,
                output_tokens,
                duration: total,
                ttft,
            },
        });
    }

    Err(last_err.unwrap_or(Error::Other("max retries exhausted".into())))
}

pub(crate) async fn execute_stream(config: RequestConfig) -> Result<StreamResponse, Error> {
    let model_env = config
        .model_input
        .clone()
        .or_else(|| {
            std::env::var("SMOLLLM_MODEL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .map(|s| crate::selector::ModelInput::from(s.as_str()))
        })
        .ok_or(Error::NoValidModels)?;

    let mut selector = model_env.into_selector();
    let mut last_err: Option<Error> = None;

    while let Some(model_str) = selector.next_model() {
        match try_stream_model(&model_str, &config).await {
            Ok(stream_resp) => return Ok(stream_resp),
            Err(e) => {
                if selector.has_more() {
                    log::warn!("Model {} failed, trying fallback: {}", model_str, e);
                } else {
                    log::warn!("Model {} failed: {}", model_str, e);
                }
                last_err = Some(e);
            }
        }
    }

    Err(last_err.unwrap_or(Error::NoValidModels))
}

async fn try_stream_model(model_str: &str, config: &RequestConfig) -> Result<StreamResponse, Error> {
    let parsed = parse_model_string(model_str)?;
    let base_url = resolve_base_url(&parsed, config.base_url.as_deref())?;
    let api_key = resolve_api_key(&parsed, config.api_key.as_deref())?;
    let (chosen_key, chosen_url) = balancer::choose_pair(&api_key, &base_url)?;
    let request_url = build_request_url(&chosen_url, &parsed.provider_name);
    let body = build_request_body(&parsed.model_name, config)?;
    let input_tokens = estimate_tokens(&serde_json::to_string(&body).unwrap_or_default());

    log::info!(
        "Sending stream request: url={} model={} key={} approx_tokens={}",
        request_url,
        parsed.model_name,
        preview_api_key(&chosen_key),
        input_tokens,
    );

    let mut last_err: Option<Error> = None;
    for attempt in 0..MAX_RETRIES {
        if attempt > 0 {
            let delay = retry_delay(attempt);
            log::warn!(
                "Retrying stream after transient error: model={} attempt={} delay={:?}",
                model_str,
                attempt + 1,
                delay,
            );
            tokio::time::sleep(delay).await;
        }

        let client = config
            .http_client
            .as_ref()
            .cloned()
            .unwrap_or_else(reqwest::Client::new);

        let resp = client
            .post(&request_url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {chosen_key}"))
            .timeout(config.timeout)
            .json(&body)
            .send()
            .await;

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(Error::Request(e));
                if last_err.as_ref().is_some_and(|e| e.is_retryable()) && attempt < MAX_RETRIES - 1
                {
                    continue;
                }
                return Err(last_err.unwrap());
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

        let byte_stream = resp.bytes_stream();
        let (tx, rx) = mpsc::channel::<Result<StreamChunk, Error>>(32);

        tokio::spawn(async move {
            process_sse_task(byte_stream, tx).await;
        });

        return Ok(StreamResponse::new(
            rx,
            model_str.to_string(),
            parsed.model_name.clone(),
            parsed.provider_name.clone(),
            input_tokens,
        ));
    }

    Err(last_err.unwrap_or(Error::Other("max retries exhausted".into())))
}

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
struct ChatCompletionRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

// --- SSE parsing ---

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
            if chunk.is_empty() {
                continue;
            }
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
    Ok((content.trim().to_string(), reasoning.trim().to_string(), ttft))
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
                        Ok(chunk) if !chunk.is_empty() => {
                            if tx.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                        _ => {}
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
        if !chunk.is_empty() {
            let _ = tx.send(Ok(chunk)).await;
        }
    }
}

struct SseParser {
    buffer: String,
    think_filter: ThinkTagFilter,
}

impl SseParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            think_filter: ThinkTagFilter::new(),
        }
    }

    fn feed(&mut self, bytes: &[u8]) -> Vec<Result<StreamChunk, Error>> {
        let text = String::from_utf8_lossy(bytes);
        self.buffer.push_str(&text);

        let mut chunks = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.trim().is_empty() {
                continue;
            }

            let trimmed = line.trim();
            if trimmed == "data: [DONE]" {
                continue;
            }

            if let Some(data) = trimmed.strip_prefix("data:") {
                let payload = data.trim();
                if payload.is_empty() {
                    continue;
                }
                match self.parse_sse_data(payload) {
                    Ok(Some(chunk)) => chunks.push(Ok(chunk)),
                    Ok(None) => {}
                    Err(e) => chunks.push(Err(e)),
                }
            }
        }

        chunks
    }

    fn flush(&mut self) -> Option<StreamChunk> {
        let chunk = self.think_filter.flush();
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }

    fn parse_sse_data(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        let json: serde_json::Value = serde_json::from_str(data)?;

        if let Some(error) = json.get("error") {
            let msg = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(Error::Stream {
                message: msg.to_string(),
                partial: String::new(),
            });
        }

        let delta = match json.pointer("/choices/0/delta") {
            Some(d) => d,
            None => return Ok(None),
        };

        let content = delta
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let reasoning_content = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(|r| r.as_str())
            .unwrap_or("");

        if content.is_empty() && reasoning_content.is_empty() {
            return Ok(None);
        }

        let raw = StreamChunk {
            content: content.to_string(),
            reasoning: reasoning_content.to_string(),
        };

        let filtered = self.think_filter.feed(raw);
        if filtered.is_empty() {
            Ok(None)
        } else {
            Ok(Some(filtered))
        }
    }
}

fn retry_delay(attempt: usize) -> Duration {
    let mut d = RETRY_BASE_DELAY;
    for _ in 0..attempt {
        d = d.saturating_mul(RETRY_BACKOFF_SCALE);
    }
    d.min(RETRY_MAX_DELAY)
}

// --- StreamResponse ---

use std::pin::Pin;
use std::task::{Context, Poll};

pub struct StreamResponse {
    rx: tokio_stream::wrappers::ReceiverStream<Result<StreamChunk, Error>>,
    pub model: String,
    pub model_name: String,
    pub provider: String,
    pub reasoning: String,
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
            input_chars: input_tokens * 4,
            output_chars: 0,
            start: Instant::now(),
            first_token_time: None,
            metrics_logged: false,
        }
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
