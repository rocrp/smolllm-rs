use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::StreamExt;

use super::{process_sse_task, SharedTail, StreamInit, StreamResponse, StreamTail};

/// A byte stream that yields one buffer and then ends, standing in for a
/// provider's SSE response.
struct StaticByteStream {
    chunks: std::vec::IntoIter<Bytes>,
}

impl Stream for StaticByteStream {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.get_mut().chunks.next().map(Ok))
    }
}

fn shared_tail() -> SharedTail {
    std::sync::Arc::new(std::sync::Mutex::new(StreamTail::default()))
}

/// Init for a stream the test drives directly, standing in for a dispatch.
fn test_init(rx: mpsc::Receiver<Result<crate::StreamChunk, crate::Error>>, tail: SharedTail) -> StreamInit {
    StreamInit {
        rx,
        tail,
        model: "custom/model".into(),
        model_name: "model".into(),
        provider: "custom".into(),
        api_key_hint: crate::utils::preview_api_key("sk-1234567890abcdef"),
        input_tokens: 0,
        start: std::time::Instant::now(),
        first: None,
        hook: None,
    }
}

/// Runs the SSE task over one fixed buffer, handing back what a dispatch would.
fn spawn_sse(
    bytes: &'static [u8],
) -> (
    mpsc::Receiver<Result<crate::StreamChunk, crate::Error>>,
    SharedTail,
) {
    let stream = StaticByteStream {
        chunks: vec![Bytes::from_static(bytes)].into_iter(),
    };
    let (tx, rx) = mpsc::channel(16);
    let tail = shared_tail();
    tokio::spawn(process_sse_task(stream, tx, std::sync::Arc::clone(&tail)));
    (rx, tail)
}

/// A `StreamResponse` fed by one fixed SSE buffer.
async fn collect_stream(bytes: &'static [u8]) -> StreamResponse {
    let (rx, tail) = spawn_sse(bytes);
    StreamResponse::new(test_init(rx, tail))
}

struct OneThenPendingByteStream {
    next: Option<Bytes>,
}

impl Stream for OneThenPendingByteStream {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut()
            .next
            .take()
            .map_or(Poll::Pending, |bytes| Poll::Ready(Some(Ok(bytes))))
    }
}

#[tokio::test]
async fn streaming_task_stops_when_response_is_dropped_after_content() {
    let stream = OneThenPendingByteStream {
        next: Some(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
        )),
    };
    let (tx, rx) = mpsc::channel(1);
    let tail = shared_tail();
    let task = tokio::spawn(process_sse_task(stream, tx, std::sync::Arc::clone(&tail)));
    let mut response = StreamResponse::new(test_init(rx, tail));

    let chunk = timeout(Duration::from_secs(1), response.next())
        .await
        .expect("SSE task must forward the first content chunk")
        .unwrap()
        .unwrap();
    assert_eq!(chunk.content, "hello");

    drop(response);

    timeout(Duration::from_secs(1), task)
        .await
        .expect("SSE task must stop when its response receiver is dropped")
        .unwrap();
}

// --- rs#4: escape hatch, finish_reason, null content ------------------------

use serde_json::json;

use super::{build_request_body, SseParser};
use crate::request::RequestConfig;
use crate::types::Prompt;

fn config_with(prompt: &str) -> RequestConfig {
    RequestConfig::new(Prompt::Text(prompt.into()))
}

fn body_json(config: &RequestConfig) -> serde_json::Value {
    build_request_body("m", None, config).expect("body builds")
}

/// The body a model spec produces, with its `!effort` suffix applied.
fn body_for_spec(spec: &str, config: &RequestConfig) -> serde_json::Value {
    let parsed = crate::provider::parse_model_string(spec).expect("spec parses");
    build_request_body(
        &parsed.model_name,
        parsed.reasoning_effort.as_deref(),
        config,
    )
    .expect("body builds")
}

#[test]
fn extra_body_fields_reach_the_payload() {
    let mut config = config_with("hi");
    config.extra_body = Some(json!({"tools": [{"type": "function"}], "max_tokens": 256}));

    let body = body_json(&config);
    assert_eq!(body["max_tokens"], json!(256));
    assert_eq!(body["tools"][0]["type"], json!("function"));
}

#[test]
fn extra_body_wins_over_library_defaults() {
    let mut config = config_with("hi");
    config.temperature = Some(0.1);
    config.extra_body = Some(json!({"temperature": 1.5}));

    assert_eq!(body_json(&config)["temperature"], json!(1.5));
}

#[test]
fn extra_body_is_absent_when_not_set() {
    assert!(body_json(&config_with("hi")).get("tools").is_none());
}

#[test]
fn extra_body_rejects_reserved_keys_in_the_setter() {
    for key in ["stream", "stream_options", "messages", "model"] {
        let err = crate::ask("hi")
            .extra_body(json!({ key: "x" }))
            .err()
            .expect("reserved key must be rejected");
        assert!(
            matches!(&err, crate::Error::InvalidParam(msg) if msg.contains(key)),
            "expected InvalidParam naming {key}, got {err}"
        );
    }
}

#[test]
fn extra_body_names_every_reserved_offender() {
    let err = crate::ask("hi")
        .extra_body(json!({"model": "x", "messages": "y", "tools": "z"}))
        .err()
        .expect("reserved keys must be rejected");
    let crate::Error::InvalidParam(msg) = &err else {
        panic!("expected InvalidParam, got {err}")
    };
    assert!(msg.contains("messages") && msg.contains("model"), "{msg}");
}

#[test]
fn extra_body_rejects_non_objects() {
    let err = crate::ask("hi")
        .extra_body(json!([1, 2, 3]))
        .err()
        .expect("extra_body must be an object");
    assert!(matches!(err, crate::Error::InvalidParam(_)));
}

#[test]
fn parser_surfaces_finish_reason_verbatim() {
    let mut parser = SseParser::new();
    parser.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n");
    parser.feed(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n");
    assert_eq!(parser.finish_reason(), Some("tool_calls"));
}

#[test]
fn parser_has_no_finish_reason_when_provider_omits_it() {
    let mut parser = SseParser::new();
    parser.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n");
    assert_eq!(parser.finish_reason(), None);
}

#[test]
fn null_content_delta_is_not_an_error() {
    let mut parser = SseParser::new();
    let results = parser.feed(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n");
    assert!(
        results.iter().all(Result::is_ok),
        "a null content delta must parse: real tool-call deltas carry exactly that"
    );
    assert!(results.is_empty(), "an empty delta yields no chunk");
}

// --- rs#5: tool calls -------------------------------------------------------

use crate::toolcall::{ToolCall, ToolCallFunction};

fn feed_frame(parser: &mut SseParser, payload: &str) -> Vec<Result<crate::StreamChunk, crate::Error>> {
    parser.feed(format!("data: {payload}\n").as_bytes())
}

#[test]
fn accumulator_merges_argument_fragments() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}"#,
    );

    let calls = parser.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[0].kind, "function");
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn accumulator_keeps_parallel_calls_ordered() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"second"}},{"index":0,"id":"a","type":"function","function":{"name":"first"}}]}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
    );

    let calls = parser.tool_calls();
    assert_eq!(
        calls.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(calls[0].function.arguments, "{}");
}

#[test]
fn accumulator_without_index_starts_new_call_on_id() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f"}}]}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"x\":1}"}}]}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"id":"b","type":"function","function":{"name":"g"}}]}}]}"#,
    );

    let calls = parser.tool_calls();
    assert_eq!(
        calls.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(calls[0].function.arguments, r#"{"x":1}"#);
}

#[test]
fn accumulator_keeps_provider_extras() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{}"},"extra_content":{"google":{"thought_signature":"sig"}}}]}}]}"#,
    );

    let calls = parser.tool_calls();
    assert_eq!(
        calls[0].extra.get("extra_content"),
        Some(&json!({"google":{"thought_signature":"sig"}}))
    );
}

#[test]
fn tool_call_deltas_are_never_forwarded_as_chunks() {
    let mut parser = SseParser::new();
    let results = feed_frame(
        &mut parser,
        r#"{"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#,
    );
    assert!(results.is_empty(), "partial tool calls must not reach consumers");
    assert_eq!(parser.tool_calls().len(), 1);
}

#[test]
fn tool_call_round_trips_extras_through_json() {
    let raw = json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "sig"}}
    });
    let call: ToolCall = serde_json::from_value(raw.clone()).expect("decodes");
    assert_eq!(call.id, "call_1");
    assert_eq!(serde_json::to_value(&call).expect("encodes"), raw);
}

#[test]
fn replayed_tool_conversation_serializes_losslessly() {
    let call = ToolCall {
        id: "call_1".into(),
        kind: "function".into(),
        function: ToolCallFunction {
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
        },
        extra: json!({"extra_content": {"google": {"thought_signature": "sig"}}})
            .as_object()
            .cloned()
            .expect("object"),
    };
    let config = RequestConfig::new(Prompt::Messages(vec![
        crate::Message::user("weather in Paris?"),
        crate::Message::assistant_tool_calls("", vec![call.clone()]),
        crate::Message::tool("call_1", r#"{"temp_c":18}"#),
    ]));

    let messages = body_json(&config)["messages"].clone();
    assert_eq!(messages[1]["role"], json!("assistant"));
    assert_eq!(messages[1]["content"], json!(null));
    assert_eq!(messages[1]["tool_calls"][0]["id"], json!("call_1"));
    assert_eq!(
        messages[1]["tool_calls"][0]["extra_content"],
        json!({"google": {"thought_signature": "sig"}})
    );
    assert_eq!(messages[2]["role"], json!("tool"));
    assert_eq!(messages[2]["tool_call_id"], json!("call_1"));
    assert_eq!(messages[2]["content"], json!(r#"{"temp_c":18}"#));
}

// --- rs#6: ResolvedModel ----------------------------------------------------

#[test]
fn parser_reports_the_model_the_server_says_answered() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"model":"gpt-5!high","choices":[{"delta":{"content":"hi"}}]}"#,
    );
    assert_eq!(parser.resolved_model(), Some("gpt-5!high"));
}

#[test]
fn parser_keeps_the_last_reported_model() {
    // A relay that fell back after a reasoning-only leg names that leg in its
    // early frames; the finish and usage frames name the leg that answered.
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"model":"failed","choices":[{"delta":{"reasoning_content":"a"}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"model":"answered","choices":[{"delta":{"content":"b"}}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"model":"answered","choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    );
    feed_frame(
        &mut parser,
        r#"{"model":"answered","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#,
    );
    feed_frame(&mut parser, r#"{"model":"keepalive","choices":[]}"#);
    assert_eq!(parser.resolved_model(), Some("answered"));
    assert_eq!(
        parser.reported_usage(),
        super::ReportedUsage {
            prompt_tokens: Some(3),
            completion_tokens: Some(1)
        }
    );
}

#[tokio::test]
async fn stream_response_ends_with_the_model_of_the_last_frame() {
    let mut response = collect_stream(
        concat!(
            r#"data: {"model":"failed","choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
            "\n",
            r#"data: {"model":"answered","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#,
            "\n",
            r#"data: {"model":"answered","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#,
            "\n",
            "data: [DONE]\n",
        )
        .as_bytes(),
    )
    .await;
    while response.next().await.is_some() {}
    assert_eq!(response.resolved_model().as_deref(), Some("answered"));
    assert!(!response.usage().estimated);
}

#[test]
fn parser_ignores_the_omlx_keepalive_sentinel() {
    let mut parser = SseParser::new();
    feed_frame(
        &mut parser,
        r#"{"model":"keepalive","choices":[{"delta":{"content":""}}]}"#,
    );
    assert_eq!(parser.resolved_model(), None, "keepalive is transport, not identity");
    feed_frame(
        &mut parser,
        r#"{"model":"Qwen3.8-27B-4bit","choices":[{"delta":{"content":"hello"}}]}"#,
    );
    assert_eq!(parser.resolved_model(), Some("Qwen3.8-27B-4bit"));
}

#[test]
fn parser_ignores_an_empty_model_field() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"model":"","choices":[{"delta":{"content":"hi"}}]}"#);
    assert_eq!(parser.resolved_model(), None);
}

#[test]
fn parser_has_no_resolved_model_when_frames_omit_it() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"choices":[{"delta":{"content":"hi"}}]}"#);
    assert_eq!(parser.resolved_model(), None);
}

#[tokio::test]
async fn stream_response_reports_the_resolved_model_before_it_is_exhausted() {
    let stream = OneThenPendingByteStream {
        next: Some(Bytes::from_static(
            b"data: {\"model\":\"gpt-5!high\",\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
        )),
    };
    let (tx, rx) = mpsc::channel(1);
    let tail = shared_tail();
    tokio::spawn(process_sse_task(stream, tx, std::sync::Arc::clone(&tail)));
    let mut response = StreamResponse::new(StreamInit {
        model: "smolserver/summary".into(),
        model_name: "summary".into(),
        provider: "smolserver".into(),
        ..test_init(rx, tail)
    });

    let chunk = timeout(Duration::from_secs(1), response.next())
        .await
        .expect("first chunk arrives")
        .unwrap()
        .unwrap();
    assert_eq!(chunk.content, "hello");
    assert_eq!(response.resolved_model().as_deref(), Some("gpt-5!high"));
    assert_eq!(response.actual_model(), "gpt-5!high");
}

#[tokio::test]
async fn actual_model_falls_back_to_the_requested_spec() {
    let (tx, rx) = mpsc::channel::<Result<crate::StreamChunk, crate::Error>>(1);
    drop(tx);
    let response = StreamResponse::new(StreamInit {
        model: "gemini/flash".into(),
        model_name: "flash".into(),
        provider: "gemini".into(),
        ..test_init(rx, shared_tail())
    });

    assert_eq!(response.resolved_model(), None);
    assert_eq!(response.actual_model(), "gemini/flash");
}

// --- rs#7: real usage -------------------------------------------------------

#[test]
fn requests_ask_the_endpoint_to_report_usage() {
    // Every request this library sends streams on the wire, ask included, so
    // both paths can be handed real counts.
    assert_eq!(
        body_json(&config_with("hi"))["stream_options"],
        json!({"include_usage": true})
    );
}

#[test]
fn dropping_stream_options_reports_whether_it_was_there() {
    let mut body = body_json(&config_with("hi"));
    assert!(super::drop_stream_options(&mut body), "first drop removes it");
    assert!(body.get("stream_options").is_none());
    assert!(!super::drop_stream_options(&mut body), "second drop finds nothing");
}

#[test]
fn parser_reads_the_usage_frame() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"choices":[{"delta":{"content":"hi"}}]}"#);
    feed_frame(
        &mut parser,
        r#"{"choices":[],"usage":{"prompt_tokens":9182,"completion_tokens":1104}}"#,
    );
    let reported = parser.reported_usage();
    assert_eq!(reported.prompt_tokens, Some(9182));
    assert_eq!(reported.completion_tokens, Some(1104));
}

#[test]
fn parser_merges_usage_fields_reported_across_frames() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"choices":[],"usage":{"prompt_tokens":10}}"#);
    feed_frame(&mut parser, r#"{"choices":[],"usage":{"completion_tokens":20}}"#);
    let reported = parser.reported_usage();
    assert_eq!(reported.prompt_tokens, Some(10));
    assert_eq!(reported.completion_tokens, Some(20));
}

#[test]
fn parser_reports_no_usage_when_the_provider_omits_it() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"choices":[{"delta":{"content":"hi"}}]}"#);
    assert_eq!(parser.reported_usage(), super::ReportedUsage::default());
}

#[test]
fn reported_counts_win_and_missing_ones_fall_back_to_estimates() {
    use super::{resolve_usage_tokens, ReportedUsage};

    let both = ReportedUsage {
        prompt_tokens: Some(100),
        completion_tokens: Some(7),
    };
    assert_eq!(resolve_usage_tokens(both, 4, 3), (100, 7, false));

    let partial = ReportedUsage {
        prompt_tokens: Some(100),
        completion_tokens: None,
    };
    assert_eq!(
        resolve_usage_tokens(partial, 4, 3),
        (100, 3, true),
        "one estimated field makes the whole usage estimated"
    );

    assert_eq!(
        resolve_usage_tokens(ReportedUsage::default(), 4, 3),
        (4, 3, true)
    );
}

#[tokio::test]
async fn stream_usage_is_exact_once_the_provider_reports_it() {
    let bytes = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9182,\"completion_tokens\":1104}}\n",
    );
    let mut response = collect_stream(bytes.as_bytes()).await;
    while response.next().await.is_some() {}

    let usage = response.usage();
    assert_eq!(usage.input_tokens, 9182);
    assert_eq!(usage.output_tokens, 1104);
    assert!(!usage.estimated);
    assert_eq!(usage.api_key_hint, "sk-12...cdef");
}

#[tokio::test]
async fn stream_usage_is_estimated_when_the_provider_reports_none() {
    let mut response =
        collect_stream(b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n").await;
    while response.next().await.is_some() {}

    let usage = response.usage();
    assert!(usage.estimated);
    assert_eq!(usage.output_tokens, "hello".len() / 4);
}

#[test]
fn estimated_metrics_are_marked_approximate() {
    let exact = crate::utils::format_metrics("m", 10, 5, Duration::from_secs(1), None, false);
    let approx = crate::utils::format_metrics("m", 10, 5, Duration::from_secs(1), None, true);
    assert!(!exact.contains('~'), "{exact}");
    assert!(approx.contains("~15tok"), "{approx}");
}

// --- rs#8: stream fallback, honest timing, hook after exhaustion ------------

use crate::request::EventHook;
use crate::types::RequestEvent;
use std::sync::{Arc, Mutex};

fn recording_hook() -> (Arc<EventHook>, Arc<Mutex<Vec<RequestEvent>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let hook: Arc<EventHook> = Arc::new(move |event: RequestEvent| {
        sink.lock().expect("hook sink").push(event);
    });
    (hook, seen)
}

#[tokio::test]
async fn a_failure_before_any_chunk_is_reported_so_the_chain_can_advance() {
    let (mut rx, tail) = spawn_sse(b"data: {\"error\":{\"message\":\"upstream exploded\"}}\n");

    let err = super::take_first_chunk(&mut rx, &tail, "x/a")
        .await
        .expect_err("an error frame before any content fails the leg");
    assert!(err.to_string().contains("upstream exploded"), "{err}");
}

#[tokio::test]
async fn a_stream_that_ends_without_chunks_is_an_empty_response() {
    let (mut rx, tail) = spawn_sse(b"data: [DONE]\n");

    let err = super::take_first_chunk(&mut rx, &tail, "x/a")
        .await
        .expect_err("a 200 that streams nothing is a failed leg, not a success");
    assert!(
        matches!(&err, crate::Error::EmptyResponse { model } if model == "x/a"),
        "{err}"
    );
}

#[tokio::test]
async fn the_first_chunk_is_replayed_before_the_rest_of_the_stream() {
    let (mut rx, tail) = spawn_sse(
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"one \"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"two\"}}]}\n",
        )
        .as_bytes(),
    );
    let first = super::take_first_chunk(&mut rx, &tail, "x/a")
        .await
        .expect("first chunk");
    let mut response = StreamResponse::new(StreamInit {
        first,
        ..test_init(rx, tail)
    });

    let mut text = String::new();
    while let Some(chunk) = response.next().await {
        text.push_str(&chunk.expect("chunk").content);
    }
    assert_eq!(text, "one two", "the peeked chunk must not be swallowed");
}

#[tokio::test]
async fn an_error_after_the_first_chunk_carries_the_partial_output() {
    let mut response = collect_stream(
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"half an answer\"}}]}\n",
            "data: {\"error\":{\"message\":\"connection reset\"}}\n",
        )
        .as_bytes(),
    )
    .await;

    let first = response.next().await.expect("chunk").expect("ok");
    assert_eq!(first.content, "half an answer");

    let err = response
        .next()
        .await
        .expect("the error reaches the consumer")
        .expect_err("second item is the failure");
    let crate::Error::Stream { message, partial, .. } = err else {
        panic!("post-content failures keep the partial output: {err:?}")
    };
    assert!(message.contains("connection reset"), "{message}");
    assert_eq!(partial, "half an answer");
}

#[tokio::test]
async fn a_timeout_after_content_keeps_its_type_and_partial_output() {
    // Create the deadline, then let it expire before polling the request.
    // reqwest checks the deadline before its network future, so no server is needed.
    let pending = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("request client")
        .get("http://127.0.0.1:1")
        .timeout(Duration::ZERO)
        .send();
    tokio::time::sleep(Duration::from_millis(5)).await;
    let timeout_error = pending.await.expect_err("expired request deadline");
    assert!(timeout_error.is_timeout(), "{timeout_error:?}");

    let bytes = b"data: {\"choices\":[{\"delta\":{\"content\":\"half an answer\"}}]}\n";
    let byte_stream = tokio_stream::iter(vec![Ok(Bytes::from_static(bytes)), Err(timeout_error)]);
    let (tx, rx) = mpsc::channel(2);
    let tail = shared_tail();
    tokio::spawn(process_sse_task(
        byte_stream,
        tx,
        std::sync::Arc::clone(&tail),
    ));
    let mut response = StreamResponse::new(test_init(rx, tail));

    assert_eq!(
        response
            .next()
            .await
            .expect("first chunk")
            .expect("content")
            .content,
        "half an answer"
    );
    let error = response
        .next()
        .await
        .expect("timeout reaches the consumer")
        .expect_err("body read failed");
    assert!(error.is_timeout(), "post-content timeout must remain typed");
    let crate::Error::Stream {
        partial,
        source: Some(source),
        ..
    } = error
    else {
        panic!("post-content failure must retain its source: {error:?}")
    };
    assert_eq!(partial, "half an answer");
    assert!(matches!(*source, crate::Error::Request(_)));
}

#[tokio::test]
async fn the_hook_fires_once_the_stream_is_exhausted() {
    let (hook, seen) = recording_hook();
    let (rx, tail) = spawn_sse(
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":22}}\n",
        )
        .as_bytes(),
    );
    let mut response = StreamResponse::new(StreamInit {
        hook: Some(hook),
        ..test_init(rx, tail)
    });

    assert!(seen.lock().unwrap().is_empty(), "nothing fires mid-stream");
    while response.next().await.is_some() {}

    let events = seen.lock().unwrap();
    assert_eq!(events.len(), 1, "exactly one event per attempt");
    assert!(events[0].error.is_none());
    assert_eq!(events[0].usage.output_tokens, 22, "real counts, not zeros");
    assert_eq!(events[0].usage.model, "custom/model");
}

#[tokio::test]
async fn the_hook_fires_with_the_error_when_a_stream_fails_midway() {
    let (hook, seen) = recording_hook();
    let (rx, tail) = spawn_sse(
        concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"half\"}}]}\n",
            "data: {\"error\":{\"message\":\"boom\"}}\n",
        )
        .as_bytes(),
    );
    let mut response = StreamResponse::new(StreamInit {
        hook: Some(hook),
        ..test_init(rx, tail)
    });

    while response.next().await.is_some() {}

    let events = seen.lock().unwrap();
    assert_eq!(events.len(), 1);
    assert!(
        events[0].error.as_deref().is_some_and(|e| e.contains("boom")),
        "{:?}",
        events[0].error
    );
}

#[tokio::test]
async fn stream_timing_starts_before_the_request_is_sent() {
    let start = std::time::Instant::now()
        .checked_sub(Duration::from_millis(500))
        .expect("clock supports a half-second offset");
    let (rx, tail) = spawn_sse(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n");
    let mut response = StreamResponse::new(StreamInit {
        start,
        ..test_init(rx, tail)
    });

    while response.next().await.is_some() {}

    assert!(
        response.usage().duration >= Duration::from_millis(500),
        "duration must include connect and header wait, got {:?}",
        response.usage().duration
    );
}

// --- rs#9: truncation -------------------------------------------------------

#[test]
fn truncation_is_a_cap_hit_or_a_stream_that_lost_its_final_frame() {
    use super::is_truncated;

    assert!(is_truncated(Some("length"), true), "the provider hit its cap");
    assert!(!is_truncated(Some("stop"), true), "a natural ending is not truncation");
    assert!(!is_truncated(Some("tool_calls"), true));
    assert!(
        is_truncated(None, true),
        "content but no finish reason means the stream was cut off"
    );
    assert!(
        !is_truncated(None, false),
        "nothing at all is the empty-response case, not truncation"
    );
    assert!(!is_truncated(Some("length"), false));
}

#[tokio::test]
async fn a_stream_capped_by_the_provider_reports_truncated() {
    let mut response = collect_stream(concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"cut off mid-\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n",
    ).as_bytes())
    .await;
    while response.next().await.is_some() {}

    assert!(response.truncated());
}

#[tokio::test]
async fn a_stream_that_ended_naturally_is_not_truncated() {
    let mut response = collect_stream(concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"all of it\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
    ).as_bytes())
    .await;
    while response.next().await.is_some() {}

    assert!(!response.truncated());
}

#[tokio::test]
async fn a_stream_that_stopped_without_a_finish_reason_is_truncated() {
    let mut response =
        collect_stream(b"data: {\"choices\":[{\"delta\":{\"content\":\"and then\"}}]}\n").await;
    while response.next().await.is_some() {}

    assert!(
        response.truncated(),
        "an SSE stream that ends without its terminal frame was cut off"
    );
}

// --- rs#11: per-leg !effort suffix ------------------------------------------

#[test]
fn a_leg_suffix_sets_the_reasoning_effort_and_leaves_the_model_clean() {
    let body = body_for_spec("x/m!high", &config_with("hi"));
    assert_eq!(body["model"], json!("m"), "the suffix never reaches the wire");
    assert_eq!(body["reasoning_effort"], json!("high"));
}

#[test]
fn a_leg_suffix_overrides_the_call_level_effort() {
    let mut config = config_with("hi");
    config.reasoning_effort = Some("low".into());

    assert_eq!(body_for_spec("x/a!none", &config)["reasoning_effort"], json!("none"));
    assert_eq!(
        body_for_spec("x/b", &config)["reasoning_effort"],
        json!("low"),
        "a leg without a suffix keeps the call-level effort"
    );
}

#[test]
fn a_bare_model_takes_a_suffix_too() {
    let parsed = crate::provider::parse_model_string("qwen3!none").expect("parses");
    assert_eq!(parsed.model_name, "qwen3");
    assert_eq!(parsed.provider_name, "");
    assert_eq!(parsed.reasoning_effort.as_deref(), Some("none"));
}

#[test]
fn an_empty_effort_suffix_is_rejected_at_parse_time() {
    let err = crate::provider::parse_model_string("x/m!").expect_err("empty effort is a typo");
    assert!(
        matches!(&err, crate::Error::InvalidModel(msg) if msg.contains("x/m!")),
        "{err}"
    );
}

#[test]
fn resolved_endpoints_report_the_model_without_its_suffix() {
    let endpoints = crate::resolve_endpoints("x/m!high", Some("https://gateway.example"))
        .expect("resolves");
    assert_eq!(endpoints[0].model, "m");
    assert_eq!(
        endpoints[0].url,
        "https://gateway.example/v1/chat/completions"
    );
}

// --- rs#12: parity hygiene --------------------------------------------------

#[test]
fn an_empty_leg_is_rejected_instead_of_shortening_the_chain() {
    let err = crate::selector::ModelInput::from("x/a,,x/b")
        .validate()
        .expect_err("an empty entry is a typo, not a two-leg chain");
    assert!(
        matches!(&err, crate::Error::InvalidModelList { reason } if reason.contains("empty entry")),
        "{err}"
    );
    crate::selector::ModelInput::from("x/a,x/b")
        .validate()
        .expect("a well-formed chain still validates");
}

#[test]
fn a_stream_error_frame_is_redacted_and_capped() {
    let mut parser = SseParser::new();
    let results = feed_frame(
        &mut parser,
        r#"{"error":{"message":"rejected Authorization: Bearer sk-live-abcdef123456"}}"#,
    );
    let err = results
        .into_iter()
        .next()
        .expect("an error frame yields an error")
        .expect_err("it is an error");
    let text = err.to_string();
    assert!(!text.contains("sk-live-abcdef123456"), "{text}");
    assert!(text.contains("[REDACTED_CREDENTIAL]"), "{text}");
}

#[test]
fn a_rate_limited_leg_yields_to_the_next_model_instead_of_waiting() {
    use super::should_retry;
    let rate_limited = crate::Error::Http {
        status: 429,
        body: "slow down".into(),
    };

    assert!(
        !should_retry(&rate_limited, 0, true),
        "another model is sitting right behind this one; do not sleep on it"
    );
    assert!(
        should_retry(&rate_limited, 0, false),
        "on the last leg the backoff is all there is"
    );
    assert!(
        !should_retry(&rate_limited, super::MAX_RETRIES - 1, false),
        "retry budget spent"
    );
    assert!(
        !should_retry(
            &crate::Error::Http {
                status: 400,
                body: "nope".into()
            },
            0,
            false
        ),
        "a 400 is not transient"
    );
}

#[tokio::test]
async fn a_turn_that_only_requests_tool_calls_is_not_an_empty_response() {
    // Tool-call deltas never surface as chunks, so this stream yields nothing
    // at all — but it is a complete answer, and failing the leg would throw the
    // assembled call away and move on to another model.
    let (mut rx, tail) = spawn_sse(
        concat!(
            r#"data: {"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]}}]}"#,
            "\n",
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            "\n",
            "data: [DONE]\n",
        )
        .as_bytes(),
    );

    let first = super::take_first_chunk(&mut rx, &tail, "x/a")
        .await
        .expect("a tool-call answer is a success, not an empty response");
    assert!(first.is_none(), "there is no first chunk to replay");

    let response = StreamResponse::new(StreamInit {
        first,
        ..test_init(rx, tail)
    });
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[tokio::test]
async fn truncation_counts_reasoning_and_tool_calls_as_output() {
    let reasoning_only = collect_stream(
        b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n",
    )
    .await;
    let mut reasoning_only = reasoning_only;
    while reasoning_only.next().await.is_some() {}
    assert!(
        reasoning_only.truncated(),
        "a reasoning-only stream that lost its final frame was still cut off"
    );

    let (rx, tail) = spawn_sse(
        concat!(
            r#"data: {"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#,
            "\n",
        )
        .as_bytes(),
    );
    let mut tools_only = StreamResponse::new(test_init(rx, tail));
    while tools_only.next().await.is_some() {}
    assert!(
        tools_only.truncated(),
        "half-written argument JSON with no finish reason is a cut-off answer"
    );
}

#[test]
fn a_malformed_effort_suffix_fails_the_call_instead_of_falling_through() {
    let err = crate::selector::ModelInput::from("x/a!,x/b")
        .validate()
        .expect_err("a typo must not be mistaken for a failed leg");
    assert!(
        matches!(&err, crate::Error::InvalidModel(msg) if msg.contains("x/a!")),
        "{err}"
    );
}

#[test]
fn an_effort_suffix_is_lowercased_for_the_wire() {
    assert_eq!(
        body_for_spec("x/m!HIGH", &config_with("hi"))["reasoning_effort"],
        json!("high")
    );
}

// --- Commit rule: answer content delivered to an `ask` handler -------------

async fn consume_with(
    bytes: &'static [u8],
    handler: Option<&(dyn Fn(&crate::StreamChunk) + Send + Sync)>,
) -> Result<super::SseOutcome, crate::Error> {
    let stream = StaticByteStream {
        chunks: vec![Bytes::from_static(bytes)].into_iter(),
    };
    super::consume_sse_stream(stream, handler, std::time::Instant::now()).await
}

const CONTENT_THEN_ERROR: &[u8] = concat!(
    r#"data: {"model":"a","choices":[{"delta":{"content":"half"}}]}"#,
    "\n",
    r#"data: {"model":"a","choices":[{"delta":{},"finish_reason":"error"}],"error":{"message":"upstream died","type":"upstream_error"}}"#,
    "\n",
    "data: [DONE]\n",
)
.as_bytes();

#[tokio::test]
async fn content_delivered_to_a_handler_ends_the_chain() {
    let seen = Mutex::new(String::new());
    let handler = |chunk: &crate::StreamChunk| seen.lock().unwrap().push_str(&chunk.content);

    let error = consume_with(CONTENT_THEN_ERROR, Some(&handler))
        .await
        .err()
        .expect("the error frame fails the leg");

    assert!(super::delivered_content(&error), "{error}");
    assert!(error.to_string().contains("upstream died"), "{error}");
    match error {
        crate::Error::Stream { partial, .. } => assert_eq!(partial, "half"),
        other => panic!("expected a stream error, got {other}"),
    }
    assert_eq!(*seen.lock().unwrap(), "half");
}

#[tokio::test]
async fn without_a_handler_the_chain_advances() {
    let error = consume_with(CONTENT_THEN_ERROR, None)
        .await
        .err()
        .expect("the error frame fails the leg");
    assert!(
        !super::delivered_content(&error),
        "nothing reached the caller: {error}"
    );
}

#[tokio::test]
async fn reasoning_delivered_to_a_handler_does_not_commit() {
    let handler = |_: &crate::StreamChunk| {};
    let error = consume_with(
        concat!(
            r#"data: {"model":"a","choices":[{"delta":{"reasoning_content":"thinking"}}]}"#,
            "\n",
            r#"data: {"model":"a","choices":[{"delta":{},"finish_reason":"error"}],"error":{"message":"upstream died"}}"#,
            "\n",
        )
        .as_bytes(),
        Some(&handler),
    )
    .await
    .err()
    .expect("the error frame fails the leg");
    assert!(!super::delivered_content(&error), "{error}");
}

#[test]
fn a_truncated_answer_a_handler_saw_ends_the_chain() {
    let truncated = || crate::Error::Truncated {
        model: "x/a".into(),
    };
    assert!(super::delivered_content(&super::commit_to_handler(
        truncated(),
        true,
        "cut"
    )));
    assert!(!super::delivered_content(&super::commit_to_handler(
        truncated(),
        false,
        "cut"
    )));
    assert!(!super::delivered_content(&super::commit_to_handler(
        truncated(),
        true,
        ""
    )));
}
