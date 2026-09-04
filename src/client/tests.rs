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
    build_request_body("m", config).expect("body builds")
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
fn parser_keeps_the_first_reported_model() {
    let mut parser = SseParser::new();
    feed_frame(&mut parser, r#"{"model":"first","choices":[{"delta":{"content":"a"}}]}"#);
    feed_frame(&mut parser, r#"{"model":"second","choices":[{"delta":{"content":"b"}}]}"#);
    assert_eq!(parser.resolved_model(), Some("first"));
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
    let (mut rx, _tail) = spawn_sse(b"data: {\"error\":{\"message\":\"upstream exploded\"}}\n");

    let err = super::take_first_chunk(&mut rx, "x/a")
        .await
        .expect_err("an error frame before any content fails the leg");
    assert!(err.to_string().contains("upstream exploded"), "{err}");
}

#[tokio::test]
async fn a_stream_that_ends_without_chunks_is_an_empty_response() {
    let (mut rx, _tail) = spawn_sse(b"data: [DONE]\n");

    let err = super::take_first_chunk(&mut rx, "x/a")
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
    let first = super::take_first_chunk(&mut rx, "x/a")
        .await
        .expect("first chunk");
    let mut response = StreamResponse::new(StreamInit {
        first: Some(first),
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
    let crate::Error::Stream { message, partial } = err else {
        panic!("post-content failures keep the partial output: {err:?}")
    };
    assert!(message.contains("connection reset"), "{message}");
    assert_eq!(partial, "half an answer");
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
