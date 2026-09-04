use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_stream::StreamExt;

use super::{process_sse_task, StreamResponse};

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
    let tail: super::SharedTail =
        std::sync::Arc::new(std::sync::Mutex::new(super::StreamTail::default()));
    let task = tokio::spawn(process_sse_task(stream, tx, std::sync::Arc::clone(&tail)));
    let mut response = StreamResponse::new(
        rx,
        "custom/model".into(),
        "model".into(),
        "custom".into(),
        0,
        tail,
    );

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
    let tail: super::SharedTail =
        std::sync::Arc::new(std::sync::Mutex::new(super::StreamTail::default()));
    tokio::spawn(process_sse_task(stream, tx, std::sync::Arc::clone(&tail)));
    let mut response = StreamResponse::new(
        rx,
        "smolserver/summary".into(),
        "summary".into(),
        "smolserver".into(),
        0,
        tail,
    );

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
    let tail: super::SharedTail =
        std::sync::Arc::new(std::sync::Mutex::new(super::StreamTail::default()));
    let response = StreamResponse::new(rx, "gemini/flash".into(), "flash".into(), "gemini".into(), 0, tail);

    assert_eq!(response.resolved_model(), None);
    assert_eq!(response.actual_model(), "gemini/flash");
}
