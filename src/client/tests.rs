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
