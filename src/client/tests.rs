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
    let task = tokio::spawn(process_sse_task(stream, tx));
    let mut response = StreamResponse::new(
        rx,
        "custom/model".into(),
        "model".into(),
        "custom".into(),
        0,
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
