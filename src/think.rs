use crate::types::StreamChunk;

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

pub struct ThinkTagFilter {
    inside_think: bool,
    buffer: String,
    disabled: bool,
}

impl ThinkTagFilter {
    pub fn new() -> Self {
        Self {
            inside_think: false,
            buffer: String::new(),
            disabled: false,
        }
    }

    pub fn feed(&mut self, chunk: StreamChunk) -> StreamChunk {
        if !chunk.reasoning.is_empty() {
            self.disabled = true;
            return chunk;
        }
        if self.disabled {
            return chunk;
        }

        let mut text = std::mem::take(&mut self.buffer);
        text.push_str(&chunk.content);

        let mut reasoning_parts = Vec::new();
        let mut content_parts = Vec::new();

        while !text.is_empty() {
            if self.inside_think {
                if let Some(end) = text.find(CLOSE_TAG) {
                    reasoning_parts.push(text[..end].to_string());
                    text = text[end + CLOSE_TAG.len()..].to_string();
                    self.inside_think = false;
                } else {
                    let buffered = partial_suffix(&text, CLOSE_TAG);
                    if buffered > 0 {
                        self.buffer = text[text.len() - buffered..].to_string();
                        text.truncate(text.len() - buffered);
                    }
                    if !text.is_empty() {
                        reasoning_parts.push(text);
                    }
                    break;
                }
            } else if let Some(start) = text.find(OPEN_TAG) {
                if start > 0 {
                    content_parts.push(text[..start].to_string());
                }
                text = text[start + OPEN_TAG.len()..].to_string();
                self.inside_think = true;
            } else {
                let buffered = partial_suffix(&text, OPEN_TAG);
                if buffered > 0 {
                    self.buffer = text[text.len() - buffered..].to_string();
                    text.truncate(text.len() - buffered);
                }
                if !text.is_empty() {
                    content_parts.push(text);
                }
                break;
            }
        }

        StreamChunk {
            content: content_parts.join(""),
            reasoning: reasoning_parts.join(""),
        }
    }

    pub fn flush(&mut self) -> StreamChunk {
        if self.buffer.is_empty() {
            return StreamChunk::default();
        }
        let buf = std::mem::take(&mut self.buffer);
        if self.inside_think {
            StreamChunk { content: String::new(), reasoning: buf }
        } else {
            StreamChunk { content: buf, reasoning: String::new() }
        }
    }
}

fn partial_suffix(text: &str, tag: &str) -> usize {
    let max_len = (tag.len() - 1).min(text.len());
    for i in (1..=max_len).rev() {
        if text.ends_with(&tag[..i]) {
            return i;
        }
    }
    0
}

pub fn extract_think_tags(text: &str) -> (String, String) {
    let mut reasoning_parts = Vec::new();
    let mut remaining = text.to_string();
    let mut clean = String::new();

    loop {
        if let Some(open) = remaining.find(OPEN_TAG) {
            clean.push_str(&remaining[..open]);
            let after_open = &remaining[open + OPEN_TAG.len()..];
            if let Some(close) = after_open.find(CLOSE_TAG) {
                reasoning_parts.push(after_open[..close].trim().to_string());
                remaining = after_open[close + CLOSE_TAG.len()..].to_string();
            } else {
                reasoning_parts.push(after_open.trim().to_string());
                break;
            }
        } else {
            clean.push_str(&remaining);
            break;
        }
    }

    (reasoning_parts.join("\n\n"), clean.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_think_tag_filter_basic() {
        let mut filter = ThinkTagFilter::new();
        let chunk = StreamChunk { content: "<think>reasoning here</think>actual content".into(), reasoning: String::new() };
        let result = filter.feed(chunk);
        assert_eq!(result.reasoning, "reasoning here");
        assert_eq!(result.content, "actual content");
    }

    #[test]
    fn test_think_tag_filter_split_across_chunks() {
        let mut filter = ThinkTagFilter::new();

        let c1 = filter.feed(StreamChunk { content: "<think>part1".into(), reasoning: String::new() });
        assert_eq!(c1.reasoning, "part1");
        assert!(c1.content.is_empty());

        let c2 = filter.feed(StreamChunk { content: " part2</think>content".into(), reasoning: String::new() });
        assert_eq!(c2.reasoning, " part2");
        assert_eq!(c2.content, "content");
    }

    #[test]
    fn test_think_tag_filter_auto_disable() {
        let mut filter = ThinkTagFilter::new();
        let chunk = StreamChunk { content: "<think>test</think>".into(), reasoning: "backend reasoning".into() };
        let result = filter.feed(chunk);
        assert_eq!(result.reasoning, "backend reasoning");
        assert_eq!(result.content, "<think>test</think>");

        let c2 = filter.feed(StreamChunk { content: "<think>more</think>".into(), reasoning: String::new() });
        assert_eq!(c2.content, "<think>more</think>");
        assert!(c2.reasoning.is_empty());
    }

    #[test]
    fn test_extract_think_tags() {
        let (reasoning, content) = extract_think_tags("before<think>thought</think>after");
        assert_eq!(reasoning, "thought");
        assert_eq!(content, "beforeafter");
    }

    #[test]
    fn test_flush() {
        let mut filter = ThinkTagFilter::new();
        let result = filter.feed(StreamChunk { content: "<think>partial".into(), reasoning: String::new() });
        assert_eq!(result.reasoning, "partial");
        assert!(result.content.is_empty());

        // Buffer holds partial close tag at chunk boundary
        let mut filter2 = ThinkTagFilter::new();
        let _ = filter2.feed(StreamChunk { content: "hello<".into(), reasoning: String::new() });
        let flushed = filter2.flush();
        assert_eq!(flushed.content, "<");
    }

    #[test]
    fn test_partial_suffix() {
        assert_eq!(partial_suffix("<thi", OPEN_TAG), 4);
        assert_eq!(partial_suffix("abc<", OPEN_TAG), 1);
        assert_eq!(partial_suffix("abc", OPEN_TAG), 0);
    }
}
