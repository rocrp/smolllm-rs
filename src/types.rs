use std::fmt;
use std::time::Duration;

use crate::toolcall::ToolCall;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Developer,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Developer => "developer",
            MessageRole::Tool => "tool",
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    /// None on an assistant turn that only requests tool calls.
    pub content: Option<String>,
    /// Set on an assistant turn replaying the calls the model asked for.
    pub tool_calls: Vec<ToolCall>,
    /// Set on a tool result, naming the call it answers.
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Developer,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// An assistant turn replaying the tool calls the model asked for. Pass
    /// empty text when the turn carried no content.
    pub fn assistant_tool_calls(text: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        let text = text.into();
        Self {
            role: MessageRole::Assistant,
            content: (!text.is_empty()).then_some(text),
            tool_calls: calls,
            tool_call_id: None,
            name: None,
        }
    }

    /// A tool result answering one call.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    Messages(Vec<Message>),
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Prompt::Text(s.to_string())
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Prompt::Text(s)
    }
}

impl From<Vec<Message>> for Prompt {
    fn from(messages: Vec<Message>) -> Self {
        Prompt::Messages(messages)
    }
}

#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    pub content: String,
    pub reasoning: String,
}

impl StreamChunk {
    pub fn is_empty(&self) -> bool {
        self.content.is_empty() && self.reasoning.is_empty()
    }
}

impl fmt::Display for StreamChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.content)
    }
}

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub text: String,
    pub reasoning: String,
    /// Verbatim provider string explaining why generation ended; never normalized.
    pub finish_reason: Option<String>,
    /// The model the server reported as having produced this response; differs
    /// from the requested `model` behind aliases and proxies. None when the
    /// backend reports no model.
    pub resolved_model: Option<String>,
    /// Empty unless the model answered with tool calls. Executing them and
    /// replaying the result is the caller's job — the library runs no loop.
    pub tool_calls: Vec<ToolCall>,
    pub model: String,
    pub model_name: String,
    pub provider: String,
    pub usage: Usage,
}

impl LLMResponse {
    /// Best available identity of the model that produced this response: the
    /// ResolvedModel when the server named one, else the requested spec.
    pub fn actual_model(&self) -> &str {
        self.resolved_model.as_deref().unwrap_or(&self.model)
    }
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub provider: String,
    pub model: String,
    pub model_name: String,
    pub api_key_hint: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub duration: Duration,
    pub ttft: Option<Duration>,
}

impl Default for Usage {
    fn default() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            model_name: String::new(),
            api_key_hint: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            duration: Duration::ZERO,
            ttft: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestEvent {
    pub usage: Usage,
    pub error: Option<String>,
}
