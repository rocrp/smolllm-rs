use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Developer,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Developer => "developer",
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
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }

    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Developer,
            content: content.into(),
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
    pub model: String,
    pub model_name: String,
    pub provider: String,
    pub usage: Usage,
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
