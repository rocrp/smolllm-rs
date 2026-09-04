use std::time::Duration;

use crate::selector::ModelInput;
use crate::types::*;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) type ChunkHandler = dyn Fn(&StreamChunk) + Send + Sync;
pub(crate) type EventHook = dyn Fn(RequestEvent) + Send + Sync;

pub(crate) struct RequestConfig {
    pub prompt: Prompt,
    pub system_prompt: Option<String>,
    pub model_input: Option<ModelInput>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Duration,
    pub remove_backticks: bool,
    pub image_paths: Vec<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub reasoning_effort: Option<String>,
    /// Raw request fields the library does not model, merged into the payload
    /// last so the caller wins. Validated by the builder setter.
    pub extra_body: Option<serde_json::Value>,
    pub handler: Option<Box<ChunkHandler>>,
    pub hook: Option<std::sync::Arc<EventHook>>,
    pub http_client: Option<reqwest::Client>,
}

impl RequestConfig {
    pub(crate) fn new(prompt: Prompt) -> Self {
        Self {
            prompt,
            system_prompt: None,
            model_input: None,
            api_key: None,
            base_url: None,
            timeout: DEFAULT_TIMEOUT,
            remove_backticks: false,
            image_paths: Vec::new(),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            extra_body: None,
            handler: None,
            hook: None,
            http_client: None,
        }
    }

    /// Resolve the configured model input, falling back to the `SMOLLLM_MODEL`
    /// env var if no model was set on the builder.
    pub(crate) fn resolve_model_input(&self) -> Option<ModelInput> {
        if let Some(input) = self.model_input.clone() {
            return Some(input);
        }
        std::env::var("SMOLLLM_MODEL").ok().and_then(|s| {
            let trimmed = s.trim();
            (!trimmed.is_empty()).then(|| ModelInput::from(trimmed))
        })
    }
}
