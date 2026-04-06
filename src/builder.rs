use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::time::Duration;

use crate::client::{self, RequestConfig, StreamResponse};
use crate::error::Error;
use crate::selector::ModelInput;
use crate::types::*;

pub struct AskBuilder {
    config: RequestConfig,
}

impl AskBuilder {
    pub(crate) fn new(prompt: Prompt) -> Self {
        Self {
            config: RequestConfig {
                prompt,
                system_prompt: None,
                model_input: None,
                api_key: None,
                base_url: None,
                timeout: Duration::from_secs(120),
                remove_backticks: false,
                image_paths: Vec::new(),
                temperature: None,
                top_p: None,
                reasoning_effort: None,
                handler: None,
                hook: None,
                http_client: None,
            },
        }
    }

    pub fn model(mut self, model: &str) -> Self {
        self.config.model_input = Some(ModelInput::from(model));
        self
    }

    pub fn model_set(mut self, models: &[&str]) -> Self {
        self.config.model_input = Some(ModelInput::Random(
            models.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn model_weights(mut self, models: &[(&str, f64)]) -> Self {
        self.config.model_input = Some(ModelInput::Weighted(
            models
                .iter()
                .map(|(s, w)| (s.to_string(), *w))
                .collect(),
        ));
        self
    }

    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.config.system_prompt = Some(prompt.to_string());
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.config.api_key = Some(key.to_string());
        self
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.config.base_url = Some(url.to_string());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn remove_backticks(mut self) -> Self {
        self.config.remove_backticks = true;
        self
    }

    pub fn image_paths(mut self, paths: &[&str]) -> Self {
        self.config.image_paths = paths.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn temperature(mut self, value: f64) -> Self {
        assert!(
            (0.0..=2.0).contains(&value) && !value.is_nan(),
            "temperature must be between 0 and 2"
        );
        self.config.temperature = Some(value);
        self
    }

    pub fn top_p(mut self, value: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&value) && !value.is_nan(),
            "top_p must be between 0 and 1"
        );
        self.config.top_p = Some(value);
        self
    }

    pub fn reasoning_effort(mut self, value: &str) -> Self {
        assert!(!value.trim().is_empty(), "reasoning_effort must not be empty");
        self.config.reasoning_effort = Some(value.to_string());
        self
    }

    pub fn handler(mut self, handler: impl Fn(&StreamChunk) + Send + Sync + 'static) -> Self {
        self.config.handler = Some(Box::new(handler));
        self
    }

    pub fn hook(mut self, hook: impl Fn(RequestEvent) + Send + Sync + 'static) -> Self {
        self.config.hook = Some(Box::new(hook));
        self
    }

    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.config.http_client = Some(client);
        self
    }

    pub async fn send(self) -> Result<LLMResponse, Error> {
        client::execute_ask(self.config).await
    }
}

impl IntoFuture for AskBuilder {
    type Output = Result<LLMResponse, Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}

pub struct StreamBuilder {
    config: RequestConfig,
}

impl StreamBuilder {
    pub(crate) fn new(prompt: Prompt) -> Self {
        Self {
            config: RequestConfig {
                prompt,
                system_prompt: None,
                model_input: None,
                api_key: None,
                base_url: None,
                timeout: Duration::from_secs(120),
                remove_backticks: false,
                image_paths: Vec::new(),
                temperature: None,
                top_p: None,
                reasoning_effort: None,
                handler: None,
                hook: None,
                http_client: None,
            },
        }
    }

    pub fn model(mut self, model: &str) -> Self {
        self.config.model_input = Some(ModelInput::from(model));
        self
    }

    pub fn model_set(mut self, models: &[&str]) -> Self {
        self.config.model_input = Some(ModelInput::Random(
            models.iter().map(|s| s.to_string()).collect(),
        ));
        self
    }

    pub fn model_weights(mut self, models: &[(&str, f64)]) -> Self {
        self.config.model_input = Some(ModelInput::Weighted(
            models
                .iter()
                .map(|(s, w)| (s.to_string(), *w))
                .collect(),
        ));
        self
    }

    pub fn system_prompt(mut self, prompt: &str) -> Self {
        self.config.system_prompt = Some(prompt.to_string());
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.config.api_key = Some(key.to_string());
        self
    }

    pub fn base_url(mut self, url: &str) -> Self {
        self.config.base_url = Some(url.to_string());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn image_paths(mut self, paths: &[&str]) -> Self {
        self.config.image_paths = paths.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn temperature(mut self, value: f64) -> Self {
        assert!(
            (0.0..=2.0).contains(&value) && !value.is_nan(),
            "temperature must be between 0 and 2"
        );
        self.config.temperature = Some(value);
        self
    }

    pub fn top_p(mut self, value: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&value) && !value.is_nan(),
            "top_p must be between 0 and 1"
        );
        self.config.top_p = Some(value);
        self
    }

    pub fn reasoning_effort(mut self, value: &str) -> Self {
        assert!(!value.trim().is_empty(), "reasoning_effort must not be empty");
        self.config.reasoning_effort = Some(value.to_string());
        self
    }

    pub fn hook(mut self, hook: impl Fn(RequestEvent) + Send + Sync + 'static) -> Self {
        self.config.hook = Some(Box::new(hook));
        self
    }

    pub fn http_client(mut self, client: reqwest::Client) -> Self {
        self.config.http_client = Some(client);
        self
    }

    pub async fn send(self) -> Result<StreamResponse, Error> {
        client::execute_stream(self.config).await
    }
}

impl IntoFuture for StreamBuilder {
    type Output = Result<StreamResponse, Error>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.send())
    }
}
