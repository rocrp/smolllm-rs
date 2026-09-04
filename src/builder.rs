use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::time::Duration;

use crate::client::{self, StreamResponse};
use crate::error::Error;
use crate::request::RequestConfig;
use crate::selector::ModelInput;
use crate::types::*;

pub struct AskBuilder {
    config: RequestConfig,
}

pub struct StreamBuilder {
    config: RequestConfig,
}

/// Fields the library machinery reads back: the stream parser, usage collection
/// and routing all depend on them, so a caller override would silently break them.
const RESERVED_EXTRA_BODY_KEYS: [&str; 4] = ["stream", "stream_options", "messages", "model"];

macro_rules! shared_setters {
    ($t:ty) => {
        impl $t {
            pub fn model(mut self, model: impl AsRef<str>) -> Self {
                self.config.model_input = Some(ModelInput::from(model.as_ref()));
                self
            }

            pub fn model_set<S: AsRef<str>>(mut self, models: &[S]) -> Self {
                self.config.model_input = Some(ModelInput::Random(
                    models.iter().map(|s| s.as_ref().to_string()).collect(),
                ));
                self
            }

            pub fn model_weights<S: AsRef<str>>(mut self, models: &[(S, f64)]) -> Self {
                self.config.model_input = Some(ModelInput::Weighted(
                    models
                        .iter()
                        .map(|(s, w)| (s.as_ref().to_string(), *w))
                        .collect(),
                ));
                self
            }

            pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
                self.config.system_prompt = Some(prompt.into());
                self
            }

            pub fn api_key(mut self, key: impl Into<String>) -> Self {
                self.config.api_key = Some(key.into());
                self
            }

            pub fn base_url(mut self, url: impl Into<String>) -> Self {
                self.config.base_url = Some(url.into());
                self
            }

            pub fn timeout(mut self, timeout: Duration) -> Self {
                self.config.timeout = timeout;
                self
            }

            pub fn image_paths<I, S>(mut self, paths: I) -> Self
            where
                I: IntoIterator<Item = S>,
                S: Into<String>,
            {
                self.config.image_paths = paths.into_iter().map(Into::into).collect();
                self
            }

            pub fn temperature(mut self, value: f64) -> Result<Self, Error> {
                if !(0.0..=2.0).contains(&value) || value.is_nan() {
                    return Err(Error::InvalidParam(format!(
                        "temperature must be between 0 and 2, got {value}"
                    )));
                }
                self.config.temperature = Some(value);
                Ok(self)
            }

            pub fn top_p(mut self, value: f64) -> Result<Self, Error> {
                if !(0.0..=1.0).contains(&value) || value.is_nan() {
                    return Err(Error::InvalidParam(format!(
                        "top_p must be between 0 and 1, got {value}"
                    )));
                }
                self.config.top_p = Some(value);
                Ok(self)
            }

            pub fn reasoning_effort(mut self, value: impl Into<String>) -> Result<Self, Error> {
                let s = value.into();
                if s.trim().is_empty() {
                    return Err(Error::InvalidParam(
                        "reasoning_effort must not be empty".into(),
                    ));
                }
                self.config.reasoning_effort = Some(s);
                Ok(self)
            }

            /// Set raw request fields the library does not model (`tools`,
            /// `response_format`, `max_tokens`, provider-private params). Merged
            /// into the payload last, so these win over library defaults.
            ///
            /// Rejected here rather than at send time: every error inside the
            /// fallback loop is retried on each leg and reported from the last.
            pub fn extra_body(mut self, fields: serde_json::Value) -> Result<Self, Error> {
                let object = fields.as_object().ok_or_else(|| {
                    Error::InvalidParam("extra_body must be a JSON object".into())
                })?;
                let reserved: Vec<&str> = RESERVED_EXTRA_BODY_KEYS
                    .iter()
                    .copied()
                    .filter(|key| object.contains_key(*key))
                    .collect();
                if !reserved.is_empty() {
                    return Err(Error::InvalidParam(format!(
                        "extra_body may not set {}",
                        reserved.join(", ")
                    )));
                }
                self.config.extra_body = Some(fields);
                Ok(self)
            }

            pub fn hook(mut self, hook: impl Fn(RequestEvent) + Send + Sync + 'static) -> Self {
                self.config.hook = Some(std::sync::Arc::new(hook));
                self
            }

            pub fn http_client(mut self, client: reqwest::Client) -> Self {
                self.config.http_client = Some(client);
                self
            }
        }
    };
}

shared_setters!(AskBuilder);
shared_setters!(StreamBuilder);

impl AskBuilder {
    pub(crate) fn new(prompt: Prompt) -> Self {
        Self {
            config: RequestConfig::new(prompt),
        }
    }

    pub fn remove_backticks(mut self) -> Self {
        self.config.remove_backticks = true;
        self
    }

    pub fn handler(mut self, handler: impl Fn(&StreamChunk) + Send + Sync + 'static) -> Self {
        self.config.handler = Some(Box::new(handler));
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

impl StreamBuilder {
    pub(crate) fn new(prompt: Prompt) -> Self {
        Self {
            config: RequestConfig::new(prompt),
        }
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
