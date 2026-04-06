mod balancer;
mod builder;
mod client;
mod error;
mod image;
mod provider;
mod selector;
mod think;
mod types;
mod utils;

pub use builder::{AskBuilder, StreamBuilder};
pub use client::StreamResponse;
pub use error::Error;
pub use selector::ModelInput;
pub use think::extract_think_tags;
pub use types::*;

pub fn ask(prompt: impl Into<Prompt>) -> AskBuilder {
    AskBuilder::new(prompt.into())
}

pub fn stream(prompt: impl Into<Prompt>) -> StreamBuilder {
    StreamBuilder::new(prompt.into())
}

pub fn validate(model: &str, api_key: Option<&str>, base_url: Option<&str>) -> Result<(), Error> {
    let input = ModelInput::from(model);
    let mut selector = input.into_selector();

    while let Some(model_str) = selector.next_model() {
        let parsed = provider::parse_model_string(&model_str)?;
        let resolved_url = provider::resolve_base_url(&parsed, base_url)?;
        let resolved_key = provider::resolve_api_key(&parsed, api_key)?;
        balancer::choose_pair(&resolved_key, &resolved_url)?;
    }

    Ok(())
}
