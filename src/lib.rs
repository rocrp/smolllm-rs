mod balancer;
mod builder;
mod client;
mod endpoint;
mod error;
mod image;
mod provider;
mod request;
mod selector;
mod think;
mod types;
mod utils;

pub use builder::{AskBuilder, StreamBuilder};
pub use client::StreamResponse;
pub use endpoint::{resolve_endpoints, ResolvedEndpoint};
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
    for resolved in endpoint::resolve_models(model, base_url)? {
        let resolved_keys =
            utils::parse_api_key_list(&provider::resolve_api_key(&resolved.parsed, api_key)?)?;
        balancer::validate_pairs(&resolved_keys, &resolved.base_urls)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_unknown_provider_with_explicit_base_url() {
        assert!(validate(
            "smolllm-issue-2-validate/qwen3",
            Some("test-key"),
            Some("https://my.host"),
        )
        .is_ok());
    }

    #[test]
    fn endpoint_resolution_and_validation_do_not_change_balancer_usage() {
        let model = "smolllm-issue-3-state/model";
        let api_key = "smolllm-issue-3-key";
        let base_url = "https://issue-3.example/v2";
        let usage_before = balancer::usage_for(api_key, base_url);

        for _ in 0..3 {
            resolve_endpoints(model, Some(base_url)).unwrap();
            validate(model, Some(api_key), Some(base_url)).unwrap();
        }

        assert_eq!(balancer::usage_for(api_key, base_url), usage_before);
    }
}
