use crate::provider::{build_request_url, parse_model_string, resolve_base_url, ParsedModel};
use crate::utils::{parse_base_url_list, parse_model_list};
use crate::Error;

/// The concrete request target resolved from a model specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedEndpoint {
    /// Full URL that the request will POST to.
    pub url: String,
    /// Resolved provider name; empty for bare model names (no `provider/` prefix).
    pub provider: String,
    /// Model name with the provider prefix removed.
    pub model: String,
}

pub(crate) struct ResolvedModel {
    pub parsed: ParsedModel,
    pub base_urls: Vec<String>,
}

impl ResolvedModel {
    pub(crate) fn endpoint_for(&self, base_url: &str) -> ResolvedEndpoint {
        ResolvedEndpoint {
            url: build_request_url(base_url, &self.parsed.provider_name),
            provider: self.parsed.provider_name.clone(),
            model: self.parsed.model_name.clone(),
        }
    }

    fn into_endpoint(self) -> Result<ResolvedEndpoint, Error> {
        if self.base_urls.len() != 1 {
            return Err(Error::AmbiguousBaseUrls {
                provider: self.parsed.provider_name.clone(),
                candidates: self.base_urls.len(),
            });
        }
        Ok(self.endpoint_for(&self.base_urls[0]))
    }
}

pub(crate) fn resolve_model(model: &str, base_url: Option<&str>) -> Result<ResolvedModel, Error> {
    let parsed = parse_model_string(model)?;
    let base_urls = parse_base_url_list(&resolve_base_url(&parsed, base_url)?)?;
    Ok(ResolvedModel { parsed, base_urls })
}

pub(crate) fn resolve_models(
    model: &str,
    base_url: Option<&str>,
) -> Result<Vec<ResolvedModel>, Error> {
    let mut resolved_models = Vec::new();

    for model in parse_model_list(model)? {
        resolved_models.push(resolve_model(&model, base_url)?);
    }

    Ok(resolved_models)
}

/// Resolves one request endpoint per comma-separated model without API-key
/// lookup, network I/O, or balancer mutation.
pub fn resolve_endpoints(
    model: &str,
    base_url: Option<&str>,
) -> Result<Vec<ResolvedEndpoint>, Error> {
    let models = resolve_models(model, base_url)?;
    models
        .into_iter()
        .map(ResolvedModel::into_endpoint)
        .collect()
}
