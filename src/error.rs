#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },

    #[error("empty response from model {model}")]
    EmptyResponse { model: String },

    #[error("missing API key for provider '{provider}'. Set {env_var} or use .api_key()")]
    MissingApiKey { provider: String, env_var: String },

    #[error("missing base URL for provider '{provider}'. Pass base_url or set {env_var}")]
    MissingBaseUrl { provider: String, env_var: String },

    #[error("missing API key for bare model '{model}'. Pass api_key or use provider/model format")]
    MissingApiKeyBare { model: String },

    #[error(
        "missing base URL for bare model '{model}'. Pass base_url or use provider/model format"
    )]
    MissingBaseUrlBare { model: String },

    #[error("invalid model string: {0}")]
    InvalidModel(String),

    #[error("invalid model list: {reason}")]
    InvalidModelList { reason: String },

    #[error("invalid API key list: {reason}")]
    InvalidApiKeyList { reason: String },

    #[error("invalid base URL list: {reason}")]
    InvalidBaseUrlList { reason: String },

    #[error(
        "cannot resolve one endpoint for provider '{provider}': \
         found {candidates} base URL candidates; pass a single base_url"
    )]
    AmbiguousBaseUrls { provider: String, candidates: usize },

    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("no valid models configured")]
    NoValidModels,

    #[error("model {model} was cut off before finishing")]
    Truncated { model: String },

    #[error("stream error: {message}")]
    Stream { message: String, partial: String },

    #[error("image error: {0}")]
    Image(String),

    #[error("mismatched key/URL counts: {keys} keys vs {urls} URLs")]
    MismatchedPairs { keys: usize, urls: usize },

    #[error("{0}")]
    Request(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Http { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 529),
            _ => false,
        }
    }
}
