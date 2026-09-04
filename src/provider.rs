use std::collections::HashMap;
use std::env;
use std::sync::LazyLock;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub base_url: &'static str,
}

const fn p(name: &'static str, base_url: &'static str) -> ProviderInfo {
    ProviderInfo { name, base_url }
}

static PROVIDER_LIST: &[ProviderInfo] = &[
    p("aihubmix", "https://aihubmix.com"),
    p("anthropic", "https://api.anthropic.com/"),
    p("azure-openai", ""),
    p("baichuan", "https://api.baichuan-ai.com"),
    p("baidu-cloud", "https://qianfan.baidubce.com/v2/"),
    p("cerebras", "https://api.cerebras.ai"),
    p(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/",
    ),
    p("deepseek", "https://api.deepseek.com"),
    p("dmxapi", "https://www.dmxapi.cn"),
    p("doubao", "https://ark.cn-beijing.volces.com/api/v3/"),
    p("fireworks", "https://api.fireworks.ai/inference"),
    p("gemini", "https://generativelanguage.googleapis.com"),
    p("gitee-ai", "https://ai.gitee.com"),
    p("github", "https://models.inference.ai.azure.com/"),
    p("graphrag-kylin-mountain", ""),
    p("grok", "https://api.x.ai"),
    p("groq", "https://api.groq.com/openai"),
    p("hunyuan", "https://api.hunyuan.cloud.tencent.com"),
    p("hyperbolic", "https://api.hyperbolic.xyz"),
    p("infini", "https://cloud.infini-ai.com/maas"),
    p("jina", "https://api.jina.ai"),
    p("lmstudio", "http://localhost:1234"),
    p("minimax", "https://api.minimax.chat/v1/"),
    p("mistral", "https://api.mistral.ai"),
    p("modelscope", "https://api-inference.modelscope.cn/v1/"),
    p("moonshot", "https://api.moonshot.cn"),
    p("nvidia", "https://integrate.api.nvidia.com"),
    p("o3", "https://api.o3.fan"),
    p("ocoolai", "https://api.ocoolai.com"),
    p("ollama", "http://localhost:11434"),
    p("openai", "https://api.openai.com"),
    p("openrouter", "https://openrouter.ai/api/v1/"),
    p("perplexity", "https://api.perplexity.ai/"),
    p("ppio", "https://api.ppinfra.com/v3/openai"),
    p("silicon", "https://api.siliconflow.cn"),
    p("stepfun", "https://api.stepfun.com"),
    p("tencent-cloud-ti", "https://api.lkeap.cloud.tencent.com"),
    p("together", "https://api.together.xyz"),
    p("xirang", "https://wishub-x1.ctyun.cn"),
    p("yi", "https://api.lingyiwanwu.com"),
    p("zhinao", "https://api.360.cn"),
    p("zhipu", "https://open.bigmodel.cn/api/paas/v4/"),
];

static PROVIDERS: LazyLock<HashMap<&'static str, &'static ProviderInfo>> =
    LazyLock::new(|| PROVIDER_LIST.iter().map(|info| (info.name, info)).collect());

/// A parsed model specification. `provider_name` is empty for bare model
/// names (no `/` in the input); bare models resolve only against explicit
/// base URLs and API keys — never environment variables.
#[derive(Debug, Clone)]
pub struct ParsedModel {
    pub provider_name: String,
    pub model_name: String,
    pub base_url: String,
    /// The leg's own `!effort` suffix, overriding the call-level setting.
    pub reasoning_effort: Option<String>,
}

/// Splits a `!effort` suffix off a model spec. The suffix belongs to this leg
/// alone, so a chain can mix efforts: `proxy/qwen3!none,proxy/gpt-5!high`.
fn split_effort_suffix(spec: &str) -> Result<(&str, Option<String>), Error> {
    let Some((model, effort)) = spec.split_once('!') else {
        return Ok((spec, None));
    };
    let effort = effort.trim();
    if effort.is_empty() {
        return Err(Error::InvalidModel(format!(
            "missing reasoning effort after '!' in '{spec}'"
        )));
    }
    Ok((model.trim(), Some(effort.to_string())))
}

pub fn parse_model_string(model: &str) -> Result<ParsedModel, Error> {
    let model = model.trim();
    if model.is_empty() {
        return Err(Error::InvalidModel("model string must not be empty".into()));
    }
    let (model, reasoning_effort) = split_effort_suffix(model)?;
    if model.is_empty() {
        return Err(Error::InvalidModel("model string must not be empty".into()));
    }

    let Some((provider_name, raw_model_name)) = model.split_once('/') else {
        // Bare model name: no provider.
        return Ok(ParsedModel {
            provider_name: String::new(),
            model_name: model.to_string(),
            base_url: String::new(),
            reasoning_effort,
        });
    };

    if provider_name.is_empty() {
        return Err(Error::InvalidModel(format!(
            "missing provider before '/' in '{model}'"
        )));
    }
    let model_name = raw_model_name.trim();
    if model_name.is_empty() {
        return Err(Error::InvalidModel(format!(
            "missing model name after '/' in '{model}'"
        )));
    }

    let base_url = PROVIDERS
        .get(provider_name)
        .map_or("", |info| info.base_url);

    Ok(ParsedModel {
        provider_name: provider_name.to_string(),
        model_name: model_name.to_string(),
        base_url: base_url.to_string(),
        reasoning_effort,
    })
}

pub fn resolve_base_url(parsed: &ParsedModel, explicit: Option<&str>) -> Result<String, Error> {
    if let Some(url) = explicit {
        let url = url.trim();
        if !url.is_empty() {
            return Ok(url.to_string());
        }
    }

    if parsed.provider_name.is_empty() {
        return Err(Error::MissingBaseUrlBare {
            model: parsed.model_name.clone(),
        });
    }

    let env_key = provider_env_key(&parsed.provider_name, "BASE_URL");
    if let Ok(val) = env::var(&env_key) {
        let val = val.trim().to_string();
        if !val.is_empty() {
            return Ok(val);
        }
    }

    if parsed.base_url.trim().is_empty() {
        return Err(Error::MissingBaseUrl {
            provider: parsed.provider_name.clone(),
            env_var: env_key,
        });
    }

    Ok(parsed.base_url.clone())
}

pub fn resolve_api_key(parsed: &ParsedModel, explicit: Option<&str>) -> Result<String, Error> {
    if let Some(key) = explicit {
        let key = key.trim();
        if !key.is_empty() {
            return Ok(key.to_string());
        }
    }

    if parsed.provider_name.is_empty() {
        return Err(Error::MissingApiKeyBare {
            model: parsed.model_name.clone(),
        });
    }

    let env_key = provider_env_key(&parsed.provider_name, "API_KEY");
    if let Ok(val) = env::var(&env_key) {
        let val = val.trim().to_string();
        if !val.is_empty() {
            return Ok(val);
        }
    }

    if parsed.provider_name == "ollama" {
        return Ok("ollama".to_string());
    }

    Err(Error::MissingApiKey {
        provider: parsed.provider_name.clone(),
        env_var: env_key,
    })
}

/// A path ends in a version segment only when its last part is `v` followed by
/// digits alone, so `/v1` and `/v3` count while `/v1beta` does not — the same
/// `/v\d+$` rule the Python port uses.
fn has_version_suffix(url: &str) -> bool {
    let trimmed = url.trim_end_matches('/');
    let last = match trimmed.rsplit('/').next() {
        Some(s) => s,
        None => return false,
    };
    let Some(digits) = last.strip_prefix('v') else {
        return false;
    };
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

pub fn build_request_url(base_url: &str, provider_name: &str) -> String {
    let base = base_url.trim();
    let endpoint = "chat/completions";
    let provider_version = match provider_name {
        "anthropic" => Some("v1"),
        "gemini" => Some("v1beta/openai"),
        _ => None,
    };

    if let Some(version) = provider_version {
        let stripped = base.trim_end_matches('/');
        if has_version_suffix(stripped) {
            format!("{stripped}/{endpoint}")
        } else {
            format!("{stripped}/{version}/{endpoint}")
        }
    } else if base.ends_with('#') {
        base.trim_end_matches('#').to_string()
    } else if base.ends_with('/') {
        format!("{base}{endpoint}")
    } else if has_version_suffix(base) {
        format!("{base}/{endpoint}")
    } else {
        format!("{base}/v1/{endpoint}")
    }
}

pub fn provider_env_key(provider_name: &str, suffix: &str) -> String {
    let base = provider_name.to_uppercase().replace('-', "_");
    format!("{base}_{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_string() {
        let parsed = parse_model_string("gemini/gemini-flash-lite-latest").unwrap();
        assert_eq!(parsed.provider_name, "gemini");
        assert_eq!(parsed.model_name, "gemini-flash-lite-latest");
        assert!(!parsed.base_url.is_empty());
    }

    #[test]
    fn test_parse_model_empty() {
        assert!(parse_model_string("").is_err());
    }

    #[test]
    fn test_parse_bare_model_has_no_provider() {
        // Even names matching known providers are plain model names when bare.
        for bare in ["gpt-4", "gemini", "openai", "ollama"] {
            let parsed = parse_model_string(bare).unwrap();
            assert_eq!(parsed.provider_name, "");
            assert_eq!(parsed.model_name, bare);
            assert!(parsed.base_url.is_empty());
        }
    }

    #[test]
    fn test_parse_rejects_empty_provider_or_model() {
        assert_eq!(
            parse_model_string("/gpt-4").unwrap_err().to_string(),
            "invalid model string: missing provider before '/' in '/gpt-4'"
        );
        assert_eq!(
            parse_model_string("gpt-4/").unwrap_err().to_string(),
            "invalid model string: missing model name after '/' in 'gpt-4/'"
        );
    }

    #[test]
    fn test_bare_model_resolves_only_explicit_base_url() {
        let parsed = parse_model_string("gpt-4").unwrap();
        assert_eq!(
            resolve_base_url(&parsed, Some(" https://my.host ")).unwrap(),
            "https://my.host"
        );
        assert_eq!(
            resolve_base_url(&parsed, None).unwrap_err().to_string(),
            "missing base URL for bare model 'gpt-4'. Pass base_url or use provider/model format"
        );
    }

    #[test]
    fn test_bare_model_resolves_only_explicit_api_key() {
        let parsed = parse_model_string("gpt-4").unwrap();
        assert_eq!(
            resolve_api_key(&parsed, Some(" sk-key ")).unwrap(),
            "sk-key"
        );
        assert_eq!(
            resolve_api_key(&parsed, None).unwrap_err().to_string(),
            "missing API key for bare model 'gpt-4'. Pass api_key or use provider/model format"
        );
    }

    #[test]
    fn test_bare_model_never_reads_underscore_env_vars() {
        // provider_env_key("") would yield "_BASE_URL"/"_API_KEY"; bare mode
        // must error before any env lookup happens.
        let previous_url = env::var_os("_BASE_URL");
        let previous_key = env::var_os("_API_KEY");
        env::set_var("_BASE_URL", "https://env.host");
        env::set_var("_API_KEY", "env-key");

        let parsed = parse_model_string("gpt-4").unwrap();
        let base_url = resolve_base_url(&parsed, None);
        let api_key = resolve_api_key(&parsed, None);

        for (var, previous) in [("_BASE_URL", previous_url), ("_API_KEY", previous_key)] {
            if let Some(value) = previous {
                env::set_var(var, value);
            } else {
                env::remove_var(var);
            }
        }

        assert!(matches!(
            base_url.unwrap_err(),
            Error::MissingBaseUrlBare { model } if model == "gpt-4"
        ));
        assert!(matches!(
            api_key.unwrap_err(),
            Error::MissingApiKeyBare { model } if model == "gpt-4"
        ));
    }

    #[test]
    fn test_bare_ollama_gets_no_literal_key() {
        // The ollama literal-key fallback applies to the ollama provider only,
        // never to a bare model that happens to be named "ollama".
        let parsed = parse_model_string("ollama").unwrap();
        assert!(matches!(
            resolve_api_key(&parsed, None).unwrap_err(),
            Error::MissingApiKeyBare { model } if model == "ollama"
        ));
    }

    #[test]
    fn test_unknown_provider_resolves_explicit_base_url() {
        let parsed = parse_model_string("smolllm-issue-2-explicit/qwen3").unwrap();

        assert_eq!(parsed.provider_name, "smolllm-issue-2-explicit");
        assert_eq!(parsed.model_name, "qwen3");
        assert!(parsed.base_url.is_empty());
        assert_eq!(
            resolve_base_url(&parsed, Some(" https://my.host ")).unwrap(),
            "https://my.host"
        );
    }

    #[test]
    fn test_unknown_provider_env_is_read_only_during_resolution() {
        let provider_name = "smolllm-issue-2-env";
        let env_key = provider_env_key(provider_name, "BASE_URL");
        let previous = env::var_os(&env_key);
        env::set_var(&env_key, " https://env.host ");

        let parsed = parse_model_string(&format!("{provider_name}/qwen3"));
        let resolved = parsed
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|parsed| resolve_base_url(parsed, None).map_err(|err| err.to_string()));

        if let Some(value) = previous {
            env::set_var(&env_key, value);
        } else {
            env::remove_var(&env_key);
        }

        let parsed = parsed.unwrap();
        assert!(
            parsed.base_url.is_empty(),
            "model parsing must not read {env_key}"
        );
        assert_eq!(resolved.unwrap(), "https://env.host");
    }

    #[test]
    fn test_unknown_provider_error_names_both_base_url_remedies() {
        let provider_name = "smolllm-issue-2-missing";
        let env_key = provider_env_key(provider_name, "BASE_URL");
        let previous = env::var_os(&env_key);
        env::remove_var(&env_key);

        let result = parse_model_string(&format!("{provider_name}/qwen3"))
            .and_then(|parsed| resolve_base_url(&parsed, None));

        if let Some(value) = previous {
            env::set_var(&env_key, value);
        }

        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "missing base URL for provider '{provider_name}'. Pass base_url or set {env_key}"
            )
        );
    }

    #[test]
    fn test_build_request_url_default() {
        assert_eq!(
            build_request_url("https://api.openai.com", "openai"),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_request_url_gemini() {
        assert_eq!(
            build_request_url("https://generativelanguage.googleapis.com", "gemini"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
    }

    #[test]
    fn test_build_request_url_with_version() {
        assert_eq!(
            build_request_url("https://api.groq.com/openai/v1", "groq"),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_request_url_hash_override() {
        assert_eq!(
            build_request_url("https://custom.api.com/endpoint#", "custom"),
            "https://custom.api.com/endpoint"
        );
    }

    #[test]
    fn test_build_request_url_trailing_slash() {
        assert_eq!(
            build_request_url("https://api.example.com/v1/", "example"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_provider_env_key() {
        assert_eq!(
            provider_env_key("tencent-cloud-ti", "API_KEY"),
            "TENCENT_CLOUD_TI_API_KEY"
        );
    }

    #[test]
    fn test_has_version_suffix() {
        assert!(has_version_suffix("https://api.com/v1"));
        assert!(has_version_suffix("https://api.com/v2/"));
        assert!(has_version_suffix("https://api.com/v10"));
        assert!(!has_version_suffix("https://api.com/openai"));
        assert!(!has_version_suffix("https://api.com"));
        assert!(!has_version_suffix("https://api.com/v"));
    }
}
