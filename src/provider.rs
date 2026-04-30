use std::collections::HashMap;
use std::env;
use std::sync::LazyLock;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub base_url: &'static str,
    pub default_model: Option<&'static str>,
}

const fn p(
    name: &'static str,
    base_url: &'static str,
    default_model: Option<&'static str>,
) -> ProviderInfo {
    ProviderInfo {
        name,
        base_url,
        default_model,
    }
}

static PROVIDER_LIST: &[ProviderInfo] = &[
    p("aihubmix", "https://aihubmix.com", None),
    p("anthropic", "https://api.anthropic.com/", None),
    p("azure-openai", "", None),
    p("baichuan", "https://api.baichuan-ai.com", None),
    p("baidu-cloud", "https://qianfan.baidubce.com/v2/", None),
    p("cerebras", "https://api.cerebras.ai", None),
    p(
        "dashscope",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/",
        None,
    ),
    p("deepseek", "https://api.deepseek.com", None),
    p("dmxapi", "https://www.dmxapi.cn", None),
    p("doubao", "https://ark.cn-beijing.volces.com/api/v3/", None),
    p("fireworks", "https://api.fireworks.ai/inference", None),
    p(
        "gemini",
        "https://generativelanguage.googleapis.com",
        Some("gemini-2.0-flash"),
    ),
    p("gitee-ai", "https://ai.gitee.com", None),
    p("github", "https://models.inference.ai.azure.com/", None),
    p("graphrag-kylin-mountain", "", None),
    p("grok", "https://api.x.ai", None),
    p("groq", "https://api.groq.com/openai", None),
    p("hunyuan", "https://api.hunyuan.cloud.tencent.com", None),
    p("hyperbolic", "https://api.hyperbolic.xyz", None),
    p("infini", "https://cloud.infini-ai.com/maas", None),
    p("jina", "https://api.jina.ai", None),
    p("lmstudio", "http://localhost:1234", None),
    p("minimax", "https://api.minimax.chat/v1/", None),
    p("mistral", "https://api.mistral.ai", None),
    p(
        "modelscope",
        "https://api-inference.modelscope.cn/v1/",
        None,
    ),
    p("moonshot", "https://api.moonshot.cn", None),
    p("nvidia", "https://integrate.api.nvidia.com", None),
    p("o3", "https://api.o3.fan", None),
    p("ocoolai", "https://api.ocoolai.com", None),
    p("ollama", "http://localhost:11434", None),
    p("openai", "https://api.openai.com", None),
    p("openrouter", "https://openrouter.ai/api/v1/", None),
    p("perplexity", "https://api.perplexity.ai/", None),
    p("ppio", "https://api.ppinfra.com/v3/openai", None),
    p("silicon", "https://api.siliconflow.cn", None),
    p("stepfun", "https://api.stepfun.com", None),
    p(
        "tencent-cloud-ti",
        "https://api.lkeap.cloud.tencent.com",
        None,
    ),
    p("together", "https://api.together.xyz", None),
    p("xirang", "https://wishub-x1.ctyun.cn", None),
    p("yi", "https://api.lingyiwanwu.com", None),
    p("zhinao", "https://api.360.cn", None),
    p("zhipu", "https://open.bigmodel.cn/api/paas/v4/", None),
];

static PROVIDERS: LazyLock<HashMap<&'static str, &'static ProviderInfo>> =
    LazyLock::new(|| PROVIDER_LIST.iter().map(|info| (info.name, info)).collect());

#[derive(Debug, Clone)]
pub struct ParsedModel {
    pub provider_name: String,
    pub model_name: String,
    pub base_url: String,
}

pub fn parse_model_string(model: &str) -> Result<ParsedModel, Error> {
    let model = model.trim();
    if model.is_empty() {
        return Err(Error::InvalidModel("model string must not be empty".into()));
    }

    let (provider_name, raw_model_name) = match model.split_once('/') {
        Some((prov, name)) => (prov.to_string(), name.trim().to_string()),
        None => (model.to_string(), String::new()),
    };

    let (base_url, model_name) = if let Some(info) = PROVIDERS.get(provider_name.as_str()) {
        let model_name = if raw_model_name.is_empty() {
            info.default_model.map(str::to_string).ok_or_else(|| {
                Error::InvalidModel(format!("model name missing for provider '{provider_name}'"))
            })?
        } else {
            raw_model_name
        };
        (info.base_url.to_string(), model_name)
    } else {
        let env_key = provider_env_key(&provider_name, "BASE_URL");
        let env_val = env::var(&env_key).unwrap_or_default();
        if env_val.trim().is_empty() {
            return Err(Error::InvalidModel(format!(
                "unknown provider '{provider_name}' and {env_key} not set"
            )));
        }
        if raw_model_name.is_empty() {
            return Err(Error::InvalidModel(format!(
                "model name missing for provider '{provider_name}'"
            )));
        }
        (env_val, raw_model_name)
    };

    Ok(ParsedModel {
        provider_name,
        model_name,
        base_url,
    })
}

pub fn resolve_base_url(parsed: &ParsedModel, explicit: Option<&str>) -> Result<String, Error> {
    if let Some(url) = explicit {
        let url = url.trim();
        if !url.is_empty() {
            return Ok(url.to_string());
        }
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

fn has_version_suffix(url: &str) -> bool {
    let trimmed = url.trim_end_matches('/');
    let last = match trimmed.rsplit('/').next() {
        Some(s) => s,
        None => return false,
    };
    let mut chars = last.chars();
    matches!(chars.next(), Some('v')) && chars.next().is_some_and(|c| c.is_ascii_digit())
}

pub fn build_request_url(base_url: &str, provider_name: &str) -> String {
    let base = base_url.trim();

    match provider_name {
        "anthropic" => {
            let stripped = base.trim_end_matches('/');
            if has_version_suffix(stripped) {
                format!("{stripped}/chat/completions")
            } else {
                format!("{stripped}/v1/chat/completions")
            }
        }
        "gemini" => {
            let stripped = base.trim_end_matches('/');
            if has_version_suffix(stripped) {
                format!("{stripped}/chat/completions")
            } else {
                format!("{stripped}/v1beta/openai/chat/completions")
            }
        }
        _ => {
            if base.ends_with('#') {
                base.trim_end_matches('#').to_string()
            } else if base.ends_with('/') {
                format!("{base}chat/completions")
            } else if has_version_suffix(base) {
                format!("{base}/chat/completions")
            } else {
                format!("{base}/v1/chat/completions")
            }
        }
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
    fn test_parse_model_default() {
        let parsed = parse_model_string("gemini").unwrap();
        assert_eq!(parsed.model_name, "gemini-2.0-flash");
    }

    #[test]
    fn test_parse_model_empty() {
        assert!(parse_model_string("").is_err());
    }

    #[test]
    fn test_parse_model_no_default() {
        assert!(parse_model_string("openai").is_err());
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
