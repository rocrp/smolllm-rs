use std::sync::LazyLock;
use std::collections::HashMap;
use std::env;

use crate::error::Error;

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub base_url: &'static str,
    pub default_model: &'static str,
}

static PROVIDERS: LazyLock<HashMap<&'static str, ProviderInfo>> = LazyLock::new(|| {
    let entries: Vec<ProviderInfo> = vec![
        ProviderInfo { name: "aihubmix",                base_url: "https://aihubmix.com",                           default_model: "" },
        ProviderInfo { name: "anthropic",               base_url: "https://api.anthropic.com/",                      default_model: "" },
        ProviderInfo { name: "azure-openai",            base_url: "",                                                default_model: "" },
        ProviderInfo { name: "baichuan",                base_url: "https://api.baichuan-ai.com",                     default_model: "" },
        ProviderInfo { name: "baidu-cloud",             base_url: "https://qianfan.baidubce.com/v2/",                default_model: "" },
        ProviderInfo { name: "cerebras",                base_url: "https://api.cerebras.ai",                         default_model: "" },
        ProviderInfo { name: "dashscope",               base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1/", default_model: "" },
        ProviderInfo { name: "deepseek",                base_url: "https://api.deepseek.com",                        default_model: "" },
        ProviderInfo { name: "dmxapi",                  base_url: "https://www.dmxapi.cn",                           default_model: "" },
        ProviderInfo { name: "doubao",                  base_url: "https://ark.cn-beijing.volces.com/api/v3/",       default_model: "" },
        ProviderInfo { name: "fireworks",               base_url: "https://api.fireworks.ai/inference",              default_model: "" },
        ProviderInfo { name: "gemini",                  base_url: "https://generativelanguage.googleapis.com",       default_model: "gemini-2.0-flash" },
        ProviderInfo { name: "gitee-ai",                base_url: "https://ai.gitee.com",                           default_model: "" },
        ProviderInfo { name: "github",                  base_url: "https://models.inference.ai.azure.com/",          default_model: "" },
        ProviderInfo { name: "graphrag-kylin-mountain", base_url: "",                                                default_model: "" },
        ProviderInfo { name: "grok",                    base_url: "https://api.x.ai",                                default_model: "" },
        ProviderInfo { name: "groq",                    base_url: "https://api.groq.com/openai",                     default_model: "" },
        ProviderInfo { name: "hunyuan",                 base_url: "https://api.hunyuan.cloud.tencent.com",           default_model: "" },
        ProviderInfo { name: "hyperbolic",              base_url: "https://api.hyperbolic.xyz",                      default_model: "" },
        ProviderInfo { name: "infini",                  base_url: "https://cloud.infini-ai.com/maas",                default_model: "" },
        ProviderInfo { name: "jina",                    base_url: "https://api.jina.ai",                             default_model: "" },
        ProviderInfo { name: "lmstudio",                base_url: "http://localhost:1234",                           default_model: "" },
        ProviderInfo { name: "minimax",                 base_url: "https://api.minimax.chat/v1/",                    default_model: "" },
        ProviderInfo { name: "mistral",                 base_url: "https://api.mistral.ai",                          default_model: "" },
        ProviderInfo { name: "modelscope",              base_url: "https://api-inference.modelscope.cn/v1/",         default_model: "" },
        ProviderInfo { name: "moonshot",                base_url: "https://api.moonshot.cn",                         default_model: "" },
        ProviderInfo { name: "nvidia",                  base_url: "https://integrate.api.nvidia.com",                default_model: "" },
        ProviderInfo { name: "o3",                      base_url: "https://api.o3.fan",                              default_model: "" },
        ProviderInfo { name: "ocoolai",                 base_url: "https://api.ocoolai.com",                         default_model: "" },
        ProviderInfo { name: "ollama",                  base_url: "http://localhost:11434",                          default_model: "" },
        ProviderInfo { name: "openai",                  base_url: "https://api.openai.com",                          default_model: "" },
        ProviderInfo { name: "openrouter",              base_url: "https://openrouter.ai/api/v1/",                   default_model: "" },
        ProviderInfo { name: "perplexity",              base_url: "https://api.perplexity.ai/",                      default_model: "" },
        ProviderInfo { name: "ppio",                    base_url: "https://api.ppinfra.com/v3/openai",               default_model: "" },
        ProviderInfo { name: "silicon",                 base_url: "https://api.siliconflow.cn",                      default_model: "" },
        ProviderInfo { name: "stepfun",                 base_url: "https://api.stepfun.com",                         default_model: "" },
        ProviderInfo { name: "tencent-cloud-ti",        base_url: "https://api.lkeap.cloud.tencent.com",             default_model: "" },
        ProviderInfo { name: "together",                base_url: "https://api.together.xyz",                        default_model: "" },
        ProviderInfo { name: "xirang",                  base_url: "https://wishub-x1.ctyun.cn",                      default_model: "" },
        ProviderInfo { name: "yi",                      base_url: "https://api.lingyiwanwu.com",                     default_model: "" },
        ProviderInfo { name: "zhinao",                  base_url: "https://api.360.cn",                              default_model: "" },
        ProviderInfo { name: "zhipu",                   base_url: "https://open.bigmodel.cn/api/paas/v4/",           default_model: "" },
    ];
    entries.into_iter().map(|p| (p.name, p)).collect()
});

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

    let (provider_name, model_name) = if let Some(pos) = model.find('/') {
        let prov = &model[..pos];
        let name = model[pos + 1..].trim();
        (prov.to_string(), name.to_string())
    } else {
        (model.to_string(), String::new())
    };

    let base_url;
    let final_model_name;

    if let Some(info) = PROVIDERS.get(provider_name.as_str()) {
        base_url = info.base_url.to_string();
        final_model_name = if model_name.is_empty() {
            if info.default_model.is_empty() {
                return Err(Error::InvalidModel(format!(
                    "model name missing for provider '{provider_name}'"
                )));
            }
            info.default_model.to_string()
        } else {
            model_name
        };
    } else {
        let env_key = provider_env_key(&provider_name, "BASE_URL");
        let env_val = env::var(&env_key).unwrap_or_default();
        if env_val.trim().is_empty() {
            return Err(Error::InvalidModel(format!(
                "unknown provider '{provider_name}' and {env_key} not set"
            )));
        }
        base_url = env_val;
        final_model_name = if model_name.is_empty() {
            return Err(Error::InvalidModel(format!(
                "model name missing for provider '{provider_name}'"
            )));
        } else {
            model_name
        };
    }

    Ok(ParsedModel {
        provider_name,
        model_name: final_model_name,
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
    if let Some(last_seg) = trimmed.rsplit('/').next() {
        if last_seg.starts_with('v') && last_seg.len() > 1 {
            return last_seg[1..].chars().all(|c| c.is_ascii_digit());
        }
        if last_seg.starts_with('v') && last_seg.len() > 1 {
            let rest = &last_seg[1..];
            return rest.chars().next().map_or(false, |c| c.is_ascii_digit());
        }
    }
    false
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
        assert_eq!(provider_env_key("tencent-cloud-ti", "API_KEY"), "TENCENT_CLOUD_TI_API_KEY");
    }

    #[test]
    fn test_has_version_suffix() {
        assert!(has_version_suffix("https://api.com/v1"));
        assert!(has_version_suffix("https://api.com/v2/"));
        assert!(!has_version_suffix("https://api.com/openai"));
        assert!(!has_version_suffix("https://api.com"));
    }
}
