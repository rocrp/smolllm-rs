use std::time::Duration;

use crate::Error;

fn split_comma_list(items: &str) -> Result<Vec<String>, String> {
    let items = items.trim();
    if items.is_empty() {
        return Err("value must not be empty".into());
    }
    let result: Vec<String> = items
        .split(',')
        .map(|item| item.trim().to_string())
        .collect();
    if result.iter().any(String::is_empty) {
        return Err("list contains empty entry".into());
    }
    Ok(result)
}

pub fn parse_model_list(items: &str) -> Result<Vec<String>, Error> {
    split_comma_list(items).map_err(|reason| Error::InvalidModelList { reason })
}

pub fn parse_api_key_list(items: &str) -> Result<Vec<String>, Error> {
    split_comma_list(items).map_err(|reason| Error::InvalidApiKeyList { reason })
}

pub fn parse_base_url_list(items: &str) -> Result<Vec<String>, Error> {
    split_comma_list(items).map_err(|reason| Error::InvalidBaseUrlList { reason })
}

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.len() / 4
    }
}

pub fn strip_backticks(text: &str) -> String {
    let trimmed = text.trim();
    if !trimmed.starts_with("```") || !trimmed.ends_with("```") || trimmed.len() < 6 {
        return text.to_string();
    }
    let after_open = match trimmed[3..].find('\n') {
        Some(n) => &trimmed[3 + n + 1..],
        None => return text.to_string(),
    };
    let body = match after_open.trim_end().strip_suffix("```") {
        Some(b) => b,
        None => return text.to_string(),
    };
    body.trim_end_matches('\n').to_string()
}

pub fn preview_api_key(key: &str) -> String {
    if key.len() <= 9 {
        key.to_string()
    } else {
        format!("{}...{}", &key[..5], &key[key.len() - 4..])
    }
}

pub fn format_metrics(
    model_name: &str,
    input_tokens: usize,
    output_tokens: usize,
    total: Duration,
    ttft: Option<Duration>,
    estimated: bool,
) -> String {
    let total_tokens = input_tokens + output_tokens;
    let tok_per_sec = if total.as_secs_f64() > 0.0 && output_tokens > 0 {
        (output_tokens as f64 / total.as_secs_f64()) as usize
    } else {
        0
    };

    let approx = if estimated { "~" } else { "" };
    let mut s = format!(
        "\u{1f4ca}{model_name} {approx}{total_tokens}tok (\u{2191}{input_tokens} \u{2193}{output_tokens})"
    );

    if let Some(ttft) = ttft {
        s.push_str(&format!(" | \u{1f680}{}", format_duration(ttft)));
    }

    s.push_str(&format!(
        " | \u{1f40e}{tok_per_sec}tok/s | \u{231b}{}",
        format_duration(total)
    ));
    s
}

fn format_duration(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else if ms >= 100.0 {
        format!("{:.0}ms", ms)
    } else if ms >= 10.0 {
        format!("{:.1}ms", ms)
    } else {
        format!("{:.2}ms", ms)
    }
}

/// Provider error bodies get logged and stored on errors, and they routinely
/// echo the request's own credentials back. 500 characters is enough to
/// diagnose one and short enough to keep out of a terminal-sized error line.
const MAX_ERROR_DETAIL: usize = 500;
const REDACTED: &str = "[REDACTED_CREDENTIAL]";

/// Word characters end at anything a JSON body or header line uses to separate
/// a value from its surroundings, so a redacted value stops at the quote or
/// comma that follows it.
fn is_delimiter(c: char) -> bool {
    c.is_whitespace() || matches!(c, ',' | '"' | '\'' | '{' | '}' | '[' | ']' | '(' | ')' | ':' | '=' | ';')
}

/// Names that introduce a credential when followed by `:` or `=`.
fn is_credential_name(word: &str) -> bool {
    [
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "access-token",
        "accesstoken",
        "token",
        "key",
    ]
    .iter()
    .any(|name| word.eq_ignore_ascii_case(name))
}

/// Shapes that are a secret wherever they appear, with no name to introduce them.
fn looks_like_secret(word: &str) -> bool {
    if word.starts_with("AIza") && word.len() >= 16 {
        return true;
    }
    ["sk-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        .iter()
        .any(|prefix| word.starts_with(prefix) && word.len() >= prefix.len() + 10)
}

/// Replaces credentials a provider echoed back with a marker. Handles the three
/// shapes that actually turn up in error bodies: a `Bearer` token, a named
/// assignment such as `api_key=...`, and bare keys with a known prefix.
pub fn redact_credentials(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut redact_this_word = false;

    while !rest.is_empty() {
        let word_start = rest.find(|c: char| !is_delimiter(c)).unwrap_or(rest.len());
        let (delimiters, tail) = rest.split_at(word_start);
        out.push_str(delimiters);
        if tail.is_empty() {
            break;
        }

        let word_end = tail.find(is_delimiter).unwrap_or(tail.len());
        let (word, after) = tail.split_at(word_end);
        if redact_this_word || looks_like_secret(word) {
            out.push_str(REDACTED);
        } else {
            out.push_str(word);
        }

        let next_word_at = after.find(|c: char| !is_delimiter(c)).unwrap_or(after.len());
        let separator = &after[..next_word_at];
        redact_this_word = word.eq_ignore_ascii_case("bearer")
            || (is_credential_name(word) && separator.contains([':', '=']));
        rest = after;
    }
    out
}

/// A provider error body made safe to keep: credentials removed, whitespace
/// collapsed, and capped so one HTML error page cannot flood a log.
pub fn brief_error_detail(text: &str) -> String {
    let detail = redact_credentials(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if detail.is_empty() {
        return "no detail".to_string();
    }
    if detail.chars().count() <= MAX_ERROR_DETAIL {
        return detail;
    }
    let head: String = detail.chars().take(MAX_ERROR_DETAIL - 3).collect();
    format!("{head}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_backticks() {
        assert_eq!(strip_backticks("hello"), "hello");
        assert_eq!(
            strip_backticks("```rust\nfn main() {}\n```"),
            "fn main() {}"
        );
        assert_eq!(strip_backticks("```\ncode\n```"), "code");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn credentials_never_survive_an_error_body() {
        let redacted = redact_credentials(
            r#"{"error":"bad key","headers":{"Authorization":"Bearer sk-live-abcdef123456"}}"#,
        );
        assert!(!redacted.contains("sk-live-abcdef123456"), "{redacted}");
        assert!(redacted.contains(REDACTED));

        assert!(!redact_credentials("api_key=AIzaSyA1234567890abcdef").contains("AIzaSyA"));
        assert!(!redact_credentials("used sk-0123456789abcdef here").contains("sk-01234"));
    }

    #[test]
    fn error_detail_is_collapsed_and_capped() {
        assert_eq!(brief_error_detail("  bad\n  request\t "), "bad request");
        assert_eq!(brief_error_detail("   "), "no detail");

        let long = brief_error_detail(&"x".repeat(2_000));
        assert_eq!(long.chars().count(), MAX_ERROR_DETAIL);
        assert!(long.ends_with("..."));
    }

    #[test]
    fn test_preview_api_key() {
        assert_eq!(preview_api_key("short"), "short");
        assert_eq!(preview_api_key("sk-1234567890abcdef"), "sk-12...cdef");
    }
}
