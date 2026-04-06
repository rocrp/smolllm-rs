use std::time::Duration;

pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() { 0 } else { text.len() / 4 }
}

pub fn strip_backticks(text: &str) -> String {
    if !text.starts_with("```") || !text.ends_with("```") {
        return text.to_string();
    }
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.is_empty() {
        return String::new();
    }

    let start = if lines[0].starts_with("```") { 1 } else { 0 };
    let end = if lines.len() > start {
        let last = lines[lines.len() - 1];
        if last == "```" {
            lines.len() - 1
        } else if last.ends_with("```") {
            lines.len()
        } else {
            lines.len()
        }
    } else {
        lines.len()
    };

    let mut result: Vec<&str> = lines[start..end].to_vec();
    if let Some(last) = result.last_mut() {
        if last.ends_with("```") && *last != "```" {
            // Trim trailing ``` from the last line
            let trimmed = &last[..last.len() - 3];
            *last = trimmed;
        }
    }
    result.join("\n")
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
) -> String {
    let total_tokens = input_tokens + output_tokens;
    let tok_per_sec = if total.as_secs_f64() > 0.0 && output_tokens > 0 {
        (output_tokens as f64 / total.as_secs_f64()) as usize
    } else {
        0
    };

    let mut s = format!(
        "\u{1f4ca}{model_name} {total_tokens}tok (\u{2191}{input_tokens} \u{2193}{output_tokens})"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_backticks() {
        assert_eq!(strip_backticks("hello"), "hello");
        assert_eq!(strip_backticks("```rust\nfn main() {}\n```"), "fn main() {}");
        assert_eq!(strip_backticks("```\ncode\n```"), "code");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_preview_api_key() {
        assert_eq!(preview_api_key("short"), "short");
        assert_eq!(preview_api_key("sk-1234567890abcdef"), "sk-12...cdef");
    }
}
