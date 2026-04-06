use std::path::Path;

use crate::error::Error;

pub fn image_to_data_url(path: &str) -> Result<String, Error> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(Error::Image("image path cannot be empty".into()));
    }

    if trimmed.starts_with("data:") {
        return Ok(trimmed.to_string());
    }

    let content = std::fs::read(trimmed)
        .map_err(|e| Error::Image(format!("failed to read image '{}': {}", trimmed, e)))?;

    let mime = mime_type_for(trimmed, &content)
        .ok_or_else(|| Error::Image(format!("unable to determine MIME type for '{}'", trimmed)))?;

    let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &content);
    Ok(format!("data:{mime};base64,{encoded}"))
}

fn mime_type_for(path: &str, content: &[u8]) -> Option<String> {
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        let mime = match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            "bmp" => "image/bmp",
            "ico" => "image/x-icon",
            "tiff" | "tif" => "image/tiff",
            _ => "",
        };
        if !mime.is_empty() {
            return Some(mime.to_string());
        }
    }

    // Content sniffing fallback
    if content.len() >= 4 {
        if content.starts_with(b"\x89PNG") {
            return Some("image/png".into());
        }
        if content.starts_with(b"\xFF\xD8\xFF") {
            return Some("image/jpeg".into());
        }
        if content.starts_with(b"GIF8") {
            return Some("image/gif".into());
        }
        if content.len() >= 12 && &content[0..4] == b"RIFF" && &content[8..12] == b"WEBP" {
            return Some("image/webp".into());
        }
    }

    None
}
