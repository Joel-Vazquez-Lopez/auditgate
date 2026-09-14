use super::{SearchResult, SourceContentType, SourceDocument};

use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, CONTENT_TYPE, USER_AGENT};

pub struct SourceFetcher {
    client: Client,
}

impl SourceFetcher {
    pub fn new() -> Result<Self, String> {
        let client = Client::builder()
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|error| format!("Could not create HTTP client: {}", error))?;

        Ok(Self { client })
    }

    pub fn fetch(&self, result: &SearchResult) -> Result<SourceDocument, String> {
        let response = self
            .client
            .get(&result.url)
            .header(USER_AGENT, "AuditGate/0.1 evidence-acquisition")
            .header(
                ACCEPT,
                "text/html,application/xhtml+xml,application/pdf,text/plain",
            )
            .send()
            .map_err(|error| format!("Could not fetch {}: {}", result.url, error))?;

        if !response.status().is_success() {
            return Err(format!(
                "{} returned HTTP {}",
                result.url,
                response.status()
            ));
        }

        let content_type_header = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let content_type = detect_content_type(&content_type_header, &result.url);

        let text = response
            .text()
            .map_err(|error| format!("Could not read {}: {}", result.url, error))?;

        Ok(SourceDocument {
            title: result.title.clone(),
            url: result.url.clone(),
            source_name: result.source_name.clone(),
            content_type,
            text,
        })
    }
}

fn detect_content_type(header: &str, url: &str) -> SourceContentType {
    if header.contains("application/pdf") || url.to_lowercase().ends_with(".pdf") {
        return SourceContentType::Pdf;
    }

    if header.contains("text/html") || header.contains("application/xhtml+xml") {
        return SourceContentType::Html;
    }

    if header.contains("text/plain") {
        return SourceContentType::PlainText;
    }

    SourceContentType::Unknown
}
