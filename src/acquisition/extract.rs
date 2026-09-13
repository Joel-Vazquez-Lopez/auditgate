use super::{
    EvidencePassage,
    SourceContentType,
    SourceDocument,
};

use scraper::{
    Html,
    Selector,
};

use std::collections::HashSet;


pub fn extract_passages(
    document: &SourceDocument,
) -> Result<Vec<EvidencePassage>, String> {
    match &document.content_type {
        SourceContentType::Html => {
            extract_html_passages(document)
        }

        SourceContentType::PlainText => {
            extract_plain_text_passages(document)
        }

        SourceContentType::Pdf => {
            Err(
                "PDF extraction is not implemented yet"
                    .to_string()
            )
        }

        SourceContentType::Unknown => {
            Err(
                "Unsupported source content type"
                    .to_string()
            )
        }
    }
}


fn extract_html_passages(
    document: &SourceDocument,
) -> Result<Vec<EvidencePassage>, String> {
    let html =
        Html::parse_document(&document.text);

    let paragraph_selector =
        Selector::parse("p")
            .map_err(|error| {
                format!(
                    "Could not create HTML selector: {}",
                    error
                )
            })?;

    let mut passages = Vec::new();
    let mut seen = HashSet::new();

    for paragraph in html.select(&paragraph_selector) {
        let text = paragraph
            .text()
            .collect::<Vec<_>>()
            .join(" ");

        let text =
            normalize_whitespace(&text);

        // Ignore navigation fragments, captions,
        // tiny boilerplate, etc.
        if text.len() < 80 {
            continue;
        }

        if !seen.insert(text.clone()) {
            continue;
        }

        passages.push(
            EvidencePassage {
                source_url:
                    document.url.clone(),

                source_title:
                    document.title.clone(),

                source_name:
                    document.source_name.clone(),

                text,
            }
        );
    }

    Ok(passages)
}


fn extract_plain_text_passages(
    document: &SourceDocument,
) -> Result<Vec<EvidencePassage>, String> {
    let mut passages = Vec::new();

    for paragraph in document.text.split("\n\n") {
        let text =
            normalize_whitespace(paragraph);

        if text.len() < 80 {
            continue;
        }

        passages.push(
            EvidencePassage {
                source_url:
                    document.url.clone(),

                source_title:
                    document.title.clone(),

                source_name:
                    document.source_name.clone(),

                text,
            }
        );
    }

    Ok(passages)
}


fn normalize_whitespace(
    text: &str,
) -> String {
    text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}