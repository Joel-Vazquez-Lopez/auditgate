use crate::Document;

pub fn search<'a>(
    claim: &str,
    documents: &'a [Document],
    limit: usize,
) -> Vec<(&'a Document, usize)> {
    let query_words: Vec<String> = claim
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect();

    let mut results = Vec::new();

    for document in documents {
        let text = format!(
            "{} {}",
            document.title,
            document
                .passages
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        )
        .to_lowercase();

        let score = query_words
            .iter()
            .filter(|word| text.contains(word.as_str()))
            .count();

        if score > 0 {
            results.push((document, score));
        }
    }

    results.sort_by(|a, b| b.1.cmp(&a.1));
    results.truncate(limit);

    results
}

pub fn claim_coverage<'a, I>(
    claim: &str,
    passages: I,
) -> f64
where
    I: IntoIterator<Item = &'a str>,
{
    let claim_tokens = meaningful_tokens(claim);

    if claim_tokens.is_empty() {
        return 0.0;
    }

    let evidence_text = passages
        .into_iter()
        .collect::<Vec<_>>()
        .join(" ");

    let evidence_tokens = meaningful_tokens(&evidence_text);

    let covered = claim_tokens
        .iter()
        .filter(|token| evidence_tokens.contains(*token))
        .count();

    covered as f64 / claim_tokens.len() as f64
}


fn meaningful_tokens(text: &str) -> std::collections::HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "a", "an", "the",
        "and", "or", "but",
        "of", "to", "in", "on", "for",
        "is", "are", "was", "were",
        "be", "been", "being",
        "with", "by", "from", "as",
        "that", "this", "these", "those",
    ];

    text.split_whitespace()
        .map(|token| {
            token
                .chars()
                .filter(|character| character.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|token| {
            token.len() >= 3
                && !STOPWORDS.contains(&token.as_str())
        })
        .collect()
}