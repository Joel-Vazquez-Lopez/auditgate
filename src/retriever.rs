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