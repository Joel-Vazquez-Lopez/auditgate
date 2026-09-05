use crate::{Document, Passage};
use std::collections::{HashMap, HashSet};

pub struct BM25 {
    documents: Vec<Document>,
    words: Vec<Vec<String>>,
    document_frequency: HashMap<String, usize>,
    average_length: f64,
}

pub struct EvidenceCandidate<'a> {
    pub document: &'a Document,
    pub passage: &'a Passage,
    pub document_score: f64,
    pub passage_score: f64,
    pub combined_score: f64,
}

impl BM25 {
    pub fn new(documents: Vec<Document>) -> Self {
        let words: Vec<Vec<String>> = documents
            .iter()
            .map(|doc| {
                let text = format!(
                    "{} {}",
                    doc.title,
                    doc.passages
                        .iter()
                        .map(|p| p.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                );

                tokenize(&text)
            })
            .collect();

        let average_length =
            words.iter().map(|w| w.len()).sum::<usize>() as f64
                / words.len() as f64;

        let mut document_frequency = HashMap::new();

        for document_words in &words {
            let unique: HashSet<&String> =
                document_words.iter().collect();

            for word in unique {
                *document_frequency
                    .entry(word.clone())
                    .or_insert(0) += 1;
            }
        }

        Self {
            documents,
            words,
            document_frequency,
            average_length,
        }
    }

    pub fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<(&Document, f64)> {
        let query_words = tokenize(query);
        let n = self.documents.len() as f64;

        let mut results = Vec::new();

        for (i, words) in self.words.iter().enumerate() {
            let score = bm25_score(
                &query_words,
                words,
                &self.document_frequency,
                n,
                self.average_length,
            );

            if score > 0.0 {
                results.push((&self.documents[i], score));
            }
        }

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);

        results
    }

    pub fn search_passages<'a>(
    &'a self,
    query: &str,
    documents: &[(&'a Document, f64)],
    limit: usize,
) -> Vec<EvidenceCandidate<'a>> {
    let query_words = tokenize(query);

    // Build passage-level statistics across all candidate documents.
    let all_passage_words: Vec<Vec<String>> = documents
        .iter()
        .flat_map(|(document, _)| {
            document
                .passages
                .iter()
                .map(|passage| tokenize(&passage.text))
        })
        .collect();

    let average_passage_length = if all_passage_words.is_empty() {
        1.0
    } else {
        all_passage_words
            .iter()
            .map(|words| words.len())
            .sum::<usize>() as f64
            / all_passage_words.len() as f64
    };

    let mut passage_df: HashMap<String, usize> =
        HashMap::new();

    for words in &all_passage_words {
        let unique: HashSet<&String> =
            words.iter().collect();

        for word in unique {
            *passage_df
                .entry(word.clone())
                .or_insert(0) += 1;
        }
    }

    let n_passages = all_passage_words.len() as f64;

    let mut candidates = Vec::new();

    // IMPORTANT:
    // Select the best passage independently from each document.
    for (document, document_score) in documents {
        let mut best_passage: Option<(&Passage, f64)> = None;

        for passage in &document.passages {
            let passage_words = tokenize(&passage.text);

            let passage_score = bm25_score(
                &query_words,
                &passage_words,
                &passage_df,
                n_passages,
                average_passage_length,
            );

            match best_passage {
                None => {
                    best_passage =
                        Some((passage, passage_score));
                }

                Some((_, best_score))
                    if passage_score > best_score =>
                {
                    best_passage =
                        Some((passage, passage_score));
                }

                _ => {}
            }
        }

        if let Some((passage, passage_score)) = best_passage {
            candidates.push(EvidenceCandidate {
                document,
                passage,
                document_score: *document_score,
                passage_score,

                // We keep this field because the struct
                // already has it, but we're no longer using
                // a magic weighted combination for selection.
                combined_score: passage_score,
            });
        }
    }

    // Preserve document ranking.
    candidates.sort_by(|a, b| {
        b.document_score.total_cmp(&a.document_score)
    });

    candidates.truncate(limit);

    candidates
}
}

fn bm25_score(
    query_words: &[String],
    words: &[String],
    document_frequency: &HashMap<String, usize>,
    n: f64,
    average_length: f64,
) -> f64 {
    if words.is_empty() || n == 0.0 {
        return 0.0;
    }

    let mut score = 0.0;

    for query_word in query_words {
        let frequency = words
            .iter()
            .filter(|word| *word == query_word)
            .count() as f64;

        if frequency == 0.0 {
            continue;
        }

        let df = *document_frequency
            .get(query_word)
            .unwrap_or(&0) as f64;

        let idf =
            ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        let k1 = 1.5;
        let b = 0.75;
        let length = words.len() as f64;

        score += idf
            * (frequency * (k1 + 1.0))
            / (
                frequency
                    + k1
                        * (
                            1.0
                                - b
                                + b
                                    * length
                                    / average_length
                        )
            );
    }

    score
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(String::from)
        .collect()
}
