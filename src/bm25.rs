use crate::Document;
use std::collections::{HashMap, HashSet};

pub struct BM25 {
    documents: Vec<Document>,
    words: Vec<Vec<String>>,
    document_frequency: HashMap<String, usize>,
    average_length: f64,
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
            let mut score = 0.0;

            for query_word in &query_words {
                let frequency = words
                    .iter()
                    .filter(|word| *word == query_word)
                    .count() as f64;

                if frequency == 0.0 {
                    continue;
                }

                let df = *self
                    .document_frequency
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
                                + b * length / self.average_length
                            )
                    );
            }

            if score > 0.0 {
                results.push((&self.documents[i], score));
            }
        }

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(limit);

        results
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(String::from)
        .collect()
}