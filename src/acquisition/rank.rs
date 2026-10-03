use super::EvidencePassage;

use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ScoredEvidencePassage {
    pub passage: EvidencePassage,
    pub relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct RerankerResponse {
    scores: Vec<f64>,
}

pub fn rank_passages(
    claim: &str,
    passages: &[EvidencePassage],
) -> Result<Vec<ScoredEvidencePassage>, String> {
    if passages.is_empty() {
        return Ok(Vec::new());
    }

    let passage_texts: Vec<&str> = passages
        .iter()
        .map(|passage| passage.text.as_str())
        .collect();

    let input = json!({
        "claim": claim,
        "passages": passage_texts,
    });

    let response = reqwest::blocking::Client::new()
        .post("http://127.0.0.1:8001/rerank")
        .json(&input)
        .send()
        .map_err(|error| {
            format!(
                "Failed to call reranker model service: {}",
                error
            )
        })?
        .error_for_status()
        .map_err(|error| {
            format!(
                "Reranker model service returned an error: {}",
                error
            )
        })?
        .json::<RerankerResponse>()
        .map_err(|error| {
            format!(
                "Invalid reranker model service response: {}",
                error
            )
        })?;

    if response.scores.len() != passages.len() {
        return Err(format!(
            "Reranker returned {} scores for {} passages",
            response.scores.len(),
            passages.len()
        ));
    }

    let mut scored: Vec<ScoredEvidencePassage> = passages
        .iter()
        .cloned()
        .zip(response.scores)
        .map(|(passage, relevance_score)| ScoredEvidencePassage {
            passage,
            relevance_score,
        })
        .collect();

    scored.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));

    Ok(scored)
}

pub fn relevance_concentration(passages: &[ScoredEvidencePassage]) -> f64 {
    if passages.is_empty() {
        return 0.0;
    }

    let max_score = passages
        .iter()
        .map(|passage| passage.relevance_score)
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = passages
        .iter()
        .map(|passage| (passage.relevance_score - max_score).exp())
        .collect();

    let total: f64 = weights.iter().sum();

    if total == 0.0 {
        return 0.0;
    }

    let top_mass: f64 = weights.iter().take(3).sum();

    top_mass / total
}

pub fn source_diversity(passages: &[ScoredEvidencePassage]) -> f64 {
    let useful: Vec<_> = passages
        .iter()
        .filter(|passage| passage.relevance_score > 0.0)
        .collect();

    if useful.is_empty() {
        return 0.0;
    }

    let unique_sources: std::collections::HashSet<&str> = useful
        .iter()
        .map(|passage| passage.passage.source_url.as_str())
        .collect();

    unique_sources.len() as f64 / useful.len() as f64
}
