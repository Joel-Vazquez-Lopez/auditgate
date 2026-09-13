use super::EvidencePassage;

use serde::Deserialize;
use serde_json::json;

use std::io::Write;
use std::process::{
    Command,
    Stdio,
};


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

    let passage_texts: Vec<&str> =
        passages
            .iter()
            .map(|passage| passage.text.as_str())
            .collect();

    let input = json!({
        "claim": claim,
        "passages": passage_texts,
    });

    let mut child =
        Command::new("python")
            .arg("reranker/rerank.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                format!(
                    "Could not start reranker: {}",
                    error
                )
            })?;

    {
        let stdin =
            child
                .stdin
                .as_mut()
                .ok_or(
                    "Could not open reranker stdin"
                        .to_string()
                )?;

        stdin
            .write_all(
                input.to_string().as_bytes()
            )
            .map_err(|error| {
                format!(
                    "Could not send passages to reranker: {}",
                    error
                )
            })?;
    }

    let output =
        child
            .wait_with_output()
            .map_err(|error| {
                format!(
                    "Reranker process failed: {}",
                    error
                )
            })?;

    if !output.status.success() {
        return Err(format!(
            "Reranker exited with status {}",
            output.status
        ));
    }

    let response: RerankerResponse =
        serde_json::from_slice(
            &output.stdout
        )
        .map_err(|error| {
            format!(
                "Could not parse reranker output: {}",
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

    let mut scored: Vec<ScoredEvidencePassage> =
        passages
            .iter()
            .cloned()
            .zip(response.scores)
            .map(
                |(passage, relevance_score)| {
                    ScoredEvidencePassage {
                        passage,
                        relevance_score,
                    }
                }
            )
            .collect();

    scored.sort_by(
        |a, b| {
            b.relevance_score
                .total_cmp(&a.relevance_score)
        }
    );

    Ok(scored)
}

pub fn relevance_concentration(
    passages: &[ScoredEvidencePassage],
) -> f64 {
    if passages.is_empty() {
        return 0.0;
    }

    let max_score = passages
        .iter()
        .map(|passage| passage.relevance_score)
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = passages
        .iter()
        .map(|passage| {
            (passage.relevance_score - max_score).exp()
        })
        .collect();

    let total: f64 = weights.iter().sum();

    if total == 0.0 {
        return 0.0;
    }

    let top_mass: f64 = weights
        .iter()
        .take(3)
        .sum();

    top_mass / total
}


pub fn source_diversity(
    passages: &[ScoredEvidencePassage],
) -> f64 {
    let useful: Vec<_> = passages
        .iter()
        .filter(|passage| {
            passage.relevance_score > 0.0
        })
        .collect();

    if useful.is_empty() {
        return 0.0;
    }

    let unique_sources: std::collections::HashSet<&str> =
        useful
            .iter()
            .map(|passage| {
                passage.passage.source_url.as_str()
            })
            .collect();

    unique_sources.len() as f64
        / useful.len() as f64
}