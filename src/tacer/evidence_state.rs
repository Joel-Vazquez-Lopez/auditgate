use crate::acquisition::{ScoredEvidencePassage, relevance_concentration, source_diversity};
use crate::overlap;
use crate::tacer::types::EvidenceState;

fn source_quality_score(url: &str) -> f64 {
    let url = url.to_lowercase();

    if url.contains("pubmed.ncbi.nlm.nih.gov")
        || url.contains("pmc.ncbi.nlm.nih.gov")
        || url.contains(".gov/")
        || url.contains(".gov.")
    {
        return 1.0;
    }

    if url.contains(".edu/")
        || url.contains(".edu.")
        || url.contains(".ac.uk")
        || url.contains("doi.org")
    {
        return 0.9;
    }

    if url.contains("biorxiv.org") || url.contains("medrxiv.org") || url.contains("arxiv.org") {
        return 0.75;
    }

    0.5
}

fn source_quality(passages: &[ScoredEvidencePassage]) -> f64 {
    let mut sources: std::collections::HashMap<&str, (f64, f64)> = std::collections::HashMap::new();

    for passage in passages
        .iter()
        .filter(|passage| passage.relevance_score > 0.0)
    {
        let url = passage.passage.source_url.as_str();
        let quality = source_quality_score(url);

        let entry = sources
            .entry(url)
            .or_insert((passage.relevance_score, quality));

        if passage.relevance_score > entry.0 {
            entry.0 = passage.relevance_score;
        }
    }

    if sources.is_empty() {
        return 0.0;
    }

    let total_relevance: f64 = sources.values().map(|(relevance, _)| relevance).sum();

    if total_relevance == 0.0 {
        return 0.0;
    }

    sources
        .values()
        .map(|(relevance, quality)| relevance * quality)
        .sum::<f64>()
        / total_relevance
}

pub fn build_evidence_state(
    claim: &str,
    passages: &[ScoredEvidencePassage],
    retrieval_novelty: f64,
    iteration: usize,
) -> EvidenceState {
    let positive_passages: Vec<&str> = passages
        .iter()
        .filter(|passage| passage.relevance_score > 0.0)
        .map(|passage| passage.passage.text.as_str())
        .collect();

    let claim_coverage = overlap::claim_coverage(claim, positive_passages);

    let relevance_concentration = relevance_concentration(passages);

    let source_diversity = source_diversity(passages);
    let source_quality = source_quality(passages);

    EvidenceState {
        candidate_count: passages.len(),
        relevance_concentration,
        claim_coverage,

        ranker_agreement: f64::NAN,
        source_diversity,
        source_quality,

        support_strength: f64::NAN,
        contradiction_strength: f64::NAN,
        evidence_conflict: f64::NAN,

        retrieval_novelty,
        iteration,
    }
}
