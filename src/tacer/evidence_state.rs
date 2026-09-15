use crate::acquisition::{ScoredEvidencePassage, relevance_concentration, source_diversity};
use crate::overlap;
use crate::tacer::types::EvidenceState;

pub fn build_evidence_state(
    claim: &str,
    passages: &[ScoredEvidencePassage],
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

    EvidenceState {
        candidate_count: passages.len(),
        relevance_concentration,
        claim_coverage,

        ranker_agreement: f64::NAN,
        source_diversity,
        source_quality: f64::NAN,

        support_strength: f64::NAN,
        contradiction_strength: f64::NAN,
        evidence_conflict: f64::NAN,

        retrieval_novelty: f64::NAN,
        iteration,
    }
}
