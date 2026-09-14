use crate::tacer::types::{EvidenceState, RetrievalAction};

pub fn choose_action(state: &EvidenceState) -> RetrievalAction {
    // No evidence at all.
    if state.candidate_count == 0 {
        return RetrievalAction::BroadenSearch;
    }

    // Strong unresolved contradiction between evidence sources.
    if state.support_strength >= 0.80 && state.contradiction_strength >= 0.80 {
        return RetrievalAction::DiversifySources;
    }

    // Retrieved evidence is relevant but does not cover
    // enough of the claim.
    if state.relevance_concentration >= 0.60 && state.claim_coverage < 0.50 {
        return RetrievalAction::SeekComplementaryEvidence;
    }

    // Retrieval appears broadly poor.
    if state.relevance_concentration < 0.30 {
        return RetrievalAction::ReformulateQuery;
    }

    // Evidence is useful but ranking is unstable.
    if state.ranker_agreement < 0.40 {
        return RetrievalAction::Refine;
    }

    // Evidence comes from too narrow a source pool.
    if state.source_diversity < 0.30 {
        return RetrievalAction::DiversifySources;
    }

    // We have repeatedly searched without finding
    // much genuinely new information.
    if state.iteration >= 3 && state.retrieval_novelty < 0.15 {
        return RetrievalAction::Abstain;
    }

    RetrievalAction::UseCurrent
}
