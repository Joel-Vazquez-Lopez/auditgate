#[derive(Debug, Clone)]
pub struct RetrievalSignals {
    pub top_score: f64,
    pub mean_score: f64,
    pub score_std: f64,
    pub gap_1_2: f64,
    pub gap_3_4: f64,
    pub gap_5_6: f64,
    pub gap_1_5: f64,
    pub top1_mass: f64,
    pub top3_mass: f64,
    pub top5_mass: f64,
    pub top1_to_top5: f64,
    pub top3_to_top8: f64,
    pub entropy: f64,
    pub normalized_entropy: f64,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceRoute {
    Compact,
    Expanded,
    AcquireMore,
}

#[derive(Debug, Clone)]
pub struct TacerDecision {
    pub route: EvidenceRoute,
    pub signals: RetrievalSignals,
    pub reason: String,
}

// ==================================================
// TACER V2 — general adaptive retrieval controller
// ==================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetrievalAction {
    /// Current evidence is good enough to continue
    /// to verification / decision.
    UseCurrent,

    /// Keep the same evidence pool but improve
    /// selection or ranking.
    Refine,

    /// Retrieve more evidence using a broader search.
    BroadenSearch,

    /// Generate a better search query from the claim
    /// and current evidence gaps.
    ReformulateQuery,

    /// Search for evidence from different sources
    /// or domains.
    DiversifySources,

    /// Search specifically for evidence covering
    /// a missing part of the claim.
    SeekComplementaryEvidence,

    SeekOpposingEvidence,

    /// Search specifically for stronger-quality evidence
    /// when current evidence is relevant but has weak provenance.
    SeekHigherQualityEvidence,

    /// Evidence acquisition has failed or remains
    /// too uncertain to justify a commitment.
    Abstain,
}

#[derive(Debug, Clone)]
pub struct EvidenceState {
    /// Number of candidate passages currently available.
    pub candidate_count: usize,

    /// How strongly evidence is concentrated around
    /// the highest-ranked candidates.
    pub relevance_concentration: f64,

    /// Estimated coverage of the claim by retrieved evidence.
    pub claim_coverage: f64,

    /// Agreement between independent retrieval/ranking signals.
    pub ranker_agreement: f64,

    /// Diversity of independent sources represented.
    pub source_diversity: f64,

    /// Quality / trust estimate of the available sources.
    pub source_quality: f64,

    /// Strength of supporting evidence.
    pub support_strength: f64,

    /// Strength of contradicting evidence.
    pub contradiction_strength: f64,

    /// Degree of disagreement or conflict in evidence.
    pub evidence_conflict: f64,

    /// How much genuinely new information recent retrieval
    /// added compared with previous iterations.
    pub retrieval_novelty: f64,

    /// Number of retrieval iterations already attempted.
    pub iteration: usize,
}
