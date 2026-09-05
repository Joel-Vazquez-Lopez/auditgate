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