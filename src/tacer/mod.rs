pub mod coverage;
pub mod features;
pub mod router;
pub mod types;

pub use coverage::{
    coverage_signals,
    CoverageSignals,
};

pub use features::retrieval_signals;

pub use router::{
    route_after_expansion,
    route_initial,
    TACER_A_THRESHOLD,
};

pub use types::{
    EvidenceRoute,
    RetrievalSignals,
    TacerDecision,
};