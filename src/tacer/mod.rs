pub mod controller;
pub mod coverage;
pub mod evidence_state;
pub mod features;
pub mod router;
pub mod types;

pub use coverage::{CoverageSignals, coverage_signals};

pub use features::retrieval_signals;

pub use router::{TACER_A_THRESHOLD, route_after_expansion, route_initial};

pub use types::{EvidenceRoute, RetrievalSignals, TacerDecision};

pub use controller::choose_action;

pub use evidence_state::build_evidence_state;
