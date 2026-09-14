use crate::tacer::types::EvidenceRoute;

/// Frozen TACER-A operating threshold used in the
/// SciFact adaptive-retrieval experiments.
pub const TACER_A_THRESHOLD: f64 = 0.55;

/// Route the initial compact retrieval state.
///
/// This function deliberately does not compute
/// P(sufficient). The probability must come from
/// the trained TACER-A model.
pub fn route_initial(probability_sufficient: f64) -> EvidenceRoute {
    if probability_sufficient >= TACER_A_THRESHOLD {
        EvidenceRoute::Compact
    } else {
        EvidenceRoute::Expanded
    }
}

/// Route after local expansion.
///
/// TACER-A V1 was trained on the compact depth-5 state,
/// so we do not pretend that its probability is calibrated
/// for the depth-20 state.
///
/// For now, the caller supplies whether expansion produced
/// a credible resolved evidence state. Later this can be
/// replaced by a learned multi-depth controller.
pub fn route_after_expansion(resolved: bool) -> EvidenceRoute {
    if resolved {
        EvidenceRoute::Expanded
    } else {
        EvidenceRoute::AcquireMore
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sufficient_initial_state_stays_compact() {
        assert_eq!(route_initial(0.80), EvidenceRoute::Compact);
    }

    #[test]
    fn insufficient_initial_state_expands() {
        assert_eq!(route_initial(0.20), EvidenceRoute::Expanded);
    }

    #[test]
    fn threshold_is_inclusive() {
        assert_eq!(route_initial(TACER_A_THRESHOLD), EvidenceRoute::Compact);
    }

    #[test]
    fn unresolved_expansion_acquires_more() {
        assert_eq!(route_after_expansion(false), EvidenceRoute::AcquireMore);
    }

    #[test]
    fn resolved_expansion_continues() {
        assert_eq!(route_after_expansion(true), EvidenceRoute::Expanded);
    }
}
