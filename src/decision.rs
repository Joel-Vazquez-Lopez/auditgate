use crate::VerificationResult;

pub struct AuditDecision {
    pub decision: String,
    pub reason: String,
}

pub fn decide(
    results: &[VerificationResult],
) -> AuditDecision {
    let threshold = 0.90;

    let strong_support = results.iter().any(|result| {
        result.label == "supported"
            && result.confidence >= threshold
    });

    let strong_contradiction = results.iter().any(|result| {
        result.label == "contradicted"
            && result.confidence >= threshold
    });

    match (strong_support, strong_contradiction) {
        (true, false) => AuditDecision {
            decision: "ALLOW".to_string(),
            reason:
                "Retrieved evidence strongly supports the claim."
                    .to_string(),
        },

        (false, true) => AuditDecision {
            decision: "BLOCK".to_string(),
            reason:
                "Retrieved evidence strongly contradicts the claim."
                    .to_string(),
        },

        (true, true) => AuditDecision {
            decision: "REVIEW".to_string(),
            reason:
                "Retrieved evidence contains conflicting strong signals."
                    .to_string(),
        },

        (false, false) => AuditDecision {
            decision: "REVIEW".to_string(),
            reason:
                "Retrieved evidence does not establish strong support or contradiction."
                    .to_string(),
        },
    }
}