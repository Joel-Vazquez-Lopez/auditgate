use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct AlignmentSignals {
    /// Fraction of informative claim tokens found in evidence.
    pub anchor_coverage: f64,

    /// Fraction of claim tokens beginning with an uppercase character
    /// that also occur in the evidence.
    ///
    /// This is only a lightweight proxy for named/scientific entities.
    pub entity_coverage: f64,

    /// Whether the strongest distinctive claim anchor appears.
    pub primary_anchor_present: bool,

    /// Combined diagnostic score.
    ///
    /// IMPORTANT:
    /// This is not yet used for routing or final decisions.
    pub alignment_score: f64,
}

pub fn alignment_signals(
    claim: &str,
    evidence: &str,
) -> AlignmentSignals {
    let claim_tokens = tokenize(claim);
    let evidence_tokens = tokenize(evidence);

    let evidence_set: HashSet<&str> =
        evidence_tokens
            .iter()
            .map(String::as_str)
            .collect();

    // --------------------------------------------------
    // Informative claim anchors
    // --------------------------------------------------

    let informative: Vec<&String> = claim_tokens
        .iter()
        .filter(|token| !is_stopword(token))
        .collect();

    let anchor_matches = informative
        .iter()
        .filter(|token| {
            evidence_set.contains(token.as_str())
        })
        .count();

    let anchor_coverage =
        safe_ratio(anchor_matches, informative.len());

    // --------------------------------------------------
    // Lightweight entity / identifier anchors
    // --------------------------------------------------
    //
    // We preserve the original claim here because
    // lowercasing would destroy capitalization.
    // This is intentionally heuristic for V1.
    // --------------------------------------------------

    let entity_tokens = entity_like_tokens(claim);

    let entity_matches = entity_tokens
        .iter()
        .filter(|token| {
            evidence_set.contains(token.as_str())
        })
        .count();

    let entity_coverage =
        safe_ratio(entity_matches, entity_tokens.len());

    // --------------------------------------------------
    // Primary anchor
    // --------------------------------------------------
    //
    // For V1, use the least generic informative token.
    // We approximate this using:
    //   1. entity-like token first
    //   2. otherwise longest informative token
    //
    // Later this can become a learned or NLP-derived
    // proposition argument.
    // --------------------------------------------------

    let primary_anchor =
        choose_primary_anchor(
            &entity_tokens,
            &informative,
        );

    let primary_anchor_present =
        primary_anchor
            .as_ref()
            .map(|anchor| {
                evidence_set.contains(anchor.as_str())
            })
            .unwrap_or(false);

    // --------------------------------------------------
    // Diagnostic alignment score
    // --------------------------------------------------
    //
    // Do NOT treat these weights as calibrated.
    // We only want an interpretable diagnostic signal
    // before training/evaluating anything.
    // --------------------------------------------------

    let primary_score =
        if primary_anchor_present {
            1.0
        } else {
            0.0
        };

    let alignment_score =
        0.40 * anchor_coverage
        + 0.35 * entity_coverage
        + 0.25 * primary_score;

    AlignmentSignals {
        anchor_coverage,
        entity_coverage,
        primary_anchor_present,
        alignment_score,
    }
}

fn choose_primary_anchor(
    entity_tokens: &[String],
    informative: &[&String],
) -> Option<String> {
    if let Some(token) = entity_tokens
        .iter()
        .max_by_key(|token| token.len())
    {
        return Some(token.clone());
    }

    informative
        .iter()
        .max_by_key(|token| token.len())
        .map(|token| (*token).clone())
}

fn entity_like_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|token| !token.is_empty())
        .filter(|token| {
            let has_letter =
                token.chars().any(|c| c.is_alphabetic());

            let has_upper =
                token.chars().any(|c| c.is_uppercase());

            let has_digit =
                token.chars().any(|c| c.is_ascii_digit());

            has_letter && (has_upper || has_digit)
        })
        .map(|token| token.to_lowercase())
        .filter(|token| !is_stopword(token))
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|token| !token.is_empty())
        .map(String::from)
        .collect()
}

fn safe_ratio(
    numerator: usize,
    denominator: usize,
) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "a"
            | "an"
            | "the"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "being"
            | "with"
            | "of"
            | "in"
            | "on"
            | "for"
            | "to"
            | "and"
            | "or"
            | "as"
            | "by"
            | "from"
            | "that"
            | "this"
            | "these"
            | "those"
    )
}