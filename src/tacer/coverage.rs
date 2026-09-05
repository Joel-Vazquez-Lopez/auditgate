use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct CoverageSignals {
    pub exact_overlap: f64,
    pub rare_overlap: f64,
    pub phrase_overlap: f64,
    pub density: f64,
    pub position_bonus: f64,
    pub evidence_score: f64,
}

pub fn coverage_signals(
    claim: &str,
    passage: &str,
    position: usize,
) -> CoverageSignals {
    let claim_tokens = tokenize(claim);
    let passage_tokens = tokenize(passage);

    let claim_terms: HashSet<&String> =
        claim_tokens.iter().collect();

    let passage_terms: HashSet<&String> =
        passage_tokens.iter().collect();

    if claim_terms.is_empty() || passage_terms.is_empty() {
        return CoverageSignals {
            exact_overlap: 0.0,
            rare_overlap: 0.0,
            phrase_overlap: 0.0,
            density: 0.0,
            position_bonus: 0.0,
            evidence_score: 0.0,
        };
    }

    let overlap_terms: HashSet<_> = claim_terms
        .intersection(&passage_terms)
        .copied()
        .collect();

    // TACER exact query-term coverage.
    let exact_overlap =
        overlap_terms.len() as f64 / claim_terms.len() as f64;

    // Term counts inside the candidate passage.
    let mut term_counts: HashMap<&String, usize> =
        HashMap::new();

    for token in &passage_tokens {
        *term_counts.entry(token).or_insert(0) += 1;
    }

    // TACER rare-term overlap.
    let rare_sum: f64 = overlap_terms
        .iter()
        .map(|term| {
            let count = *term_counts.get(*term).unwrap_or(&1);
            1.0 / (count as f64).sqrt()
        })
        .sum();

    let rare_overlap =
        rare_sum / claim_terms.len() as f64;

    let phrase_overlap =
        phrase_overlap_score(&claim_tokens, &passage_tokens);

    let density =
        overlap_terms.len() as f64 / passage_terms.len() as f64;

    let position_bonus =
        1.0 / (1.0 + position as f64);

    // These are the weights from the original TACER
    // evidence_sentences() implementation.
    let evidence_score =
        (0.40 * exact_overlap)
        + (0.25 * rare_overlap)
        + (0.20 * phrase_overlap)
        + (0.10 * density)
        + (0.05 * position_bonus);

    CoverageSignals {
        exact_overlap,
        rare_overlap,
        phrase_overlap,
        density,
        position_bonus,
        evidence_score,
    }
}

fn phrase_overlap_score(
    claim_tokens: &[String],
    passage_tokens: &[String],
) -> f64 {
    let claim_bigrams = ngrams(claim_tokens, 2);
    let claim_trigrams = ngrams(claim_tokens, 3);

    if claim_bigrams.is_empty() && claim_trigrams.is_empty() {
        return 0.0;
    }

    let passage_bigrams = ngrams(passage_tokens, 2);
    let passage_trigrams = ngrams(passage_tokens, 3);

    let bigram_overlap = if claim_bigrams.is_empty() {
        0.0
    } else {
        claim_bigrams
            .intersection(&passage_bigrams)
            .count() as f64
            / claim_bigrams.len() as f64
    };

    let trigram_overlap = if claim_trigrams.is_empty() {
        0.0
    } else {
        claim_trigrams
            .intersection(&passage_trigrams)
            .count() as f64
            / claim_trigrams.len() as f64
    };

    (0.40 * bigram_overlap)
        + (0.60 * trigram_overlap)
}

fn ngrams(
    tokens: &[String],
    size: usize,
) -> HashSet<Vec<String>> {
    if tokens.len() < size {
        return HashSet::new();
    }

    tokens
        .windows(size)
        .map(|window| window.to_vec())
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(String::from)
        .collect()
}