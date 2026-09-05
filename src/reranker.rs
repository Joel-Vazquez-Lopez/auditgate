use crate::bm25::EvidenceCandidate;

pub struct RerankedEvidence<'a> {
    pub candidate: &'a EvidenceCandidate<'a>,
    pub relevance_score: f64,
}

pub fn rerank<'a>(
    claim: &str,
    candidates: &'a [EvidenceCandidate<'a>],
) -> Vec<RerankedEvidence<'a>> {
    let claim_terms = important_terms(claim);

    let mut results: Vec<RerankedEvidence<'a>> = candidates
        .iter()
        .map(|candidate| {
            let passage_terms =
                important_terms(&candidate.passage.text);

            let matched = claim_terms
                .iter()
                .filter(|term| passage_terms.contains(term))
                .count();

            let relevance_score =
                if claim_terms.is_empty() {
                    0.0
                } else {
                    matched as f64 / claim_terms.len() as f64
                };

            RerankedEvidence {
                candidate,
                relevance_score,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.relevance_score
            .total_cmp(&a.relevance_score)
            .then_with(|| {
                b.candidate
                    .document_score
                    .total_cmp(&a.candidate.document_score)
            })
    });

    results
}

fn important_terms(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() > 2)
        .filter(|word| {
            !matches!(
                *word,
                "the"
                    | "and"
                    | "are"
                    | "was"
                    | "were"
                    | "with"
                    | "for"
                    | "from"
                    | "that"
                    | "this"
                    | "has"
                    | "have"
                    | "had"
                    | "into"
                    | "its"
            )
        })
        .map(String::from)
        .collect()
}