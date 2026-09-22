use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct SourceUnit {
    pub id: usize,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ExtractionWireSource<'a> {
    pub id: usize,
    pub text: &'a str,
}

#[derive(Debug, Serialize)]
pub struct ExtractionWireRequest<'a> {
    pub sources: Vec<ExtractionWireSource<'a>>,
}

#[derive(Debug, Deserialize)]
pub struct ExtractionWireClaim {
    pub text: String,
    pub source_id: usize,
    pub kind: String,
}

#[derive(Debug, Deserialize)]
pub struct ExtractionWireResponse {
    pub claims: Vec<ExtractionWireClaim>,
}

#[derive(Debug, Clone)]
pub struct ExtractionRequest {
    pub sources: Vec<SourceUnit>,
}

pub trait ClaimExtractor {
    type Output;

    fn extract(
        &self,
        request: &ExtractionRequest,
    ) -> Result<Vec<Self::Output>, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticClaimKind {
    Verifiable,
    NonVerifiable,
}

/// Semantic claim extraction contract.
///
/// A semantic extractor should:
///
/// 1. ATOMICITY
///    Each verifiable output represents one independently checkable proposition.
///    Compound assertions should be decomposed when their components can be
///    independently true or false.
///
/// 2. FAITHFULNESS
///    Extracted claims must preserve the meaning of the source text.
///    The extractor must not introduce facts, entities, causal relations,
///    certainty, or specificity that the source did not assert.
///
/// 3. VERIFIABILITY
///    Statements that can in principle be supported or contradicted by
///    external evidence are Verifiable.
///
/// 4. NON-VERIFIABLE CONTENT
///    Pure opinions, preferences, rhetorical statements, commands, questions,
///    and other content without an externally checkable proposition are
///    NonVerifiable.
///
/// 5. EPISTEMIC QUALIFICATION
///    Hedging, attribution, uncertainty, and modality are part of the claim
///    when they materially affect its meaning. They must not be silently
///    strengthened into unconditional factual assertions.
///
/// 6. PROVENANCE
///    Every output identifies the source unit from which it was derived.
///    Source text itself is owned by AuditGate and must not be generated
///    or rewritten by the semantic extractor.
///    Multiple atomic claims may share the same source unit.
/// 7. NO FACT CHECKING DURING EXTRACTION
///    Extraction identifies what the text asserts. It does not decide whether
///    the assertion is true. Truth assessment belongs to AuditGate downstream.

pub struct SemanticClaim {
    pub text: String,
    pub source_id: usize,
    pub kind: SemanticClaimKind,
}
fn validate_wire_response(
    request: &ExtractionRequest,
    response: ExtractionWireResponse,
) -> Result<Vec<SemanticClaim>, String> {
    let valid_source_ids: std::collections::HashSet<usize> = request
        .sources
        .iter()
        .map(|source| source.id)
        .collect();

    let mut claims = Vec::new();

    for claim in response.claims {
        let text = claim.text.trim();

        if text.is_empty() {
            return Err("Extractor returned an empty claim".to_string());
        }

        if !valid_source_ids.contains(&claim.source_id) {
            return Err(format!(
                "Extractor returned unknown source_id: {}",
                claim.source_id
            ));
        }

        let kind = match claim.kind.as_str() {
            "verifiable" => SemanticClaimKind::Verifiable,
            "non_verifiable" => SemanticClaimKind::NonVerifiable,
            other => {
                return Err(format!(
                    "Extractor returned unknown claim kind: {}",
                    other
                ));
            }
        };

        claims.push(SemanticClaim {
            text: text.to_string(),
            source_id: claim.source_id,
            kind,
        });
    }

    Ok(claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct ExtractionCase {
        name: &'static str,
        input: &'static str,
        expected_verifiable: &'static [&'static str],
        expected_non_verifiable: &'static [&'static str],
    }

    fn regression_cases() -> Vec<ExtractionCase> {
        vec![
            ExtractionCase {
                name: "simple factual claim",
                input: "The Eiffel Tower was completed in 1889.",
                expected_verifiable: &[
                    "The Eiffel Tower was completed in 1889.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "compound factual claim",
                input: "The Eiffel Tower is in Paris and was completed in 1889.",
                expected_verifiable: &[
                    "The Eiffel Tower is in Paris.",
                    "The Eiffel Tower was completed in 1889.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "pure opinion",
                input: "I think the Eiffel Tower is beautiful.",
                expected_verifiable: &[],
                expected_non_verifiable: &[
                    "I think the Eiffel Tower is beautiful.",
                ],
            },
            ExtractionCase {
                name: "mixed fact and opinion",
                input: "The Eiffel Tower is in Paris, and I think it is beautiful.",
                expected_verifiable: &[
                    "The Eiffel Tower is in Paris.",
                ],
                expected_non_verifiable: &[
                    "I think the Eiffel Tower is beautiful.",
                ],
            },
            ExtractionCase {
                name: "numerical claim",
                input: "The experiment included 240 participants.",
                expected_verifiable: &[
                    "The experiment included 240 participants.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "false but verifiable claim",
                input: "The Moon is made primarily of cheese.",
                expected_verifiable: &[
                    "The Moon is made primarily of cheese.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "hedged claim",
                input: "Researchers suggest that the treatment may reduce mortality.",
                expected_verifiable: &[
                    "Researchers suggest that the treatment may reduce mortality.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "causal claim",
                input: "Smoking increases the risk of lung cancer.",
                expected_verifiable: &[
                    "Smoking increases the risk of lung cancer.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "question",
                input: "Is the Eiffel Tower in Paris?",
                expected_verifiable: &[],
                expected_non_verifiable: &[
                    "Is the Eiffel Tower in Paris?",
                ],
            },
            ExtractionCase {
                name: "command",
                input: "Visit the Eiffel Tower when you go to Paris.",
                expected_verifiable: &[],
                expected_non_verifiable: &[
                    "Visit the Eiffel Tower when you go to Paris.",
                ],
            },
            ExtractionCase {
                name: "attributed claim",
                input: "The WHO reports that global life expectancy increased between 2000 and 2019.",
                expected_verifiable: &[
                    "The WHO reports that global life expectancy increased between 2000 and 2019.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "uncertain prediction",
                input: "The new policy could reduce emissions by 2030.",
                expected_verifiable: &[
                    "The new policy could reduce emissions by 2030.",
                ],
                expected_non_verifiable: &[],
            },
            ExtractionCase {
                name: "fact followed by subjective judgement",
                input: "The study included 500 participants, and the design was excellent.",
                expected_verifiable: &[
                    "The study included 500 participants.",
                ],
                expected_non_verifiable: &[
                    "The design was excellent.",
                ],
            },
            ExtractionCase {
                name: "two independently falsifiable claims",
                input: "The drug reduced blood pressure but increased heart rate.",
                expected_verifiable: &[
                    "The drug reduced blood pressure.",
                    "The drug increased heart rate.",
                ],
                expected_non_verifiable: &[],
            },
        ]
    }

    #[test]
    fn regression_suite_is_defined() {
        let cases = regression_cases();

        assert!(cases.len() >= 10);

        for case in cases {
            assert!(!case.name.is_empty());
            assert!(!case.input.is_empty());

            assert!(
                !case.expected_verifiable.is_empty()
                    || !case.expected_non_verifiable.is_empty()
            );
        }
    }
    #[test]
fn rejects_unknown_source_id() {
    let request = ExtractionRequest {
        sources: vec![SourceUnit {
            id: 0,
            text: "The Eiffel Tower is in Paris.".to_string(),
        }],
    };

    let response = ExtractionWireResponse {
        claims: vec![ExtractionWireClaim {
            text: "The Eiffel Tower is in Paris.".to_string(),
            source_id: 99,
            kind: "verifiable".to_string(),
        }],
    };

    let result = validate_wire_response(&request, response);

    assert!(result.is_err());
}

#[test]
fn rejects_unknown_claim_kind() {
    let request = ExtractionRequest {
        sources: vec![SourceUnit {
            id: 0,
            text: "The Eiffel Tower is in Paris.".to_string(),
        }],
    };

    let response = ExtractionWireResponse {
        claims: vec![ExtractionWireClaim {
            text: "The Eiffel Tower is in Paris.".to_string(),
            source_id: 0,
            kind: "probably_true".to_string(),
        }],
    };

    let result = validate_wire_response(&request, response);

    assert!(result.is_err());
}

#[test]
fn rejects_empty_claim() {
    let request = ExtractionRequest {
        sources: vec![SourceUnit {
            id: 0,
            text: "The Eiffel Tower is in Paris.".to_string(),
        }],
    };

    let response = ExtractionWireResponse {
        claims: vec![ExtractionWireClaim {
            text: "   ".to_string(),
            source_id: 0,
            kind: "verifiable".to_string(),
        }],
    };

    let result = validate_wire_response(&request, response);

    assert!(result.is_err());
}
#[test]
fn accepts_valid_semantic_claims() {
    let request = ExtractionRequest {
        sources: vec![SourceUnit {
            id: 0,
            text: "The drug reduced blood pressure but increased heart rate.".to_string(),
        }],
    };

    let response = ExtractionWireResponse {
        claims: vec![
            ExtractionWireClaim {
                text: "The drug reduced blood pressure.".to_string(),
                source_id: 0,
                kind: "verifiable".to_string(),
            },
            ExtractionWireClaim {
                text: "The drug increased heart rate.".to_string(),
                source_id: 0,
                kind: "verifiable".to_string(),
            },
        ],
    };

    let claims = validate_wire_response(&request, response)
        .expect("Valid extraction should be accepted");

    assert_eq!(claims.len(), 2);

    assert_eq!(claims[0].text, "The drug reduced blood pressure.");
    assert_eq!(claims[0].source_id, 0);
    assert_eq!(claims[0].kind, SemanticClaimKind::Verifiable);

    assert_eq!(claims[1].text, "The drug increased heart rate.");
    assert_eq!(claims[1].source_id, 0);
    assert_eq!(claims[1].kind, SemanticClaimKind::Verifiable);
}


}