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

#[derive(Debug, Clone)]
pub struct SemanticClaim {
    pub text: String,
    pub source_id: usize,
    pub kind: SemanticClaimKind,
}
#[derive(Debug, Clone, PartialEq)]
pub enum FaithfulnessDecision {
    Faithful,
    Unsafe,
}

#[derive(Debug, Clone)]
pub struct FaithfulnessResult {
    pub decision: FaithfulnessDecision,
    pub reason: String,
}

pub trait FaithfulnessGate {
    fn evaluate(
        &self,
        source: &SourceUnit,
        claim: &SemanticClaim,
    ) -> Result<FaithfulnessResult, String>;
}

#[derive(Debug, Clone)]
pub enum FaithfulnessOutcome {
    Accepted(SemanticClaim),
    SourceFallback(SourceUnit),
}

fn apply_faithfulness(
    source: &SourceUnit,
    claim: &SemanticClaim,
    result: &FaithfulnessResult,
) -> Result<FaithfulnessOutcome, String> {
        if source.id != claim.source_id {
        return Err(format!(
            "Faithfulness source mismatch: source_id {} != claim source_id {}",
            source.id,
            claim.source_id
        ));
    }

    match result.decision {
        FaithfulnessDecision::Faithful => {
            Ok(FaithfulnessOutcome::Accepted(claim.clone()))
        }
        FaithfulnessDecision::Unsafe => {
            Ok(FaithfulnessOutcome::SourceFallback(source.clone()))
        }
    }
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

#[derive(Debug, Clone)]
pub struct T5ClaimExtractor;

impl ClaimExtractor for T5ClaimExtractor {
    type Output = SemanticClaim;

    fn extract(
        &self,
        request: &ExtractionRequest,
    ) -> Result<Vec<Self::Output>, String> {
        let wire_request = ExtractionWireRequest {
            sources: request
                .sources
                .iter()
                .map(|source| ExtractionWireSource {
                    id: source.id,
                    text: source.text.as_str(),
                })
                .collect(),
        };

        let input = serde_json::to_string(&wire_request)
            .map_err(|error| format!("Failed to serialize extraction request: {}", error))?;

        let mut child = std::process::Command::new("python")
            .arg("extractor/extract.py")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|error| format!("Failed to start claim extractor: {}", error))?;

        {
            use std::io::Write;

            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| "Failed to open claim extractor stdin".to_string())?;

            stdin
                .write_all(input.as_bytes())
                .map_err(|error| format!("Failed to send extraction request: {}", error))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|error| format!("Claim extractor process failed: {}", error))?;

        if !output.status.success() {
            return Err(format!(
                "Claim extractor exited with status {}",
                output.status
            ));
        }

        let response: ExtractionWireResponse = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("Invalid claim extractor response: {}", error))?;

        validate_wire_response(request, response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
        struct FaithfulnessCase {
        name: &'static str,
        source: &'static str,
        candidate: &'static str,
        expected: FaithfulnessDecision,
    }
        #[test]
    fn faithful_candidate_is_accepted() {
        let source = SourceUnit {
            id: 7,
            text: "The Eiffel Tower is in Paris and was completed in 1889.".to_string(),
        };

        let claim = SemanticClaim {
            text: "The Eiffel Tower was completed in 1889.".to_string(),
            source_id: 7,
            kind: SemanticClaimKind::Verifiable,
        };

        let result = FaithfulnessResult {
            decision: FaithfulnessDecision::Faithful,
            reason: "Candidate preserves the source assertion.".to_string(),
        };

        let outcome = apply_faithfulness(&source, &claim, &result)
        .expect("Matching source and claim IDs should succeed");

        match outcome {
            FaithfulnessOutcome::Accepted(accepted) => {
                assert_eq!(
                    accepted.text,
                    "The Eiffel Tower was completed in 1889."
                );
                assert_eq!(accepted.source_id, 7);
            }
            FaithfulnessOutcome::SourceFallback(_) => {
                panic!("Faithful candidate should be accepted");
            }
        }
    }

    #[test]
    fn unsafe_candidate_falls_back_to_exact_source() {
        let source = SourceUnit {
            id: 9,
            text: "Is the Eiffel Tower in Paris?".to_string(),
        };

        let claim = SemanticClaim {
            text: "The Eiffel Tower is in Paris.".to_string(),
            source_id: 9,
            kind: SemanticClaimKind::Verifiable,
        };

        let result = FaithfulnessResult {
            decision: FaithfulnessDecision::Unsafe,
            reason: "Question was converted into an assertion.".to_string(),
        };

        let outcome = apply_faithfulness(&source, &claim, &result)
        .expect("Matching source and claim IDs should succeed");

        match outcome {
            FaithfulnessOutcome::SourceFallback(fallback) => {
                assert_eq!(fallback.id, 9);
                assert_eq!(fallback.text, "Is the Eiffel Tower in Paris?");
            }
            FaithfulnessOutcome::Accepted(_) => {
                panic!("Unsafe candidate should not be accepted");
            }
        }
    }
    
    #[test]
fn faithfulness_rejects_source_mismatch() {
    let source = SourceUnit {
        id: 1,
        text: "The Eiffel Tower is in Paris.".to_string(),
    };

    let claim = SemanticClaim {
        text: "The Eiffel Tower is in Paris.".to_string(),
        source_id: 2,
        kind: SemanticClaimKind::Verifiable,
    };

    let result = FaithfulnessResult {
        decision: FaithfulnessDecision::Faithful,
        reason: "Candidate appears faithful.".to_string(),
    };

    let error = apply_faithfulness(&source, &claim, &result)
        .expect_err("Mismatched source IDs must be rejected");

    assert!(error.contains("source mismatch"));
}


    fn faithfulness_cases() -> Vec<FaithfulnessCase> {
        vec![
            FaithfulnessCase {
                name: "unchanged factual claim",
                source: "The Eiffel Tower was completed in 1889.",
                candidate: "The Eiffel Tower was completed in 1889.",
                expected: FaithfulnessDecision::Faithful,
            },
            FaithfulnessCase {
                name: "valid atomic decomposition",
                source: "The Eiffel Tower is in Paris and was completed in 1889.",
                candidate: "The Eiffel Tower was completed in 1889.",
                expected: FaithfulnessDecision::Faithful,
            },
            FaithfulnessCase {
                name: "preserved hedge",
                source: "Researchers suggest that the treatment may reduce mortality.",
                candidate: "Researchers suggest that the treatment may reduce mortality.",
                expected: FaithfulnessDecision::Faithful,
            },
            FaithfulnessCase {
                name: "question converted to assertion",
                source: "Is the Eiffel Tower in Paris?",
                candidate: "The Eiffel Tower is in Paris.",
                expected: FaithfulnessDecision::Unsafe,
            },
            FaithfulnessCase {
                name: "conditional converted to assertion",
                source: "If temperatures continue to rise, sea levels could increase substantially by 2100.",
                candidate: "Temperatures continue to rise.",
                expected: FaithfulnessDecision::Unsafe,
            },
            FaithfulnessCase {
                name: "conditional dependency removed",
                source: "If temperatures continue to rise, sea levels could increase substantially by 2100.",
                candidate: "Sea levels could increase substantially by 2100.",
                expected: FaithfulnessDecision::Unsafe,
            },
            FaithfulnessCase {
                name: "causal relation removed",
                source: "The model achieved 92% accuracy because it learned robust representations.",
                candidate: "The model learned robust representations.",
                expected: FaithfulnessDecision::Unsafe,
            },
        ]
    }

    #[test]
    fn faithfulness_suite_is_defined() {
        let cases = faithfulness_cases();

        assert!(cases.iter().any(|case| {
            case.expected == FaithfulnessDecision::Faithful
        }));

        assert!(cases.iter().any(|case| {
            case.expected == FaithfulnessDecision::Unsafe
        }));

        for case in cases {
            assert!(!case.name.is_empty());
            assert!(!case.source.is_empty());
            assert!(!case.candidate.is_empty());
        }
    }
    #[test]
    fn t5_extractor_runs_end_to_end() {
        let extractor = T5ClaimExtractor;

        let request = ExtractionRequest {
            sources: vec![SourceUnit {
                id: 42,
                text: "The Eiffel Tower is in Paris and was completed in 1889.".to_string(),
            }],
        };

        let claims = extractor
            .extract(&request)
            .expect("T5 claim extraction should succeed");

        assert_eq!(claims.len(), 2);

        assert_eq!(claims[0].text, "The Eiffel Tower is in Paris.");
        assert_eq!(claims[0].source_id, 42);

        assert_eq!(
            claims[1].text,
            "The Eiffel Tower was completed in 1889."
        );
        assert_eq!(claims[1].source_id, 42);
    }

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
fn t5_extractor_regression_baseline() {
    let cases = regression_cases();
    let extractor = T5ClaimExtractor;

    let request = ExtractionRequest {
        sources: cases
            .iter()
            .enumerate()
            .map(|(id, case)| SourceUnit {
                id,
                text: case.input.to_string(),
            })
            .collect(),
    };

    let claims = extractor
        .extract(&request)
        .expect("T5 claim extraction should succeed");

    for (id, case) in cases.iter().enumerate() {
        let actual: Vec<&str> = claims
            .iter()
            .filter(|claim| claim.source_id == id)
            .map(|claim| claim.text.as_str())
            .collect();

        let expected: Vec<&str> = case
            .expected_verifiable
            .iter()
            .chain(case.expected_non_verifiable.iter())
            .copied()
            .collect();

        println!("\nCASE: {}", case.name);
        println!("SOURCE: {}", case.input);
        println!("EXPECTED: {:?}", expected);
        println!("ACTUAL:   {:?}", actual);
    }
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
