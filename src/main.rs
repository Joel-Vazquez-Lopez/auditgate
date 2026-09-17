mod acquisition;
mod decision;
mod overlap;
mod tacer;

use acquisition::{
    SearchProvider, SearchRequest, SourceFetcher, TavilySearchProvider, extract_passages,
    rank_passages,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use tacer::types::RetrievalAction;
use tacer::{build_evidence_state, choose_action};

// --------------------------------------------------
// Verifier data structures
// --------------------------------------------------

#[derive(Serialize)]
struct VerificationInput<'a> {
    claim: &'a str,
    evidence: Vec<&'a str>,
}

#[derive(Deserialize, Debug)]
pub struct VerificationResult {
    pub label: String,
    pub confidence: f64,
}

#[derive(Deserialize, Debug)]
struct VerificationResponse {
    results: Vec<VerificationResult>,
}

// --------------------------------------------------
// Batch verifier
// --------------------------------------------------

fn verify(claim: &str, evidence: Vec<&str>) -> VerificationResponse {
    let input = VerificationInput { claim, evidence };

    let json = serde_json::to_string(&input).expect("Could not create verifier JSON");

    let mut child = Command::new("python")
        .arg("verifier/verify.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Could not start verifier");

    child
        .stdin
        .as_mut()
        .expect("Could not open verifier stdin")
        .write_all(json.as_bytes())
        .expect("Could not send input to verifier");

    let output = child.wait_with_output().expect("Verifier failed");

    if !output.status.success() {
        panic!(
            "Verifier exited with error:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).expect("Could not parse verifier result")
}

// --------------------------------------------------
// Acquisition failure logging
// --------------------------------------------------

fn log_acquisition_failure(url: &str, title: &str, error: &str, iteration: usize, action: &str) {
    std::fs::create_dir_all("results").expect("Could not create results directory");

    let failure = serde_json::json!({
        "url": url,
        "title": title,
        "error": error,
        "iteration": iteration,
        "action": action,
    });

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("results/acquisition_failures.jsonl")
        .expect("Could not open acquisition failure log");

    writeln!(file, "{}", failure).expect("Could not write acquisition failure");
}

// --------------------------------------------------
// CLI
// --------------------------------------------------

#[derive(Parser)]
#[command(name = "auditgate")]
#[command(about = "Evidence-grounded claim auditing")]
struct Args {
    /// Claim to audit
    claim: String,
}

// --------------------------------------------------
// Main AuditGate pipeline
// --------------------------------------------------

fn main() {
    let args = Args::parse();
    let claim = args.claim;

    // --------------------------------------------------
    // Online evidence discovery
    // --------------------------------------------------

    println!("\nONLINE EVIDENCE SEARCH");

    let search_provider =
        TavilySearchProvider::from_env().expect("Could not initialize search provider");

    let search_request = SearchRequest {
        query: claim.clone(),
        limit: 5,
    };

    let search_results = search_provider
        .search(&search_request)
        .expect("Online evidence search failed");

    println!("Sources found: {}", search_results.len());

    // --------------------------------------------------
    // Fetch and extract real source evidence
    // --------------------------------------------------

    println!("\n================================");
    println!("ONLINE EVIDENCE ACQUISITION");
    println!("================================");

    let fetcher = SourceFetcher::new().expect("Could not initialize source fetcher");

    let mut online_passages = Vec::new();

    for result in &search_results {
        match fetcher.fetch(result) {
            Ok(document) => match extract_passages(&document) {
                Ok(passages) => {
                    online_passages.extend(passages);
                }

                Err(error) => {
                    println!("Extraction skipped: {}", error);
                }
            },

            Err(error) => {
                println!("Fetch failed: {}", error);

                log_acquisition_failure(&result.url, &result.title, &error, 0, "InitialSearch");
            }
        }
    }

    println!(
        "\nTotal online evidence passages: {}",
        online_passages.len()
    );

    println!("\n================================");
    println!("ONLINE EVIDENCE RANKING");
    println!("================================");

    let ranked_online_passages =
        rank_passages(&claim, &online_passages).expect("Could not rank online evidence");

    println!("Scored passages: {}", ranked_online_passages.len());

    const MAX_TACER_ITERATIONS: usize = 3;

    let mut final_online_passages = ranked_online_passages;
    let mut iteration = 0;
    let mut retrieval_novelty = 1.0;

    loop {
        println!("\nTACER — ITERATION {}", iteration);

        let semantic_evidence: Vec<&str> = final_online_passages
            .iter()
            .take(5)
            .map(|candidate| candidate.passage.text.as_str())
            .collect();

        let semantic_verification = verify(&claim, semantic_evidence);

        let support_strength = semantic_verification
            .results
            .iter()
            .filter(|result| result.label == "supported")
            .map(|result| result.confidence)
            .fold(0.0_f64, f64::max);

        let contradiction_strength = semantic_verification
            .results
            .iter()
            .filter(|result| result.label == "contradicted")
            .map(|result| result.confidence)
            .fold(0.0_f64, f64::max);

        let evidence_conflict = support_strength.min(contradiction_strength);

        let state = build_evidence_state(
            &claim,
            &final_online_passages,
            retrieval_novelty,
            support_strength,
            contradiction_strength,
            evidence_conflict,
            iteration,
        );

        println!("Candidates:              {}", state.candidate_count);

        println!(
            "Relevance concentration: {:.3}",
            state.relevance_concentration
        );

        println!("Claim coverage:           {:.3}", state.claim_coverage);

        println!("Source diversity:         {:.3}", state.source_diversity);
        println!("Source quality:           {:.3}", state.source_quality);
        println!("Support strength:         {:.3}", state.support_strength);
        println!(
            "Contradiction strength:   {:.3}",
            state.contradiction_strength
        );
        println!("Evidence conflict:        {:.3}", state.evidence_conflict);
        println!("Retrieval novelty:        {:.3}", state.retrieval_novelty);

        let action = choose_action(&state);

        println!("TACER action:             {:?}", action);

        if action == RetrievalAction::Abstain {
            println!("TACER stopping: further evidence acquisition has diminishing returns.");
            break;
        }

        if iteration >= MAX_TACER_ITERATIONS - 1 {
            println!("TACER stopping: maximum acquisition iterations reached.");
            break;
        }

        if action != RetrievalAction::DiversifySources
            && action != RetrievalAction::ReformulateQuery
            && action != RetrievalAction::SeekComplementaryEvidence
            && action != RetrievalAction::SeekOpposingEvidence
            && action != RetrievalAction::SeekHigherQualityEvidence
        {
            println!(
                "TACER stopping: action {:?} is not yet implemented.",
                action
            );
            break;
        }
        // --------------------------------------------------
        // Diversify sources
        // --------------------------------------------------

        match action {
            RetrievalAction::DiversifySources => {
                println!("\nDIVERSIFY SOURCES");
            }

            RetrievalAction::ReformulateQuery => {
                println!("\nREFORMULATE QUERY");
            }

            RetrievalAction::SeekComplementaryEvidence => {
                println!("\nSEEK COMPLEMENTARY EVIDENCE");
            }

            RetrievalAction::SeekOpposingEvidence => {
                println!("\nSEEK OPPOSING EVIDENCE");
            }

            RetrievalAction::SeekHigherQualityEvidence => {
                println!("\nSEEK HIGHER QUALITY EVIDENCE");
            }

            _ => unreachable!(),
        }

        let acquisition_query = match action {
            RetrievalAction::DiversifySources => {
                format!(
                    "{} review evidence alternative sources independent studies",
                    claim
                )
            }

            RetrievalAction::ReformulateQuery => {
                format!("{} facts evidence explanation", claim)
            }

            RetrievalAction::SeekComplementaryEvidence => {
                format!(
                    "{} mechanism details missing evidence specific relationship",
                    claim
                )
            }

            RetrievalAction::SeekOpposingEvidence => {
                if state.support_strength > state.contradiction_strength {
                    format!(
                        "{} contrary evidence contradiction criticism evidence against",
                        claim
                    )
                } else {
                    format!("{} supporting evidence confirmation evidence for", claim)
                }
            }
            RetrievalAction::SeekHigherQualityEvidence => {
                format!(
                    "{} primary source official documentation peer reviewed research authoritative evidence",
                    claim
                )
            }

            _ => unreachable!(),
        };

        println!("Acquisition query: {}", acquisition_query);

        let diversified_request = SearchRequest {
            query: acquisition_query,
            limit: 5,
        };

        let diversified_results = search_provider
            .search(&diversified_request)
            .expect("Diversified search failed");

        let existing_urls: std::collections::HashSet<String> = final_online_passages
            .iter()
            .map(|candidate| candidate.passage.source_url.clone())
            .collect();

        let new_results: Vec<_> = diversified_results
            .into_iter()
            .filter(|result| !existing_urls.contains(&result.url))
            .collect();

        println!("New diversified sources: {}", new_results.len());

        if new_results.is_empty() {
            println!("TACER stopping: diversification found no new sources.");
            break;
        }

        println!("\nDIVERSIFIED EVIDENCE ACQUISITION");

        let mut diversified_passages = Vec::new();

        for result in &new_results {
            match fetcher.fetch(result) {
                Ok(document) => match extract_passages(&document) {
                    Ok(passages) => {
                        diversified_passages.extend(passages);
                    }

                    Err(error) => {
                        println!("Extraction failed: {}", error);
                    }
                },

                Err(error) => {
                    println!("Fetch failed: {}", error);

                    log_acquisition_failure(
                        &result.url,
                        &result.title,
                        &error,
                        iteration + 1,
                        &format!("{:?}", action),
                    );
                }
            }
        }

        println!(
            "\nNew diversified evidence passages: {}",
            diversified_passages.len()
        );

        if diversified_passages.is_empty() {
            retrieval_novelty = 0.0;
            iteration += 1;

            println!("Retrieval novelty:       {:.3}", retrieval_novelty);
            continue;
        }

        let new_passage_count = diversified_passages.len();

        online_passages.extend(diversified_passages);

        retrieval_novelty = new_passage_count as f64 / online_passages.len() as f64;

        println!("Total evidence passages: {}", online_passages.len());

        println!("Retrieval novelty:       {:.3}", retrieval_novelty);

        iteration += 1;

        println!("\nEVIDENCE RERANKING — ITERATION {}", iteration);

        final_online_passages =
            rank_passages(&claim, &online_passages).expect("Could not rerank diversified evidence");
    }

    // --------------------------------------------------
    // Evidence currently passed to verifier
    // --------------------------------------------------
    //
    // IMPORTANT:
    // We deliberately preserve the existing behavior
    // for this experiment. TACER is not controlling
    // evidence selection yet.
    // --------------------------------------------------

    let evidence: Vec<&str> = final_online_passages
        .iter()
        .take(10)
        .map(|candidate| candidate.passage.text.as_str())
        .collect();

    // --------------------------------------------------
    // Batch verification
    // --------------------------------------------------

    let verification = verify(&claim, evidence);

    // --------------------------------------------------
    // Display evidence + verification
    // --------------------------------------------------

    println!("\nEVIDENCE VERIFICATION");

    for (index, (candidate, result)) in final_online_passages
        .iter()
        .take(10)
        .zip(verification.results.iter())
        .enumerate()
    {
        println!("\n--------------------------------");

        println!("{}. {}", index + 1, candidate.passage.source_title);

        println!("URL: {}", candidate.passage.source_url);

        println!("Relevance: {:.3}", candidate.relevance_score);

        println!("{}", candidate.passage.text);

        println!("→ {} ({:.3})", result.label, result.confidence);
    }

    // --------------------------------------------------
    // AuditGate decision
    // --------------------------------------------------

    let audit_decision = decision::decide(&verification.results);

    println!("\n================================");
    println!("AUDITGATE DECISION");
    println!("================================");

    println!("Decision: {}", audit_decision.decision);

    println!("Reason: {}", audit_decision.reason);
}
