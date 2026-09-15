mod acquisition;
mod alignment;
mod bm25;
mod decision;
mod overlap;
mod reranker;
mod tacer;

use acquisition::{
    SearchProvider, SearchRequest, SourceFetcher, TavilySearchProvider, extract_passages,
    rank_passages, relevance_concentration, source_diversity,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use tacer::types::RetrievalAction;
use tacer::{build_evidence_state, choose_action};

#[derive(Deserialize)]
struct Passage {
    text: String,
}

#[derive(Deserialize)]
struct Document {
    document_id: String,
    title: String,
    passages: Vec<Passage>,
}

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
// TACER-A runtime data structures
// --------------------------------------------------

#[derive(Serialize)]
struct TacerCandidateInput<'a> {
    document_id: &'a str,
    passage_id: String,
    text: &'a str,
}

#[derive(Serialize)]
struct TacerInput<'a> {
    claim: &'a str,
    retrieved_document_scores: Vec<f64>,
    candidates: Vec<TacerCandidateInput<'a>>,
}

#[derive(Deserialize, Debug)]
struct TacerScoredCandidate {
    document_id: String,
    passage_id: String,
    msmarco_score: f64,
    alignment_score: f64,
}

#[derive(Deserialize, Debug)]
struct TacerInferenceResponse {
    probability_sufficient: f64,
    candidates: Vec<TacerScoredCandidate>,
}

// --------------------------------------------------
// Load SciFact documents
// --------------------------------------------------

fn load_documents() -> Vec<Document> {
    let file =
        File::open("data/normalized/scifact/documents.jsonl").expect("Could not open documents");

    BufReader::new(file)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).expect("Invalid document"))
        .collect()
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

fn infer_tacer<'a>(
    claim: &'a str,
    documents: &[(&'a Document, f64)],
    candidates: &[bm25::EvidenceCandidate<'a>],
) -> TacerInferenceResponse {
    let input = TacerInput {
        claim,

        retrieved_document_scores: documents.iter().map(|(_, score)| *score).collect(),

        candidates: candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| TacerCandidateInput {
                document_id: candidate.document.document_id.as_str(),

                passage_id: format!("runtime:{index}"),

                text: candidate.passage.text.as_str(),
            })
            .collect(),
    };

    let json = serde_json::to_string(&input).expect("Could not create TACER JSON");

    let mut child = Command::new("python")
        .arg("src/tacer/infer.py")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Could not start TACER inference");

    child
        .stdin
        .as_mut()
        .expect("Could not open TACER stdin")
        .write_all(json.as_bytes())
        .expect("Could not send TACER input");

    let output = child.wait_with_output().expect("TACER inference failed");

    if !output.status.success() {
        panic!(
            "TACER inference exited with error:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).expect("Could not parse TACER result")
}

fn select_top_alignment<'a>(
    candidates: Vec<bm25::EvidenceCandidate<'a>>,
    inference: &TacerInferenceResponse,
    limit: usize,
) -> Vec<bm25::EvidenceCandidate<'a>> {
    let mut ranked: Vec<_> = candidates
        .into_iter()
        .zip(inference.candidates.iter())
        .map(|(candidate, scored)| (candidate, scored.alignment_score))
        .collect();

    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    ranked.truncate(limit);

    ranked.into_iter().map(|(candidate, _)| candidate).collect()
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
    let documents = load_documents();

    println!("Loaded {} documents.", documents.len());

    let bm25 = bm25::BM25::new(documents);

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

    println!("\nTACER — ITERATION 0");

    let online_state = build_evidence_state(&claim, &ranked_online_passages, 0);

    println!("Candidates:              {}", online_state.candidate_count);

    println!(
        "Relevance concentration: {:.3}",
        online_state.relevance_concentration
    );

    println!(
        "Claim coverage:           {:.3}",
        online_state.claim_coverage
    );

    println!(
        "Source diversity:         {:.3}",
        online_state.source_diversity
    );

    let online_action = choose_action(&online_state);

    println!("TACER action:             {:?}", online_action);

    if online_action == RetrievalAction::DiversifySources {
        println!("\nDIVERSIFY SOURCES");

        let diversified_query = format!(
            "{} review evidence alternative sources independent studies",
            claim
        );

        println!("Diversified query: {}", diversified_query);

        let diversified_request = SearchRequest {
            query: diversified_query,
            limit: 5,
        };

        let diversified_results = search_provider
            .search(&diversified_request)
            .expect("Diversified search failed");

        let existing_urls: std::collections::HashSet<&str> = search_results
            .iter()
            .map(|result| result.url.as_str())
            .collect();

        let new_results: Vec<_> = diversified_results
            .into_iter()
            .filter(|result| !existing_urls.contains(result.url.as_str()))
            .collect();

        println!("New diversified sources: {}", new_results.len());

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
                }
            }
        }

        println!(
            "\nNew diversified evidence passages: {}",
            diversified_passages.len()
        );

        online_passages.extend(diversified_passages);

        println!(
            "Total evidence passages after diversification: {}",
            online_passages.len()
        );

        println!("\nEVIDENCE RERANKING — ITERATION 1");

        let reranked_after_diversification =
            rank_passages(&claim, &online_passages).expect("Could not rerank diversified evidence");

        let state_after_diversification =
            build_evidence_state(&claim, &reranked_after_diversification, 1);

        println!("\nTACER — ITERATION 1");

        println!(
            "Candidates:              {}",
            state_after_diversification.candidate_count
        );

        println!(
            "Relevance concentration: {:.3}",
            state_after_diversification.relevance_concentration
        );

        println!(
            "Claim coverage:           {:.3}",
            state_after_diversification.claim_coverage
        );

        println!(
            "Source diversity:         {:.3}",
            state_after_diversification.source_diversity
        );

        let action_after_diversification = choose_action(&state_after_diversification);

        println!(
            "TACER action:             {:?}",
            action_after_diversification
        );
    }

    // --------------------------------------------------
    // Document retrieval
    // --------------------------------------------------

    let documents = bm25.search(&claim, 5);

    println!("\nRETRIEVED DOCUMENTS");

    for (rank, (document, score)) in documents.iter().enumerate() {
        println!(
            "{}. {} | {:.3} | {}",
            rank + 1,
            document.document_id,
            score,
            document.title
        );
    }

    // --------------------------------------------------
    // Learned TACER-A inference
    // --------------------------------------------------

    let tacer_candidates = bm25.all_passages(&documents);

    let tacer_inference = infer_tacer(&claim, &documents, &tacer_candidates);

    println!("\n================================");
    println!("TACER-A");
    println!("================================");

    println!(
        "P(sufficient): {:.6}",
        tacer_inference.probability_sufficient
    );

    let tacer_route = tacer::route_initial(tacer_inference.probability_sufficient);

    println!("Route: {:?}", tacer_route);

    println!("Candidates evaluated: {}", tacer_inference.candidates.len());

    let final_passages = if tacer_route == tacer::EvidenceRoute::Expanded {
        println!("\n================================");
        println!("TACER-A EXPANSION");
        println!("================================");

        let expanded_documents = bm25.search(&claim, 20);

        println!("Expanded retrieval depth: {}", expanded_documents.len());

        let expanded_candidates = bm25.all_passages(&expanded_documents);

        let expanded_tacer_inference =
            infer_tacer(&claim, &expanded_documents, &expanded_candidates);

        println!(
            "Expanded P(sufficient): {:.6}",
            expanded_tacer_inference.probability_sufficient
        );

        println!(
            "Expanded candidates evaluated: {}",
            expanded_tacer_inference.candidates.len()
        );

        select_top_alignment(expanded_candidates, &expanded_tacer_inference, 5)
    } else {
        select_top_alignment(tacer_candidates, &tacer_inference, 5)
    };

    // --------------------------------------------------
    // Semantic reranking
    // --------------------------------------------------

    let reranked = reranker::rerank(&claim, &final_passages);

    println!("\nRERANKED EVIDENCE");

    for (rank, item) in reranked.iter().enumerate() {
        println!("\n--------------------------------");

        println!("{}. relevance={:.3}", rank + 1, item.relevance_score);

        println!("Document: {}", item.candidate.document.title);

        println!("{}", item.candidate.passage.text);
    }

    // --------------------------------------------------
    // TACER diagnostics
    // --------------------------------------------------
    //
    // TACER is diagnostic only at this stage.
    // It observes the reranked candidate evidence but
    // does NOT yet alter verification or AuditGate's
    // final decision.
    // --------------------------------------------------

    let retrieval_scores: Vec<f64> = reranked.iter().map(|item| item.relevance_score).collect();

    let retrieval_state = tacer::retrieval_signals(&retrieval_scores);

    println!("\n================================");
    println!("TACER EVIDENCE STATE");
    println!("================================");

    println!("\nRETRIEVAL CONCENTRATION");

    println!("Top score:          {:.3}", retrieval_state.top_score);

    println!("Mean score:         {:.3}", retrieval_state.mean_score);

    println!("Score std:          {:.3}", retrieval_state.score_std);

    println!("Gap 1→2:            {:.3}", retrieval_state.gap_1_2);

    println!("Gap 1→5:            {:.3}", retrieval_state.gap_1_5);

    println!("Top-1 mass:         {:.3}", retrieval_state.top1_mass);

    println!("Top-3 mass:         {:.3}", retrieval_state.top3_mass);

    println!("Top-5 mass:         {:.3}", retrieval_state.top5_mass);

    println!("Top1 / Top5:        {:.3}", retrieval_state.top1_to_top5);

    println!("Top3 / Top8 mass:   {:.3}", retrieval_state.top3_to_top8);

    println!("Entropy:            {:.3}", retrieval_state.entropy);

    println!(
        "Normalized entropy: {:.3}",
        retrieval_state.normalized_entropy
    );

    // --------------------------------------------------
    // TACER evidence coverage
    // --------------------------------------------------

    println!("\nEVIDENCE COVERAGE");

    for (index, item) in reranked.iter().enumerate() {
        let coverage = tacer::coverage_signals(&claim, &item.candidate.passage.text, index);
        let alignment = alignment::alignment_signals(&claim, &item.candidate.passage.text);

        println!("\n--------------------------------");

        println!("{}. {}", index + 1, item.candidate.document.document_id);

        println!("Document: {}", item.candidate.document.title);

        println!("Exact overlap:  {:.3}", coverage.exact_overlap);

        println!("Rare overlap:   {:.3}", coverage.rare_overlap);

        println!("Phrase overlap: {:.3}", coverage.phrase_overlap);

        println!("Density:        {:.3}", coverage.density);

        println!("Position bonus: {:.3}", coverage.position_bonus);

        println!("TACER score:    {:.3}", coverage.evidence_score);
        println!("Anchor coverage: {:.3}", alignment.anchor_coverage);

        println!("Entity coverage: {:.3}", alignment.entity_coverage);

        println!(
            "Primary anchor:  {}",
            if alignment.primary_anchor_present {
                "present"
            } else {
                "MISSING"
            }
        );

        println!("Alignment score: {:.3}", alignment.alignment_score);
        println!("{}", item.candidate.passage.text);
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

    let evidence: Vec<&str> = final_passages
        .iter()
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

    for (index, (candidate, result)) in final_passages
        .iter()
        .zip(verification.results.iter())
        .enumerate()
    {
        println!("\n--------------------------------");

        println!("{}. {}", index + 1, candidate.document.document_id);

        println!("Document: {}", candidate.document.title);

        println!(
            "Scores: document={:.3} passage={:.3} combined={:.3}",
            candidate.document_score, candidate.passage_score, candidate.combined_score
        );

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
