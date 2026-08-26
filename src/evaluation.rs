use crate::{bm25::BM25, overlap, Document};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Deserialize)]
struct Evidence {
    document_id: String,
}

#[derive(Deserialize)]
struct Claim {
    text: String,
    gold_label: String,
    gold_evidence: Vec<Evidence>,
}

fn load_claims() -> Vec<Claim> {
    let file = File::open("data/eval/auditgate_eval.jsonl")
        .expect("Could not open evaluation data");

    BufReader::new(file)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .filter(|claim: &Claim| claim.gold_label != "insufficient_evidence")
        .collect()
}

fn is_hit(results: &[&str], gold: &[&str], k: usize) -> bool {
    results
        .iter()
        .take(k)
        .any(|id| gold.contains(id))
}

fn print_results(name: &str, r1: usize, r3: usize, r5: usize, total: usize) {
    println!("\n{name}");
    println!("--------------------");
    println!("Claims: {total}");
    println!("Recall@1: {:.3}", r1 as f64 / total as f64);
    println!("Recall@3: {:.3}", r3 as f64 / total as f64);
    println!("Recall@5: {:.3}", r5 as f64 / total as f64);
}

pub fn evaluate_overlap(documents: &[Document]) {
    let claims = load_claims();

    let mut r1 = 0;
    let mut r3 = 0;
    let mut r5 = 0;

    for claim in &claims {
        let results = overlap::search(&claim.text, documents, 5);

        let retrieved: Vec<&str> = results
            .iter()
            .map(|(doc, _)| doc.document_id.as_str())
            .collect();

        let gold: Vec<&str> = claim
            .gold_evidence
            .iter()
            .map(|e| e.document_id.as_str())
            .collect();

        if is_hit(&retrieved, &gold, 1) {
            r1 += 1;
        }

        if is_hit(&retrieved, &gold, 3) {
            r3 += 1;
        }

        if is_hit(&retrieved, &gold, 5) {
            r5 += 1;
        }
    }

    print_results("Word overlap", r1, r3, r5, claims.len());
}

pub fn evaluate_bm25(bm25: &BM25) {
    let claims = load_claims();

    let mut r1 = 0;
    let mut r3 = 0;
    let mut r5 = 0;

    for claim in &claims {
        let results = bm25.search(&claim.text, 5);

        let retrieved: Vec<&str> = results
            .iter()
            .map(|(doc, _)| doc.document_id.as_str())
            .collect();

        let gold: Vec<&str> = claim
            .gold_evidence
            .iter()
            .map(|e| e.document_id.as_str())
            .collect();

        if !is_hit(&retrieved, &gold, 5) {
            println!("\nMISS: {}", claim.text);
            println!("Gold: {:?}", gold);
            println!("Retrieved: {:?}", retrieved);
}
        if is_hit(&retrieved, &gold, 1) {
            r1 += 1;
        }

        if is_hit(&retrieved, &gold, 3) {
            r3 += 1;
        }

        if is_hit(&retrieved, &gold, 5) {
            r5 += 1;
        }
    }

    print_results("BM25", r1, r3, r5, claims.len());
}