mod retriever;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};

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

fn load_documents() -> Vec<Document> {
    let file = File::open("data/normalized/scifact/documents.jsonl")
        .expect("Could not open documents");

    BufReader::new(file)
        .lines()
        .map(|line| {
            serde_json::from_str(&line.unwrap())
                .expect("Invalid document")
        })
        .collect()
}

fn main() {
    let documents = load_documents();

    println!("Loaded {} documents.", documents.len());

    let claim = "1 in 5 million in UK have abnormal PrP positivity.";

    println!("\nClaim:");
    println!("{claim}");

    println!("\nTop evidence:");

    let results = retriever::search(claim, &documents, 5);

    for (rank, (document, score)) in results.iter().enumerate() {
        println!(
            "{}. {} | score={} | {}",
            rank + 1,
            document.document_id,
            score,
            document.title
        );
    }
}