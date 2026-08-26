mod overlap;
mod evaluation;
mod bm25;


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

    evaluation::evaluate_overlap(&documents);

    let bm25 = bm25::BM25::new(documents);

    evaluation::evaluate_bm25(&bm25);
}