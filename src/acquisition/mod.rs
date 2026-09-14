pub mod extract;
pub mod fetch;
pub mod rank;
pub mod search;

pub use extract::extract_passages;
pub use fetch::SourceFetcher;
pub use rank::{ScoredEvidencePassage, rank_passages, relevance_concentration, source_diversity};
pub use search::{SearchProvider, SearchRequest, TavilySearchProvider};

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: Option<String>,
    pub source_name: Option<String>,
}

#[derive(Debug, Clone)]
pub enum SourceContentType {
    Html,
    Pdf,
    PlainText,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct SourceDocument {
    pub title: String,
    pub url: String,
    pub source_name: Option<String>,
    pub content_type: SourceContentType,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct EvidencePassage {
    pub source_url: String,
    pub source_title: String,
    pub source_name: Option<String>,
    pub text: String,
}
