use super::SearchResult;

use reqwest::blocking::Client;
use serde::{
    Deserialize,
    Serialize,
};
use std::env;


// ==================================================
// Generic search interface
// ==================================================

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
}

pub trait SearchProvider {
    fn search(
        &self,
        request: &SearchRequest,
    ) -> Result<Vec<SearchResult>, String>;
}


// ==================================================
// Tavily implementation
// ==================================================

pub struct TavilySearchProvider {
    client: Client,
    api_key: String,
}

impl TavilySearchProvider {
    pub fn from_env() -> Result<Self, String> {
        let api_key = env::var("TAVILY_API_KEY")
            .map_err(|_| {
                "TAVILY_API_KEY environment variable is not set"
                    .to_string()
            })?;

        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }
}


// --------------------------------------------------
// Tavily request / response shapes
// --------------------------------------------------

#[derive(Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    max_results: usize,
    search_depth: &'a str,
}

#[derive(Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
}

#[derive(Deserialize)]
struct TavilyResult {
    title: String,
    url: String,

    #[serde(default)]
    content: String,
}


// ==================================================
// SearchProvider implementation
// ==================================================

impl SearchProvider for TavilySearchProvider {
    fn search(
        &self,
        request: &SearchRequest,
    ) -> Result<Vec<SearchResult>, String> {
        let body = TavilyRequest {
            api_key: &self.api_key,
            query: &request.query,
            max_results: request.limit,
            search_depth: "advanced",
        };

        let response = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .map_err(|error| {
                format!(
                    "Tavily request failed: {}",
                    error
                )
            })?;

        if !response.status().is_success() {
            return Err(format!(
                "Tavily returned HTTP {}",
                response.status()
            ));
        }

        let response: TavilyResponse =
            response
                .json()
                .map_err(|error| {
                    format!(
                        "Could not parse Tavily response: {}",
                        error
                    )
                })?;

        let results = response
            .results
            .into_iter()
            .map(|result| {
                SearchResult {
                    title: result.title,
                    url: result.url,
                    snippet: if result.content.is_empty() {
                        None
                    } else {
                        Some(result.content)
                    },
                    source_name: None,
                }
            })
            .collect();

        Ok(results)
    }
}