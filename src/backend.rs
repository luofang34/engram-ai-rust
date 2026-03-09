//! Storage backend trait — one interface, multiple implementations.
//!
//! Backends:
//! - `InMemoryBackend` (pure Rust, WASM-ready, bincode persistence)
//! - `SqlxBackend<Postgres>` (shared across machines, pgvector)
//! - `SqlxBackend<Sqlite>` (edge devices, offline-first)

use chrono::{DateTime, Utc};

use crate::types::{MemoryRecord, MemoryType};

/// Core storage operations that every backend must implement.
pub trait MemoryBackend: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    // --- CRUD ---
    fn add(&mut self, record: &MemoryRecord) -> Result<(), Self::Error>;
    fn get(&self, id: &str) -> Result<Option<MemoryRecord>, Self::Error>;
    fn all(&self) -> Result<Vec<MemoryRecord>, Self::Error>;
    fn update(&mut self, record: &MemoryRecord) -> Result<(), Self::Error>;
    fn delete(&mut self, id: &str) -> Result<(), Self::Error>;

    // --- Access tracking (ACT-R) ---
    fn record_access(&mut self, id: &str) -> Result<(), Self::Error>;
    fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, Self::Error>;

    // --- Search ---
    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, Self::Error>;
    fn search_by_type(&self, memory_type: MemoryType) -> Result<Vec<MemoryRecord>, Self::Error>;
    fn search_vector(&self, query_embedding: &[f32], limit: usize, min_score: f32) -> Vec<(String, f32)>;

    // --- Hebbian ---
    fn get_hebbian_neighbors(&self, memory_id: &str) -> Result<Vec<String>, Self::Error>;
    fn record_coactivation(&mut self, ids: &[String], min_weight: f64) -> Result<(), Self::Error>;
    fn decay_hebbian_links(&mut self, factor: f64) -> Result<usize, Self::Error>;

    // --- Persistence ---
    fn flush(&self) -> Result<(), Self::Error>;
}

/// Async variant for SQLx backends (Postgres, SQLite).
#[cfg(feature = "sqlx")]
#[async_trait::async_trait]
pub trait AsyncMemoryBackend: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn add(&mut self, record: &MemoryRecord) -> Result<(), Self::Error>;
    async fn get(&self, id: &str) -> Result<Option<MemoryRecord>, Self::Error>;
    async fn all(&self) -> Result<Vec<MemoryRecord>, Self::Error>;
    async fn update(&mut self, record: &MemoryRecord) -> Result<(), Self::Error>;
    async fn delete(&mut self, id: &str) -> Result<(), Self::Error>;

    async fn record_access(&mut self, id: &str) -> Result<(), Self::Error>;
    async fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, Self::Error>;

    async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, Self::Error>;
    async fn search_by_type(&self, memory_type: MemoryType) -> Result<Vec<MemoryRecord>, Self::Error>;
    async fn search_vector(&self, query_embedding: &[f32], limit: usize, min_score: f32) -> Vec<(String, f32)>;

    async fn get_hebbian_neighbors(&self, memory_id: &str) -> Result<Vec<String>, Self::Error>;
    async fn record_coactivation(&mut self, ids: &[String], min_weight: f64) -> Result<(), Self::Error>;
    async fn decay_hebbian_links(&mut self, factor: f64) -> Result<usize, Self::Error>;

    async fn flush(&self) -> Result<(), Self::Error>;
}
