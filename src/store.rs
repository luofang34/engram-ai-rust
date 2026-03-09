//! Pure Rust in-memory storage backend with disk persistence.
//!
//! Replaces the SQLite backend for WASM compatibility.
//! Uses bincode serialization for fast, compact persistence.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::types::{MemoryRecord, MemoryType};

/// Stopwords filtered from the inverted index.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
];

/// Snapshot format for disk persistence.
#[derive(Serialize, Deserialize)]
struct StoreSnapshot {
    memories: Vec<MemoryRecord>,
    hebbian_links: Vec<(String, String, f64)>,
    version: u32,
}

/// Pure Rust in-memory memory store with optional disk persistence.
pub struct MemoryStore {
    memories: HashMap<String, MemoryRecord>,
    hebbian_links: HashMap<(String, String), f64>,
    inverted_index: HashMap<String, HashSet<String>>,
    path: Option<PathBuf>,
}

impl MemoryStore {
    /// Create a new store. If path is Some, loads existing data or creates fresh.
    pub fn new<P: Into<PathBuf>>(path: Option<P>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.map(|p| p.into());

        let mut store = Self {
            memories: HashMap::new(),
            hebbian_links: HashMap::new(),
            inverted_index: HashMap::new(),
            path,
        };

        if let Some(ref p) = store.path {
            if p.exists() {
                store.load_from_disk()?;
            }
        }

        Ok(store)
    }

    /// Add a new memory record.
    pub fn add(&mut self, record: &MemoryRecord) -> Result<(), Box<dyn std::error::Error>> {
        self.index_record(record);
        self.memories.insert(record.id.clone(), record.clone());
        self.auto_save()?;
        Ok(())
    }

    /// Get a memory by ID.
    pub fn get(&self, id: &str) -> Result<Option<MemoryRecord>, Box<dyn std::error::Error>> {
        Ok(self.memories.get(id).cloned())
    }

    /// Return all memory records.
    pub fn all(&self) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error>> {
        Ok(self.memories.values().cloned().collect())
    }

    /// Update an existing memory record.
    pub fn update(&mut self, record: &MemoryRecord) -> Result<(), Box<dyn std::error::Error>> {
        // Remove old index entries
        self.deindex_record(&record.id);
        // Re-index with new content
        self.index_record(record);
        self.memories.insert(record.id.clone(), record.clone());
        self.auto_save()?;
        Ok(())
    }

    /// Delete a memory by ID.
    pub fn delete(&mut self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.deindex_record(id);
        self.memories.remove(id);
        // Clean up hebbian links involving this id
        self.hebbian_links.retain(|(a, b), _| a != id && b != id);
        self.auto_save()?;
        Ok(())
    }

    /// Record an access timestamp for a memory.
    pub fn record_access(&mut self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(record) = self.memories.get_mut(id) {
            record.access_times.push(Utc::now());
        }
        Ok(())
    }

    /// Get access timestamps for a memory.
    pub fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, Box<dyn std::error::Error>> {
        Ok(self.memories.get(id).map(|r| r.access_times.clone()).unwrap_or_default())
    }

    /// Full-text search using the inverted index.
    pub fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error>> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        // Count matches per memory
        let mut scores: HashMap<&str, usize> = HashMap::new();
        for token in &tokens {
            if let Some(ids) = self.inverted_index.get(token) {
                for id in ids {
                    *scores.entry(id.as_str()).or_insert(0) += 1;
                }
            }
        }

        // Sort by match count descending
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        let results: Vec<MemoryRecord> = ranked
            .into_iter()
            .take(limit)
            .filter_map(|(id, _)| self.memories.get(id).cloned())
            .collect();

        Ok(results)
    }

    /// Search memories by type.
    pub fn search_by_type(&self, memory_type: MemoryType) -> Result<Vec<MemoryRecord>, Box<dyn std::error::Error>> {
        Ok(self.memories.values()
            .filter(|r| r.memory_type == memory_type)
            .cloned()
            .collect())
    }

    /// Get Hebbian neighbor IDs for a memory.
    pub fn get_hebbian_neighbors(&self, memory_id: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut neighbors = Vec::new();
        for ((a, b), strength) in &self.hebbian_links {
            if *strength > 0.0 {
                if a == memory_id {
                    neighbors.push(b.clone());
                } else if b == memory_id {
                    neighbors.push(a.clone());
                }
            }
        }
        Ok(neighbors)
    }

    /// Record co-activation between two memories.
    /// Returns true if a new link was formed (threshold reached).
    ///
    /// Encoding: negative values = tracking count (not yet linked),
    /// positive values = active link strength.
    pub fn record_coactivation(
        &mut self,
        id1: &str,
        id2: &str,
        threshold: i32,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let (a, b) = if id1 < id2 {
            (id1.to_string(), id2.to_string())
        } else {
            (id2.to_string(), id1.to_string())
        };
        let key = (a.clone(), b.clone());

        let current = self.hebbian_links.get(&key).copied();

        match current {
            None => {
                // First co-activation
                if threshold <= 1 {
                    self.hebbian_links.insert(key, 1.0);
                    Ok(true)
                } else {
                    self.hebbian_links.insert(key, -1.0);
                    Ok(false)
                }
            }
            Some(v) if v > 0.0 => {
                // Already linked, strengthen
                let new_strength = (v + 0.1).min(1.0);
                self.hebbian_links.insert(key, new_strength);
                Ok(false)
            }
            Some(v) => {
                // Tracking phase (negative = count)
                let count = (-v) as i32 + 1;
                if count >= threshold {
                    self.hebbian_links.insert(key, 1.0);
                    Ok(true)
                } else {
                    self.hebbian_links.insert(key, -(count as f64));
                    Ok(false)
                }
            }
        }
    }

    /// Decay all Hebbian links by a factor.
    pub fn decay_hebbian_links(&mut self, factor: f64) -> Result<usize, Box<dyn std::error::Error>> {
        // Decay all positive links
        for strength in self.hebbian_links.values_mut() {
            if *strength > 0.0 {
                *strength *= factor;
            }
        }
        // Prune very weak links
        let before = self.hebbian_links.len();
        self.hebbian_links.retain(|_, s| *s < 0.0 || *s >= 0.1);
        let pruned = before - self.hebbian_links.len();
        Ok(pruned)
    }

    /// Semantic search using cosine similarity against stored embeddings.
    pub fn search_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .memories
            .values()
            .filter_map(|record| {
                record.embedding.as_ref().map(|emb| {
                    let score = cosine_similarity(query_embedding, emb);
                    (record.id.clone(), score)
                })
            })
            .filter(|(_, score)| *score >= min_score)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    /// Save to disk.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref path) = self.path {
            let snapshot = StoreSnapshot {
                memories: self.memories.values().cloned().collect(),
                hebbian_links: self.hebbian_links
                    .iter()
                    .map(|((a, b), s)| (a.clone(), b.clone(), *s))
                    .collect(),
                version: 1,
            };
            let data = bincode::serialize(&snapshot)?;
            std::fs::write(path, data)?;
        }
        Ok(())
    }

    fn load_from_disk(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref path) = self.path {
            let data = std::fs::read(path)?;
            let snapshot: StoreSnapshot = bincode::deserialize(&data)?;

            for record in snapshot.memories {
                self.index_record(&record);
                self.memories.insert(record.id.clone(), record);
            }
            for (a, b, s) in snapshot.hebbian_links {
                self.hebbian_links.insert((a, b), s);
            }
        }
        Ok(())
    }

    fn auto_save(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.save()
    }

    fn index_record(&mut self, record: &MemoryRecord) {
        let tokens = tokenize(&record.content);
        for token in tokens {
            self.inverted_index
                .entry(token)
                .or_insert_with(HashSet::new)
                .insert(record.id.clone());
        }
    }

    fn deindex_record(&mut self, id: &str) {
        // Remove id from all inverted index entries
        self.inverted_index.retain(|_, ids| {
            ids.remove(id);
            !ids.is_empty()
        });
    }
}

/// Tokenize text: lowercase, split on non-alphanumeric, filter stopwords and short tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2 && !STOPWORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryLayer, MemoryRecord, MemoryType};

    fn make_record(id: &str, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            content: content.to_string(),
            memory_type: MemoryType::Factual,
            layer: MemoryLayer::Working,
            created_at: Utc::now(),
            access_times: vec![Utc::now()],
            working_strength: 1.0,
            core_strength: 0.0,
            importance: 0.5,
            pinned: false,
            consolidation_count: 0,
            last_consolidated: None,
            source: String::new(),
            contradicts: None,
            contradicted_by: None,
            metadata: None,
            embedding: None,
        }
    }

    #[test]
    fn test_add_get() {
        let mut store = MemoryStore::new::<PathBuf>(None).unwrap();
        let record = make_record("abc", "hello world");
        store.add(&record).unwrap();
        let got = store.get("abc").unwrap().unwrap();
        assert_eq!(got.content, "hello world");
    }

    #[test]
    fn test_fts() {
        let mut store = MemoryStore::new::<PathBuf>(None).unwrap();
        store.add(&make_record("1", "rust programming language")).unwrap();
        store.add(&make_record("2", "python scripting language")).unwrap();
        store.add(&make_record("3", "cooking recipes")).unwrap();

        let results = store.search_fts("programming rust", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_vector_search() {
        let mut store = MemoryStore::new::<PathBuf>(None).unwrap();
        let mut r1 = make_record("1", "first");
        r1.embedding = Some(vec![1.0, 0.0, 0.0]);
        let mut r2 = make_record("2", "second");
        r2.embedding = Some(vec![0.0, 1.0, 0.0]);
        store.add(&r1).unwrap();
        store.add(&r2).unwrap();

        let results = store.search_vector(&[1.0, 0.1, 0.0], 10, 0.0);
        assert_eq!(results[0].0, "1");
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }
}
