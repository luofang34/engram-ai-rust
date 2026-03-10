//! Main Memory API — simplified interface to Engram's cognitive models.

use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

/// Type alias for memory merge/compression functions.
pub type MergeFn = dyn Fn(&[&str]) -> String;

use crate::config::MemoryConfig;
use crate::models::{effective_strength, retrieval_activation, run_consolidation_cycle};
use crate::retrieval::{self, RetrievalConfig};
use crate::store::MemoryStore;
use crate::types::{
    LayerStats, MemoryLayer, MemoryRecord, MemoryStats, MemoryType, RecallResult, TypeStats,
};

/// Main interface to the Engram memory system.
///
/// Wraps the neuroscience math models behind a clean API.
/// All complexity is hidden — you just add, recall, and consolidate.
pub struct Memory {
    storage: MemoryStore,
    config: MemoryConfig,
    retrieval_config: RetrievalConfig,
    created_at: chrono::DateTime<Utc>,
    /// Tracks per-memory retrieval outcomes for meta-learning.
    /// Maps memory_id → (times_retrieved, times_rewarded_positive, times_rewarded_negative)
    meta_stats: HashMap<String, (u64, u64, u64)>,
}

impl Memory {
    /// Initialize Engram memory system.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to database file. Created if it doesn't exist.
    ///   Use `:memory:` for in-memory (non-persistent) operation.
    /// * `config` - MemoryConfig with tunable parameters. None = literature defaults.
    pub fn new(
        path: &str,
        config: Option<MemoryConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = if path == ":memory:" {
            MemoryStore::new::<std::path::PathBuf>(None)?
        } else {
            MemoryStore::new(Some(std::path::PathBuf::from(path)))?
        };
        let config = config.unwrap_or_default();
        let created_at = Utc::now();

        Ok(Self {
            storage,
            config,
            retrieval_config: RetrievalConfig::default(),
            created_at,
            meta_stats: HashMap::new(),
        })
    }

    /// Set custom retrieval pipeline configuration.
    pub fn set_retrieval_config(&mut self, config: RetrievalConfig) {
        self.retrieval_config = config;
    }

    /// Store a new memory. Returns memory ID.
    ///
    /// The memory is encoded with initial working_strength=1.0 (strong
    /// hippocampal trace) and core_strength=0.0 (no neocortical trace yet).
    /// Consolidation cycles will gradually transfer it to core.
    ///
    /// Noise content (greetings, acknowledgments) is silently rejected.
    pub fn add(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: Option<f64>,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Noise filter at ingest
        if retrieval::is_noise(content) {
            return Ok(String::new());
        }

        let id = format!("{}", Uuid::new_v4())[..8].to_string();
        let importance = importance.unwrap_or_else(|| memory_type.default_importance());

        let record = MemoryRecord {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            layer: MemoryLayer::Working,
            created_at: Utc::now(),
            access_times: vec![Utc::now()],
            working_strength: 1.0,
            core_strength: 0.0,
            importance,
            pinned: false,
            consolidation_count: 0,
            last_consolidated: None,
            source: source.unwrap_or("").to_string(),
            contradicts: None,
            contradicted_by: None,
            metadata,
            embedding: None,
        };

        self.storage.add(&record)?;
        Ok(id)
    }

    /// Store a memory with a pre-computed embedding vector.
    pub fn add_with_embedding(
        &mut self,
        content: &str,
        memory_type: MemoryType,
        importance: Option<f64>,
        source: Option<&str>,
        metadata: Option<serde_json::Value>,
        embedding: Vec<f32>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        if retrieval::is_noise(content) {
            return Ok(String::new());
        }

        let id = format!("{}", Uuid::new_v4())[..8].to_string();
        let importance = importance.unwrap_or_else(|| memory_type.default_importance());

        let record = MemoryRecord {
            id: id.clone(),
            content: content.to_string(),
            memory_type,
            layer: MemoryLayer::Working,
            created_at: Utc::now(),
            access_times: vec![Utc::now()],
            working_strength: 1.0,
            core_strength: 0.0,
            importance,
            pinned: false,
            consolidation_count: 0,
            last_consolidated: None,
            source: source.unwrap_or("").to_string(),
            contradicts: None,
            contradicted_by: None,
            metadata,
            embedding: Some(embedding),
        };

        self.storage.add(&record)?;
        Ok(id)
    }

    /// Retrieve relevant memories using the full pipeline:
    /// FTS + vector → RRF fusion → ACT-R cognitive scoring → MMR diversity.
    pub fn recall(
        &mut self,
        query: &str,
        limit: usize,
        context: Option<Vec<String>>,
        min_confidence: Option<f64>,
    ) -> Result<Vec<RecallResult>, Box<dyn std::error::Error>> {
        self.recall_hybrid(query, limit, context, min_confidence, None)
    }

    /// Recall with optional embedding vector for hybrid semantic+keyword+cognitive search.
    ///
    /// Uses the full retrieval pipeline:
    /// 1. Candidate generation (FTS + vector search)
    /// 2. RRF fusion (merges ranked lists)
    /// 3. Cognitive scoring (ACT-R activation blend)
    /// 4. MMR diversity selection (suppresses near-duplicates)
    pub fn recall_hybrid(
        &mut self,
        query: &str,
        limit: usize,
        context: Option<Vec<String>>,
        min_confidence: Option<f64>,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<RecallResult>, Box<dyn std::error::Error>> {
        let now = Utc::now();
        let context = context.unwrap_or_default();
        let min_conf = min_confidence.unwrap_or(0.0);

        // Pre-compute cognitive scores for all memories
        let all = self.storage.all()?;
        let cognitive_scores: HashMap<String, f64> = all
            .iter()
            .filter_map(|record| {
                let activation = retrieval_activation(
                    record,
                    &context,
                    now,
                    self.config.actr_decay,
                    self.config.context_weight,
                    self.config.importance_weight,
                    self.config.contradiction_penalty,
                );
                if activation == f64::NEG_INFINITY {
                    None
                } else {
                    Some((record.id.clone(), activation))
                }
            })
            .collect();

        // Run the retrieval pipeline
        let candidates = retrieval::retrieve(
            &self.storage,
            query,
            limit,
            query_embedding,
            &cognitive_scores,
            &self.retrieval_config,
        )?;

        // Convert to RecallResult with confidence
        let results: Vec<RecallResult> = candidates
            .into_iter()
            .map(|cand| {
                let confidence = self.compute_confidence(&cand.record, cand.score);
                let confidence_label = confidence_label(confidence);

                RecallResult {
                    record: cand.record,
                    activation: cand.score,
                    confidence,
                    confidence_label,
                }
            })
            .filter(|r| r.confidence >= min_conf)
            .collect();

        // Record access for all retrieved memories (ACT-R learning)
        for result in &results {
            self.storage.record_access(&result.record.id)?;
            // Meta-learning: track retrieval count
            let entry = self
                .meta_stats
                .entry(result.record.id.clone())
                .or_insert((0, 0, 0));
            entry.0 += 1;
        }

        // Hebbian learning: record co-activation
        if self.config.hebbian_enabled && results.len() >= 2 {
            let memory_ids: Vec<_> = results.iter().map(|r| r.record.id.clone()).collect();
            crate::models::record_coactivation(
                &mut self.storage,
                &memory_ids,
                self.config.hebbian_threshold,
            )?;
        }

        Ok(results)
    }

    /// Run a consolidation cycle ("sleep replay").
    ///
    /// Based on Murre & Chessa's Memory Chain Model, it:
    ///
    /// 1. Decays working_strength (hippocampal traces fade)
    /// 2. Transfers knowledge to core_strength (neocortical consolidation)
    /// 3. Replays archived memories (prevents catastrophic forgetting)
    /// 4. Rebalances layers (promote strong → core, demote weak → archive)
    /// 5. Runs meta-learning importance adjustment
    pub fn consolidate(&mut self, days: f64) -> Result<(), Box<dyn std::error::Error>> {
        run_consolidation_cycle(&mut self.storage, days, &self.config)?;

        // Decay Hebbian links
        if self.config.hebbian_enabled {
            self.storage
                .decay_hebbian_links(self.config.hebbian_decay)?;
        }

        // Meta-learning: adjust importance based on actual utility
        self.meta_learn()?;

        Ok(())
    }

    /// Merge similar memories into distilled summaries.
    ///
    /// This is "consolidation as compression" — instead of just decaying,
    /// find clusters of related memories and merge them.
    ///
    /// Takes an optional `merge_fn` that compresses multiple content strings
    /// into one. If None, uses simple concatenation with dedup.
    ///
    /// Returns the number of merges performed.
    pub fn compress(
        &mut self,
        similarity_threshold: f64,
        merge_fn: Option<&MergeFn>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let all = self.storage.all()?;
        if all.len() < 2 {
            return Ok(0);
        }

        let mut merged_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut merge_count = 0;

        for record in &all {
            if merged_ids.contains(&record.id) {
                continue;
            }

            // Find Hebbian neighbors
            let neighbors = self.storage.get_hebbian_neighbors(&record.id)?;
            if neighbors.is_empty() {
                continue;
            }

            // Check content similarity with neighbors
            let mut cluster: Vec<&MemoryRecord> = vec![record];
            for nid in &neighbors {
                if merged_ids.contains(nid) {
                    continue;
                }
                if let Some(neighbor) = all.iter().find(|r| r.id == *nid) {
                    let sim = jaccard_similarity(&record.content, &neighbor.content);
                    if sim >= similarity_threshold {
                        cluster.push(neighbor);
                    }
                }
            }

            if cluster.len() < 2 {
                continue;
            }

            // Merge the cluster
            let contents: Vec<&str> = cluster.iter().map(|r| r.content.as_str()).collect();
            let merged_content = match merge_fn {
                Some(f) => f(&contents),
                None => default_merge(&contents),
            };

            // Keep the strongest memory as the survivor
            let survivor_idx = cluster
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let sa = a.working_strength + a.core_strength;
                    let sb = b.working_strength + b.core_strength;
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let survivor = cluster[survivor_idx];
            let mut updated = survivor.clone();
            updated.content = merged_content;
            updated.importance = (updated.importance * 1.1).min(1.0);
            for (i, r) in cluster.iter().enumerate() {
                if i != survivor_idx {
                    updated.core_strength += r.core_strength * 0.5;
                    merged_ids.insert(r.id.clone());
                }
            }

            self.storage.update(&updated)?;

            for (i, r) in cluster.iter().enumerate() {
                if i != survivor_idx {
                    self.storage.delete(&r.id)?;
                }
            }

            merge_count += 1;
        }

        Ok(merge_count)
    }

    /// Forget a specific memory or prune all below threshold.
    pub fn forget(
        &mut self,
        memory_id: Option<&str>,
        threshold: Option<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threshold = threshold.unwrap_or(self.config.forget_threshold);

        if let Some(id) = memory_id {
            self.storage.delete(id)?;
        } else {
            let now = Utc::now();
            let all = self.storage.all()?;
            for record in all {
                if !record.pinned
                    && effective_strength(&record, now) < threshold
                    && record.layer != MemoryLayer::Archive
                {
                    let mut updated = record;
                    updated.layer = MemoryLayer::Archive;
                    self.storage.update(&updated)?;
                }
            }
        }

        Ok(())
    }

    /// Process user feedback as a dopaminergic reward signal.
    ///
    /// Detects positive/negative sentiment and applies reward modulation
    /// to recently accessed memories. Also updates meta-learning stats.
    pub fn reward(
        &mut self,
        feedback: &str,
        recent_n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let polarity = detect_feedback_polarity(feedback);

        if polarity == 0.0 {
            return Ok(());
        }

        let all = self.storage.all()?;
        let mut recent: Vec<_> = all
            .into_iter()
            .filter(|r| !r.access_times.is_empty())
            .collect();
        recent.sort_by_key(|r| std::cmp::Reverse(r.access_times.last().cloned()));

        for mut record in recent.into_iter().take(recent_n) {
            if polarity > 0.0 {
                record.working_strength += self.config.reward_magnitude * polarity;
                record.working_strength = record.working_strength.min(2.0);
                let entry = self
                    .meta_stats
                    .entry(record.id.clone())
                    .or_insert((0, 0, 0));
                entry.1 += 1;
            } else {
                record.working_strength *= 1.0 + polarity * 0.1;
                record.working_strength = record.working_strength.max(0.0);
                let entry = self
                    .meta_stats
                    .entry(record.id.clone())
                    .or_insert((0, 0, 0));
                entry.2 += 1;
            }
            self.storage.update(&record)?;
        }

        Ok(())
    }

    /// Global synaptic downscaling — normalize all memory weights.
    pub fn downscale(&mut self, factor: Option<f64>) -> Result<usize, Box<dyn std::error::Error>> {
        let factor = factor.unwrap_or(self.config.downscale_factor);
        let all = self.storage.all()?;
        let mut count = 0;

        for mut record in all {
            if !record.pinned {
                record.working_strength *= factor;
                record.core_strength *= factor;
                self.storage.update(&record)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Memory system statistics.
    pub fn stats(&self) -> Result<MemoryStats, Box<dyn std::error::Error>> {
        let all = self.storage.all()?;
        let now = Utc::now();

        let mut by_type: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut by_layer: HashMap<String, Vec<&MemoryRecord>> = HashMap::new();
        let mut pinned = 0;

        for record in &all {
            by_type
                .entry(record.memory_type.to_string())
                .or_default()
                .push(record);
            by_layer
                .entry(record.layer.to_string())
                .or_default()
                .push(record);
            if record.pinned {
                pinned += 1;
            }
        }

        let type_stats: HashMap<String, TypeStats> = by_type
            .into_iter()
            .map(|(type_name, records)| {
                let count = records.len();
                let avg_strength = records
                    .iter()
                    .map(|r| effective_strength(r, now))
                    .sum::<f64>()
                    / count as f64;
                let avg_importance =
                    records.iter().map(|r| r.importance).sum::<f64>() / count as f64;

                (
                    type_name,
                    TypeStats {
                        count,
                        avg_strength,
                        avg_importance,
                    },
                )
            })
            .collect();

        let layer_stats: HashMap<String, LayerStats> = by_layer
            .into_iter()
            .map(|(layer_name, records)| {
                let count = records.len();
                let avg_working =
                    records.iter().map(|r| r.working_strength).sum::<f64>() / count as f64;
                let avg_core = records.iter().map(|r| r.core_strength).sum::<f64>() / count as f64;

                (
                    layer_name,
                    LayerStats {
                        count,
                        avg_working,
                        avg_core,
                    },
                )
            })
            .collect();

        let uptime_hours = (now - self.created_at).num_seconds() as f64 / 3600.0;

        Ok(MemoryStats {
            total_memories: all.len(),
            by_type: type_stats,
            by_layer: layer_stats,
            pinned,
            uptime_hours,
        })
    }

    /// Pin a memory — it won't decay or be pruned.
    pub fn pin(&mut self, memory_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut record) = self.storage.get(memory_id)? {
            record.pinned = true;
            self.storage.update(&record)?;
        }
        Ok(())
    }

    /// Unpin a memory — it will resume normal decay.
    pub fn unpin(&mut self, memory_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(mut record) = self.storage.get(memory_id)? {
            record.pinned = false;
            self.storage.update(&record)?;
        }
        Ok(())
    }

    /// Get Hebbian links for a specific memory.
    pub fn hebbian_links(
        &self,
        memory_id: &str,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        self.storage.get_hebbian_neighbors(memory_id)
    }

    /// Export all memories for sync (content-addressed snapshot).
    pub fn export_snapshot(&self) -> Result<crate::sync::Snapshot, Box<dyn std::error::Error>> {
        crate::sync::Snapshot::from_store(&self.storage)
    }

    /// Import and merge a remote snapshot (CRDT union merge).
    pub fn import_snapshot(
        &mut self,
        snapshot: &crate::sync::Snapshot,
    ) -> Result<crate::sync::MergeReport, Box<dyn std::error::Error>> {
        crate::sync::merge_snapshot(&mut self.storage, snapshot)
    }

    /// Meta-learning: adjust importance based on retrieval/reward patterns.
    ///
    /// Memories that are frequently retrieved and positively rewarded
    /// get their importance boosted. Memories retrieved but negatively
    /// rewarded get suppressed.
    fn meta_learn(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let updates: Vec<(String, f64)> = self
            .meta_stats
            .iter()
            .filter_map(|(id, (retrieved, positive, negative))| {
                if *retrieved < 2 {
                    return None;
                }
                let total_feedback = *positive + *negative;
                if total_feedback == 0 {
                    return None;
                }

                let reward_ratio = *positive as f64 / total_feedback as f64;
                let adjustment = (reward_ratio - 0.5) * 0.05;
                Some((id.clone(), adjustment))
            })
            .collect();

        for (id, adjustment) in updates {
            if let Some(mut record) = self.storage.get(&id)? {
                record.importance = (record.importance + adjustment).clamp(0.05, 1.0);
                self.storage.update(&record)?;
            }
        }

        Ok(())
    }

    fn compute_confidence(&self, record: &MemoryRecord, activation: f64) -> f64 {
        let normalized_activation = (activation + 10.0) / 20.0;
        let confidence = (normalized_activation.clamp(0.0, 1.0) * 0.7) + (record.importance * 0.3);
        confidence.clamp(0.0, 1.0)
    }
}

/// Default merge function: deduplicate sentences, concatenate.
fn default_merge(contents: &[&str]) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();

    for content in contents {
        for sentence in content.split(['.', ';', '\n']) {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() && seen.insert(trimmed.to_lowercase()) {
                merged.push(trimmed);
            }
        }
    }

    merged.join(". ")
}

/// Jaccard similarity between two content strings (word-level).
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();
    let words_b: std::collections::HashSet<&str> = b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 2)
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let union = words_a.union(&words_b).count() as f64;
    intersection / union
}

fn confidence_label(confidence: f64) -> String {
    match confidence {
        c if c >= 0.8 => "high".to_string(),
        c if c >= 0.5 => "medium".to_string(),
        c if c >= 0.2 => "low".to_string(),
        _ => "very low".to_string(),
    }
}

fn detect_feedback_polarity(feedback: &str) -> f64 {
    let lower = feedback.to_lowercase();
    let positive = [
        "good",
        "great",
        "excellent",
        "correct",
        "right",
        "yes",
        "nice",
        "perfect",
    ];
    let negative = [
        "bad",
        "wrong",
        "incorrect",
        "no",
        "error",
        "mistake",
        "poor",
    ];

    let pos_count = positive.iter().filter(|&w| lower.contains(w)).count();
    let neg_count = negative.iter().filter(|&w| lower.contains(w)).count();

    if pos_count > neg_count {
        1.0
    } else if neg_count > pos_count {
        -1.0
    } else {
        0.0
    }
}
