//! Content-addressed sync layer with CRDT merge semantics.
//!
//! Enables cross-device memory synchronization without a centralized server.
//!
//! Design principles (inspired by Merkle-CRDTs):
//! - Each memory gets a content hash (CID) for deduplication
//! - Snapshots are serializable blobs that can travel over any transport
//! - Merge is a CRDT union: new memories = G-Set (grow-only), strength = max-wins
//! - No conflict: two devices adding different memories just union them
//! - Idempotent: importing the same snapshot twice is a no-op
//!
//! Transport-agnostic: snapshots can sync via:
//! - File copy (USB, shared folder)
//! - HTTP API
//! - iroh (content-addressed blobs, P2P)
//! - IPFS / libp2p
//! - Any message bus

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::store::MemoryStore;
use crate::types::MemoryRecord;

/// A content hash for deduplication (SHA-256 of normalized content).
pub type ContentHash = String;

/// A portable snapshot of memory state for sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// Device/node ID that produced this snapshot.
    pub node_id: String,
    /// Timestamp of snapshot creation.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// All memory records, keyed by content hash.
    pub memories: HashMap<ContentHash, MemoryRecord>,
    /// Hebbian links as (id_a, id_b, strength).
    pub hebbian_links: Vec<(String, String, f64)>,
}

/// Report from a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeReport {
    /// Number of new memories added from the remote snapshot.
    pub added: usize,
    /// Number of existing memories updated (strength merged).
    pub updated: usize,
    /// Number of memories that were identical (skipped).
    pub skipped: usize,
    /// Number of Hebbian links merged.
    pub links_merged: usize,
}

impl Snapshot {
    /// Create a snapshot from the current store state.
    pub fn from_store(store: &MemoryStore) -> Result<Self, Box<dyn std::error::Error>> {
        let all = store.all()?;
        let mut memories = HashMap::new();

        for record in all {
            let hash = content_hash(&record);
            memories.insert(hash, record);
        }

        // Export Hebbian links
        let hebbian_links = store.export_hebbian_links();

        Ok(Self {
            version: 1,
            node_id: generate_node_id(),
            created_at: chrono::Utc::now(),
            memories,
            hebbian_links,
        })
    }

    /// Serialize to bytes (JSON for portability across platforms and versions).
    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(data)?)
    }
}

/// Merge a remote snapshot into the local store using CRDT semantics.
///
/// Merge rules:
/// - **New memories** (hash not in local): add them (G-Set union)
/// - **Existing memories** (same hash): merge strengths using max-wins
/// - **Hebbian links**: union with max-strength
pub fn merge_snapshot(
    store: &mut MemoryStore,
    snapshot: &Snapshot,
) -> Result<MergeReport, Box<dyn std::error::Error>> {
    let mut report = MergeReport::default();

    // Build hash map of local memories
    let local_all = store.all()?;
    let local_hashes: HashMap<ContentHash, MemoryRecord> = local_all
        .into_iter()
        .map(|r| (content_hash(&r), r))
        .collect();

    // Also build id→hash map for conflict detection
    let local_ids: HashMap<String, ContentHash> = local_hashes
        .iter()
        .map(|(hash, r)| (r.id.clone(), hash.clone()))
        .collect();

    for (hash, remote_record) in &snapshot.memories {
        if let Some(local_record) = local_hashes.get(hash) {
            // Same content hash — merge strengths (max-wins LWW)
            let needs_update = remote_record.working_strength > local_record.working_strength
                || remote_record.core_strength > local_record.core_strength
                || remote_record.importance > local_record.importance
                || remote_record.access_times.len() > local_record.access_times.len();

            if needs_update {
                let mut merged = local_record.clone();
                merged.working_strength =
                    merged.working_strength.max(remote_record.working_strength);
                merged.core_strength = merged.core_strength.max(remote_record.core_strength);
                merged.importance = merged.importance.max(remote_record.importance);
                // Merge access times (union, deduplicated by timestamp)
                let mut times = merged.access_times.clone();
                for t in &remote_record.access_times {
                    if !times.contains(t) {
                        times.push(*t);
                    }
                }
                times.sort();
                merged.access_times = times;
                // Keep the higher consolidation count
                merged.consolidation_count = merged
                    .consolidation_count
                    .max(remote_record.consolidation_count);
                // Pinned if either side pinned
                merged.pinned = merged.pinned || remote_record.pinned;

                store.update(&merged)?;
                report.updated += 1;
            } else {
                report.skipped += 1;
            }
        } else {
            // New memory — check for ID collision
            let mut record = remote_record.clone();
            if local_ids.contains_key(&record.id) {
                // ID collision with different content — generate new ID
                record.id = format!("{}", uuid::Uuid::new_v4())[..8].to_string();
            }
            store.add(&record)?;
            report.added += 1;
        }
    }

    // Merge Hebbian links
    for (a, b, strength) in &snapshot.hebbian_links {
        if *strength > 0.0 {
            // Try to form/strengthen the link
            let _ = store.record_coactivation(a, b, 1);
            report.links_merged += 1;
        }
    }

    Ok(report)
}

/// Compute a content hash for deduplication.
///
/// Uses a simple hash of normalized content + type + source.
/// This is NOT cryptographically secure — it's for dedup only.
fn content_hash(record: &MemoryRecord) -> ContentHash {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    record.content.trim().to_lowercase().hash(&mut hasher);
    record.memory_type.to_string().hash(&mut hasher);
    record.source.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Generate a random node ID for this device.
fn generate_node_id() -> String {
    format!("{}", uuid::Uuid::new_v4())[..12].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryLayer, MemoryType};
    use chrono::Utc;
    use std::path::PathBuf;

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
    fn test_snapshot_roundtrip() {
        let mut store = MemoryStore::new::<PathBuf>(None).unwrap();
        store.add(&make_record("a1", "hello world")).unwrap();
        store.add(&make_record("b2", "rust programming")).unwrap();

        let snapshot = Snapshot::from_store(&store).unwrap();
        assert_eq!(snapshot.memories.len(), 2);

        // Serialize and deserialize
        let bytes = snapshot.to_bytes().unwrap();
        let restored = Snapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored.memories.len(), 2);
    }

    #[test]
    fn test_merge_new_memories() {
        let mut local = MemoryStore::new::<PathBuf>(None).unwrap();
        local.add(&make_record("a1", "local memory")).unwrap();

        let mut remote = MemoryStore::new::<PathBuf>(None).unwrap();
        remote.add(&make_record("b2", "remote memory")).unwrap();

        let snapshot = Snapshot::from_store(&remote).unwrap();
        let report = merge_snapshot(&mut local, &snapshot).unwrap();

        assert_eq!(report.added, 1);
        assert_eq!(local.all().unwrap().len(), 2);
    }

    #[test]
    fn test_merge_idempotent() {
        let mut local = MemoryStore::new::<PathBuf>(None).unwrap();
        local.add(&make_record("a1", "shared memory")).unwrap();

        let snapshot = Snapshot::from_store(&local).unwrap();

        // Import same snapshot twice
        let _report1 = merge_snapshot(&mut local, &snapshot).unwrap();
        let report2 = merge_snapshot(&mut local, &snapshot).unwrap();

        // Second import should skip everything
        assert_eq!(report2.added, 0);
        assert_eq!(local.all().unwrap().len(), 1);
    }

    #[test]
    fn test_merge_strength_max_wins() {
        let mut local = MemoryStore::new::<PathBuf>(None).unwrap();
        let mut rec = make_record("a1", "test memory");
        rec.working_strength = 0.5;
        rec.core_strength = 0.3;
        local.add(&rec).unwrap();

        // Remote has higher strengths
        let mut remote = MemoryStore::new::<PathBuf>(None).unwrap();
        let mut remote_rec = make_record("a1", "test memory");
        remote_rec.working_strength = 0.8;
        remote_rec.core_strength = 0.1; // lower core
        remote.add(&remote_rec).unwrap();

        let snapshot = Snapshot::from_store(&remote).unwrap();
        let report = merge_snapshot(&mut local, &snapshot).unwrap();

        assert_eq!(report.updated, 1);
        let merged = local.all().unwrap();
        assert_eq!(merged.len(), 1);
        // Max-wins: working=0.8 (remote), core=0.3 (local)
        assert!((merged[0].working_strength - 0.8).abs() < 0.01);
        assert!((merged[0].core_strength - 0.3).abs() < 0.01);
    }
}
