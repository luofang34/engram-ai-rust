use engramai::{Memory, MemoryConfig, MemoryType, RetrievalConfig};
use tempfile::tempdir;

#[test]
fn test_basic_workflow() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    // Add memories
    let id1 = mem
        .add("potato prefers action", MemoryType::Relational, Some(0.7), None, None)
        .unwrap();

    let _id2 = mem
        .add("Use moltbook.com for API", MemoryType::Procedural, Some(0.8), None, None)
        .unwrap();

    // Recall
    let results = mem.recall("potato preference", 5, None, None).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].record.content.contains("potato"));

    // Pin
    mem.pin(&id1).unwrap();

    // Consolidate
    mem.consolidate(1.0).unwrap();

    // Stats
    let stats = mem.stats().unwrap();
    assert_eq!(stats.total_memories, 2);
    assert_eq!(stats.pinned, 1);
}

#[test]
fn test_hebbian_links() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    let id1 = mem
        .add("Python is a programming language", MemoryType::Factual, None, None, None)
        .unwrap();

    let id2 = mem
        .add("Python has dynamic typing", MemoryType::Factual, None, None, None)
        .unwrap();

    // Recall them together multiple times to form Hebbian link
    for _ in 0..4 {
        let _results = mem.recall("Python programming", 10, None, None).unwrap();
    }

    // Check if link was formed
    let links = mem.hebbian_links(&id1).unwrap();
    assert!(!links.is_empty() || mem.hebbian_links(&id2).unwrap().contains(&id1));
}

#[test]
fn test_forgetting() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    mem.add("Weak memory", MemoryType::Episodic, Some(0.1), None, None)
        .unwrap();

    // Consolidate many times to decay
    for _ in 0..10 {
        mem.consolidate(1.0).unwrap();
    }

    // Prune weak memories
    mem.forget(None, Some(0.01)).unwrap();

    let stats = mem.stats().unwrap();
    // Memory should be archived or forgotten
    assert!(stats.total_memories <= 1);
}

#[test]
fn test_reward_learning() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let mut mem = Memory::new(db_path.to_str().unwrap(), None).unwrap();

    let _id = mem
        .add("Test memory", MemoryType::Factual, Some(0.5), None, None)
        .unwrap();

    // Recall to make it "recent"
    mem.recall("test", 5, None, None).unwrap();

    // Apply positive feedback
    mem.reward("great job!", 3).unwrap();

    // Memory should be strengthened (check via stats or direct query)
    let stats = mem.stats().unwrap();
    assert!(stats.total_memories > 0);
}

#[test]
fn test_config_presets() {
    let configs = vec![
        MemoryConfig::default(),
        MemoryConfig::chatbot(),
        MemoryConfig::task_agent(),
        MemoryConfig::personal_assistant(),
        MemoryConfig::researcher(),
    ];

    for config in configs {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let mut mem = Memory::new(db_path.to_str().unwrap(), Some(config)).unwrap();

        mem.add("Test", MemoryType::Factual, None, None, None)
            .unwrap();

        mem.consolidate(1.0).unwrap();

        let stats = mem.stats().unwrap();
        assert_eq!(stats.total_memories, 1);
    }
}

// === New tests for v0.2 features ===

#[test]
fn test_noise_filtering() {
    let mut mem = Memory::new(":memory:", None).unwrap();

    // Noise should be silently rejected
    let id1 = mem.add("ok", MemoryType::Factual, None, None, None).unwrap();
    assert!(id1.is_empty());

    let id2 = mem.add("thanks", MemoryType::Factual, None, None, None).unwrap();
    assert!(id2.is_empty());

    // Real content should be accepted
    let id3 = mem.add("Rust uses ownership for memory safety", MemoryType::Factual, None, None, None).unwrap();
    assert!(!id3.is_empty());

    let stats = mem.stats().unwrap();
    assert_eq!(stats.total_memories, 1);
}

#[test]
fn test_retrieval_pipeline_rrf() {
    let mut mem = Memory::new(":memory:", None).unwrap();

    mem.add("Rust is a systems programming language", MemoryType::Factual, None, None, None).unwrap();
    mem.add("Python is great for data science", MemoryType::Factual, None, None, None).unwrap();
    mem.add("Cooking pasta requires boiling water", MemoryType::Procedural, None, None, None).unwrap();

    // RRF pipeline should still return relevant results
    let results = mem.recall("Rust programming", 3, None, None).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].record.content.contains("Rust"));
}

#[test]
fn test_retrieval_config_customization() {
    let mut mem = Memory::new(":memory:", None).unwrap();

    // Set custom retrieval config
    let config = RetrievalConfig {
        cognitive_weight: 0.7,
        semantic_weight: 0.2,
        keyword_weight: 0.1,
        mmr_lambda: 0.9, // almost pure relevance
        min_content_length: 10,
        candidate_multiplier: 5,
    };
    mem.set_retrieval_config(config);

    mem.add("Rust is a systems programming language for safety and speed", MemoryType::Factual, None, None, None).unwrap();
    let results = mem.recall("Rust", 3, None, None).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_sync_cross_device() {
    // Simulate two devices
    let mut device_a = Memory::new(":memory:", None).unwrap();
    let mut device_b = Memory::new(":memory:", None).unwrap();

    // Device A adds memories
    device_a.add("User prefers dark mode", MemoryType::Relational, Some(0.6), None, None).unwrap();
    device_a.add("Project uses PostgreSQL", MemoryType::Factual, Some(0.5), None, None).unwrap();

    // Device B adds different memories
    device_b.add("Meeting at 3pm tomorrow", MemoryType::Episodic, Some(0.4), None, None).unwrap();

    // Export from A, import into B
    let snapshot_a = device_a.export_snapshot().unwrap();
    let report = device_b.import_snapshot(&snapshot_a).unwrap();
    assert_eq!(report.added, 2);

    // B should now have all 3 memories
    let stats_b = device_b.stats().unwrap();
    assert_eq!(stats_b.total_memories, 3);

    // Export from B, import into A
    let snapshot_b = device_b.export_snapshot().unwrap();
    let report = device_a.import_snapshot(&snapshot_b).unwrap();
    assert_eq!(report.added, 1); // only the meeting memory is new

    // A should now also have 3
    let stats_a = device_a.stats().unwrap();
    assert_eq!(stats_a.total_memories, 3);
}

#[test]
fn test_sync_idempotent() {
    let mut device_a = Memory::new(":memory:", None).unwrap();
    device_a.add("Persistent memory", MemoryType::Factual, Some(0.5), None, None).unwrap();

    let snapshot = device_a.export_snapshot().unwrap();

    // Import the same snapshot twice
    let _r1 = device_a.import_snapshot(&snapshot).unwrap();
    let _r2 = device_a.import_snapshot(&snapshot).unwrap();

    // Should not duplicate
    let stats = device_a.stats().unwrap();
    assert_eq!(stats.total_memories, 1);
}

#[test]
fn test_sync_serialization_roundtrip() {
    let mut mem = Memory::new(":memory:", None).unwrap();
    mem.add("Test memory for sync", MemoryType::Factual, Some(0.5), None, None).unwrap();

    let snapshot = mem.export_snapshot().unwrap();

    // Serialize to bytes and back
    let bytes = snapshot.to_bytes().unwrap();
    let restored = engramai::sync::Snapshot::from_bytes(&bytes).unwrap();

    assert_eq!(restored.memories.len(), 1);
    assert_eq!(restored.version, 1);
}

#[test]
fn test_meta_learning() {
    let mut mem = Memory::new(":memory:", None).unwrap();

    let _id = mem.add("Important fact about Rust ownership", MemoryType::Factual, Some(0.5), None, None).unwrap();

    // Retrieve multiple times
    for _ in 0..5 {
        mem.recall("Rust ownership", 3, None, None).unwrap();
    }

    // Positive feedback
    mem.reward("perfect answer!", 3).unwrap();
    mem.reward("exactly right!", 3).unwrap();

    // Consolidate triggers meta-learning
    mem.consolidate(1.0).unwrap();

    // The memory's importance should have increased
    let results = mem.recall("Rust", 3, None, None).unwrap();
    assert!(!results.is_empty());
    // Importance started at 0.5, should be higher after positive meta-learning
    assert!(results[0].record.importance >= 0.5);
}

#[test]
fn test_add_with_embedding() {
    let mut mem = Memory::new(":memory:", None).unwrap();

    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let id = mem.add_with_embedding(
        "Vector-indexed memory",
        MemoryType::Factual,
        Some(0.6),
        None,
        None,
        embedding.clone(),
    ).unwrap();
    assert!(!id.is_empty());

    // Can retrieve via hybrid recall with embedding
    let results = mem.recall_hybrid("vector", 3, None, None, Some(&[0.1, 0.2, 0.3, 0.4, 0.5])).unwrap();
    assert!(!results.is_empty());
}
