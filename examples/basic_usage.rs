//! Basic usage example demonstrating Engram AI's core API.

use engramai::{Memory, MemoryType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Engram AI Demo ===\n");

    // Create in-memory database
    let mut mem = Memory::new(":memory:", None)?;

    // --- Add memories ---
    println!("Adding memories...");
    let _id1 = mem.add(
        "potato prefers action over discussion",
        MemoryType::Relational,
        Some(0.7),
        None,
        None,
    )?;
    let _id2 = mem.add(
        "SaltyHall uses Supabase for database",
        MemoryType::Factual,
        Some(0.5),
        None,
        None,
    )?;
    let _id3 = mem.add(
        "Use www.moltbook.com not moltbook.com",
        MemoryType::Procedural,
        Some(0.8),
        None,
        None,
    )?;
    let id4 = mem.add(
        "potato said I kinda like you",
        MemoryType::Emotional,
        Some(0.95),
        None,
        None,
    )?;
    let _id5 = mem.add(
        "Saw a funny cat meme",
        MemoryType::Episodic,
        Some(0.1),
        None,
        None,
    )?;

    // Noise is filtered automatically
    let noise_id = mem.add("ok", MemoryType::Factual, None, None, None)?;
    assert!(noise_id.is_empty(), "Noise should be filtered");
    println!("  Added 5 memories (noise filtered automatically)\n");

    // --- Recall with RRF + MMR pipeline ---
    println!("--- Recall: 'what does potato like?' ---");
    let results = mem.recall("what does potato like?", 3, None, None)?;
    for r in &results {
        println!(
            "  [{:10}] conf={:.2} score={:.2} | {}",
            r.confidence_label,
            r.confidence,
            r.activation,
            &r.record.content[..r.record.content.len().min(50)]
        );
    }

    println!();
    println!("--- Recall: 'moltbook API' ---");
    let results = mem.recall("moltbook API", 3, None, None)?;
    for r in &results {
        println!(
            "  [{:10}] conf={:.2} score={:.2} | {}",
            r.confidence_label,
            r.confidence,
            r.activation,
            &r.record.content[..r.record.content.len().min(50)]
        );
    }

    // --- Reward + Meta-learning ---
    println!("\n--- Applying positive feedback ---");
    mem.reward("good job, that's exactly right!", 3)?;

    // --- Consolidate ---
    println!("--- Running consolidation (3 days) ---");
    for day in 1..=3 {
        mem.consolidate(1.0)?;
        println!("  Day {}/3 complete (meta-learning active)", day);
    }

    // Pin emotional memory
    mem.pin(&id4)?;
    println!("\n--- Pinned emotional memory ---");

    // --- Cross-device sync ---
    println!("\n--- Cross-device sync demo ---");
    let mut device_b = Memory::new(":memory:", None)?;
    device_b.add("Remote device note", MemoryType::Episodic, Some(0.4), None, None)?;

    // Export from main, import into device B
    let snapshot = mem.export_snapshot()?;
    let bytes = snapshot.to_bytes()?;
    println!("  Snapshot: {} bytes ({} memories)", bytes.len(), snapshot.memories.len());

    let restored = engramai::sync::Snapshot::from_bytes(&bytes)?;
    let report = device_b.import_snapshot(&restored)?;
    println!("  Merged: {} added, {} updated, {} skipped", report.added, report.updated, report.skipped);

    let stats_b = device_b.stats()?;
    println!("  Device B now has {} memories", stats_b.total_memories);

    // --- Stats ---
    println!("\n--- Memory Statistics ---");
    let stats = mem.stats()?;
    println!("  Total: {} memories", stats.total_memories);
    println!("  Pinned: {}", stats.pinned);
    println!("  Uptime: {:.1} hours", stats.uptime_hours);

    for (type_name, info) in &stats.by_type {
        println!(
            "  {:12}: {} entries, avg_str={:.3}, avg_imp={:.2}",
            type_name, info.count, info.avg_strength, info.avg_importance
        );
    }

    println!();
    for (layer_name, info) in &stats.by_layer {
        println!(
            "  {:12}: {} entries, avg_working={:.3}, avg_core={:.3}",
            layer_name, info.count, info.avg_working, info.avg_core
        );
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}
