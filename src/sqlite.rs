//! SQLite storage backend via SQLx.
//!
//! For edge devices and offline-first operation.
//! Enable with `features = ["sqlx-sqlite"]`.

#![cfg(feature = "sqlx-sqlite")]

use chrono::{DateTime, Utc};
use sqlx::{SqlitePool, Row, sqlite::SqliteRow};

use crate::types::{MemoryLayer, MemoryRecord, MemoryType};

/// SQLite-backed memory storage via SQLx (async, no C rusqlite dep).
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Connect to SQLite database and run migrations.
    /// Use `:memory:` for in-memory, or a file path for persistent.
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let url = if database_url == ":memory:" {
            "sqlite::memory:".to_string()
        } else if database_url.starts_with("sqlite:") {
            database_url.to_string()
        } else {
            format!("sqlite:{}?mode=rwc", database_url)
        };
        let pool = SqlitePool::connect(&url).await?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    async fn migrate(&self) -> Result<(), sqlx::Error> {
        sqlx::raw_sql(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS engram_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                working_strength REAL NOT NULL DEFAULT 1.0,
                core_strength REAL NOT NULL DEFAULT 0.0,
                importance REAL NOT NULL DEFAULT 0.3,
                pinned INTEGER NOT NULL DEFAULT 0,
                consolidation_count INTEGER NOT NULL DEFAULT 0,
                last_consolidated TEXT,
                source TEXT DEFAULT '',
                contradicts TEXT,
                contradicted_by TEXT,
                metadata TEXT,
                embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS engram_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL REFERENCES engram_memories(id) ON DELETE CASCADE,
                accessed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_engram_access_memory ON engram_access_log(memory_id);

            CREATE TABLE IF NOT EXISTS engram_hebbian_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (source_id, target_id)
            );

            -- FTS5 for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS engram_fts USING fts5(
                content, content=engram_memories, content_rowid=rowid
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS engram_fts_insert AFTER INSERT ON engram_memories BEGIN
                INSERT INTO engram_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS engram_fts_delete AFTER DELETE ON engram_memories BEGIN
                INSERT INTO engram_fts(engram_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS engram_fts_update AFTER UPDATE ON engram_memories BEGIN
                INSERT INTO engram_fts(engram_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                INSERT INTO engram_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // --- CRUD ---

    pub async fn add(&self, record: &MemoryRecord) -> Result<(), sqlx::Error> {
        let embedding_blob = record.embedding.as_ref().map(|v| {
            v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()
        });

        sqlx::query(
            r#"INSERT INTO engram_memories
                (id, content, memory_type, layer, created_at, working_strength, core_strength,
                 importance, pinned, consolidation_count, last_consolidated, source,
                 contradicts, contradicted_by, metadata, embedding)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)"#,
        )
        .bind(&record.id)
        .bind(&record.content)
        .bind(record.memory_type.to_string())
        .bind(record.layer.to_string())
        .bind(record.created_at.to_rfc3339())
        .bind(record.working_strength)
        .bind(record.core_strength)
        .bind(record.importance)
        .bind(record.pinned)
        .bind(record.consolidation_count)
        .bind(record.last_consolidated.map(|t| t.to_rfc3339()))
        .bind(&record.source)
        .bind(&record.contradicts)
        .bind(&record.contradicted_by)
        .bind(record.metadata.as_ref().map(|m| m.to_string()))
        .bind(&embedding_blob)
        .execute(&self.pool)
        .await?;

        sqlx::query("INSERT INTO engram_access_log (memory_id, accessed_at) VALUES ($1, $2)")
            .bind(&record.id)
            .bind(record.created_at.to_rfc3339())
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn get(&self, id: &str) -> Result<Option<MemoryRecord>, sqlx::Error> {
        let row = sqlx::query("SELECT * FROM engram_memories WHERE id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await?;

        match row {
            Some(row) => {
                let access_times = self.get_access_times(id).await?;
                Ok(Some(row_to_record(&row, access_times)))
            }
            None => Ok(None),
        }
    }

    pub async fn all(&self) -> Result<Vec<MemoryRecord>, sqlx::Error> {
        let rows = sqlx::query("SELECT * FROM engram_memories")
            .fetch_all(&self.pool)
            .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
            let access_times = self.get_access_times(&id).await?;
            records.push(row_to_record(&row, access_times));
        }
        Ok(records)
    }

    pub async fn update(&self, record: &MemoryRecord) -> Result<(), sqlx::Error> {
        let embedding_blob = record.embedding.as_ref().map(|v| {
            v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()
        });

        sqlx::query(
            r#"UPDATE engram_memories SET
                content = $2, memory_type = $3, layer = $4,
                working_strength = $5, core_strength = $6, importance = $7,
                pinned = $8, consolidation_count = $9, last_consolidated = $10,
                source = $11, contradicts = $12, contradicted_by = $13, metadata = $14,
                embedding = COALESCE($15, embedding)
               WHERE id = $1"#,
        )
        .bind(&record.id)
        .bind(&record.content)
        .bind(record.memory_type.to_string())
        .bind(record.layer.to_string())
        .bind(record.working_strength)
        .bind(record.core_strength)
        .bind(record.importance)
        .bind(record.pinned)
        .bind(record.consolidation_count)
        .bind(record.last_consolidated.map(|t| t.to_rfc3339()))
        .bind(&record.source)
        .bind(&record.contradicts)
        .bind(&record.contradicted_by)
        .bind(record.metadata.as_ref().map(|m| m.to_string()))
        .bind(&embedding_blob)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete(&self, id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("DELETE FROM engram_memories WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // --- Access tracking ---

    pub async fn record_access(&self, id: &str) -> Result<(), sqlx::Error> {
        sqlx::query("INSERT INTO engram_access_log (memory_id, accessed_at) VALUES ($1, $2)")
            .bind(id)
            .bind(Utc::now().to_rfc3339())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn get_access_times(&self, id: &str) -> Result<Vec<DateTime<Utc>>, sqlx::Error> {
        let rows = sqlx::query(
            "SELECT accessed_at FROM engram_access_log WHERE memory_id = $1 ORDER BY accessed_at",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .iter()
            .filter_map(|r| {
                let s: String = r.get("accessed_at");
                DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))
            })
            .collect())
    }

    // --- Search ---

    pub async fn search_fts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, sqlx::Error> {
        // Clean query for FTS5
        let words: Vec<&str> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 2)
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = words.join(" OR ");

        let rows = sqlx::query(
            r#"SELECT m.* FROM engram_memories m
               JOIN engram_fts f ON m.rowid = f.rowid
               WHERE engram_fts MATCH $1
               LIMIT $2"#,
        )
        .bind(&fts_query)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
            let access_times = self.get_access_times(&id).await?;
            records.push(row_to_record(&row, access_times));
        }
        Ok(records)
    }

    pub async fn search_by_type(
        &self,
        memory_type: MemoryType,
    ) -> Result<Vec<MemoryRecord>, sqlx::Error> {
        let rows = sqlx::query("SELECT * FROM engram_memories WHERE memory_type = $1")
            .bind(memory_type.to_string())
            .fetch_all(&self.pool)
            .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
            let access_times = self.get_access_times(&id).await?;
            records.push(row_to_record(&row, access_times));
        }
        Ok(records)
    }

    /// Vector similarity search using brute-force cosine similarity.
    /// (No pgvector — SQLite stores embeddings as BLOB, searched in Rust.)
    pub async fn search_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<(String, f32)> {
        let rows = sqlx::query("SELECT id, embedding FROM engram_memories WHERE embedding IS NOT NULL")
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();

        let mut scored: Vec<(String, f32)> = rows
            .iter()
            .filter_map(|row| {
                let id: String = row.get("id");
                let blob: Vec<u8> = row.get("embedding");
                let emb = blob_to_f32(&blob);
                let score = cosine_similarity(query_embedding, &emb);
                if score >= min_score {
                    Some((id, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    // --- Hebbian ---

    pub async fn get_hebbian_neighbors(
        &self,
        memory_id: &str,
    ) -> Result<Vec<String>, sqlx::Error> {
        let rows = sqlx::query(
            r#"SELECT target_id AS neighbor FROM engram_hebbian_links WHERE source_id = $1 AND weight > 0.1
               UNION
               SELECT source_id AS neighbor FROM engram_hebbian_links WHERE target_id = $1 AND weight > 0.1"#,
        )
        .bind(memory_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| r.get("neighbor")).collect())
    }

    pub async fn record_coactivation(
        &self,
        ids: &[String],
        min_weight: f64,
    ) -> Result<(), sqlx::Error> {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = if ids[i] < ids[j] {
                    (&ids[i], &ids[j])
                } else {
                    (&ids[j], &ids[i])
                };
                sqlx::query(
                    r#"INSERT INTO engram_hebbian_links (source_id, target_id, weight)
                       VALUES ($1, $2, $3)
                       ON CONFLICT (source_id, target_id)
                       DO UPDATE SET weight = MIN(engram_hebbian_links.weight + $3, 1.0)"#,
                )
                .bind(a)
                .bind(b)
                .bind(min_weight)
                .execute(&self.pool)
                .await?;
            }
        }
        Ok(())
    }

    pub async fn decay_hebbian_links(&self, factor: f64) -> Result<usize, sqlx::Error> {
        let result = sqlx::query(
            "UPDATE engram_hebbian_links SET weight = weight * $1 WHERE weight > 0.01",
        )
        .bind(factor)
        .execute(&self.pool)
        .await?;

        sqlx::query("DELETE FROM engram_hebbian_links WHERE weight < 0.01")
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() as usize)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

// --- Helpers ---

fn blob_to_f32(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

fn row_to_record(row: &SqliteRow, access_times: Vec<DateTime<Utc>>) -> MemoryRecord {
    let memory_type_str: String = row.get("memory_type");
    let layer_str: String = row.get("layer");
    let created_str: String = row.get("created_at");
    let last_consol_str: Option<String> = row.get("last_consolidated");
    let metadata_str: Option<String> = row.get("metadata");
    let embedding_blob: Option<Vec<u8>> = row.get("embedding");

    let memory_type = match memory_type_str.as_str() {
        "factual" => MemoryType::Factual,
        "episodic" => MemoryType::Episodic,
        "relational" => MemoryType::Relational,
        "emotional" => MemoryType::Emotional,
        "procedural" => MemoryType::Procedural,
        "opinion" => MemoryType::Opinion,
        "causal" => MemoryType::Causal,
        _ => MemoryType::Factual,
    };

    let layer = match layer_str.as_str() {
        "core" => MemoryLayer::Core,
        "working" => MemoryLayer::Working,
        "archive" => MemoryLayer::Archive,
        _ => MemoryLayer::Working,
    };

    let pinned_int: i32 = row.get("pinned");

    MemoryRecord {
        id: row.get("id"),
        content: row.get("content"),
        memory_type,
        layer,
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        access_times,
        working_strength: row.get("working_strength"),
        core_strength: row.get("core_strength"),
        importance: row.get("importance"),
        pinned: pinned_int != 0,
        consolidation_count: row.get("consolidation_count"),
        last_consolidated: last_consol_str
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&Utc))),
        source: row.get("source"),
        contradicts: row.get("contradicts"),
        contradicted_by: row.get("contradicted_by"),
        metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
        embedding: embedding_blob.map(|b| blob_to_f32(&b)),
    }
}
