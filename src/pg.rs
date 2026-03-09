//! PostgreSQL storage backend via SQLx.
//!
//! Shared across machines. Uses pgvector for embedding search.
//! Enable with `features = ["postgres"]`.

#![cfg(feature = "postgres")]

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row, postgres::PgRow};

use crate::types::{MemoryLayer, MemoryRecord, MemoryType};

/// PostgreSQL-backed memory storage with pgvector support.
pub struct PgStore {
    pool: PgPool,
}

impl PgStore {
    /// Connect to PostgreSQL and run migrations.
    pub async fn new(database_url: &str) -> Result<Self, sqlx::Error> {
        let pool = PgPool::connect(database_url).await?;
        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    /// Create from an existing pool.
    pub fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    async fn migrate(&self) -> Result<(), sqlx::Error> {
        // Core schema — pgvector extension + tables
        sqlx::raw_sql(
            r#"
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS engram_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                working_strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                core_strength DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                importance DOUBLE PRECISION NOT NULL DEFAULT 0.3,
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                consolidation_count INTEGER NOT NULL DEFAULT 0,
                last_consolidated TIMESTAMPTZ,
                source TEXT DEFAULT '',
                contradicts TEXT,
                contradicted_by TEXT,
                metadata JSONB,
                embedding vector(768),
                ts_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            );

            CREATE INDEX IF NOT EXISTS idx_engram_memories_ts ON engram_memories USING GIN(ts_content);
            CREATE INDEX IF NOT EXISTS idx_engram_memories_type ON engram_memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_engram_memories_layer ON engram_memories(layer);
            CREATE INDEX IF NOT EXISTS idx_engram_memories_embedding ON engram_memories
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

            CREATE TABLE IF NOT EXISTS engram_access_log (
                id SERIAL PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES engram_memories(id) ON DELETE CASCADE,
                accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_engram_access_memory ON engram_access_log(memory_id);

            CREATE TABLE IF NOT EXISTS engram_hebbian_links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                PRIMARY KEY (source_id, target_id)
            );
            "#,
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // --- CRUD ---

    pub async fn add(&self, record: &MemoryRecord) -> Result<(), sqlx::Error> {
        let embedding_str = record.embedding.as_ref().map(|v| {
            format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
        });

        sqlx::query(
            r#"INSERT INTO engram_memories
                (id, content, memory_type, layer, created_at, working_strength, core_strength,
                 importance, pinned, consolidation_count, last_consolidated, source,
                 contradicts, contradicted_by, metadata, embedding)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                       CASE WHEN $16::text IS NOT NULL THEN $16::vector ELSE NULL END)"#,
        )
        .bind(&record.id)
        .bind(&record.content)
        .bind(record.memory_type.to_string())
        .bind(record.layer.to_string())
        .bind(record.created_at)
        .bind(record.working_strength)
        .bind(record.core_strength)
        .bind(record.importance)
        .bind(record.pinned)
        .bind(record.consolidation_count)
        .bind(record.last_consolidated)
        .bind(&record.source)
        .bind(&record.contradicts)
        .bind(&record.contradicted_by)
        .bind(&record.metadata)
        .bind(&embedding_str)
        .execute(&self.pool)
        .await?;

        // Insert initial access time
        sqlx::query("INSERT INTO engram_access_log (memory_id, accessed_at) VALUES ($1, $2)")
            .bind(&record.id)
            .bind(record.created_at)
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
                Ok(Some(row_to_record(&row, access_times)?))
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
            records.push(row_to_record(&row, access_times)?);
        }
        Ok(records)
    }

    pub async fn update(&self, record: &MemoryRecord) -> Result<(), sqlx::Error> {
        let embedding_str = record.embedding.as_ref().map(|v| {
            format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
        });

        sqlx::query(
            r#"UPDATE engram_memories SET
                content = $2, memory_type = $3, layer = $4,
                working_strength = $5, core_strength = $6, importance = $7,
                pinned = $8, consolidation_count = $9, last_consolidated = $10,
                source = $11, contradicts = $12, contradicted_by = $13, metadata = $14,
                embedding = CASE WHEN $15::text IS NOT NULL THEN $15::vector ELSE embedding END
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
        .bind(record.last_consolidated)
        .bind(&record.source)
        .bind(&record.contradicts)
        .bind(&record.contradicted_by)
        .bind(&record.metadata)
        .bind(&embedding_str)
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
        sqlx::query("INSERT INTO engram_access_log (memory_id) VALUES ($1)")
            .bind(id)
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

        Ok(rows.iter().map(|r| r.get("accessed_at")).collect())
    }

    // --- Search ---

    /// Full-text search using PostgreSQL ts_vector.
    pub async fn search_fts(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryRecord>, sqlx::Error> {
        let tsquery = query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" | ");

        let rows = sqlx::query(
            r#"SELECT *, ts_rank(ts_content, to_tsquery('english', $1)) AS rank
               FROM engram_memories
               WHERE ts_content @@ to_tsquery('english', $1)
               ORDER BY rank DESC
               LIMIT $2"#,
        )
        .bind(&tsquery)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let mut records = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
            let access_times = self.get_access_times(&id).await?;
            records.push(row_to_record(&row, access_times)?);
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
            records.push(row_to_record(&row, access_times)?);
        }
        Ok(records)
    }

    /// Vector similarity search using pgvector cosine distance.
    pub async fn search_vector(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_score: f32,
    ) -> Vec<(String, f32)> {
        let emb_str = format!(
            "[{}]",
            query_embedding.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")
        );

        // pgvector: 1 - cosine_distance = cosine_similarity
        let rows = sqlx::query(
            r#"SELECT id, 1 - (embedding <=> $1::vector) AS score
               FROM engram_memories
               WHERE embedding IS NOT NULL
               ORDER BY embedding <=> $1::vector
               LIMIT $2"#,
        )
        .bind(&emb_str)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        rows.iter()
            .filter_map(|r| {
                let id: String = r.get("id");
                let score: f64 = r.get("score");
                if score as f32 >= min_score {
                    Some((id, score as f32))
                } else {
                    None
                }
            })
            .collect()
    }

    // --- Hebbian ---

    pub async fn get_hebbian_neighbors(
        &self,
        memory_id: &str,
    ) -> Result<Vec<String>, sqlx::Error> {
        let rows = sqlx::query(
            r#"SELECT target_id FROM engram_hebbian_links WHERE source_id = $1 AND weight > 0.1
               UNION
               SELECT source_id FROM engram_hebbian_links WHERE target_id = $1 AND weight > 0.1"#,
        )
        .bind(memory_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter().map(|r| {
            let col: String = r.try_get("target_id").unwrap_or_else(|_| r.get("source_id"));
            col
        }).collect())
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
                       DO UPDATE SET weight = LEAST(engram_hebbian_links.weight + $3, 1.0)"#,
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

        // Clean up near-zero links
        sqlx::query("DELETE FROM engram_hebbian_links WHERE weight < 0.01")
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() as usize)
    }

    /// Get the connection pool for direct access.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

// --- Row conversion ---

fn row_to_record(row: &PgRow, access_times: Vec<DateTime<Utc>>) -> Result<MemoryRecord, sqlx::Error> {
    let memory_type_str: String = row.get("memory_type");
    let layer_str: String = row.get("layer");

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

    Ok(MemoryRecord {
        id: row.get("id"),
        content: row.get("content"),
        memory_type,
        layer,
        created_at: row.get("created_at"),
        access_times,
        working_strength: row.get("working_strength"),
        core_strength: row.get("core_strength"),
        importance: row.get("importance"),
        pinned: row.get("pinned"),
        consolidation_count: row.get("consolidation_count"),
        last_consolidated: row.get("last_consolidated"),
        source: row.get("source"),
        contradicts: row.get("contradicts"),
        contradicted_by: row.get("contradicted_by"),
        metadata: row.get("metadata"),
        embedding: None, // pgvector type not directly deserializable to Vec<f32>
    })
}
