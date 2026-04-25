"""Schema and migrations for the SQLite memory store."""

MEMORY_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    embedding_id TEXT,
    somatic_tag TEXT,
    affect_primary TEXT,
    affect_valence REAL,
    affect_arousal REAL,
    affect_certainty REAL,
    affect_intensity REAL,
    affect_social_target TEXT,
    affect_tags TEXT,
    salience REAL NOT NULL DEFAULT 1.0,
    compacted INTEGER NOT NULL DEFAULT 0,
    identity_core INTEGER NOT NULL DEFAULT 0,
    confidence_score REAL NOT NULL DEFAULT 0.8,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    used_successfully INTEGER NOT NULL DEFAULT 0,
    used_unsuccessfully INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_compacted ON episodes(compacted);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_id TEXT,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    salience REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience);

CREATE TABLE IF NOT EXISTS compactions (
    compaction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    episode_ids TEXT NOT NULL,
    summary TEXT NOT NULL,
    removed_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_edges (
    edge_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'semantic',
    semantic_weight REAL NOT NULL DEFAULT 0.0,
    emotional_weight REAL NOT NULL DEFAULT 0.0,
    recency_weight REAL NOT NULL DEFAULT 0.0,
    structural_weight REAL NOT NULL DEFAULT 0.0,
    salience_weight REAL NOT NULL DEFAULT 0.0,
    causal_weight REAL NOT NULL DEFAULT 0.0,
    verification_weight REAL NOT NULL DEFAULT 0.0,
    actor_affinity_weight REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, target_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON episode_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON episode_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_confidence ON episode_edges(confidence);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON episode_edges(kind);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content,
    content='episodes',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS episodes_fts_insert AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_delete AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_update AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO episodes_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

MEMORY_STORE_MIGRATIONS = [
    "ALTER TABLE episodes ADD COLUMN affect_primary TEXT",
    "ALTER TABLE episodes ADD COLUMN affect_valence REAL",
    "ALTER TABLE episodes ADD COLUMN affect_arousal REAL",
    "ALTER TABLE episodes ADD COLUMN affect_certainty REAL",
    "ALTER TABLE episodes ADD COLUMN affect_intensity REAL",
    "ALTER TABLE episodes ADD COLUMN affect_social_target TEXT",
    "ALTER TABLE episodes ADD COLUMN affect_tags TEXT",
    "ALTER TABLE episodes ADD COLUMN identity_core INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE episode_edges ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'",
    "ALTER TABLE episodes ADD COLUMN confidence_score REAL NOT NULL DEFAULT 0.8",
    "ALTER TABLE episodes ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE episodes ADD COLUMN last_accessed TEXT",
    "ALTER TABLE episodes ADD COLUMN used_successfully INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE episodes ADD COLUMN used_unsuccessfully INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE episode_edges ADD COLUMN salience_weight REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE episode_edges ADD COLUMN causal_weight REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE episode_edges ADD COLUMN verification_weight REAL NOT NULL DEFAULT 0.0",
    "ALTER TABLE episode_edges ADD COLUMN actor_affinity_weight REAL NOT NULL DEFAULT 0.0",
]
