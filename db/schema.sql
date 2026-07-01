CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  outcome TEXT,              -- 'closed' | 'lost' | 'human-closed' | 'abandoned'
  turn_count INTEGER,
  user_turn_count INTEGER,
  drop_off_turn INTEGER,
  intervention_flag BOOLEAN,
  is_ai_conversation BOOLEAN,
  platform TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS turns (
  id TEXT PRIMARY KEY,       -- conversation_id + ':' + turn_index
  conversation_id TEXT REFERENCES conversations(id),
  turn_index INTEGER,
  role TEXT,                 -- 'agent' | 'client' | 'human' | 'system' | 'tool'
  raw_text TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  label TEXT,
  cluster_type TEXT,         -- 'failure' | 'faq'
  turn_count INTEGER
);

CREATE TABLE IF NOT EXISTS cluster_members (
  cluster_id TEXT REFERENCES clusters(id),
  turn_id TEXT REFERENCES turns(id),
  PRIMARY KEY (cluster_id, turn_id)
);

CREATE TABLE IF NOT EXISTS turn_coords (
  turn_id TEXT PRIMARY KEY REFERENCES turns(id),
  cluster_type TEXT,   -- 'failure' | 'faq'
  x REAL,
  y REAL,
  cluster_id TEXT      -- NULL if noise point
);

CREATE TABLE IF NOT EXISTS annotations (
  id TEXT PRIMARY KEY,
  conversation_id TEXT,
  failure_type TEXT,
  note TEXT,
  annotated_at TEXT
);

CREATE TABLE IF NOT EXISTS tags (
  id TEXT PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS turn_signals (
  turn_id TEXT PRIMARY KEY REFERENCES turns(id),
  conversation_id TEXT,
  turn_index INTEGER,
  role TEXT,                 -- 'client' | 'agent'
  intent TEXT,
  intent_confidence REAL,
  sentiment REAL,            -- client turns only; NULL for agent
  sentiment_confidence REAL,
  discovered INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turn_signals_conv ON turn_signals(conversation_id);
