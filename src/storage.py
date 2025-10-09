import sqlite3
from pathlib import Path

DB_PATH = Path("data/ata.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE,
  name TEXT,
  club TEXT,
  state TEXT,
  start_date TEXT,
  end_date TEXT,
  last_seen TEXT
);

CREATE TABLE IF NOT EXISTS shooters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ata_no TEXT,             -- if present
  name TEXT,
  profile_url TEXT UNIQUE, -- if present
  last_seen TEXT
);

CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_url TEXT,          -- FK by URL (simplifies ingestion)
  shooter_profile TEXT,    -- FK by profile URL or name
  shooter_name TEXT,
  ata_no TEXT,
  discipline TEXT,         -- Singles/Handicap/Doubles
  yardage REAL,
  score INTEGER,
  targets INTEGER,
  class TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_results_event ON results(event_url);
CREATE INDEX IF NOT EXISTS idx_results_shooter ON results(shooter_profile);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as c:
        c.executescript(DDL)
