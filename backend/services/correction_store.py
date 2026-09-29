"""Stores expert corrections as labelled training pairs.

- If DATABASE_URL is set (e.g. on Render, pointing at Neon Postgres): saved in the
  `corrections` table, so they survive restarts and redeploys.
- Otherwise (local development): appended to backend/data/corrections.jsonl.
"""
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "corrections.jsonl"
DATABASE_URL = os.getenv("DATABASE_URL")

FIELDS = ("source_text", "source_script", "model_output", "corrected_output",
          "engine", "language_code", "document_id")

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_text      TEXT NOT NULL,
    source_script    TEXT NOT NULL,
    model_output     TEXT NOT NULL,
    corrected_output TEXT NOT NULL,
    engine           TEXT NOT NULL,
    language_code    TEXT,
    document_id      TEXT
)
"""

_table_ready = False


def storage_backend() -> str:
    return "postgres" if DATABASE_URL else "file"


def _connect():
    import psycopg  # only needed when a database is configured
    return psycopg.connect(DATABASE_URL, connect_timeout=15)


def _ensure_table(conn):
    global _table_ready
    if not _table_ready:
        conn.execute(SCHEMA)
        _table_ready = True


# ---- Learned corrections -------------------------------------------------
# {(language_code, roman word lower-case): Counter({corrected_output: times chosen})}
_learned = None
MIN_AGREEMENT = 2   # corrections outside the model's own candidates need this many votes


def _remember(record: dict):
    if _learned is None or record.get("engine") != "indicxlit" or not record.get("language_code"):
        return
    key = (record["language_code"], record["source_text"].strip().lower())
    _learned[key][record["corrected_output"].strip()] += 1


def _load_learned():
    global _learned
    _learned = defaultdict(Counter)
    try:
        if DATABASE_URL:
            with _connect() as conn:
                _ensure_table(conn)
                rows = conn.execute("SELECT language_code, source_text, corrected_output, engine "
                                    "FROM corrections ORDER BY id").fetchall()
            rows = [dict(zip(("language_code", "source_text", "corrected_output", "engine"), r)) for r in rows]
        elif STORE.exists():
            with STORE.open(encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = []
        for r in rows:
            _remember(r)
    except Exception:
        logging.getLogger(__name__).exception("Could not load learned corrections")


def learned_for(language_code: str, word: str) -> list:
    """[(corrected_output, votes), ...] most-voted first, for this word in this language."""
    if _learned is None:
        _load_learned()
    return _learned.get((language_code, word.strip().lower()), Counter()).most_common()


def save_correction(record: dict) -> int:
    """Save one correction; returns the total number stored."""
    if DATABASE_URL:
        columns = ", ".join(FIELDS)
        placeholders = ", ".join(["%s"] * len(FIELDS))
        with _connect() as conn:  # commits automatically on success
            _ensure_table(conn)
            conn.execute(f"INSERT INTO corrections ({columns}) VALUES ({placeholders})",
                         [record.get(f) for f in FIELDS])
            total = conn.execute("SELECT count(*) FROM corrections").fetchone()[0]
        _remember(record)
        return total

    STORE.parent.mkdir(parents=True, exist_ok=True)
    row = {f: record.get(f) for f in FIELDS}
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "ts": time.time()}, ensure_ascii=False) + "\n")
    _remember(row)
    return count_corrections()


def count_corrections() -> int:
    if DATABASE_URL:
        with _connect() as conn:
            _ensure_table(conn)
            return conn.execute("SELECT count(*) FROM corrections").fetchone()[0]
    if not STORE.exists():
        return 0
    with STORE.open(encoding="utf-8") as f:
        return sum(1 for _ in f)
