"""
database.py: a small SQLite database that records

  uploads        every image a user uploads for retraining (file, label, whether it
                 has been used in a retraining run yet)
  training_runs  every retraining run (trigger, status, metrics before and after,
                 whether the new model was promoted)
  predictions    every prediction (label, confidence, latency) for monitoring

SQLite is a single file (data/cassava.db), needs no server, and is part of Python.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    used_in_training INTEGER NOT NULL DEFAULT 0,
    training_run_id INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_new_images INTEGER,
    n_replay_images INTEGER,
    metrics_before TEXT,
    metrics_after TEXT,
    promoted INTEGER,
    model_version INTEGER,
    message TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    latency_ms REAL,
    model_version INTEGER
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")   # lets reads and writes happen together
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)


# ---- uploads ---------------------------------------------------------------
def add_upload(original_name, label, path):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO uploads (original_name, label, path, uploaded_at) VALUES (?,?,?,?)",
            (original_name, label, str(path), now()))
        return cur.lastrowid


def pending_uploads():
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM uploads WHERE used_in_training = 0 ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def upload_counts():
    with connect() as con:
        rows = con.execute(
            "SELECT label, used_in_training, COUNT(*) AS n FROM uploads GROUP BY label, used_in_training"
        ).fetchall()
    out = {"pending": {}, "used": {}}
    for r in rows:
        out["used" if r["used_in_training"] else "pending"][r["label"]] = r["n"]
    out["pending_total"] = sum(out["pending"].values())
    out["used_total"] = sum(out["used"].values())
    return out


def mark_uploads_used(ids, run_id, new_paths=None):
    with connect() as con:
        for i, upload_id in enumerate(ids):
            if new_paths:
                con.execute("UPDATE uploads SET used_in_training=1, training_run_id=?, path=? WHERE id=?",
                            (run_id, str(new_paths[i]), upload_id))
            else:
                con.execute("UPDATE uploads SET used_in_training=1, training_run_id=? WHERE id=?",
                            (run_id, upload_id))


#  TRAINING RUNS 
def start_run(trigger):
    with connect() as con:
        cur = con.execute(
            "INSERT INTO training_runs (trigger, status, started_at) VALUES (?, 'running', ?)",
            (trigger, now()))
        return cur.lastrowid


def finish_run(run_id, status, message="", **fields):
    for key in ("metrics_before", "metrics_after"):
        if key in fields and fields[key] is not None:
            fields[key] = json.dumps(fields[key])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(
            f"UPDATE training_runs SET status=?, finished_at=?, message=?{', ' + cols if cols else ''} WHERE id=?",
            (status, now(), message, *fields.values(), run_id))


def list_runs(limit=20):
    with connect() as con:
        rows = con.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("metrics_before", "metrics_after"):
            if d.get(key):
                d[key] = json.loads(d[key])
        out.append(d)
    return out


#  PREDICTIONS 
def log_prediction(label, confidence, latency_ms, model_version):
    with connect() as con:
        con.execute(
            "INSERT INTO predictions (ts, label, confidence, latency_ms, model_version) VALUES (?,?,?,?,?)",
            (now(), label, confidence, latency_ms, model_version))


def prediction_summary():
    with connect() as con:
        total = con.execute("SELECT COUNT(*), AVG(latency_ms) FROM predictions").fetchone()
        by_label = con.execute(
            "SELECT label, COUNT(*) AS n FROM predictions GROUP BY label").fetchall()
    return {"total": total[0], "avg_latency_ms": round(total[1] or 0, 1),
            "by_label": {r["label"]: r["n"] for r in by_label}}
