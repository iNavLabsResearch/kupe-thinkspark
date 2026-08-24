#!/usr/bin/env python3
"""SQLite mirror of ThinkSpark generation trackers — fast stats / resume."""

from __future__ import annotations

import csv
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config.paths import BATCH_CSV, GEN_COST_CSV, RAW_CORPUS, ROW_CSV, SQLITE_PATH, TRACKER_DIR

_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def connect(path: Path = SQLITE_PATH) -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    with _LOCK:
        conn = sqlite3.connect(str(path), timeout=60)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    if own:
        ctx = connect()
        conn = ctx.__enter__()
    assert conn is not None
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS batches (
            batch_id INTEGER PRIMARY KEY,
            batch_num INTEGER,
            language TEXT,
            topic TEXT,
            register_mix TEXT,
            focus TEXT,
            status TEXT,
            stream_tag TEXT,
            model TEXT,
            items_requested INTEGER,
            items_kept INTEGER,
            items_rejected INTEGER,
            prompt_tokens INTEGER,
            cached_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            cache_hit_pct REAL,
            cost_inr REAL,
            cum_cost_inr REAL,
            error_type TEXT,
            error_message TEXT,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
        CREATE INDEX IF NOT EXISTS idx_batches_lang ON batches(language);

        CREATE TABLE IF NOT EXISTS rows (
            row_id INTEGER PRIMARY KEY,
            language TEXT,
            batch_id INTEGER,
            status TEXT,
            fail_reason TEXT,
            intent TEXT,
            input_preview TEXT,
            jsonl_line INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_rows_lang ON rows(language);
        CREATE INDEX IF NOT EXISTS idx_rows_status ON rows(status);
        CREATE INDEX IF NOT EXISTS idx_rows_batch ON rows(batch_id);

        CREATE TABLE IF NOT EXISTS lang_stats (
            language TEXT PRIMARY KEY,
            target INTEGER,
            kept INTEGER,
            rejected INTEGER,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    if own:
        ctx.__exit__(None, None, None)


def upsert_batch(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = [
        "batch_id", "batch_num", "language", "topic", "register_mix", "focus",
        "status", "stream_tag", "model", "items_requested", "items_kept",
        "items_rejected", "prompt_tokens", "cached_tokens", "completion_tokens",
        "total_tokens", "cache_hit_pct", "cost_inr", "cum_cost_inr",
        "error_type", "error_message", "started_at", "finished_at", "updated_at",
    ]
    data = {c: row.get(c) for c in cols}
    data["updated_at"] = data.get("updated_at") or utc_now()
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "batch_id")
    conn.execute(
        f"INSERT INTO batches ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(batch_id) DO UPDATE SET {updates}",
        [data[c] for c in cols],
    )


def upsert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = [
        "row_id", "language", "batch_id", "status", "fail_reason", "intent",
        "input_preview", "jsonl_line", "created_at", "updated_at",
    ]
    data = {c: row.get(c) for c in cols}
    data["updated_at"] = data.get("updated_at") or utc_now()
    if not data.get("created_at"):
        data["created_at"] = data["updated_at"]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "row_id")
    conn.execute(
        f"INSERT INTO rows ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(row_id) DO UPDATE SET {updates}",
        [data[c] for c in cols],
    )


def upsert_lang_stat(conn: sqlite3.Connection, lang: str, *, target: int, kept: int, rejected: int) -> None:
    conn.execute(
        """
        INSERT INTO lang_stats (language, target, kept, rejected, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(language) DO UPDATE SET
            target=excluded.target,
            kept=excluded.kept,
            rejected=excluded.rejected,
            updated_at=excluded.updated_at
        """,
        (lang, target, kept, rejected, utc_now()),
    )


def next_batch_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(batch_id), 0) + 1 FROM batches").fetchone()
    return int(row[0])


def next_row_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(row_id), 0) + 1 FROM rows").fetchone()
    return int(row[0])


def max_batch_num(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(batch_num), 0) FROM batches").fetchone()
    return int(row[0])


def lang_kept_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT language, COUNT(*) AS n FROM rows WHERE status='kept' GROUP BY language"
    ).fetchall()
    return {str(r["language"]): int(r["n"]) for r in rows}


def lang_rejected_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT language, COUNT(*) AS n FROM rows WHERE status='rejected' GROUP BY language"
    ).fetchall()
    return {str(r["language"]): int(r["n"]) for r in rows}


def batch_status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM batches GROUP BY status"
    ).fetchall()
    return {str(r["status"] or "unknown"): int(r["n"]) for r in rows}


def reject_reason_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT fail_reason, COUNT(*) AS n FROM rows "
        "WHERE status='rejected' AND fail_reason IS NOT NULL AND fail_reason != '' "
        "GROUP BY fail_reason ORDER BY n DESC"
    ).fetchall()
    return {str(r["fail_reason"]): int(r["n"]) for r in rows}


def recent_failed_batches(conn: sqlite3.Connection, limit: int = 8) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT batch_id, batch_num, language, topic, stream_tag, error_type, error_message, finished_at
        FROM batches
        WHERE status='failed'
        ORDER BY batch_id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def summarize(conn: sqlite3.Connection, targets: dict[str, int] | None = None) -> dict[str, Any]:
    targets = targets or {}
    kept = lang_kept_counts(conn)
    rejected = lang_rejected_counts(conn)
    langs = sorted(set(kept) | set(rejected) | set(targets))
    lang_rows = []
    for lang in langs:
        k = kept.get(lang, 0)
        r = rejected.get(lang, 0)
        tgt = targets.get(lang, 0)
        lang_rows.append({
            "language": lang,
            "kept": k,
            "rejected": r,
            "target": tgt,
            "left": max(tgt - k, 0),
            "pass_rate": (100.0 * k / (k + r)) if (k + r) else 0.0,
        })
    cost_row = conn.execute(
        "SELECT COALESCE(MAX(cum_cost_inr), 0) AS cum FROM batches"
    ).fetchone()
    return {
        "langs": lang_rows,
        "batch_status": batch_status_counts(conn),
        "reject_reasons": reject_reason_counts(conn),
        "recent_failed": recent_failed_batches(conn),
        "total_kept": sum(kept.values()),
        "total_rejected": sum(rejected.values()),
        "cum_cost_inr": float(cost_row["cum"] or 0) if cost_row else 0.0,
        "max_batch_num": max_batch_num(conn),
        "max_row_id": next_row_id(conn) - 1,
    }


def _csv_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def needs_csv_import() -> bool:
    if not SQLITE_PATH.exists():
        return True
    db_mtime = SQLITE_PATH.stat().st_mtime
    for path in (BATCH_CSV, ROW_CSV, GEN_COST_CSV, RAW_CORPUS):
        if path.exists() and path.stat().st_mtime > db_mtime:
            return True
    return False


def import_batch_csv(conn: sqlite3.Connection, path: Path = BATCH_CSV) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            upsert_batch(conn, {
                "batch_id": int(row.get("batch_id") or 0),
                "batch_num": int(row.get("batch_num") or 0),
                "language": row.get("language") or "",
                "topic": row.get("topic") or "",
                "register_mix": row.get("register_mix") or "",
                "focus": row.get("focus") or "",
                "status": row.get("status") or "done",
                "stream_tag": row.get("stream_tag") or "",
                "model": row.get("model") or "",
                "items_requested": int(row.get("items_requested") or 0),
                "items_kept": int(row.get("items_kept") or 0),
                "items_rejected": int(row.get("items_rejected") or 0),
                "prompt_tokens": int(row.get("prompt_tokens") or 0),
                "cached_tokens": int(row.get("cached_tokens") or 0),
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "total_tokens": int(row.get("total_tokens") or 0),
                "cache_hit_pct": float(row.get("cache_hit_pct") or 0),
                "cost_inr": float(row.get("cost_inr") or 0),
                "cum_cost_inr": float(row.get("cum_cost_inr") or 0),
                "error_type": row.get("error_type") or "",
                "error_message": row.get("error_message") or "",
                "started_at": row.get("started_at") or "",
                "finished_at": row.get("finished_at") or row.get("timestamp_utc") or "",
            })
            n += 1
    return n


def import_row_csv(conn: sqlite3.Connection, path: Path = ROW_CSV) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            upsert_row(conn, {
                "row_id": int(row.get("row_id") or 0),
                "language": row.get("language") or "",
                "batch_id": int(row.get("batch_id") or 0) or None,
                "status": row.get("status") or "kept",
                "fail_reason": row.get("fail_reason") or "",
                "intent": row.get("intent") or "",
                "input_preview": row.get("input_preview") or "",
                "jsonl_line": int(row.get("jsonl_line") or 0) or None,
                "created_at": row.get("created_at") or "",
            })
            n += 1
    return n


def import_cost_csv(conn: sqlite3.Connection, path: Path = GEN_COST_CSV) -> int:
    """Backfill batches from legacy generation_costs.csv when batch_tracker is empty."""
    if conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]:
        return 0
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            upsert_batch(conn, {
                "batch_id": i,
                "batch_num": int(row.get("batch_num") or i),
                "language": row.get("language") or "",
                "topic": "",
                "register_mix": row.get("register_mix") or "",
                "focus": row.get("focus") or "",
                "status": "done",
                "stream_tag": "",
                "model": "",
                "items_requested": 0,
                "items_kept": int(row.get("items_kept") or 0),
                "items_rejected": 0,
                "prompt_tokens": int(row.get("prompt_tokens") or 0),
                "cached_tokens": int(row.get("cached_tokens") or 0),
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "total_tokens": int(row.get("total_tokens") or 0),
                "cache_hit_pct": float(row.get("cache_hit_pct") or 0),
                "cost_inr": float(row.get("cost_inr") or 0),
                "cum_cost_inr": float(row.get("cum_cost_inr") or 0),
                "finished_at": row.get("timestamp_utc") or "",
            })
            n += 1
    return n


def sync_lang_stats(conn: sqlite3.Connection, targets: dict[str, int]) -> None:
    kept = lang_kept_counts(conn)
    rejected = lang_rejected_counts(conn)
    langs = set(targets) | set(kept) | set(rejected)
    for lang in langs:
        upsert_lang_stat(
            conn, lang,
            target=targets.get(lang, 0),
            kept=kept.get(lang, 0),
            rejected=rejected.get(lang, 0),
        )
