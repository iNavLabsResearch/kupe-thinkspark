#!/usr/bin/env python3
"""CSV trackers for ThinkSpark generation (source of truth) + SQLite mirror."""

from __future__ import annotations

import csv
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config.paths import (
    BATCH_CSV, GEN_COST_CSV, RAW_CORPUS, ROW_CSV, RUN_STATE_JSON, SQLITE_PATH, TRACKER_DIR,
)
from config.taxonomy import LANGUAGES

from data_scripts import db as dbmod

BATCH_FIELDS = [
    "batch_id", "batch_num", "language", "topic", "register_mix", "focus",
    "status", "stream_tag", "model", "items_requested", "items_kept", "items_rejected",
    "prompt_tokens", "cached_tokens", "completion_tokens", "total_tokens",
    "cache_hit_pct", "cost_inr", "cum_cost_inr",
    "error_type", "error_message", "started_at", "finished_at", "timestamp_utc",
]

ROW_FIELDS = [
    "row_id", "language", "batch_id", "status", "fail_reason", "intent",
    "input_preview", "jsonl_line", "created_at",
]


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def init() -> None:
    dbmod.ensure_dirs()
    dbmod.init_db()
    for path, fields in ((BATCH_CSV, BATCH_FIELDS), (ROW_CSV, ROW_FIELDS)):
        if not path.exists() or path.stat().st_size == 0:
            _write_csv(path, fields, [])


def _mirror_batch(row: dict[str, Any]) -> None:
    try:
        with dbmod.connect() as conn:
            dbmod.upsert_batch(conn, row)
    except Exception:
        pass


def _mirror_row(row: dict[str, Any]) -> None:
    try:
        with dbmod.connect() as conn:
            dbmod.upsert_row(conn, row)
    except Exception:
        pass


def upsert_batch(row: dict[str, Any]) -> None:
    row = dict(row)
    row["timestamp_utc"] = row.get("timestamp_utc") or row.get("finished_at") or dbmod.utc_now()
    bid = int(row.get("batch_id") or 0)
    with _locked(BATCH_CSV):
        rows = _read_csv(BATCH_CSV)
        merged = {int(r.get("batch_id") or 0): r for r in rows if r.get("batch_id")}
        merged[bid] = {k: str(row.get(k, "")) for k in BATCH_FIELDS}
        _write_csv(BATCH_CSV, BATCH_FIELDS, sorted(merged.values(), key=lambda r: int(r["batch_id"])))
    _mirror_batch(row)


def upsert_row(row: dict[str, Any]) -> None:
    row = dict(row)
    rid = int(row.get("row_id") or 0)
    if not row.get("created_at"):
        row["created_at"] = dbmod.utc_now()
    with _locked(ROW_CSV):
        rows = _read_csv(ROW_CSV)
        merged = {int(r.get("row_id") or 0): r for r in rows if r.get("row_id")}
        merged[rid] = {k: str(row.get(k, "")) for k in ROW_FIELDS}
        _write_csv(ROW_CSV, ROW_FIELDS, sorted(merged.values(), key=lambda r: int(r["row_id"])))
    _mirror_row(row)


def begin_batch(
    *, batch_id: int, batch_num: int, language: str, topic: str,
    register_mix: str, focus: list[str], stream_tag: str, model: str,
    items_requested: int,
) -> None:
    upsert_batch({
        "batch_id": batch_id,
        "batch_num": batch_num,
        "language": language,
        "topic": topic,
        "register_mix": register_mix,
        "focus": "|".join(focus),
        "status": "in_progress",
        "stream_tag": stream_tag,
        "model": model,
        "items_requested": items_requested,
        "items_kept": 0,
        "items_rejected": 0,
        "started_at": dbmod.utc_now(),
    })


def finish_batch(batch_id: int, row: dict[str, Any]) -> None:
    payload = dict(row)
    payload["batch_id"] = batch_id
    payload["status"] = payload.get("status") or "done"
    payload["finished_at"] = payload.get("finished_at") or dbmod.utc_now()
    upsert_batch(payload)


def record_row(
    *, row_id: int, language: str, batch_id: int | None, status: str,
    fail_reason: str = "", intent: str = "", input_preview: str = "",
    jsonl_line: int | None = None,
) -> None:
    upsert_row({
        "row_id": row_id,
        "language": language,
        "batch_id": batch_id or "",
        "status": status,
        "fail_reason": fail_reason,
        "intent": intent,
        "input_preview": input_preview[:120],
        "jsonl_line": jsonl_line or "",
        "created_at": dbmod.utc_now(),
    })


def allocate_batch_id() -> int:
    with dbmod.connect() as conn:
        return dbmod.next_batch_id(conn)


def allocate_row_id() -> int:
    with dbmod.connect() as conn:
        return dbmod.next_row_id(conn)


def max_batch_num() -> int:
    with dbmod.connect() as conn:
        return dbmod.max_batch_num(conn)


def lang_kept_counts() -> dict[str, int]:
    with dbmod.connect() as conn:
        counts = dbmod.lang_kept_counts(conn)
    for code in LANGUAGES:
        counts.setdefault(code, 0)
    return counts


def lang_rejected_counts() -> dict[str, int]:
    with dbmod.connect() as conn:
        return dbmod.lang_rejected_counts(conn)


def total_corpus_rows() -> int:
    with dbmod.connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM rows WHERE status='kept'").fetchone()
        return int(row[0])


def summarize(targets: dict[str, int] | None = None) -> dict[str, Any]:
    sync_sqlite(targets)
    with dbmod.connect() as conn:
        return dbmod.summarize(conn, targets)


def sync_sqlite(targets: dict[str, int] | None = None) -> None:
    init()
    with dbmod.connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0:
            dbmod.import_cost_csv(conn, GEN_COST_CSV)
        if dbmod.needs_csv_import():
            dbmod.import_row_csv(conn)
            dbmod.import_batch_csv(conn)
        if targets:
            dbmod.sync_lang_stats(conn, targets)


def hydrate_from_corpus(corpus_path: Path = RAW_CORPUS, targets: dict[str, int] | None = None) -> int:
    """Index existing JSONL rows into row_tracker + SQLite (resume-safe)."""
    init()
    targets = targets or {}
    if not corpus_path.exists():
        sync_sqlite(targets)
        return 0

    with dbmod.connect() as conn:
        existing_lines = {
            int(r["jsonl_line"])
            for r in conn.execute(
                "SELECT jsonl_line FROM rows WHERE status='kept' AND jsonl_line IS NOT NULL"
            ).fetchall()
            if r["jsonl_line"]
        }
        next_id = dbmod.next_row_id(conn)
        added = 0
        with corpus_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line_no in existing_lines:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lang = rec.get("language") or ""
                upsert_row({
                    "row_id": next_id,
                    "language": lang,
                    "batch_id": "",
                    "status": "kept",
                    "fail_reason": "",
                    "intent": rec.get("intent") or "",
                    "input_preview": (rec.get("input") or "")[:120],
                    "jsonl_line": line_no,
                    "created_at": dbmod.utc_now(),
                })
                next_id += 1
                added += 1
        if targets:
            dbmod.sync_lang_stats(conn, targets)
    return added


def write_run_state(payload: dict[str, Any]) -> None:
    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STATE_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_all() -> None:
    for path in (BATCH_CSV, ROW_CSV, SQLITE_PATH, RUN_STATE_JSON):
        if path.exists():
            path.unlink()
    for name in ("batch_tracker.csv.lock", "row_tracker.csv.lock"):
        lock = TRACKER_DIR / name
        if lock.exists():
            lock.unlink()
    init()
