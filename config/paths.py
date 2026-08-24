#!/usr/bin/env python3
"""Shared paths + generation plan for ThinkSpark."""

from __future__ import annotations

from pathlib import Path

# kupe-thinkspark/
ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_SPLITS = DATA / "splits"
DATA_VOCAB = DATA / "vocab"
REPORTS = ROOT / "reports"
ENV_FILE = ROOT / ".env"

# Raw generated corpus + running cost ledger (mirrors kupe-tts layout)
RAW_CORPUS = DATA_RAW / "thinkspark_corpus.jsonl"
GEN_COST_CSV = DATA_RAW / "generation_costs.csv"

# Indexed tracker (CSV source of truth + SQLite mirror — like kupe-tts/soniox)
TRACKER_DIR = DATA_RAW / "tracker"
BATCH_CSV = TRACKER_DIR / "batch_tracker.csv"
ROW_CSV = TRACKER_DIR / "row_tracker.csv"
SQLITE_PATH = TRACKER_DIR / "thinkspark.sqlite"
RUN_STATE_JSON = TRACKER_DIR / "run_state.json"

# Post-processing artifacts
CLEAN_CORPUS = DATA_RAW / "thinkspark_corpus_clean.jsonl"
TRAIN_JSONL = DATA_SPLITS / "train.jsonl"
VAL_JSONL = DATA_SPLITS / "val.jsonl"
TEST_JSONL = DATA_SPLITS / "test.jsonl"

# Vocab / label maps for training
LABELMAPS_JSON = DATA_VOCAB / "label_maps.json"
FILLER_DICT_JSON = DATA_VOCAB / "filler_dictionary.json"

# Reports
EDA_HTML = REPORTS / "data_eda_report.html"
TRAIN_CURVES_PNG = REPORTS / "training_curves.png"
CONFUSION_PNG = REPORTS / "confusion_intent.png"


def load_env() -> None:
    """Load ROOT/.env into os.environ (never overwrite an existing var)."""
    import os
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
