#!/usr/bin/env python3
"""Stage 1 — generate synthetic training data via Sarvam (SSE streaming).

Resume is default — continues from indexed JSONL + SQLite tracker.
Use --fresh to wipe and restart.

Run stats in another terminal:
    python scripts/02_stats.py

Examples:
    python scripts/01_generate_data.py
    python scripts/01_generate_data.py --concurrency 50
    python scripts/01_generate_data.py --langs hi,as
    python scripts/01_generate_data.py --fresh
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_scripts.generate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
