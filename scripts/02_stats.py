#!/usr/bin/env python3
"""Stage 2 — live generation stats (pass/fail batches + validation rejects).

Run while generation is active:

    python scripts/02_stats.py
    python scripts/02_stats.py --once
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_scripts.stats import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
