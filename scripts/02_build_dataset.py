#!/usr/bin/env python3
"""Stage 2 — validate, dedupe, stratified-split, build vocab + filler dictionary.

    python scripts/02_build_dataset.py --val 0.1 --test 0.1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_scripts.build_dataset import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
