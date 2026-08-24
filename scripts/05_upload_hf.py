#!/usr/bin/env python3
"""Upload built ThinkSpark dataset to Hugging Face.

Uses HF_TOKEN from kupe-tts/.env (same write token as kupe-tts uploads).

Examples:
    python scripts/05_upload_hf.py
    python scripts/05_upload_hf.py --include-raw
    python scripts/05_upload_hf.py --repo anuj-inavlabs/kupe-thinkspark
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_scripts.hf_upload import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
