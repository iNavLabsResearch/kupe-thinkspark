#!/usr/bin/env python3
"""SSE data generation entrypoint (Sarvam · gemma4).

Resume is default — indexes existing JSONL into SQLite and continues toward targets.
Use --fresh only to wipe corpus + tracker and start over.

    python data_scripts/generate.py
    python data_scripts/generate.py --langs hi,as
    python data_scripts/generate.py --fresh

Run stats in another terminal:
    python data_scripts/stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_scripts.data_gen_agent import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
