#!/usr/bin/env python3
"""End-to-end smoke test — NO API, runs in well under a minute on an M1.

Proves the whole ThinkSpark pipeline is wired correctly:
    1. offline synthetic corpus            (smoke_gen)
    2. validate / split / vocab            (build_dataset)
    3. train a few epochs on MPS/CPU       (trainer)
    4. load checkpoint + run inference     (ThinkSparkPredictor)

    python scripts/smoke_test.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(desc: str, *cmd: str) -> None:
    print(f"\n\033[1m=== {desc} ===\033[0m", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {desc}")


def main() -> int:
    run("1/4 generate synthetic corpus", "scripts/smoke_gen.py", "--per-cell", "8")
    run("2/4 build dataset (validate + split + vocab)",
        "scripts/02_build_dataset.py", "--val", "0.15", "--test", "0.15")
    run("3/4 train (smoke config)",
        "scripts/03_train.py", "--config", "configs/thinkspark_smoke.yaml")

    print("\n\033[1m=== 4/4 inference ===\033[0m", flush=True)
    from config.paths import FILLER_DICT_JSON
    from thinkspark.infer import ThinkSparkPredictor

    ckpt = ROOT / "artifacts/thinkspark_smoke/best"
    if not (ckpt / "model.pt").exists():
        ckpt = sorted((ROOT / "artifacts/thinkspark_smoke").glob("epoch-*"))[-1]
    pred = ThinkSparkPredictor(ckpt, FILLER_DICT_JSON, seed=0)

    trials = [
        ("हम्म अच्छा एक सेकंड", "user is thinking about a purchase"),
        ("yaar ye refund abhi tak nahi aaya", "user is angrily chasing a delayed refund"),
        ("えっと ちょっと待って", "user is unsure which plan to pick"),
        ("okay okay let me see", ""),
    ]
    for inp, ctx in trials:
        r = pred.predict(inp, ctx)
        spark = r["spark"] or "· (silence)"
        print(f"  input={inp!r}\n    context={ctx!r}\n    -> spark={spark!r} "
              f"intent={r['intent']} lang={r['language']} type={r['filler_type']}\n")

    print("\033[1;32mSMOKE TEST PASSED ✅\033[0m  full pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
