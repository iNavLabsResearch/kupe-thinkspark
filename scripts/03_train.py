#!/usr/bin/env python3
"""Stage 3 — train ThinkSpark locally on Mac M1 (MPS/CPU).

    python scripts/03_train.py --config configs/thinkspark_tiny.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinkspark.config import TrainConfig  # noqa: E402
from thinkspark.trainer import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thinkspark_tiny.yaml")
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config)
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
