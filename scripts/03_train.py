#!/usr/bin/env python3
"""Stage 3 — train ThinkSpark (Mac MPS / Colab 1-GPU / Kaggle T4x2 DDP).

Fetches splits from Hugging Face if they are not already on disk, then trains.

    python scripts/03_train.py
    python scripts/03_train.py --config configs/thinkspark_tiny.yaml
    python scripts/03_train.py --gpus 2            # force DDP (Kaggle T4 x2)
    python scripts/03_train.py --distributed off   # single GPU even if 2 are visible
    python scripts/03_train.py --refresh-data      # re-download from HF
    python scripts/03_train.py --epochs 15         # override yaml (early-stop still on)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config.paths import load_env  # noqa: E402
from thinkspark.config import TrainConfig  # noqa: E402
from thinkspark.trainer import train  # noqa: E402

load_env()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/thinkspark_tiny.yaml")
    ap.add_argument("--hf-repo", default="", help="HF dataset id (default: config / anuj-inavlabs/kupe-thinkspark)")
    ap.add_argument("--no-fetch", action="store_true", help="do not download; require local splits")
    ap.add_argument("--refresh-data", action="store_true", help="re-download splits from HF even if local files exist")
    ap.add_argument("--gpus", type=int, default=None, help="0=auto all CUDA GPUs, 1=single, 2=DDP on two GPUs")
    ap.add_argument("--distributed", choices=["auto", "on", "off"], default=None)
    ap.add_argument("--no-amp", action="store_true", help="disable CUDA mixed precision")
    ap.add_argument("--epochs", type=int, default=None, help="override yaml epochs (early-stop still on)")
    args = ap.parse_args()

    cfg = TrainConfig.load(args.config)
    if args.hf_repo:
        cfg.data.hf_repo = args.hf_repo
    if args.no_fetch:
        cfg.data.hf_fetch = False
    if args.refresh_data:
        cfg.data.hf_refresh = True
    if args.gpus is not None:
        cfg.run.gpus = args.gpus
    if args.distributed is not None:
        cfg.run.distributed = args.distributed
    if args.no_amp:
        cfg.run.amp = False
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
