#!/usr/bin/env python3
"""Live generation stats — pass/fail batches + validation rejects (kupe-tts/soniox style).

Run in a second terminal while generate.py is streaming:

    python scripts/02_stats.py
    python scripts/02_stats.py --once
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RAW_CORPUS, SQLITE_PATH, TRACKER_DIR  # noqa: E402
from config.taxonomy import LANGUAGES  # noqa: E402
from data_scripts.gen_config import LANG_TARGETS, total_target  # noqa: E402
from data_scripts import tracker  # noqa: E402


class C:
    R = "\033[0m"
    B = "\033[1m"
    D = "\033[2m"
    RED = "\033[91m"
    GRN = "\033[92m"
    YLW = "\033[93m"
    CYN = "\033[96m"
    DIM = "\033[2m"


def term_size() -> tuple[int, int]:
    try:
        sz = shutil.get_terminal_size(fallback=(100, 36))
        return max(72, sz.columns), max(24, sz.lines)
    except OSError:
        return 100, 36


def bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(pct, 100.0))
    filled = int(round(width * pct / 100.0))
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1f}%"


def render(targets: dict[str, int]) -> str:
    s = tracker.summarize(targets)
    cols, _ = term_size()
    w = cols - 2
    lines: list[str] = []
    lines.append(f"{C.B}{C.CYN}ThinkSpark generation stats{C.R}  {C.DIM}{SQLITE_PATH}{C.R}")
    lines.append("─" * min(w, 90))
    lines.append(
        f"Corpus kept {C.GRN}{s['total_kept']:,}{C.R}  "
        f"rejected {C.RED}{s['total_rejected']:,}{C.R}  "
        f"plan {total_target():,}  "
        f"₹{s['cum_cost_inr']:.2f}  "
        f"rows indexed {s['max_row_id']:,}"
    )

    bs = s["batch_status"]
    lines.append("")
    lines.append(f"{C.B}Batches{C.R}  "
                 f"{C.GRN}done {bs.get('done', 0):,}{C.R}  "
                 f"{C.RED}failed {bs.get('failed', 0):,}{C.R}  "
                 f"{C.YLW}in_progress {bs.get('in_progress', 0):,}{C.R}  "
                 f"total #{s['max_batch_num']:,}")

    lines.append("")
    lines.append(f"{C.B}Per language (kept / target · pass rate){C.R}")
    by_code = {r["language"]: r for r in s["langs"]}
    # every planned language, plan order (not truncated)
    ordered = [c for c in targets if c in by_code] + [
        r["language"] for r in s["langs"] if r["language"] not in targets
    ]
    for code in ordered:
        r = by_code[code]
        name = LANGUAGES.get(r["language"], {}).get("name", r["language"])[:12]
        col = C.GRN if r["left"] == 0 else (C.YLW if r["kept"] else C.DIM)
        lines.append(
            f"  {r['language']:<7} {name:<12} "
            f"{col}{r['kept']:>5,}{C.R}/{r['target']:<5,} "
            f"left {r['left']:>5,}  "
            f"pass {r['pass_rate']:5.1f}%  "
            f"rej {r['rejected']:,}"
        )

    reasons = s["reject_reasons"]
    if reasons:
        lines.append("")
        lines.append(f"{C.B}Validation rejects (failed rows){C.R}")
        for reason, n in list(reasons.items())[:10]:
            lines.append(f"  {C.RED}{reason:<22}{C.R} {n:,}")

    failed = s["recent_failed"]
    if failed:
        lines.append("")
        lines.append(f"{C.B}Recent failed batches{C.R}")
        for b in failed[:6]:
            topic = (b.get("topic") or "")[:18]
            err = (b.get("error_message") or b.get("error_type") or "?")[:48]
            lines.append(
                f"  {C.RED}#{b.get('batch_num', '?'):>4}{C.R} "
                f"{b.get('language', '?'):<6} {topic:<18} {C.DIM}{err}{C.R}"
            )

    lines.append("")
    lines.append(f"{C.DIM}jsonl {RAW_CORPUS} · tracker {TRACKER_DIR}{C.R}")
    return "\n".join(lines)


def parse_args():
    ap = argparse.ArgumentParser(description="Live ThinkSpark generation stats dashboard.")
    ap.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--langs", default="", help="optional comma subset for targets display")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    langs = [c.strip() for c in args.langs.split(",") if c.strip()]
    targets = {c: LANG_TARGETS[c] for c in (langs or LANG_TARGETS)} if langs else dict(LANG_TARGETS)

    tracker.init()
    tracker.hydrate_from_corpus(targets=targets)

    if args.once:
        print(render(targets))
        return 0

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(render(targets))
            time.sleep(max(args.interval, 0.5))
    except KeyboardInterrupt:
        print(f"\n{C.DIM}stats stopped{C.R}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
