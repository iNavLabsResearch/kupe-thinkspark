#!/usr/bin/env python3
"""Offline synthetic corpus generator — NO API needed.

Builds a small, script-valid corpus straight from the curated LEXICON so the
whole pipeline (build_dataset -> train -> infer) can be smoke-tested without
spending a Sarvam call. Real training data comes from scripts/01_generate_data.py.

    python scripts/smoke_gen.py --per-cell 6
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import RAW_CORPUS  # noqa: E402
from config.taxonomy import EMOTIONS, LANGUAGES, LEXICON, REGISTERS  # noqa: E402

def _pool(lang: str) -> list[str]:
    out = []
    for intent_tbl in LEXICON[lang].values():
        for forms in intent_tbl.values():
            out.extend([f for f in forms if f])
    return out or ["hmm", "okay"]


def make_input(lang: str, rng: random.Random) -> str:
    """A short pseudo-utterance in the language's native script, assembled from
    lexicon surface forms (guarantees the right script for smoke validation)."""
    pool = _pool(lang)
    return " ".join(rng.choice(pool) for _ in range(rng.randint(2, 4)))


def make_context(lang: str, rng: random.Random) -> tuple[str, list[str]]:
    """A multi-turn, sometimes multilingual past-conversation transcript.

    Exercises the real contract: mostly the target language + native script, but
    occasionally the Agent turn is English (code-switch). Returns (text, langs).
    """
    if rng.random() < 0.2:
        return "", []                       # cold start
    pool = _pool(lang)
    en_pool = _pool("en")
    n_turns = rng.randint(1, 4)
    langs = {lang}
    lines = []
    for t in range(n_turns):
        speaker = "User" if t % 2 == 0 else "Agent"
        mix_en = speaker == "Agent" and rng.random() < 0.35   # agent code-switches
        src = en_pool if mix_en else pool
        if mix_en:
            langs.add("en")
        turn = " ".join(rng.choice(src) for _ in range(rng.randint(2, 4)))
        lines.append(f"{speaker}: {turn}")
    return "\n".join(lines), sorted(langs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=6, help="rows per (lang,intent)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with RAW_CORPUS.open("w", encoding="utf-8") as f:
        for lang, intents in LEXICON.items():
            script = LANGUAGES[lang]["script"]
            for intent, type_tbl in intents.items():
                types = [t for t, forms in type_tbl.items() if forms]
                if not types:
                    continue
                for _ in range(args.per_cell):
                    ftype = rng.choice(types)
                    cands = list(type_tbl[ftype])
                    if intent == "no_filler":
                        ftype, cands = "none", []
                    weights = ([round(1 / len(cands), 4)] * len(cands)) if cands else []
                    ctx, ctx_langs = make_context(lang, rng)
                    rec = {
                        "input": make_input(lang, rng),
                        "context": ctx,
                        "context_langs": ctx_langs,
                        "language": lang,
                        "script": script,
                        "register": rng.choice(REGISTERS),
                        "intent": intent,
                        "emotion": rng.choice(EMOTIONS),
                        "filler_type": ftype,
                        "filler_candidates": cands,
                        "filler_weights": weights,
                        "notes": "synthetic-smoke",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    print(f"[smoke_gen] wrote {n} synthetic rows -> {RAW_CORPUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
