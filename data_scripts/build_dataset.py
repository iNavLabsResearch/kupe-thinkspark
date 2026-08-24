#!/usr/bin/env python3
"""
Validate -> dedupe -> split -> build label maps + filler dictionary.

Reads   data/raw/thinkspark_corpus.jsonl   (from data_gen_agent.py or the
        synthetic smoke generator)
Writes  data/raw/thinkspark_corpus_clean.jsonl
        data/splits/{train,val,test}.jsonl        (stratified by language+intent)
        data/vocab/label_maps.json                (label id maps + tokenizer meta)
        data/vocab/filler_dictionary.json         (aggregated surface forms)

Usage:
    python data_scripts/build_dataset.py
    python data_scripts/build_dataset.py --val 0.1 --test 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import (  # noqa: E402
    CLEAN_CORPUS, FILLER_DICT_JSON, LABELMAPS_JSON, RAW_CORPUS,
    TEST_JSONL, TRAIN_JSONL, VAL_JSONL,
)
from config.taxonomy import (  # noqa: E402
    EMOTION2ID, FILLERTYPE2ID, INTENT2ID, LANG2ID, REGISTER2ID,
    EMOTIONS, FILLER_TYPES, INTENTS, LANG_LIST, REGISTERS, LANGUAGES,
)
from data_scripts.script_utils import input_language_ok  # noqa: E402

REQUIRED = ("input", "language", "intent", "filler_type")


def load_raw(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        sys.exit(f"No corpus at {path} — run data_gen_agent.py or smoke_gen first.")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def clean(rows: list[dict]) -> tuple[list[dict], Counter]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    drop = Counter()
    for r in rows:
        if any(not r.get(k) and r.get(k) != "" for k in REQUIRED):
            drop["missing_field"] += 1
            continue
        lang = r.get("language")
        if lang not in LANGUAGES:
            drop["bad_lang"] += 1
            continue
        if r.get("intent") not in INTENTS:
            drop["bad_intent"] += 1
            continue
        inp = (r.get("input") or "").strip()
        if len(inp) < 2:
            drop["short_input"] += 1
            continue
        if not input_language_ok(inp, lang):
            drop["wrong_language"] += 1
            continue
        key = (lang, inp)
        if key in seen:
            drop["dup"] += 1
            continue
        seen.add(key)
        # normalise optional fields
        r.setdefault("context", "")
        cl = r.get("context_langs") or []
        r["context_langs"] = [c for c in cl if c in LANGUAGES] if isinstance(cl, list) else []
        if not r["context_langs"] and r["context"]:
            r["context_langs"] = [lang]
        r.setdefault("emotion", "neutral")
        r.setdefault("register", "casual")
        if r["emotion"] not in EMOTIONS:
            r["emotion"] = "neutral"
        if r["register"] not in REGISTERS:
            r["register"] = "casual"
        if r["filler_type"] not in FILLER_TYPES:
            r["filler_type"] = "none" if r["intent"] == "no_filler" else "word"
        r.setdefault("filler_candidates", [])
        r.setdefault("filler_weights", [])
        r["script"] = LANGUAGES[lang]["script"]
        kept.append(r)
    return kept, drop


def stratified_split(rows, val_frac, test_frac, seed):
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["language"], r["intent"])].append(r)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(int(round(n * test_frac)), 1 if n >= 5 else 0)
        n_val = max(int(round(n * val_frac)), 1 if n >= 5 else 0)
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def build_filler_dict(rows) -> dict:
    """Aggregate (lang -> intent -> type -> {surface: weight}) from the corpus,
    so inference can sample real observed fillers, not only the seed lexicon."""
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for r in rows:
        cands = r.get("filler_candidates") or []
        weights = r.get("filler_weights") or [1.0] * len(cands)
        for c, w in zip(cands, weights):
            try:
                w = float(w)
            except (TypeError, ValueError):
                w = 1.0
            agg[r["language"]][r["intent"]][r["filler_type"]][c] += w
    # to plain dict with normalised weights
    out: dict = {}
    for lang, itbl in agg.items():
        out[lang] = {}
        for intent, ttbl in itbl.items():
            out[lang][intent] = {}
            for ftype, counter in ttbl.items():
                tot = sum(counter.values()) or 1.0
                out[lang][intent][ftype] = {
                    s: round(w / tot, 4) for s, w in counter.most_common()
                }
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = load_raw(RAW_CORPUS)
    print(f"[load] {len(raw):,} raw rows")
    rows, drop = clean(raw)
    print(f"[clean] kept {len(rows):,} | dropped {sum(drop.values()):,} {dict(drop)}")
    if not rows:
        sys.exit("No valid rows after cleaning.")

    write_jsonl(CLEAN_CORPUS, rows)

    train, val, test = stratified_split(rows, args.val, args.test, args.seed)
    write_jsonl(TRAIN_JSONL, train)
    write_jsonl(VAL_JSONL, val)
    write_jsonl(TEST_JSONL, test)
    print(f"[split] train={len(train):,} val={len(val):,} test={len(test):,}")

    filler_dict = build_filler_dict(rows)
    FILLER_DICT_JSON.parent.mkdir(parents=True, exist_ok=True)
    FILLER_DICT_JSON.write_text(json.dumps(filler_dict, ensure_ascii=False, indent=1), encoding="utf-8")

    label_maps = {
        "lang2id": LANG2ID, "register2id": REGISTER2ID, "intent2id": INTENT2ID,
        "emotion2id": EMOTION2ID, "fillertype2id": FILLERTYPE2ID,
        "lang_list": LANG_LIST, "registers": REGISTERS, "intents": INTENTS,
        "emotions": EMOTIONS, "filler_types": FILLER_TYPES,
        "tokenizer": {"kind": "byte", "vocab_size": 259},  # 256 bytes + PAD/BOS/EOS
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "by_language": dict(Counter(r["language"] for r in rows)),
            "by_intent": dict(Counter(r["intent"] for r in rows)),
        },
    }
    LABELMAPS_JSON.write_text(json.dumps(label_maps, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[vocab] label_maps -> {LABELMAPS_JSON.name} | filler_dictionary -> {FILLER_DICT_JSON.name}")
    print("[done] dataset ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
