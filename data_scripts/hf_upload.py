#!/usr/bin/env python3
"""Upload built ThinkSpark dataset to Hugging Face (uses HF_TOKEN from kupe-tts/.env)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import (  # noqa: E402
    CLEAN_CORPUS,
    FILLER_DICT_JSON,
    LABELMAPS_JSON,
    RAW_CORPUS,
    ROOT,
    TEST_JSONL,
    TRAIN_JSONL,
    VAL_JSONL,
)

KUPE_TTS_ENV = ROOT.parent / "kupe-tts" / ".env"
HF_REPO_DEFAULT = "kupe-thinkspark"


def load_hf_token() -> str:
    import os

    if KUPE_TTS_ENV.exists():
        for raw in KUPE_TTS_ENV.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                _, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value.startswith("hf_") and len(value) > 20:
                    os.environ["HF_TOKEN"] = value
                    break
    key = (
        __import__("os").environ.get("HF_TOKEN", "").strip()
        or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
    )
    if not key or not key.startswith("hf_"):
        raise RuntimeError(
            f"Missing HF_TOKEN in {KUPE_TTS_ENV}. "
            "Add a write token from https://huggingface.co/settings/tokens"
        )
    return key


def resolve_repo_id(token: str, repo: str | None) -> str:
    from huggingface_hub import HfApi

    if repo and "/" in repo:
        return repo
    user = HfApi(token=token).whoami().get("name") or ""
    if not user:
        raise RuntimeError("Could not resolve Hugging Face username from token.")
    slug = repo or HF_REPO_DEFAULT
    return f"{user}/{slug}"


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def dataset_readme(repo_id: str, stats: dict[str, int]) -> str:
    total = stats.get("clean", stats.get("raw", 0))
    return f"""---
license: apache-2.0
language:
- multilingual
task_categories:
- text-classification
- token-classification
tags:
- backchannel
- filler-words
- voice-assistant
- multilingual
- thinkspark
size_categories:
- 10K<n<100K
---

# ThinkSpark training corpus

Synthetic multilingual training data for **ThinkSpark** — a tiny model that predicts
the human *thinking sound* / backchannel a voice assistant should emit between STT
and the main LLM reply.

Generated with Sarvam `sarvam-105b` (`reasoning_effort=low`) via the kupe-thinkspark pipeline.

## Files

| path | rows | description |
|------|-----:|-------------|
| `corpus/thinkspark_corpus_clean.jsonl` | {stats.get('clean', 0):,} | validated + deduped full corpus |
| `corpus/thinkspark_corpus.jsonl` | {stats.get('raw', 0):,} | raw generated corpus (includes rejects filtered at build) |
| `splits/train.jsonl` | {stats.get('train', 0):,} | stratified train split |
| `splits/val.jsonl` | {stats.get('val', 0):,} | validation split |
| `splits/test.jsonl` | {stats.get('test', 0):,} | test split |
| `vocab/label_maps.json` | — | language / intent / register / emotion label maps |
| `vocab/filler_dictionary.json` | — | per-language filler surface forms |

**Total clean rows:** {total:,}

## Row schema

Each JSONL line:

```json
{{
  "input": "user's current utterance",
  "context": "prior conversation transcript (may be empty)",
  "context_langs": ["hi"],
  "language": "hi",
  "script": "Deva",
  "register": "casual",
  "intent": "thinking",
  "emotion": "neutral",
  "filler_type": "sound",
  "filler_candidates": ["हम्म...", "अच्छा"],
  "filler_weights": [0.6, 0.4],
  "notes": "..."
}}
```

## Source

Built locally with [kupe-thinkspark](https://github.com/anuj-inavlabs/kupe) — mirrors the kupe-tts data generation workflow.

Uploaded: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
"""


def upload_files(repo_id: str, token: str, *, include_raw: bool) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    stats = {
        "clean": count_jsonl(CLEAN_CORPUS),
        "raw": count_jsonl(RAW_CORPUS),
        "train": count_jsonl(TRAIN_JSONL),
        "val": count_jsonl(VAL_JSONL),
        "test": count_jsonl(TEST_JSONL),
    }
    uploads: list[tuple[Path, str]] = [
        (CLEAN_CORPUS, "corpus/thinkspark_corpus_clean.jsonl"),
        (TRAIN_JSONL, "splits/train.jsonl"),
        (VAL_JSONL, "splits/val.jsonl"),
        (TEST_JSONL, "splits/test.jsonl"),
        (LABELMAPS_JSON, "vocab/label_maps.json"),
        (FILLER_DICT_JSON, "vocab/filler_dictionary.json"),
    ]
    if include_raw:
        uploads.append((RAW_CORPUS, "corpus/thinkspark_corpus.jsonl"))

    missing = [str(p) for p, _ in uploads if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files to upload:\n  " + "\n  ".join(missing))

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    ops: list[CommitOperationAdd] = []
    for local, remote in uploads:
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))

    readme = dataset_readme(repo_id, stats)
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))

    meta = {
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo_id,
        "stats": stats,
        "files": [remote for _, remote in uploads],
    }
    ops.append(
        CommitOperationAdd(
            path_in_repo="meta/upload_stats.json",
            path_or_fileobj=json.dumps(meta, indent=2).encode("utf-8"),
        )
    )

    msg = (
        f"Upload ThinkSpark corpus ({stats['clean']:,} clean, "
        f"{stats['train']:,}/{stats['val']:,}/{stats['test']:,} splits)"
    )
    api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops, commit_message=msg)
    return f"https://huggingface.co/datasets/{repo_id}"


def parse_args():
    ap = argparse.ArgumentParser(description="Upload ThinkSpark data to Hugging Face.")
    ap.add_argument(
        "--repo", default=HF_REPO_DEFAULT,
        help=f"HF dataset slug or full id (default: {{user}}/{HF_REPO_DEFAULT})",
    )
    ap.add_argument(
        "--include-raw", action="store_true",
        help="also upload raw thinkspark_corpus.jsonl (~40k rows)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    token = load_hf_token()
    repo_id = resolve_repo_id(token, args.repo)
    print(f"Uploading to {repo_id} …")
    url = upload_files(repo_id, token, include_raw=args.include_raw)
    print(f"Done: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
