"""Fetch ThinkSpark splits from Hugging Face if they are not already local.

Used by training so the same command works on a Mac, a PC, Colab, and Kaggle:
download once, then reuse the files on disk.
"""

from __future__ import annotations

import os
from pathlib import Path

from config.paths import FILLER_DICT_JSON, LABELMAPS_JSON, ROOT, TEST_JSONL, TRAIN_JSONL, VAL_JSONL, load_env

DEFAULT_HF_REPO = "anuj-inavlabs/kupe-thinkspark"

# HF dataset layout -> local project paths
HF_FILES = (
    ("splits/train.jsonl", TRAIN_JSONL),
    ("splits/val.jsonl", VAL_JSONL),
    ("splits/test.jsonl", TEST_JSONL),
    ("vocab/label_maps.json", LABELMAPS_JSON),
    ("vocab/filler_dictionary.json", FILLER_DICT_JSON),
)


def _ok(path: Path, min_bytes: int = 64) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def local_training_ready(train_jsonl: str | Path, val_jsonl: str | Path,
                         test_jsonl: str | Path, label_maps: str | Path) -> bool:
    return all(_ok(resolve_data_path(p)) for p in (train_jsonl, val_jsonl, test_jsonl, label_maps))


def maybe_hf_token() -> str | None:
    """Best-effort token: env, thinkspark/.env, kupe-tts/.env. Public datasets work without one."""
    load_env()
    tts_env = ROOT.parent / "kupe-tts" / ".env"
    if tts_env.exists():
        for raw in tts_env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                _, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value.startswith("hf_") and len(value) > 20:
                    os.environ.setdefault("HF_TOKEN", value)
                    break
    key = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if key.startswith("hf_") and len(key) > 20:
        return key
    return None


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def fetch_hf_dataset(
    repo: str | None = None,
    *,
    refresh: bool = False,
    token: str | None | bool = None,
) -> dict[str, Path]:
    """Download splits + vocab into data/. Skip files that already exist unless refresh=True."""
    repo = (repo or os.environ.get("HF_DATASET_ID") or DEFAULT_HF_REPO).strip()
    if token is None:
        token = maybe_hf_token()

    needed = [(remote, dest) for remote, dest in HF_FILES if refresh or not _ok(dest)]
    if not needed:
        return {remote: dest for remote, dest in HF_FILES}

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch training data. "
            "pip install huggingface_hub"
        ) from exc

    print(f"[data] fetching {repo} ({len(needed)} file(s) missing locally)")
    dests: dict[str, Path] = {}
    for remote, dest in HF_FILES:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not refresh and _ok(dest):
            dests[remote] = dest
            continue
        path = hf_hub_download(
            repo_id=repo,
            filename=remote,
            repo_type="dataset",
            token=token,
        )
        src = Path(path)
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        dests[remote] = dest
        extra = f"  ({_count_lines(dest):,} rows)" if dest.suffix == ".jsonl" else ""
        print(f"[data]   {remote} -> {dest.relative_to(ROOT)}{extra}")
    return dests


def ensure_training_data(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    test_jsonl: str | Path,
    label_maps: str | Path,
    *,
    repo: str | None = None,
    fetch: bool = True,
    refresh: bool = False,
) -> None:
    """Use local splits if present; otherwise pull them from Hugging Face."""
    train_p = resolve_data_path(train_jsonl)
    val_p = resolve_data_path(val_jsonl)
    test_p = resolve_data_path(test_jsonl)
    maps_p = resolve_data_path(label_maps)

    ready = all(_ok(p) for p in (train_p, val_p, test_p, maps_p))
    if ready and not refresh:
        print(
            f"[data] local splits ready "
            f"(train={_count_lines(train_p):,} val={_count_lines(val_p):,} "
            f"test={_count_lines(test_p):,}) — skip HF fetch"
        )
        return
    if not fetch:
        missing = [str(p) for p in (train_p, val_p, test_p, maps_p) if not _ok(p)]
        raise FileNotFoundError(
            "Training data not found and HF fetch is disabled. Missing:\n  "
            + "\n  ".join(missing)
        )
    fetch_hf_dataset(repo, refresh=refresh or not ready)
    if not all(_ok(p) for p in (train_p, val_p, test_p, maps_p)):
        raise FileNotFoundError(
            f"HF fetch finished but training files are still missing under {ROOT / 'data'}. "
            f"Check --hf-repo (tried {repo or DEFAULT_HF_REPO})."
        )
    print(
        f"[data] ready train={_count_lines(train_p):,} "
        f"val={_count_lines(val_p):,} test={_count_lines(test_p):,}"
    )
