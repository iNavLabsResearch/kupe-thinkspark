"""JSONL dataset + collate for ThinkSpark.

Each row -> two byte sequences (input, context) and five integer labels
(intent, language, register, emotion, filler_type).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config.taxonomy import (
    EMOTION2ID, FILLERTYPE2ID, INTENT2ID, LANG2ID, REGISTER2ID,
)
from . import tokenizer as tok


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ThinkSparkDataset(Dataset):
    def __init__(self, rows: list[dict], max_input_len: int, max_context_len: int):
        self.rows = rows
        self.max_input_len = max_input_len
        self.max_context_len = max_context_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "input_ids": tok.encode(r["input"], self.max_input_len),
            "context_ids": tok.encode(r.get("context") or "", self.max_context_len),
            "intent": INTENT2ID[r["intent"]],
            "language": LANG2ID[r["language"]],
            "register": REGISTER2ID.get(r.get("register", "casual"), REGISTER2ID["casual"]),
            "emotion": EMOTION2ID.get(r.get("emotion", "neutral"), EMOTION2ID["neutral"]),
            "filler_type": FILLERTYPE2ID.get(r.get("filler_type", "none"), FILLERTYPE2ID["none"]),
        }


def collate(batch: list[dict]) -> dict:
    inp_ids, inp_mask = tok.pad_batch([b["input_ids"] for b in batch])
    ctx_ids, ctx_mask = tok.pad_batch([b["context_ids"] for b in batch])
    out = {
        "input_ids": torch.tensor(inp_ids, dtype=torch.long),
        "input_mask": torch.tensor(inp_mask, dtype=torch.bool),
        "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
        "context_mask": torch.tensor(ctx_mask, dtype=torch.bool),
    }
    for k in ("intent", "language", "register", "emotion", "filler_type"):
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    return out
