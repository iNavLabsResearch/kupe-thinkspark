"""Byte-level tokenizer.

Every language and script on earth is just UTF-8 bytes, so a byte vocab gives
ZERO out-of-vocabulary across Devanagari, Tamil, kana, Arabic, Latin, emoji —
with a tiny 259-row embedding table. Perfect for a "crazy multilingual" model
that must never choke on an unseen script.

  ids 0..255  -> raw UTF-8 byte value
  256 PAD, 257 BOS, 258 EOS
"""

from __future__ import annotations

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259


def encode(text: str, max_len: int, add_special: bool = True) -> list[int]:
    body = list(text.encode("utf-8"))
    if add_special:
        budget = max_len - 2
        body = body[:budget]
        ids = [BOS_ID] + body + [EOS_ID]
    else:
        ids = body[:max_len]
    return ids


def pad_batch(seqs: list[list[int]], pad_to: int | None = None) -> tuple[list[list[int]], list[list[int]]]:
    """Return (padded_ids, attention_mask) as python lists."""
    n = pad_to or max((len(s) for s in seqs), default=1)
    ids, mask = [], []
    for s in seqs:
        s = s[:n]
        pad = n - len(s)
        ids.append(s + [PAD_ID] * pad)
        mask.append([1] * len(s) + [0] * pad)
    return ids, mask
