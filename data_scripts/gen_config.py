#!/usr/bin/env python3
"""Generation plan for ThinkSpark: how many examples per language, and the
register / intent mixes each batch asks the LLM for.

The generator (data_gen_agent.py) walks LANG_PLAN, and for each language rotates
through REGISTER_MIXES x INTENT_FOCUS so the corpus is diverse and every intent —
especially `no_filler` — is well represented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.taxonomy import INTENTS, LANGUAGES  # noqa: E402

# Examples generated per LLM batch (one API call -> up to this many rows).
# Bigger batch = fewer calls. Warm uses WARM_BATCH_SIZE=1 (see data_gen_agent.py).
BATCH_SIZE = 40

# Per-language target counts. Indian languages get the most; foreign a solid base.
# Tune freely — the generator resumes toward these targets.
_TIER_A = 4000   # Hindi, Hinglish, English  (flagship)
_TIER_B = 2000   # major Indian languages
_TIER_C = 1000   # smaller Indian + big foreign
_TIER_D = 600    # remaining foreign

LANG_TARGETS: dict[str, int] = {
    "hi": _TIER_A, "hi_en": _TIER_A, "en": _TIER_A,
    "mr": _TIER_B, "bn": _TIER_B, "gu": _TIER_B, "ta": _TIER_B,
    "te": _TIER_B, "kn": _TIER_B, "ml": _TIER_B, "pa": _TIER_B, "ur": _TIER_B,
    "or": _TIER_C, "as": _TIER_C, "es": _TIER_C, "fr": _TIER_C,
    "de": _TIER_C, "ja": _TIER_C, "zh": _TIER_C,
    "pt": _TIER_D, "ar": _TIER_D, "ru": _TIER_D,
}
# Any language present in taxonomy but missing above -> small default.
for _c in LANGUAGES:
    LANG_TARGETS.setdefault(_c, _TIER_D)

# Register mixes rotated per batch (sums ~1.0, phrased for the prompt).
REGISTER_MIXES = [
    "70% casual spoken, 20% formal, 10% urban_mixed (English loanwords)",
    "50% urban_mixed code-switched with English loanwords, 30% casual, 20% formal",
    "60% formal polite, 30% casual, 10% urban_mixed",
]

# Each batch is biased toward a small set of intents so coverage stays balanced.
# `no_filler` is deliberately its own focus so ~25% of the corpus is the silence
# negative class (the reference plan's most-important class).
INTENT_FOCUS = [
    ["thinking", "hesitating", "clarifying_question"],
    ["agreeing", "positive_ack", "encouraging"],
    ["surprised", "excited", "empathetic"],
    ["negative_ack", "skeptical", "impatient"],
    ["apologetic", "calming", "sad_acknowledge", "polite_interrupt"],
    ["no_filler", "no_filler", "thinking"],   # heavy silence sampling
]

# Rotating topics — only the trailing user suffix changes (stable prefix stays fixed).
TOPICS: list[str] = [
    "work and deadlines", "family and home", "money and refunds", "tech support",
    "health and appointments", "travel plans", "food and restaurants",
    "shopping and delivery", "small talk", "complaints and frustration",
    "learning and study", "weather and commute", "friends and plans",
    "banking and payments", "housing and rent", "kids and school",
]

# sanity
assert all(i in INTENTS for grp in INTENT_FOCUS for i in grp)


def total_target() -> int:
    return sum(LANG_TARGETS.values())
