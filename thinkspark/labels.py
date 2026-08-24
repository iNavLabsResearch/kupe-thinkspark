"""Authoritative label space for training/inference, derived from the taxonomy.

Why not just read the downloaded label_maps.json? Because we now relabel on the
fly (fine intents -> super intents, emotion fix) WITHOUT regenerating data. The
code taxonomy is the single source of truth for the label space; the HF splits
stay untouched and are mapped at load time.

`scheme`:
  "super"  -> 9 coarse agent-reactions (default; higher accuracy)
  "fine"   -> the original 17 intents (for comparison)
"""

from __future__ import annotations

from config.taxonomy import (
    EMOTIONS, FILLER_TYPES, INTENT_TO_SUPER, INTENTS, LANG_LIST, REGISTERS,
    SUPER_INTENTS, fix_emotion,
)


class LabelScheme:
    def __init__(self, scheme: str = "super"):
        if scheme not in ("super", "fine"):
            raise ValueError(f"unknown intent scheme {scheme!r}")
        self.scheme = scheme
        self.intents = SUPER_INTENTS if scheme == "super" else INTENTS
        self.intent2id = {t: i for i, t in enumerate(self.intents)}
        self.id2intent = {i: t for t, i in self.intent2id.items()}
        self.emotions = EMOTIONS
        self.registers = REGISTERS
        self.filler_types = FILLER_TYPES
        self.lang_list = LANG_LIST
        self.lang2id = {c: i for i, c in enumerate(LANG_LIST)}
        self.emotion2id = {e: i for i, e in enumerate(EMOTIONS)}
        self.register2id = {r: i for i, r in enumerate(REGISTERS)}
        self.fillertype2id = {t: i for i, t in enumerate(FILLER_TYPES)}

    # ---- row-level relabel (applied at data load; no regeneration) ----
    def intent_of(self, fine_intent: str) -> str:
        if self.scheme == "fine":
            return fine_intent
        return INTENT_TO_SUPER.get(fine_intent, "thinking")

    def intent_id(self, fine_intent: str) -> int:
        return self.intent2id[self.intent_of(fine_intent)]

    def emotion_id(self, emotion: str, fine_intent: str, input_text: str, context: str) -> int:
        fixed = fix_emotion(emotion, fine_intent, input_text, context)
        return self.emotion2id.get(fixed, self.emotion2id["neutral"])

    # ---- serialisable maps saved with the checkpoint ----
    def n_classes(self) -> dict:
        return {
            "intent": len(self.intents), "language": len(self.lang_list),
            "register": len(self.registers), "emotion": len(self.emotions),
            "filler_type": len(self.filler_types),
        }

    def to_maps(self) -> dict:
        return {
            "intent_scheme": self.scheme,
            "intents": self.intents,
            "intent2id": self.intent2id,
            "lang_list": self.lang_list,
            "registers": self.registers,
            "emotions": self.emotions,
            "filler_types": self.filler_types,
        }
