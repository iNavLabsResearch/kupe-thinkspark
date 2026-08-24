"""Inference: (input utterance + optional context) -> spoken thinking spark.

Loads a trained checkpoint, predicts (intent, language, register, emotion,
filler_type), then samples a surface form. Sampling prefers the corpus-derived
filler_dictionary.json (real observed fillers) and falls back to the curated
LEXICON in config/taxonomy.py, so an untrained/edge combo still speaks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from config.taxonomy import (
    ID2EMOTION, ID2FILLERTYPE, ID2LANG, ID2REGISTER, INTENT_TO_SUPER, LANGUAGES,
    SCRIPTS, SUPER_INTENTS, sample_filler, sample_filler_super,
)
from . import tokenizer as tok
from .model import ThinkSpark

# script tag -> language codes that use it (e.g. "Deva" -> ["hi", "mr"])
_SCRIPT_TO_LANGS: dict[str, list[str]] = {}
for _code, _meta in LANGUAGES.items():
    _SCRIPT_TO_LANGS.setdefault(_meta["script"], []).append(_code)


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def _in_any_script(cp: int) -> bool:
    return any(_in_ranges(cp, r) for r in SCRIPTS.values())


def _dominant_script(text: str) -> tuple[str | None, float]:
    """Return (script_tag, ratio) of the script most of `text` uses.

    Denominator counts every 'scripted' char — letters PLUS Indic combining marks
    (matras/virama like ा ्), which are not `isalpha()` and would otherwise make
    Devanagari look Latin-dominant in code-mixed text like 'यह possible है'."""
    chars = [c for c in text if c.isalpha() or _in_any_script(ord(c))]
    if not chars:
        return None, 0.0
    best, best_r = None, 0.0
    for tag, ranges in SCRIPTS.items():
        r = sum(1 for c in chars if _in_ranges(ord(c), ranges)) / len(chars)
        if r > best_r:
            best, best_r = tag, r
    return best, best_r


class _ModelCfg:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class ThinkSparkPredictor:
    def __init__(self, ckpt_dir: str | Path, filler_dict_path: str | Path | None = None,
                 device: str = "auto", seed: int | None = None):
        ckpt_dir = Path(ckpt_dir)
        meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
        self.labels = json.loads((ckpt_dir / "label_maps.json").read_text(encoding="utf-8"))
        mcfg = _ModelCfg(meta["model_cfg"])
        dcfg = meta["data_cfg"]
        self.max_input_len = dcfg["max_input_len"]
        self.max_context_len = dcfg["max_context_len"]

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = ThinkSpark(
            mcfg, len(self.labels["intents"]), len(self.labels["lang_list"]),
            len(self.labels["registers"]), len(self.labels["emotions"]),
            len(self.labels["filler_types"]), self.max_input_len, self.max_context_len,
        ).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=self.device))
        self.model.eval()

        # intent label list from the checkpoint (super scheme by default; also
        # works for a fine-scheme checkpoint since it stores its own list)
        self.intent_labels = self.labels.get("intents", SUPER_INTENTS)
        # trust an explicit scheme; otherwise infer it (super has 'silence',
        # the old fine scheme has 'no_filler') so old checkpoints still work.
        self.intent_scheme = self.labels.get("intent_scheme") or (
            "super" if "silence" in self.intent_labels else "fine")

        fine_dict = {}
        if filler_dict_path and Path(filler_dict_path).exists():
            fine_dict = json.loads(Path(filler_dict_path).read_text(encoding="utf-8"))
        # the dictionary on disk is keyed by FINE intent; under the super scheme we
        # union each super-intent's fine members so the spark still samples real
        # observed fillers.
        self.filler_dict = self._to_super_filler_dict(fine_dict) \
            if self.intent_scheme == "super" else fine_dict
        self.rng = random.Random(seed)

    @staticmethod
    def _to_super_filler_dict(fine_dict: dict) -> dict:
        out: dict = {}
        for lang, itbl in fine_dict.items():
            out.setdefault(lang, {})
            for fine, ttbl in itbl.items():
                sup = INTENT_TO_SUPER.get(fine, fine)
                dst = out[lang].setdefault(sup, {})
                for ftype, forms in ttbl.items():
                    cell = dst.setdefault(ftype, {})
                    for surface, w in forms.items():
                        cell[surface] = cell.get(surface, 0.0) + float(w)
        return out

    @torch.no_grad()
    def predict(self, input_text: str, context: str = "", force_lang: str | None = None) -> dict:
        inp = tok.encode(input_text, self.max_input_len)
        ctx = tok.encode(context or "", self.max_context_len)
        ii, im = tok.pad_batch([inp]); ci, cm = tok.pad_batch([ctx])
        batch = {
            "input_ids": torch.tensor(ii, dtype=torch.long, device=self.device),
            "input_mask": torch.tensor(im, dtype=torch.bool, device=self.device),
            "context_ids": torch.tensor(ci, dtype=torch.long, device=self.device),
            "context_mask": torch.tensor(cm, dtype=torch.bool, device=self.device),
        }
        out = self.model(batch)
        prob = {h: torch.softmax(v, -1)[0] for h, v in out.items()}
        intent = self.intent_labels[int(prob["intent"].argmax())]
        register = ID2REGISTER[int(prob["register"].argmax())]
        emotion = ID2EMOTION[int(prob["emotion"].argmax())]
        ftype = ID2FILLERTYPE[int(prob["filler_type"].argmax())]

        lang, lang_corrected = self._resolve_language(input_text, prob["language"], force_lang)
        lang_list = self.labels["lang_list"]
        lang_conf = float(prob["language"][lang_list.index(lang)]) if lang in lang_list else \
            float(prob["language"].max())

        spark = self._surface(lang, intent, ftype)
        return {
            "spark": spark, "intent": intent, "language": lang,
            "register": register, "emotion": emotion, "filler_type": ftype,
            "language_corrected": lang_corrected,
            "confidence": {
                "intent": float(prob["intent"].max()),
                "language": lang_conf,
            },
        }

    def _resolve_language(self, input_text, lang_prob, force_lang):
        """Pick the language. A native (non-Latin) script is decisive: if the raw
        argmax disagrees with the input's dominant script, re-pick the highest-
        probability language that actually uses that script. This makes language
        essentially perfect for Devanagari/Gujarati/Tamil/… inputs and only leaves
        the genuinely ambiguous Latin cases (en vs hi_en vs es…) to the model."""
        lang_list = self.labels["lang_list"]
        if force_lang:
            return force_lang, False
        lang = ID2LANG[int(lang_prob.argmax())]
        dom, ratio = _dominant_script(input_text)
        if dom and dom != "Latn" and ratio >= 0.5:
            allowed = [c for c in _SCRIPT_TO_LANGS.get(dom, []) if c in lang_list]
            if allowed and lang not in allowed:
                best_code, best_p = lang, -1.0
                for code in allowed:
                    p = float(lang_prob[lang_list.index(code)])
                    if p > best_p:
                        best_p, best_code = p, code
                return best_code, True
        return lang, False

    def _surface(self, lang: str, intent: str, ftype: str) -> str:
        if intent in ("no_filler", "silence") or ftype == "none":
            return ""
        # 1) corpus dictionary (keyed by super-intent under the default scheme)
        cell = (self.filler_dict.get(lang, {}).get(intent, {}) or {}).get(ftype)
        if cell:
            forms = list(cell.keys()); weights = list(cell.values())
            return self.rng.choices(forms, weights=weights, k=1)[0]
        # 2) any type for this intent in the dictionary
        idict = self.filler_dict.get(lang, {}).get(intent, {})
        for t, cell in idict.items():
            if cell:
                return self.rng.choices(list(cell), weights=list(cell.values()), k=1)[0]
        # 3) curated seed lexicon (super-aware fallback)
        if self.intent_scheme == "super":
            return sample_filler_super(lang, intent, ftype, self.rng)
        return sample_filler(lang, intent, ftype, self.rng)
