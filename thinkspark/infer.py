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
    ID2EMOTION, ID2FILLERTYPE, ID2INTENT, ID2LANG, ID2REGISTER, sample_filler,
)
from . import tokenizer as tok
from .model import ThinkSpark


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

        self.filler_dict = {}
        if filler_dict_path and Path(filler_dict_path).exists():
            self.filler_dict = json.loads(Path(filler_dict_path).read_text(encoding="utf-8"))
        self.rng = random.Random(seed)

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
        intent = ID2INTENT[int(prob["intent"].argmax())]
        lang = force_lang or ID2LANG[int(prob["language"].argmax())]
        register = ID2REGISTER[int(prob["register"].argmax())]
        emotion = ID2EMOTION[int(prob["emotion"].argmax())]
        ftype = ID2FILLERTYPE[int(prob["filler_type"].argmax())]

        spark = self._surface(lang, intent, ftype)
        return {
            "spark": spark, "intent": intent, "language": lang,
            "register": register, "emotion": emotion, "filler_type": ftype,
            "confidence": {
                "intent": float(prob["intent"].max()),
                "language": float(prob["language"].max()),
            },
        }

    def _surface(self, lang: str, intent: str, ftype: str) -> str:
        if intent == "no_filler" or ftype == "none":
            return ""
        # 1) corpus dictionary
        cell = (self.filler_dict.get(lang, {}).get(intent, {}) or {}).get(ftype)
        if cell:
            forms = list(cell.keys()); weights = list(cell.values())
            return self.rng.choices(forms, weights=weights, k=1)[0]
        # 2) any type for this intent in the dictionary
        idict = self.filler_dict.get(lang, {}).get(intent, {})
        for t, cell in idict.items():
            if cell:
                return self.rng.choices(list(cell), weights=list(cell.values()), k=1)[0]
        # 3) curated seed lexicon
        return sample_filler(lang, intent, ftype, self.rng)
