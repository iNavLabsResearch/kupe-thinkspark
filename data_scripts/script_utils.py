#!/usr/bin/env python3
"""Script detection / validation shared by generator + validator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.taxonomy import LANGUAGES, SCRIPTS  # noqa: E402


def _in_ranges(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


def script_ratio(text: str, script_tag: str) -> float:
    """Fraction of *letters* in `text` that belong to `script_tag`.

    Whitespace, digits and punctuation are ignored. ASCII latin is always
    tolerated (loanwords / romanized brand names) except when checking a
    non-Latin script we still count it as 'other'.
    """
    ranges = SCRIPTS.get(script_tag)
    if not ranges:
        return 1.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters if _in_ranges(ord(c), ranges))
    return hits / len(letters)


# Shared across Indic scripts — NOT a language leak.
# U+0964 DEVANAGARI DANDA (।) / U+0965 DOUBLE DANDA (॥) are the normal
# sentence stop for bn/as/hi/mr/or and sit in the Deva block.
_SHARED_PUNCT = {0x0964, 0x0965}


def has_forbidden_indic(text: str, allowed_tag: str) -> bool:
    """True if a *letter* from a different Indic script leaks in.

    Punctuation in the Deva block (। ॥) is allowed — Bengali/Assamese
    sentences almost always end with । and that is not a Hindi leak.
    """
    indic = ["Deva", "Beng", "Gujr", "Guru", "Taml", "Telu", "Knda", "Mlym", "Orya"]
    for tag in indic:
        if tag == allowed_tag:
            continue
        ranges = SCRIPTS[tag]
        for c in text:
            cp = ord(c)
            if cp in _SHARED_PUNCT:
                continue
            if not c.isalpha():
                continue
            if _in_ranges(cp, ranges):
                return True
    return False


def valid_for_language(text: str, script_tag: str, min_ratio: float = 0.55) -> bool:
    """Accept when the native script dominates and no other Indic letters leak.

    Latin loanwords (sorry, wifi, card) do not count against Indic languages —
    ratio is scored on non-Latin letters only. Latin/code-mixed uses a looser bar.
    """
    if not text or not text.strip():
        return False
    if script_tag in ("Latn",):
        return script_ratio(text, "Latn") >= 0.30
    if has_forbidden_indic(text, script_tag):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    native = [c for c in letters if _in_ranges(ord(c), SCRIPTS.get(script_tag, ()))]
    if not native:
        return False
    latn = SCRIPTS.get("Latn", [])
    non_latin = [c for c in letters if not _in_ranges(ord(c), latn)]
    if not non_latin:
        return False
    return (len(native) / len(non_latin)) >= min_ratio


# ---------------------------------------------------------------------------
# Latin-script cross-language leak guard.
# Script alone cannot tell Spanish from English (both Latin), so for the Latin
# foreign languages we use deterministic marker words + diacritics to catch the
# common failure: the model emits English text for a Spanish/French/... row.
# ---------------------------------------------------------------------------
_EN_MARKERS = {
    " the ", " is ", " are ", " you ", " and ", " what ", " to ", " for ",
    " i ", " my ", " of ", " it ", " this ", " that ", " have ", " with ",
    " don't ", " i'm ", " can ", " will ", " not ", " your ",
}
_LATIN_LANG = {
    "es": {"chars": "ñ¿¡áéíóúü",
           "words": {" el ", " la ", " que ", " de ", " no ", " sí ", " está ",
                     " pero ", " muy ", " gracias ", " por ", " los ", " un ", " una "}},
    "fr": {"chars": "àâçéèêëîïôûùœ",
           "words": {" le ", " la ", " est ", " je ", " vous ", " pas ", " oui ",
                     " merci ", " c'est ", " un ", " une ", " des ", " et ", " ça "}},
    "de": {"chars": "äöüß",
           "words": {" ich ", " ist ", " und ", " nicht ", " ja ", " danke ",
                     " das ", " du ", " ein ", " sehr ", " mit ", " wir ", " was "}},
    "pt": {"chars": "ãõáâàéêíóôç",
           "words": {" o ", " a ", " que ", " não ", " sim ", " obrigado ",
                     " você ", " muito ", " está ", " um ", " uma ", " de ", " e "}},
}


def latin_language_ok(text: str, code: str) -> bool:
    """For Latin-script languages: English and Hinglish always pass; foreign
    Latin languages must show a target marker OR at least not read as plain
    English (reject a clear English leak, tolerate short ambiguous lines)."""
    if script_ratio(text, "Latn") < 0.30:
        return False
    if code in ("en", "hi_en"):
        return True
    spec = _LATIN_LANG.get(code)
    if not spec:
        return True  # unknown Latin language — script check only
    low = " " + text.lower() + " "
    has_target = any(ch in low for ch in spec["chars"]) or any(w in low for w in spec["words"])
    if has_target:
        return True
    english_hits = sum(1 for w in _EN_MARKERS if w in low)
    return english_hits == 0  # no target signal AND looks English -> reject


def input_language_ok(text: str, lang_code: str) -> bool:
    """Full leak check for the `input` field, keyed by language code."""
    if not text or not text.strip():
        return False
    meta = LANGUAGES.get(lang_code)
    if not meta:
        return False
    script = meta["script"]
    if script == "Latn":
        return latin_language_ok(text, lang_code)
    return valid_for_language(text, script)
