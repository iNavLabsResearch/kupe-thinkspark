#!/usr/bin/env python3
"""
ThinkSpark synthetic-data generator (Sarvam · gemma4)
=====================================================
Generates training examples for the ThinkSpark filler/backchannel model:
short spoken user utterances (INPUT) + a running conversation CONTEXT, each
labelled with (language, script, register, intent, emotion, filler_type) and a
list of native filler surface candidates.

Mirrors kupe-tts/text_scripts/text_gen_agent.py:
  * Sarvam open-source chat completions (v2) with SSE live streaming
  * KV-cache-friendly stable system prefix + rotating topic suffix
  * ThreadPool concurrency, exponential-backoff retries
  * per-batch cost metering -> generation_costs.csv
  * resume toward per-language targets
  * live split-screen (rich): LEFT progress+stats, RIGHT raw SSE token stream

Usage:
    python data_scripts/data_gen_agent.py                 # all languages
    python data_scripts/data_gen_agent.py --langs hi,en   # subset
    python data_scripts/data_gen_agent.py --max-rows 500  # global cap (smoke)
    python data_scripts/data_gen_agent.py --concurrency 50

Requires SARVAM_API_KEY (in .env or environment).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from config.paths import (  # noqa: E402
    GEN_COST_CSV, RAW_CORPUS, load_env,
)
from config.taxonomy import (  # noqa: E402
    EMOTIONS, FILLER_TYPES, INTENTS, LANGUAGES,
)
from data_scripts.gen_config import (  # noqa: E402
    BATCH_SIZE, INTENT_FOCUS, LANG_TARGETS, REGISTER_MIXES, TOPICS, total_target,
)
from data_scripts import db as dbmod  # noqa: E402
from data_scripts import tracker  # noqa: E402
from data_scripts.script_utils import input_language_ok  # noqa: E402
from data_scripts.ui import make_ui  # noqa: E402

load_env()

# ---------------------------------------------------------------------------
# Sarvam config (identical endpoint style to kupe-tts)
# ---------------------------------------------------------------------------
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
BASE_URL = "https://api.sarvam.ai/v2"
# gemma4: json_object + reasoning_effort=None → JSON in `content`.
# sarvam-105b: reasoning_effort=low; if API returns no content we auto-fallback to gemma4.
MODEL = os.environ.get("SARVAM_MODEL", "sarvam-105b")

MAX_RETRIES = 5
MAX_TOKENS = 4096
WARM_BATCH_SIZE = 1          # tiny warm — primes KV prefix only, fast first response
WARM_MAX_TOKENS = 1024       # cap warm output (not a full batch=50 JSON)
STREAM_PARSE_SEC = 5         # while SSE streaming, parse + append jsonl this often
PARSER_BUF_MAX = 48_000      # trim SSE parse tail if a stream fragment grows too large
KEEP_EMA_INIT = 0.40         # initial guess: kept rows ≈ 40% of batch request size
KEEP_EMA_ALPHA = 0.25        # smoothing for observed kept-rows-per-request
KEEP_YIELD_FLOOR = 0.12      # min kept/request ratio when sizing gen_count
DEFAULT_CONCURRENCY = int(os.environ.get("SARVAM_CONCURRENCY", "50"))
USE_STREAM = os.environ.get("SARVAM_STREAM", "1") == "1"
WARM_CACHE_FIRST = True   # 1 sequential warm call per language → KV hits on parallel batch
WARM_CALLS = 1
FSYNC_EVERY = 10

# INR per 1M tokens — docs.sarvam.ai/api/getting-started/pricing
_PRICE = {
    "gemma4": (36.60, 13.73, 91.50),
    "sarvam-105b": (29.28, 10.98, 73.20),
    "glm5.2": (128.10, 23.79, 402.60),
}
PRICE_INPUT_PER_M, PRICE_CACHED_PER_M, PRICE_OUTPUT_PER_M = _PRICE.get(
    MODEL, _PRICE["gemma4"],
)

io_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Prompt construction — long STABLE system prefix (max KV-cache reuse), only the
# trailing language/intent/register line varies per call.
# ---------------------------------------------------------------------------
_INTENT_LIST_STR = ", ".join(INTENTS)
_EMOTION_LIST_STR = ", ".join(EMOTIONS)
_FILLERTYPE_LIST_STR = ", ".join(FILLER_TYPES)

SYSTEM_PROMPT = (
    "You generate training data for ThinkSpark, a tiny model that predicts the "
    "human 'thinking sound' / backchannel a voice assistant should make in the "
    "gap between a user speaking and the assistant answering.\n"
    "For each row you invent:\n"
    "  input   = the user's CURRENT last spoken line (ASR-style: short, natural, "
    "disfluent), written in the TARGET language and its NATIVE script.\n"
    "  context = the PAST conversation so far — a multi-turn transcript leading up "
    "to `input`. Format as speaker-tagged turns separated by newlines, e.g. "
    "'User: ...\\nAgent: ...\\nUser: ...'. The last turn should flow naturally into "
    "`input`. Keep it 1-4 short turns.\n"
    "  CONTEXT LANGUAGE — realistic and VARIED, never forced: MOST turns in the "
    "TARGET language + native script, but code-switching is normal — a turn may mix "
    "languages (e.g. Hindi with English words), the agent may answer in English or "
    "another language, or the whole context may be pure target language. Sometimes "
    "(cold start / first turn) context is \"\" (empty).\n"
    "  context_langs = list of language codes that actually appear in the context "
    "(e.g. [\"hi\"], [\"hi\",\"en\"]). Empty context -> [].\n"
    "  register, intent, emotion, filler_type, filler_candidates.\n"
    "NUANCE: the reaction must fit BOTH the input AND the flow of the past "
    "conversation — e.g. if the user has repeated a complaint across turns, react "
    "apologetic/calming, not cheerful.\n"
    "The filler is what the assistant murmurs: a non-lexical SOUND, a short WORD, "
    "a SOUND+WORD blend, or a few WORDS. For intent=no_filler use filler_type='none' "
    "and filler_candidates=[].\n"
    "Filler candidates MUST be authentic to the target language+register and in "
    "the native script. Never put another language's script in a filler.\n"
    "Output ONLY raw JSON: {\"items\":[{...}, ...]}. No prose, no code fences.\n"
    "Each item has exactly: input, context, context_langs, register, intent, "
    "emotion, filler_type, filler_candidates, filler_weights, notes.\n"
    "filler_weights: floats same length as filler_candidates, summing to 1.0.\n"
    "notes: <=8 word tone tag. Vary topics widely. Never repeat near-duplicates.\n"
    f"Allowed intent values: {_INTENT_LIST_STR}.\n"
    f"Allowed emotion values: {_EMOTION_LIST_STR}.\n"
    f"Allowed filler_type values: {_FILLERTYPE_LIST_STR}.\n"
    # Static pad — identical system prefix on every call → Sarvam prompt-cache reuse.
    "REPEAT THIS CONTRACT EVERY CALL: same system instructions, same schema, "
    "only the trailing Topic line of the user message may change."
)


def _language_lock(lang_code: str) -> str:
    """Per-language OUTPUT LANGUAGE LOCK for the `input` field. Lives in the stable
    (cached) prefix, so it costs nothing against the KV cache but strongly cuts
    wrong-language leaks."""
    meta = LANGUAGES[lang_code]
    if lang_code == "hi_en":
        return (
            "OUTPUT LANGUAGE LOCK: `input` is Hinglish — romanized Hindi in Latin "
            "script mixed with English words, as urban Indians actually text/speak. "
            "Do NOT write it in Devanagari or any other Indic script.\n"
        )
    if meta["script"] == "Latn":  # en / es / fr / de / pt
        return (
            f"OUTPUT LANGUAGE LOCK: `input` MUST be natural {meta['name']} only. "
            f"Do NOT write it in English or any other language"
            + ("" if lang_code == "en" else f" — real {meta['name']} words, spelling and diacritics")
            + ". No other script.\n"
        )
    return (
        f"OUTPUT LANGUAGE LOCK: `input` MUST be 100% {meta['name']} ({meta['native']}) "
        f"in {meta['script']} script. Do NOT use English/Latin letters or any other "
        f"Indic/foreign script in `input`. Any other script makes the row invalid.\n"
    )


def stable_prefix(lang_code: str) -> str:
    """Byte-stable user prefix per language — always BATCH_SIZE so KV matches warm+parallel."""
    meta = LANGUAGES[lang_code]
    return (
        f"TARGET language: {meta['name']} ({meta['native']}), code={lang_code}, "
        f"native script={meta['script']}.\n"
        + _language_lock(lang_code)
        + f"Return exactly this JSON shape with up to {BATCH_SIZE} items:\n"
        f'{{"items":[{{"input":"...","context":"User: ...\\nAgent: ...","context_langs":["{lang_code}"],'
        f'"register":"casual","intent":"thinking","emotion":"curious","filler_type":"sound",'
        f'"filler_candidates":["..."],"filler_weights":[1.0],"notes":"tag"}}]}}\n'
        f"context uses \\n between turns and may be \"\" for a cold start.\n"
        f"Keep JSON compact. Vary openings, turn counts and context languages inside the batch.\n"
        f"Do not copy the topic wording verbatim into every line.\n"
    )


def build_user_message(
    lang_code: str, topic: str, register_mix: str, focus: list[str],
    gen_count: int | None = None,
) -> tuple[str, str, str]:
    """Returns (full_user_message, stable_prefix, variable_suffix)."""
    n = gen_count or BATCH_SIZE
    prefix = stable_prefix(lang_code)
    suffix = (
        f"\nTopic: {topic}\n"
        f"Register mix for THIS batch: {register_mix}\n"
        f"Bias intents toward: {', '.join(focus)} "
        f"(stay realistic; keep the natural silence/no_filler class where it fits)\n"
        f"Generate {n} now."
    )
    return prefix + suffix, prefix, suffix


def model_extra_body(lang_code: str, model: str) -> dict:
    extra: dict = {"prompt_cache_key": cache_key(lang_code)}
    if model == "gemma4":
        extra["reasoning_effort"] = None
    elif model == "sarvam-105b":
        extra["reasoning_effort"] = "low"
    return extra


def probe_json_model(client: OpenAI, requested: str) -> str:
    """sarvam-105b often returns reasoning-only on json_object — verify before a long run."""
    if requested != "sarvam-105b":
        return requested
    ping = stable_prefix("hi") + "\nTopic: ping\nGenerate 1 now."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ping},
    ]
    try:
        resp = client.chat.completions.create(
            model="sarvam-105b", messages=messages, temperature=0.7,
            max_tokens=WARM_MAX_TOKENS, response_format={"type": "json_object"},
            extra_body={"reasoning_effort": "low", "prompt_cache_key": "thinkspark:probe"},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content and '"items"' in content:
            return "sarvam-105b"
    except Exception:
        pass
    print(
        "NOTE: sarvam-105b returned no JSON content (reasoning-only on this endpoint). "
        "Using gemma4 for generation; warm KV still uses the same stable prefix."
    )
    return "gemma4"


def cache_key(lang_code: str) -> str:
    return f"thinkspark:{lang_code}"


# ---------------------------------------------------------------------------
# Lenient JSON recovery (truncated stream safe) — same spirit as kupe-tts
# ---------------------------------------------------------------------------
def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text


def scan_item_objects_with_ends(text: str) -> list[tuple[dict, int, int]]:
    """Find item dicts at any nesting depth; return (obj, start, end_exclusive)."""
    text = _strip_json_fence(text)
    if not text:
        return []
    try:
        data = json.loads(text)
        items = data.get("items")
        if isinstance(items, list):
            out: list[tuple[dict, int, int]] = []
            for it in items:
                if isinstance(it, dict) and "input" in it:
                    frag = json.dumps(it, ensure_ascii=False)
                    pos = text.find(frag)
                    if pos >= 0:
                        out.append((it, pos, pos + len(frag)))
            if out:
                return out
    except json.JSONDecodeError:
        pass
    found: list[tuple[dict, int, int]] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            frag = text[start:i + 1]
            try:
                obj = json.loads(frag)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "input" in obj:
                found.append((obj, start, i + 1))
    return found


def size_gen_count(
    remaining: int,
    batch_size: int,
    exp_kept_per_req: float,
    *,
    in_flight: int = 0,
) -> int:
    """Items to request in one API call given remaining target + reject yield."""
    if remaining <= 0:
        return 0
    exp_kept = max(exp_kept_per_req, 1.0)
    yield_rate = max(exp_kept / batch_size, KEEP_YIELD_FLOOR)
    slots = max(in_flight, 0) + 1
    share = remaining / slots
    return min(batch_size, max(1, math.ceil(share / yield_rate)))


def scan_item_objects(text: str) -> list[dict]:
    """Brace-scan for complete item dicts; safe on partial SSE buffers."""
    return [obj for obj, _, _ in scan_item_objects_with_ends(text)]


class IncrementalItemParser:
    """Yield newly completed item dicts as SSE chunks append; compact buffer to save RAM."""

    def __init__(self) -> None:
        self._buf = ""
        self._yielded = 0
        self._total_parsed = 0

    @property
    def total_parsed(self) -> int:
        return self._total_parsed

    @property
    def buf_len(self) -> int:
        return len(self._buf)

    def feed(self, chunk: str) -> list[dict]:
        if chunk:
            self._buf += chunk
        return self._drain()

    def flush(self) -> list[dict]:
        return self._drain()

    def _drain(self) -> list[dict]:
        found = scan_item_objects_with_ends(self._buf)
        new = found[self._yielded:]
        if not new:
            self._trim_oversize()
            return []
        self._yielded = len(found)
        self._total_parsed += len(new)
        last_end = new[-1][2]
        self._buf = self._buf[last_end:].lstrip(", \n\r\t")
        self._yielded = 0
        self._trim_oversize()
        return [obj for obj, _, _ in new]

    def _trim_oversize(self) -> None:
        if len(self._buf) > PARSER_BUF_MAX:
            self._buf = self._buf[-PARSER_BUF_MAX:]
            self._yielded = 0


def parse_items_lenient(raw: str) -> list[dict]:
    items = scan_item_objects(raw)
    if not items:
        raise ValueError(f"no recoverable items (raw_chars={len(raw or '')})")
    return items


# ---------------------------------------------------------------------------
# Validation / normalisation of a single generated item
# ---------------------------------------------------------------------------
def normalize_item_with_reason(lang_code: str, it: dict) -> tuple[dict | None, str | None]:
    meta = LANGUAGES[lang_code]
    inp = (it.get("input") or "").strip()
    if not inp or len(inp) < 2:
        return None, "empty_input"

    intent = (it.get("intent") or "").strip()
    if intent not in INTENTS:
        return None, "bad_intent"
    emotion = (it.get("emotion") or "neutral").strip()
    if emotion not in EMOTIONS:
        emotion = "neutral"
    register = (it.get("register") or "casual").strip()
    if register not in ("formal", "casual", "urban_mixed"):
        register = "casual"
    ftype = (it.get("filler_type") or "").strip()
    if ftype not in FILLER_TYPES:
        ftype = "none" if intent == "no_filler" else "word"

    if not input_language_ok(inp, lang_code):
        return None, "bad_script"

    cands = it.get("filler_candidates") or []
    if not isinstance(cands, list):
        cands = []
    cands = [str(c).strip() for c in cands if str(c).strip()]
    weights = it.get("filler_weights") or []
    if not isinstance(weights, list) or len(weights) != len(cands):
        weights = [round(1.0 / len(cands), 4)] * len(cands) if cands else []

    if intent == "no_filler":
        ftype, cands, weights = "none", [], []
    elif not cands:
        return None, "missing_filler"

    context = (it.get("context") or "").strip()
    ctx_langs = it.get("context_langs") or []
    if not isinstance(ctx_langs, list):
        ctx_langs = []
    ctx_langs = [str(c).strip() for c in ctx_langs if str(c).strip() in LANGUAGES]
    if not ctx_langs and context:
        ctx_langs = [lang_code]

    return {
        "input": inp,
        "context": context,
        "context_langs": ctx_langs,
        "language": lang_code,
        "script": meta["script"],
        "register": register,
        "intent": intent,
        "emotion": emotion,
        "filler_type": ftype,
        "filler_candidates": cands,
        "filler_weights": weights,
        "notes": (it.get("notes") or "").strip()[:80],
    }, None


def normalize_item(lang_code: str, it: dict) -> dict | None:
    rec, _ = normalize_item_with_reason(lang_code, it)
    return rec


# ---------------------------------------------------------------------------
# Cost helpers (same shape as kupe-tts)
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "timestamp_utc", "batch_num", "language", "register_mix", "focus",
    "items_kept", "prompt_tokens", "cached_tokens", "uncached_input_tokens",
    "completion_tokens", "total_tokens", "cache_hit_pct", "cache_reported",
    "prefix_share_pct", "est_cacheable_tokens", "cost_inr",
    "cost_if_prefix_cached_inr", "run_cost_inr", "cum_cost_inr", "corpus_items",
]


def usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    try:
        return usage.model_dump()
    except Exception:
        return {}


def extract_cached_tokens(usage) -> tuple[int, bool]:
    """Return (cached_tokens, reported). reported=False if Sarvam omits cache fields."""
    d = usage_to_dict(usage)
    if not d:
        return 0, False
    candidates: list[int] = []
    ptd = d.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        for key in (
            "cached_tokens", "cache_read_input_tokens",
            "cached_prompt_tokens", "prompt_cache_hit_tokens",
        ):
            if ptd.get(key) is not None:
                candidates.append(int(ptd.get(key) or 0))
    for key in ("cached_tokens", "cache_read_input_tokens", "cached_prompt_tokens"):
        if d.get(key) is not None:
            candidates.append(int(d.get(key) or 0))
    if ptd is None and not candidates:
        return 0, False
    if not candidates:
        return 0, True
    return max(candidates), True


def estimate_prefix_cacheable(
    prompt_tokens: int, stable_chars: int, total_chars: int,
) -> tuple[int, float]:
    if prompt_tokens <= 0 or total_chars <= 0:
        return 0, 0.0
    share = min(max(stable_chars / total_chars, 0.0), 1.0)
    return int(round(prompt_tokens * share)), 100.0 * share


def calc_cost_inr(prompt: int, cached: int, completion: int) -> float:
    """Sarvam gemma4 ₹/1M: input / cached / output — same as kupe-tts text_gen_agent.py."""
    cached = min(max(cached, 0), max(prompt, 0))
    uncached = max(prompt - cached, 0)
    return (uncached * PRICE_INPUT_PER_M + cached * PRICE_CACHED_PER_M
            + completion * PRICE_OUTPUT_PER_M) / 1_000_000.0


def billing_cached_tokens(
    api_cached: int,
    cache_reported: bool,
    est_cacheable: int,
    *,
    assume_prefix_hit: bool,
) -> tuple[int, str]:
    """Prefer API-reported cache; after warm, bill stable prefix at cached rate."""
    if cache_reported and api_cached > 0:
        return api_cached, "API"
    if assume_prefix_hit and est_cacheable > 0:
        return est_cacheable, "prefix-est"
    return api_cached, "none"


def ensure_cost_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_cost_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        f.flush()


def load_cum_cost(path: Path) -> float:
    if not path.exists():
        return 0.0
    last = 0.0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = float(row.get("cum_cost_inr") or 0)
            except (TypeError, ValueError):
                pass
    return last


def load_existing_counts(path: Path) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {c: 0 for c in LANGUAGES}
    n = 0
    if not path.exists():
        return counts, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[r.get("language", "")] = counts.get(r.get("language", ""), 0) + 1
            n += 1
    return counts, n


# ---------------------------------------------------------------------------
# Model call (SSE streaming with retries)
# ---------------------------------------------------------------------------
def call_model(
    client: OpenAI, lang_code: str, topic: str, register_mix: str,
    focus: list[str], ui_sink=None, tag: str = "",
    *, model: str, gen_count: int | None = None, max_tokens: int | None = None,
    on_items: Callable[[list[dict]], None] | None = None,
):
    n = gen_count or BATCH_SIZE
    cap = max_tokens or MAX_TOKENS
    user_msg, stable_prefix_text, suffix = build_user_message(
        lang_code, topic, register_mix, focus, gen_count=n,
    )
    stable_chars = len(SYSTEM_PROMPT) + len(stable_prefix_text)
    total_chars = len(SYSTEM_PROMPT) + len(user_msg)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    extra = model_extra_body(lang_code, model)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        raw = ""
        stream_chars = 0
        parser: IncrementalItemParser | None = None
        try:
            if USE_STREAM:
                if ui_sink is not None:
                    ui_sink.begin_stream(tag, f"attempt {attempt}/{MAX_RETRIES} · n={n}")
                stream = client.chat.completions.create(
                    model=model, messages=messages, temperature=1.0, top_p=0.95,
                    max_tokens=cap, response_format={"type": "json_object"},
                    stream=True, stream_options={"include_usage": True},
                    extra_body=extra,
                )
                usage = None
                got_token = False
                last_pulse = time.time()
                parser = IncrementalItemParser() if on_items else None
                pending: list[dict] = []
                last_parse_flush = time.time()
                stream_chars = 0
                parts: list[str] | None = [] if not on_items else None
                for chunk in stream:
                    now = time.time()
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        c = getattr(delta, "content", None)
                        if c:
                            got_token = True
                            stream_chars += len(c)
                            if parts is not None:
                                parts.append(c)
                            if ui_sink is not None:
                                ui_sink.feed(tag, c)
                            if parser is not None:
                                pending.extend(parser.feed(c))
                    if ui_sink is not None and not got_token:
                        if now - last_pulse >= 5.0:
                            ui_sink.pulse_stream(
                                tag,
                                f"waiting for first token ({model}, n={n}) …",
                            )
                            last_pulse = now
                    if parser is not None and on_items and now - last_parse_flush >= STREAM_PARSE_SEC:
                        if pending:
                            on_items(pending)
                            pending.clear()
                        last_parse_flush = now
                if parser is not None:
                    pending.extend(parser.flush())
                    if pending and on_items:
                        on_items(pending)
                        pending.clear()
                    items = [] if parser.total_parsed == 0 else [None]
                    raw_len = stream_chars
                else:
                    raw = "".join(parts or []).strip()
                    items = parse_items_lenient(raw)[:n] if raw else []
                    raw_len = len(raw)
                if ui_sink is not None:
                    ui_sink.end_stream(tag, raw_len, parser.total_parsed if parser else len(items))
            else:
                if ui_sink is not None:
                    ui_sink.begin_stream(tag, f"attempt {attempt}/{MAX_RETRIES} (non-stream)")
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=1.0, top_p=0.95,
                    max_tokens=cap, response_format={"type": "json_object"},
                    extra_body=extra,
                )
                raw = resp.choices[0].message.content or ""
                usage = resp.usage
                items = parse_items_lenient(raw)[:n] if raw else []
                if on_items and items:
                    on_items(items)
                if ui_sink is not None:
                    ui_sink.end_stream(tag, len(raw), len(items))
            if not items:
                hint = stream_chars if parser is not None else len(raw or "")
                raise ValueError(f"no recoverable items (raw_chars={hint})")
            return items, usage, stable_chars, total_chars
        except Exception as e:  # noqa: BLE001
            last_err = e
            if ui_sink is not None:
                ui_sink.pulse_stream(tag, f"retry in {min(2 ** attempt, 20)}s — {e}")
            wait = min(2 ** attempt, 20)
            time.sleep(wait)
    raise RuntimeError(f"failed {lang_code} after {MAX_RETRIES}: {last_err}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def print_lang_progress(counts: dict[str, int], targets: dict[str, int], langs: list[str]) -> None:
    print(f"  {'code':<7} {'language':<14} {'done':>7} {'target':>7} {'left':>7} {'pct':>6}")
    print("  " + "-" * 54)
    for c in langs:
        meta = LANGUAGES[c]
        done = min(counts.get(c, 0), targets[c])
        tgt = targets[c]
        left = max(tgt - done, 0)
        pct = (100.0 * done / tgt) if tgt else 100.0
        print(f"  {c:<7} {meta['name']:<14} {done:>7,} {tgt:>7,} {left:>7,} {pct:>5.1f}%")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate ThinkSpark training data via Sarvam gemma4.")
    ap.add_argument(
        "--langs", default="",
        help="comma list of language codes (default: all 22 languages in gen_config)",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="global cap on NEW rows this run (0 = full plan)")
    ap.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"parallel Sarvam requests (default {DEFAULT_CONCURRENCY}, env SARVAM_CONCURRENCY)",
    )
    ap.add_argument(
        "--fresh", action="store_true",
        help="delete existing corpus + cost ledger before generating",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not SARVAM_API_KEY:
        sys.exit("Set SARVAM_API_KEY (in kupe-thinkspark/.env or environment).")

    langs = [c.strip() for c in args.langs.split(",") if c.strip()] or sorted(LANG_TARGETS.keys())
    langs = [c for c in langs if c in LANGUAGES]
    targets = {c: LANG_TARGETS[c] for c in langs}

    if args.fresh:
        for path in (RAW_CORPUS, GEN_COST_CSV):
            if path.exists():
                path.unlink()
        tracker.clear_all()

    tracker.init()
    indexed = tracker.hydrate_from_corpus(RAW_CORPUS, targets)
    tracker.sync_sqlite(targets)

    client = OpenAI(
        api_key=SARVAM_API_KEY, base_url=BASE_URL,
        default_headers={"api-subscription-key": SARVAM_API_KEY},
        timeout=600.0,
    )
    gen_model = probe_json_model(client, MODEL)
    global PRICE_INPUT_PER_M, PRICE_CACHED_PER_M, PRICE_OUTPUT_PER_M
    PRICE_INPUT_PER_M, PRICE_CACHED_PER_M, PRICE_OUTPUT_PER_M = _PRICE.get(
        gen_model, _PRICE["gemma4"],
    )
    ensure_cost_csv(GEN_COST_CSV)
    RAW_CORPUS.parent.mkdir(parents=True, exist_ok=True)

    counts = tracker.lang_kept_counts()
    corpus_items = tracker.total_corpus_rows()
    cum_cost = load_cum_cost(GEN_COST_CSV)
    if cum_cost == 0.0:
        with dbmod.connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(cum_cost_inr), 0) FROM batches").fetchone()
            cum_cost = float(row[0] or 0)
    plan_total = sum(targets.values())
    have = sum(min(counts.get(c, 0), targets[c]) for c in langs)

    if indexed:
        print(f"Indexed {indexed:,} existing JSONL rows into SQLite tracker.")

    print(f"Requested model: {MODEL} @ {BASE_URL}")
    if gen_model != MODEL:
        print(f"Generation model: {gen_model} (auto-fallback)")
    else:
        print(f"Generation model: {gen_model}")
    print(f"Languages: {len(langs)} | plan {plan_total:,} rows "
          f"(all-lang plan {total_target():,}) | already have {corpus_items:,} rows")
    print(f"Prior cumulative cost ₹{cum_cost:.4f}")
    print(
        f"Pricing (₹/1M): input={PRICE_INPUT_PER_M}  "
        f"cached={PRICE_CACHED_PER_M}  output={PRICE_OUTPUT_PER_M}"
    )
    print(
        f"Concurrency={args.concurrency} batch={BATCH_SIZE} warm_n={WARM_BATCH_SIZE} "
        f"stream={'SSE' if USE_STREAM else 'off'} "
        f"jsonl_flush={STREAM_PARSE_SEC}s "
        f"warm_kv={'on' if WARM_CACHE_FIRST else 'off'} (once/lang, {WARM_CALLS} call/s)"
    )
    print("\nPer-language progress:")
    print_lang_progress(counts, targets, langs)
    print()

    reg_cycle = itertools.cycle(REGISTER_MIXES)
    focus_cycle = itertools.cycle(INTENT_FOCUS)
    topic_cycles = {c: itertools.cycle(TOPICS) for c in langs}

    if all(counts.get(c, 0) >= targets[c] for c in langs):
        print("Nothing to do — all language targets already met.")
        return 0

    run_cost = 0.0
    run_prompt = run_cached = run_completion = 0
    new_rows = 0
    run_rejected = 0
    batch_num = tracker.max_batch_num()
    commits_since_fsync = 0
    start = time.time()
    stop = False
    warmed_langs: set[str] = set()
    lang_keep_ema: dict[str, float] = {
        c: BATCH_SIZE * KEEP_EMA_INIT for c in langs
    }

    out_f = RAW_CORPUS.open("a", encoding="utf-8")
    ui = make_ui(plan_total, have, gen_model)
    ui.start()

    def write_items(lang: str, items: list[dict], *, batch_id: int | None = None) -> tuple[int, int]:
        """Append validated rows to jsonl as SSE completes them (thread-safe)."""
        nonlocal new_rows, corpus_items, stop, run_rejected, commits_since_fsync
        kept = rejected = 0
        with io_lock:
            if stop:
                return kept, rejected
            for it in items:
                rec, fail_reason = normalize_item_with_reason(lang, it)
                intent = (it.get("intent") or "").strip()
                preview = (it.get("input") or "")[:120]
                if rec is None:
                    rejected += 1
                    tracker.record_row(
                        row_id=tracker.allocate_row_id(),
                        language=lang,
                        batch_id=batch_id,
                        status="rejected",
                        fail_reason=fail_reason or "validation",
                        intent=intent,
                        input_preview=preview,
                    )
                    continue
                if counts.get(lang, 0) >= targets[lang]:
                    continue
                if args.max_rows and new_rows >= args.max_rows:
                    stop = True
                    break
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jsonl_line = corpus_items + 1
                counts[lang] = counts.get(lang, 0) + 1
                corpus_items += 1
                new_rows += 1
                kept += 1
                tracker.record_row(
                    row_id=tracker.allocate_row_id(),
                    language=lang,
                    batch_id=batch_id,
                    status="kept",
                    intent=rec.get("intent") or "",
                    input_preview=rec.get("input") or "",
                    jsonl_line=jsonl_line,
                )
            if kept:
                out_f.flush()
                commits_since_fsync += 1
                if commits_since_fsync >= FSYNC_EVERY:
                    os.fsync(out_f.fileno())
                    commits_since_fsync = 0
            run_rejected += rejected

        if kept:
            elapsed = max(time.time() - start, 1e-6)
            lang_done = min(counts.get(lang, 0), targets[lang])
            ui.advance(kept)
            ui.set_stats(
                lang=lang, lang_done=lang_done, lang_target=targets[lang],
                rows=new_rows, rejected=run_rejected, batches=batch_num,
                cache_pct=(100.0 * run_cached / run_prompt) if run_prompt else 0.0,
                prefix_pct=0.0,
                prompt_tok=run_prompt, out_tok=run_completion,
                rows_per_s=new_rows / elapsed,
                cost_per_1k=(run_cost / new_rows * 1000.0) if new_rows else 0.0,
                run_cost=run_cost, cum_cost=cum_cost,
            )
        return kept, rejected

    def finalize_batch(
        lang: str, topic: str, reg_mix: str, focus: list[str],
        usage, stable_chars: int, total_chars: int, *,
        batch_id: int, kept: int, rejected: int, assume_prefix_hit: bool,
    ) -> None:
        """Meter cost + log once per API call (rows may already be in jsonl)."""
        nonlocal run_cost, cum_cost, batch_num, run_prompt, run_cached, run_completion

        prev = lang_keep_ema.get(lang, BATCH_SIZE * KEEP_EMA_INIT)
        lang_keep_ema[lang] = KEEP_EMA_ALPHA * kept + (1.0 - KEEP_EMA_ALPHA) * prev

        pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct)) if usage else (pt + ct)
        api_cached, cache_reported = extract_cached_tokens(usage)
        est_cacheable, prefix_share_pct = estimate_prefix_cacheable(
            pt, stable_chars, total_chars,
        )
        billed_cached, cache_mode = billing_cached_tokens(
            api_cached, cache_reported, est_cacheable,
            assume_prefix_hit=assume_prefix_hit,
        )
        uncached = max(pt - billed_cached, 0)
        cost = calc_cost_inr(pt, billed_cached, ct)
        cost_if_prefix = calc_cost_inr(pt, est_cacheable, ct)

        with io_lock:
            run_cost += cost
            cum_cost += cost
            run_prompt += pt
            run_cached += billed_cached
            run_completion += ct

            append_cost_row(GEN_COST_CSV, {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "batch_num": batch_num, "language": lang,
                "register_mix": reg_mix, "focus": "|".join(focus),
                "items_kept": kept, "prompt_tokens": pt,
                "cached_tokens": billed_cached,
                "uncached_input_tokens": uncached,
                "completion_tokens": ct, "total_tokens": tt,
                "cache_hit_pct": f"{(100.0 * billed_cached / pt) if pt else 0.0:.1f}",
                "cache_reported": "1" if cache_reported else "0",
                "prefix_share_pct": f"{prefix_share_pct:.1f}",
                "est_cacheable_tokens": est_cacheable,
                "cost_inr": f"{cost:.6f}",
                "cost_if_prefix_cached_inr": f"{cost_if_prefix:.6f}",
                "run_cost_inr": f"{run_cost:.6f}", "cum_cost_inr": f"{cum_cost:.6f}",
                "corpus_items": corpus_items,
            })
            tracker.finish_batch(batch_id, {
                "batch_num": batch_num,
                "language": lang,
                "topic": topic,
                "register_mix": reg_mix,
                "focus": "|".join(focus),
                "status": "done",
                "model": gen_model,
                "items_kept": kept,
                "items_rejected": rejected,
                "prompt_tokens": pt,
                "cached_tokens": billed_cached,
                "completion_tokens": ct,
                "total_tokens": tt,
                "cache_hit_pct": (100.0 * billed_cached / pt) if pt else 0.0,
                "cost_inr": cost,
                "cum_cost_inr": cum_cost,
            })
            tracker.sync_sqlite(targets)

        hit = (100.0 * billed_cached / pt) if pt else 0.0
        lang_done = min(counts.get(lang, 0), targets[lang])
        elapsed = max(time.time() - start, 1e-6)
        run_cache_pct = (100.0 * run_cached / run_prompt) if run_prompt else 0.0
        ui.set_stats(
            lang=lang, lang_done=lang_done, lang_target=targets[lang],
            rows=new_rows, rejected=run_rejected, batches=batch_num,
            cache_pct=run_cache_pct, prefix_pct=prefix_share_pct,
            prompt_tok=run_prompt, out_tok=run_completion,
            rows_per_s=new_rows / elapsed,
            cost_per_1k=(run_cost / new_rows * 1000.0) if new_rows else 0.0,
            run_cost=run_cost, cum_cost=cum_cost,
            exp_keep=lang_keep_ema.get(lang, BATCH_SIZE * KEEP_EMA_INIT),
        )
        ui.event(
            f"[batch {batch_num:>4}] {lang:6s} +{kept:<2} rej {rejected:<2} "
            f"({lang_done:,}/{targets[lang]:,}) {topic[:16]:16s} "
            f"in={pt} out={ct} KV {hit:.0f}% ₹{cost:.3f}"
        )

    def run_batch(
        lang: str, topic: str, reg_mix: str, focus: list[str], tag: str,
        *, gen_count: int | None = None, max_tokens: int | None = None,
        assume_prefix_hit: bool,
    ) -> None:
        nonlocal batch_num
        n = gen_count or BATCH_SIZE
        batch_kept = 0
        batch_rej = 0
        with io_lock:
            nonlocal batch_num
            batch_num += 1
            batch_num_local = batch_num
            batch_id = tracker.allocate_batch_id()
        tracker.begin_batch(
            batch_id=batch_id,
            batch_num=batch_num_local,
            language=lang,
            topic=topic,
            register_mix=reg_mix,
            focus=focus,
            stream_tag=tag,
            model=gen_model,
            items_requested=n,
        )

        def on_items(new_items: list[dict]) -> None:
            nonlocal batch_kept, batch_rej
            if batch_kept >= n or not new_items:
                return
            k, r = write_items(lang, new_items[: n - batch_kept], batch_id=batch_id)
            batch_kept += k
            batch_rej += r

        try:
            _items, usage, stable_chars, total_chars = call_model(
                client, lang, topic, reg_mix, focus,
                ui_sink=ui, tag=tag, model=gen_model,
                gen_count=gen_count, max_tokens=max_tokens,
                on_items=on_items,
            )
            finalize_batch(
                lang, topic, reg_mix, focus, usage,
                stable_chars, total_chars,
                batch_id=batch_id,
                kept=batch_kept, rejected=batch_rej,
                assume_prefix_hit=assume_prefix_hit,
            )
        except Exception as exc:
            tracker.finish_batch(batch_id, {
                "batch_num": batch_num_local,
                "language": lang,
                "topic": topic,
                "register_mix": reg_mix,
                "focus": "|".join(focus),
                "status": "failed",
                "model": gen_model,
                "items_kept": batch_kept,
                "items_rejected": batch_rej,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
            raise

    def warm_language(lang: str) -> None:
        """Prime Sarvam KV once per language; warm rows are committed like any batch."""
        topic = next(topic_cycles[lang])
        reg_mix = next(reg_cycle)
        focus = next(focus_cycle)
        ui.event(
            f">>> warming KV cache [{lang}] ({WARM_CALLS}× n={WARM_BATCH_SIZE}) — "
            f"small priming call, output saved if valid"
        )
        for i in range(WARM_CALLS):
            try:
                run_batch(
                    lang, topic, reg_mix, focus, f"{lang}·warm",
                    gen_count=WARM_BATCH_SIZE,
                    max_tokens=WARM_MAX_TOKENS,
                    assume_prefix_hit=False,
                )
                ui.event(
                    f"warm [{lang}] "
                    f"({counts.get(lang, 0):,}/{targets[lang]:,}) — primed KV prefix"
                )
            except Exception as e:  # noqa: BLE001
                ui.event(f"!! FAILED warm [{lang}] pass {i + 1}: {e}")
        warmed_langs.add(lang)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for lang in langs:
            if stop or (args.max_rows and new_rows >= args.max_rows):
                break
            if counts.get(lang, 0) >= targets[lang]:
                continue

            meta = LANGUAGES[lang]
            ui.event(f"--- {lang} ({meta['name']}) {counts.get(lang, 0):,}/{targets[lang]:,} ---")

            if WARM_CACHE_FIRST and lang not in warmed_langs:
                warm_language(lang)
                if stop:
                    break

            def remaining_for_lang() -> int:
                left = targets[lang] - counts.get(lang, 0)
                if args.max_rows:
                    left = min(left, args.max_rows - new_rows)
                return max(left, 0)

            pending: dict = {}
            job_seq = 0

            def submit_one() -> bool:
                nonlocal job_seq
                if stop or remaining_for_lang() <= 0:
                    return False
                exp_kept = lang_keep_ema.get(lang, BATCH_SIZE * KEEP_EMA_INIT)
                gen_n = size_gen_count(
                    remaining_for_lang(), BATCH_SIZE, exp_kept,
                    in_flight=len(pending),
                )
                if gen_n <= 0:
                    return False
                job_seq += 1
                tag = f"{lang}·j{job_seq}"
                topic = next(topic_cycles[lang])
                rm = next(reg_cycle)
                fo = next(focus_cycle)
                fut = pool.submit(
                    run_batch, lang, topic, rm, fo, tag,
                    gen_count=gen_n,
                    assume_prefix_hit=lang in warmed_langs,
                )
                pending[fut] = gen_n
                return True

            primed = 0
            while len(pending) < args.concurrency and submit_one():
                primed += 1
            if primed:
                exp_kept = lang_keep_ema.get(lang, BATCH_SIZE * KEEP_EMA_INIT)
                ui.event(
                    f">>> up to {args.concurrency} parallel for {lang} "
                    f"(~{exp_kept:.0f} kept/req, {remaining_for_lang():,} left, continuous fill)"
                )

            while pending and not stop:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    gen_n = pending.pop(fut)
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001
                        ui.event(f"!! FAILED {lang} n={gen_n}: {e}")
                while len(pending) < args.concurrency and submit_one():
                    pass
                if remaining_for_lang() <= 0:
                    break

            if not stop:
                done = min(counts.get(lang, 0), targets[lang])
                ui.event(f"=== {lang} ({meta['name']}) complete: {done:,}/{targets[lang]:,} ===")

    if commits_since_fsync:
        os.fsync(out_f.fileno())

    ui.stop()
    out_f.close()
    elapsed = time.time() - start
    cache_pct = (100.0 * run_cached / run_prompt) if run_prompt else 0.0
    tracker.write_run_state({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "new_rows": new_rows,
        "run_rejected": run_rejected,
        "batch_num": batch_num,
        "run_cost_inr": run_cost,
        "cum_cost_inr": cum_cost,
        "elapsed_sec": elapsed,
    })
    print("\n" + "=" * 68)
    print(f"DONE. +{new_rows:,} new rows in {batch_num} API calls ({elapsed/60:.1f} min)")
    print(f"Tokens prompt={run_prompt:,} cached={run_cached:,} ({cache_pct:.1f}%) "
          f"completion={run_completion:,}")
    print(f"Cost this run ₹{run_cost:.4f} | cumulative ₹{cum_cost:.4f}")
    print(f"Corpus: {RAW_CORPUS} ({corpus_items:,} rows)")
    print("\nFinal per-language counts:")
    print_lang_progress(counts, targets, langs)
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
