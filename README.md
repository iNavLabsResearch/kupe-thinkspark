# kupe-thinkspark — a tiny multilingual "thinking sound" predictor

Voice AI agents run **STT → LLM → TTS**. Between the user finishing (STT) and the
agent's real reply (TTS) there is dead air. **ThinkSpark** fills that gap with a
*human* thinking sound / backchannel — `hmm`, `अच्छा`, `एक सेकंड`, `sí sí`,
`ええと` — in the **right language, native script, register and emotion** for the
moment. It's a ~1–3M-param model that trains **locally on a Mac M1** and infers in
single-digit milliseconds.

It mirrors the `kupe-tts` workflow: **Sarvam `gemma4`** generates the data (live
SSE streaming, tqdm, cost ledger), and everything else is local.

```
STT text ──▶  ThinkSpark  ──▶  "हम्म, एक सेकंड…"  ──▶ TTS  ──▶  (LLM answer streams in)
   (input)      + context                (spark)
```

---

## Input vs context (the important design choice)

| field | role | example |
|-------|------|---------|
| **input** | the user's current last line — the **primary** signal | `"अरे यार फिर से वही दिक्कत"` |
| **context** | the **past conversation** so far — multi-turn, **any language or mix** — **modulates** the spark | `"User: रिफंड नहीं आया\nAgent: sir I'm checking\nUser: कितनी बार बोलूँ"` |

**Context is a real past-conversation transcript you pass in**, not an English
note. It can be pure target-language (Hindi turns only), or code-switched /
multilingual (a Hindi user + an English agent turn), or empty on a cold start —
the byte-level tokenizer reads any script with zero OOV, and each row also carries
`context_langs` (the codes actually present). The model encodes input and context
**separately** and the input **cross-attends into** the context (input = query,
context = key/value), so the same input yields a different spark as the
conversation evolves — e.g. an *apologetic/calming* murmur once the user has
repeated a complaint across turns, a *curious* one during fresh small talk. See
[`thinkspark/model.py`](thinkspark/model.py).

---

## What it predicts

Five heads over a shared trunk (all in [`config/taxonomy.py`](config/taxonomy.py)):

- **intent** (headline): `thinking · agreeing · hesitating · surprised · empathetic ·
  apologetic · calming · … · no_filler` — `no_filler` is a first-class **silence**
  class (many turns need silence, not a filler).
- **language** (22): Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu,
  Kannada, Malayalam, Odia, Assamese, Urdu, Hinglish + English, Spanish, French,
  German, Portuguese, Japanese, Mandarin, Arabic, Russian — **native scripts**.
- **register**: `formal · casual · urban_mixed`
- **emotion**: `neutral · warm · concerned · apologetic · playful · …`
- **filler_type**: `sound · word · sound_word · words · none` — so the spark is
  chosen dynamically: a pure sound (`hmm`), a word (`अच्छा`), a blend
  (`hmm अच्छा`), a short phrase (`एक सेकंड`), or nothing.

The predicted `(language, intent, filler_type)` then **samples a real surface
form** from `filler_dictionary.json` (aggregated from the corpus) with a fallback
to the curated `LEXICON`. Editing a language's fillers = editing the lexicon, no
retraining.

**Byte-level tokenizer** ([`thinkspark/tokenizer.py`](thinkspark/tokenizer.py)):
every script is just UTF-8 bytes, so there is **zero OOV** across all languages
with a tiny 259-row embedding.

---

## Setup

```bash
cd kupe-thinkspark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # paste your SARVAM_API_KEY (same key kupe-tts uses)
```

No CUDA, no wandb. Training runs on **MPS** (Apple Silicon) or CPU automatically.

---

## The pipeline

### 0. Smoke test (no API, < 1 min) — do this first

```bash
python scripts/smoke_test.py
```

Generates a synthetic corpus, builds splits + vocab, trains 3 epochs on MPS, and
runs inference — proving the whole thing is wired. (Its *intent* accuracy is low
on purpose: the synthetic inputs carry no real intent signal. Real Sarvam data
does.)

### 1. Generate data — Sarvam `sarvam-105b`

```bash
python scripts/01_generate_data.py                      # full plan, all 22 languages
python scripts/01_generate_data.py --fresh              # wipe corpus + cost ledger first
python scripts/01_generate_data.py --langs hi,hi_en,en  # a subset
python scripts/01_generate_data.py --max-rows 500       # a small taste
```

**Live split-screen** (rich): the **left** pane is a progress bar + stats table
(language, rows kept/rejected, KV-cache %, tokens, rows/s, cost) + recent batch
events; the **right** pane is the **raw SSE token stream** from Sarvam, tagged per
worker, scrolling as it generates. (Falls back to a plain tqdm bar if `rich` is
missing.)

Uses **`sarvam-105b`** (cheaper *and* stronger multilingual than gemma4:
₹29.28 / ₹10.98 / ₹73.20 per 1M input/cached/output). Generation runs **one
language at a time** with a single KV warm-up call, then up to 30 concurrent
requests sharing the **same stable, cached prefix** — into which a per-language
**OUTPUT LANGUAGE LOCK** is baked (anti-leak, zero KV cost). Batches are **50 rows
per call** to amortise that cached prefix, so fewer calls, lower cost/row, faster.
Every returned row is validated (native-script / anti-English-leak guard, taxonomy
check, filler presence) with a live **`rej`** count; exponential-backoff retries;
per-batch cost ledger at `data/raw/generation_costs.csv`. Resumes toward the
per-language targets in [`data_scripts/gen_config.py`](data_scripts/gen_config.py).

### 2. Build the dataset

```bash
python scripts/02_build_dataset.py --val 0.1 --test 0.1
python data_scripts/eda_report.py        # -> reports/data_eda_report.html
```

Validates + **hard-rejects wrong-script leaks** (e.g. Gujarati inside Hindi),
dedupes, **stratifies by language × intent** into `train/val/test.jsonl`, and
writes `label_maps.json` + `filler_dictionary.json`.

### 3. Train locally (M1)

```bash
python scripts/03_train.py --config configs/thinkspark_tiny.yaml
```

Real-time in the terminal: a **tqdm bar per epoch** with running loss, intent
accuracy and LR; a printed **metrics table** after every validation (per-head
accuracy + macro-F1 + throughput + RSS); and **PNG plots** refreshed each epoch
(`training_curves.png`, `confusion_intent.png` in the run dir). Best checkpoint is
kept by val macro-F1; final held-out **test** eval prints per-intent F1.

### 4. Try it

```bash
python scripts/04_infer.py --ckpt artifacts/thinkspark/best \
  --input "अरे यार फिर से वही दिक्कत" \
  --context $'User: रिफंड नहीं आया\nAgent: sir I am checking\nUser: कितनी बार बोलूँ'

python scripts/04_infer.py --ckpt artifacts/thinkspark/best     # interactive REPL
```

In the REPL, type the user's line; prefix a line with `@` to set the context.

---

## Layout

```
config/taxonomy.py       languages, scripts, intents, emotions, filler LEXICON
config/paths.py          all paths + .env loader
data_scripts/            data_gen_agent.py (Sarvam) · build_dataset.py · eda_report.py
thinkspark/              tokenizer · dataset · model · trainer · infer · metrics · plots
configs/                 thinkspark_tiny.yaml (real) · thinkspark_smoke.yaml (fast)
scripts/                 01_generate_data · 02_build_dataset · 03_train · 04_infer · smoke_test
data/  reports/  artifacts/
```

---

## Data quality note

The Indian-language and English filler vocab is written to sound natural; the
low-resource and foreign entries are a solid starting set. As in the `kupe-tts`
plan, **spot-check `urban_mixed` and low-resource outputs with a native speaker**
before production — LLM-generated fillers can be subtly textbook. Extend coverage
by editing `LEXICON` in [`config/taxonomy.py`](config/taxonomy.py) and re-running
step 2; no retraining is needed for new surface forms.




<!-- 
cd kupe-thinkspark && pip install -r requirements.txt
python scripts/smoke_test.py                       # offline, <1 min — PASSES
python scripts/01_generate_data.py --langs hi,en   # real Sarvam data
python scripts/02_build_dataset.py
python scripts/03_train.py --config configs/thinkspark_tiny.yaml
python scripts/04_infer.py --ckpt artifacts/thinkspark/best   # interactive REPL -->