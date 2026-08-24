# kupe-thinkspark — ultra-lightweight multilingual "thinking sound" predictor

<p align="center">
  <img src="docs/architecture.svg" alt="ThinkSpark ultra-lightweight dual-encoder architecture" width="100%">
</p>

Voice AI agents run **STT → LLM → TTS**. Between the user finishing (STT) and the
agent's real reply (TTS) there is dead air. **ThinkSpark** fills that gap with a
*human* thinking sound / backchannel — `hmm`, `अच्छा`, `एक सेकंड`, `sí sí`,
`ええと` — in the **right language, native script, register and emotion** for the
moment.

It is **ultra-lightweight**: ~4.4M parameters, a 259-row byte vocab, no GPU at
inference, **2–8 ms on CPU**. Smaller than a JPEG. Trains on a Mac M1 / Kaggle
T4, then ships as ONNX into the voice worker.

It mirrors the `kupe-tts` workflow: **Sarvam** generates the data (live SSE
streaming, tqdm, cost ledger), and everything else is local.

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
[`thinkspark/model.py`](thinkspark/model.py) and the architecture SVG at the top of this README.

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

No CUDA required on a Mac. Training auto-picks **CUDA** (Kaggle / Colab / NVIDIA PC), **MPS** (Apple Silicon), or **CPU**. Two CUDA GPUs (Kaggle **T4 x2**) train with DistributedDataParallel automatically.

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

### 3. Train — fetch from Hugging Face, then train (Mac / PC / Colab / Kaggle)

```bash
python scripts/03_train.py --config configs/thinkspark_tiny.yaml
```

**Data:** if `data/splits/{train,val,test}.jsonl` and `data/vocab/label_maps.json` are already on disk, training starts immediately. If they are missing, they are downloaded from [`anuj-inavlabs/kupe-thinkspark`](https://huggingface.co/datasets/anuj-inavlabs/kupe-thinkspark) (public; `HF_TOKEN` optional). Next run reuses the local files.

```bash
python scripts/03_train.py --refresh-data     # force re-download
python scripts/03_train.py --no-fetch         # local files only
python scripts/03_train.py --gpus 1           # single GPU even if 2 are visible
python scripts/03_train.py --gpus 2           # DDP on 2 GPUs (Kaggle T4 x2)
```

| Machine | What happens |
|---------|----------------|
| **MacBook** (MPS) | skip DDP, train on Apple GPU |
| **Colab 1 GPU** | CUDA + fp16 AMP, one process |
| **Kaggle T4 x2** | auto DDP on both T4s (~2× throughput) |
| **any NVIDIA PC** | 1 GPU → single process; 2+ GPUs → DDP |

**Kaggle (T4 x2) / Colab notebook** — enable internet, then in the first cell set
credentials (or add them under **Add-ons → Secrets** as `HF_TOKEN` / `SARVAM_API_KEY`
and read them with `UserSecretsClient`):

```python
import os
os.environ["HF_TOKEN"] = "hf_..."            # write token — needed to push the model
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
# os.environ["SARVAM_API_KEY"] = "sk_..."    # only if you generate data on Kaggle
```

Clone, install, train:

```python
# clone your repo, then:
!pip install -r requirements.txt
!python scripts/03_train.py --config configs/thinkspark_tiny.yaml
```

Training is **15 epochs** with **early stopping** (patience 4, min Δ 0.002 on val
macro-F1). Extra epochs are a ceiling, not a forced march — the best checkpoint
is always kept. After training:

```python
!python scripts/export_onnx.py --ckpt artifacts/thinkspark/best
!python scripts/push_to_hf.py     # uploads weights + ONNX + architecture.svg model card
```

Real-time in the terminal: a **tqdm bar per epoch** with running loss, intent
accuracy and LR; a printed **metrics table** after every validation (per-head
accuracy + macro-F1 + throughput + RSS); and **PNG plots** refreshed each epoch
(`training_curves.png`, `confusion_intent.png` in the run dir). Best checkpoint is
kept by val macro-F1; final held-out **test** eval prints per-intent F1.

Optional: `torchrun --nproc_per_node=2 scripts/03_train.py` also works (skips auto-spawn).

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
thinkspark/              tokenizer · dataset · model · trainer · infer · metrics · plots · hf_data
configs/                 thinkspark_tiny.yaml (real, 15 epochs) · thinkspark_smoke.yaml (fast)
scripts/                 01_generate_data · 02_build_dataset · 03_train · 04_infer · push_to_hf · export_onnx
docs/architecture.svg    dual-encoder diagram (also the HF model-card hero)
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