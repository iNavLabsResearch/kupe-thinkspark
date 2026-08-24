#!/usr/bin/env python3
"""Corpus EDA -> reports/data_eda_report.html (stats + embedded plots).

    python data_scripts/eda_report.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config.paths import CLEAN_CORPUS, EDA_HTML, RAW_CORPUS  # noqa: E402
from config.taxonomy import LANGUAGES  # noqa: E402


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bar(counter: Counter, title: str, color: str) -> str:
    items = counter.most_common()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
    ax.bar(range(len(labels)), vals, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    path = CLEAN_CORPUS if CLEAN_CORPUS.exists() else RAW_CORPUS
    if not path.exists():
        sys.exit("No corpus found. Generate data first.")
    rows = _load(path)
    n = len(rows)

    by_lang = Counter(r["language"] for r in rows)
    by_intent = Counter(r["intent"] for r in rows)
    by_type = Counter(r.get("filler_type", "?") for r in rows)
    by_emotion = Counter(r.get("emotion", "?") for r in rows)
    by_register = Counter(r.get("register", "?") for r in rows)
    lengths = [len(r["input"]) for r in rows]
    ctx_share = sum(1 for r in rows if (r.get("context") or "").strip()) / max(n, 1)
    silence = by_intent.get("no_filler", 0) / max(n, 1)

    imgs = {
        "By language": _bar(by_lang, "rows per language", "#4c72b0"),
        "By intent": _bar(by_intent, "rows per intent", "#c44e52"),
        "By filler_type": _bar(by_type, "rows per filler_type", "#55a868"),
        "By emotion": _bar(by_emotion, "rows per emotion", "#8172b3"),
        "By register": _bar(by_register, "rows per register", "#ccb974"),
    }

    rows_html = "".join(
        f"<tr><td>{LANGUAGES.get(k, {}).get('name', k)}</td><td>{k}</td>"
        f"<td>{LANGUAGES.get(k, {}).get('script', '?')}</td><td>{v:,}</td></tr>"
        for k, v in by_lang.most_common()
    )
    img_html = "".join(
        f"<h3>{t}</h3><img src='data:image/png;base64,{b}'/>" for t, b in imgs.items()
    )
    avg_len = sum(lengths) / max(len(lengths), 1)
    html = f"""<!doctype html><meta charset=utf-8>
<title>ThinkSpark data EDA</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;margin:2rem;max-width:1000px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 10px}}
img{{max-width:100%;border:1px solid #eee;margin:.5rem 0}}.big{{font-size:1.4rem}}</style>
<h1>ThinkSpark — data EDA</h1>
<p class=big><b>{n:,}</b> rows · <b>{len(by_lang)}</b> languages ·
context present in <b>{ctx_share*100:.0f}%</b> ·
silence (<code>no_filler</code>) share <b>{silence*100:.1f}%</b> ·
avg input length <b>{avg_len:.0f}</b> chars</p>
<h2>Per-language coverage</h2>
<table><tr><th>Language</th><th>code</th><th>script</th><th>rows</th></tr>{rows_html}</table>
{img_html}
"""
    EDA_HTML.parent.mkdir(parents=True, exist_ok=True)
    EDA_HTML.write_text(html, encoding="utf-8")
    print(f"[eda] {n:,} rows · {len(by_lang)} langs · silence {silence*100:.1f}% "
          f"· context {ctx_share*100:.0f}%")
    print(f"[eda] wrote {EDA_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
