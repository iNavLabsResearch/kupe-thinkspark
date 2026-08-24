"""Live split-screen for data generation.

LEFT  = progress bar ("tqdm"-style) + a stats table + recent batch events.
RIGHT = the raw SSE token stream from Sarvam, tagged per worker, scrolling live.

`GenUI` uses `rich` (added to requirements). If rich is unavailable, `make_ui`
falls back to `PlainUI`, a tqdm bar with periodic prints, so generation still runs.
Both expose the same interface:

    ui.start() / ui.stop()
    ui.advance(n_rows)
    ui.set_stats(**kv)
    ui.event(line)                 # a batch-summary line
    ui.feed(tag, text_chunk)       # streamed SSE tokens
    ui.begin_stream(tag, note)     # before HTTP/SSE connect
    ui.end_stream(tag, chars, n)   # after parse
    ui.pulse_stream(tag, note)     # heartbeat while waiting for first token
"""

from __future__ import annotations

import threading
import time
from collections import deque

_WRAP = 72           # max visible chars per SSE line (content ellipsized)
_FEED_LINES = 80     # ring buffer of streamed lines (keep small for RAM)
_EVENTS = 8          # recent batch-event lines shown on the left
_CUR_MAX = 256       # max partial-line chars kept per stream tag


def _ellipsize(text: str, max_len: int = _WRAP) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 1:
        return "…"
    return text[: max_len - 1] + "…"


class PlainUI:
    """No-rich fallback: a tqdm bar + occasional prints. Same interface."""

    def __init__(self, plan_total: int, initial: int):
        from tqdm import tqdm
        self._bar = tqdm(total=plan_total, initial=initial, unit="row",
                         desc="corpus", dynamic_ncols=True, smoothing=0.3)
        self._stats = {}

    def start(self): pass

    def stop(self): self._bar.close()

    def advance(self, n): self._bar.update(n)

    def set_stats(self, **kv):
        self._stats.update(kv)
        self._bar.set_postfix_str(
            f"₹{self._stats.get('cum_cost', 0):.3f} | KV {self._stats.get('cache_pct', 0):.0f}%"
            f" | rej {self._stats.get('rejected', 0)}", refresh=False)

    def event(self, line: str):
        from tqdm import tqdm
        tqdm.write(line)

    def feed(self, tag: str, text: str):
        # streaming tokens are noisy in plain mode; drop them (bar carries progress)
        pass

    def begin_stream(self, tag: str, note: str = ""):
        from tqdm import tqdm
        tqdm.write(f"[{tag}] SSE START {note}")

    def end_stream(self, tag: str, chars: int, items: int):
        from tqdm import tqdm
        tqdm.write(f"[{tag}] SSE END items={items} chars={chars}")

    def pulse_stream(self, tag: str, note: str):
        from tqdm import tqdm
        tqdm.write(f"[{tag}] … {note}")


class GenUI:
    """rich split-screen. Renders itself: Live re-calls __rich__ each refresh so
    the stats table and SSE feed always reflect the latest state under a lock."""

    def __init__(self, plan_total: int, initial: int, model: str, refresh: int = 12):
        # import rich lazily so PlainUI can be used when rich is absent
        from rich.layout import Layout
        from rich.live import Live
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )

        self._lock = threading.Lock()
        self.model = model
        self.plan_total = plan_total
        self.progress = Progress(
            TextColumn("[bold cyan]rows"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            expand=True,
        )
        self.task = self.progress.add_task("gen", total=plan_total, completed=initial)

        self._stats: dict = {}
        self._feed: deque[str] = deque(maxlen=_FEED_LINES)
        self._cur: dict[str, str] = {}   # tag -> partial line buffer
        self._events: deque[str] = deque(maxlen=_EVENTS)
        self._stream_tag: str = ""
        self._stream_chars: dict[str, int] = {}
        self._stream_started: float = 0.0
        self._last_refresh: float = 0.0

        self._Layout = Layout
        self.layout = Layout()
        self.layout.split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )
        self.live = Live(self, refresh_per_second=refresh, screen=False)

    # ---- lifecycle ----
    def start(self): self.live.start()

    def stop(self):
        try:
            self.live.stop()
        except Exception:
            pass

    # ---- mutations (thread-safe) ----
    def advance(self, n: int):
        if n:
            self.progress.advance(self.task, n)

    def set_stats(self, **kv):
        with self._lock:
            self._stats.update(kv)

    def event(self, line: str):
        with self._lock:
            self._events.append(line)

    def _refresh(self) -> None:
        try:
            self.live.refresh()
        except Exception:
            pass

    def begin_stream(self, tag: str, note: str = ""):
        with self._lock:
            self._stream_tag = tag
            self._stream_chars[tag] = 0
            self._stream_started = time.time()
            self._cur[tag] = ""
            if len(self._cur) > 40:
                for stale in list(self._cur.keys())[:10]:
                    if stale != tag:
                        del self._cur[stale]
            line = f"▶ SSE START {note}".strip()
            self._feed.append(f"[{tag}] {line}")
            self._stats["stream_tag"] = tag
            self._stats["stream_chars"] = 0
            self._stats["stream_note"] = note or "connecting…"
        self._refresh()

    def end_stream(self, tag: str, chars: int, items: int):
        with self._lock:
            self._cur.pop(tag, None)
            self._stream_chars.pop(tag, None)
            self._feed.append(f"[{tag}] ■ SSE END items={items} chars={chars:,}")
            if self._stream_tag == tag:
                self._stream_tag = ""
                self._stats.pop("stream_tag", None)
                self._stats.pop("stream_chars", None)
                self._stats.pop("stream_note", None)
        self._refresh()

    def pulse_stream(self, tag: str, note: str):
        with self._lock:
            self._stats["stream_tag"] = tag
            self._stats["stream_note"] = note
            elapsed = time.time() - self._stream_started if self._stream_started else 0.0
            self._feed.append(f"[{tag}] … {note} ({elapsed:.0f}s)")
        self._refresh()

    def feed(self, tag: str, text: str):
        with self._lock:
            self._stream_tag = tag
            tag_chars = self._stream_chars.get(tag, 0) + len(text)
            self._stream_chars[tag] = tag_chars
            self._stats["stream_tag"] = tag
            self._stats["stream_chars"] = tag_chars
            self._stats["stream_note"] = "streaming"
            buf = self._cur.get(tag, "") + text
            while "\n" in buf or len(buf) >= _WRAP:
                nl = buf.find("\n")
                if 0 <= nl < _WRAP:
                    line, buf = buf[:nl], buf[nl + 1:]
                else:
                    line, buf = buf[:_WRAP], buf[_WRAP:]
                self._feed.append(f"[{tag}] {_ellipsize(line)}")
            if len(buf) > _CUR_MAX:
                buf = _ellipsize(buf, _CUR_MAX)
            self._cur[tag] = buf
        now = time.time()
        if now - self._last_refresh >= 0.12:
            self._last_refresh = now
            self._refresh()

    # ---- rendering ----
    def _stats_table(self):
        from rich.table import Table
        s = self._stats
        t = Table.grid(padding=(0, 1))
        t.add_column(justify="right", style="dim")
        t.add_column(justify="left")

        def row(k, v):
            t.add_row(k, v)

        row("model", f"[bold]{self.model}[/]")
        if "lang" in s:
            row("language", f"[bold yellow]{s['lang']}[/]  {s.get('lang_done', 0):,}/{s.get('lang_target', 0):,}")
        row("rows kept", f"[green]{s.get('rows', 0):,}[/]  (rej {s.get('rejected', 0):,})")
        row("batches", f"{s.get('batches', 0):,}")
        row("KV cache", f"[cyan]{s.get('cache_pct', 0):.0f}%[/]  prefix~{s.get('prefix_pct', 0):.0f}%")
        row("tokens", f"in {s.get('prompt_tok', 0):,} · out {s.get('out_tok', 0):,}")
        if "exp_keep" in s:
            row("exp keep/req", f"~{s['exp_keep']:.0f} rows")
        if s.get("stream_tag"):
            row("live SSE", f"[yellow]{s['stream_tag']}[/]  {s.get('stream_chars', 0):,} chars  {s.get('stream_note', '')}")
        row("rows/s", f"{s.get('rows_per_s', 0):.1f}")
        row("cost/1k rows", f"₹{s.get('cost_per_1k', 0):.2f}")
        row("run cost", f"₹{s.get('run_cost', 0):.4f}")
        row("cum cost", f"[bold]₹{s.get('cum_cost', 0):.4f}[/]")
        return t

    def __rich__(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        with self._lock:
            feed_lines = list(self._feed)
            # show the still-streaming partial line(s) live, with a cursor
            for tag, partial in self._cur.items():
                if partial:
                    feed_lines.append(f"[{tag}] {_ellipsize(partial, _CUR_MAX)}▏")
            feed_lines = feed_lines[-_FEED_LINES:]
            events = list(self._events)
            table = self._stats_table()

        left = Panel(
            Group(self.progress, Text(""), table, Text(""),
                  Text("recent batches", style="dim underline"),
                  Text("\n".join(events) or "…", style="dim")),
            title="[bold]progress + stats", border_style="cyan",
        )
        feed = Text("\n".join(feed_lines) or "waiting for stream…",
                    overflow="crop", no_wrap=True)
        right = Panel(feed, title=f"[bold]SSE streaming · {self.model}",
                      border_style="green")
        self.layout["left"].update(left)
        self.layout["right"].update(right)
        return self.layout


def make_ui(plan_total: int, initial: int, model: str):
    """GenUI if rich is importable, else PlainUI."""
    try:
        import rich  # noqa: F401
        return GenUI(plan_total, initial, model)
    except Exception:
        return PlainUI(plan_total, initial)
