"""Realtime training curves drawn *in the terminal* (Kaggle / Colab / SSH).

Kaggle shows cell stdout live, so we redraw compact ASCII/plotext charts after
every epoch — you watch loss fall and macro-F1 climb without opening a PNG.

Uses `plotext` when available (crisp braille plots); otherwise falls back to a
dependency-free ASCII sparkline so it never crashes a run.
"""

from __future__ import annotations

import os

# Opt-IN to plotext braille charts with THINKSPARK_PLOTEXT=1. Default is the
# dependency-free ASCII sparkline: it renders identically on Mac / Colab / Kaggle
# and never spams errors when a host ships a broken/partial `plotext` build (some
# Kaggle images expose a `plotext` with neither `.plot` nor `.subplots`).
_WANT_PLOTEXT = os.environ.get("THINKSPARK_PLOTEXT", "0") == "1"

if _WANT_PLOTEXT:
    try:
        import plotext as _plt  # type: ignore
        # sanity-check the API actually exists before we rely on it
        _HAS_PLOTEXT = all(callable(getattr(_plt, n, None)) for n in ("plot", "show"))
    except Exception:  # pragma: no cover - optional dep
        _HAS_PLOTEXT = False
else:
    _HAS_PLOTEXT = False

# flips to True after the first plotext failure so we never spam the same error
_PLOTEXT_DISABLED = False

_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(ys: list[float], lo: float | None = None, hi: float | None = None) -> str:
    if not ys:
        return ""
    lo = min(ys) if lo is None else lo
    hi = max(ys) if hi is None else hi
    rng = (hi - lo) or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((y - lo) / rng * (len(_BLOCKS) - 1)))]
                   for y in ys)


def _call(*names, **kwargs) -> None:
    """Call the first attribute in `names` that exists on plotext (API drifts
    across versions: clear_figure/clf, plotsize/plot_size, …). No-op if none."""
    args = kwargs.pop("_args", ())
    for name in names:
        fn = getattr(_plt, name, None)
        if callable(fn):
            fn(*args, **kwargs)
            return


def _one_plot(x, series, title, *, y01: bool) -> None:
    """Draw a single-axis plotext chart with one or more (ys, color, label)
    series. Avoids subplots entirely so it works on every plotext version."""
    _call("clear_figure", "clf", "cld", "clear_data")
    for ys, color, label in series:
        if ys:
            try:
                _plt.plot(x, ys, marker="braille", color=color, label=label)
            except Exception:
                _plt.plot(x, ys, label=label)
    if y01:
        try:
            _plt.ylim(0, 1)
        except Exception:
            pass
    try:
        _plt.title(title)
        _plt.xlabel("epoch")
    except Exception:
        pass
    _call("plotsize", "plot_size", _args=(100, 15))
    _call("theme", _args=("clear",))
    _plt.show()


def render(history: dict, *, epochs_total: int | None = None) -> None:
    """Draw val-loss and val-macro-F1 (headline metric) as live terminal charts.

    Plotting must NEVER crash a training run, so the whole thing is guarded and
    falls back to a dependency-free ASCII sparkline on any error.
    """
    ve = history.get("val_epoch") or []
    vloss = history.get("val_loss") or []
    vf1 = history.get("val_macro_f1") or []
    vacc = history.get("val_acc_intent") or []
    if not ve:
        return

    global _PLOTEXT_DISABLED
    if _HAS_PLOTEXT and not _PLOTEXT_DISABLED:
        try:
            _one_plot(ve, [(vloss, "red+", "val loss")], "val loss", y01=False)
            _one_plot(
                ve,
                [(vf1, "green+", "macro-F1"), (vacc, "cyan", "intent acc")],
                "val intent — macro-F1 / acc", y01=True,
            )
            return
        except Exception as e:  # never let a plot break training; report ONCE
            _PLOTEXT_DISABLED = True
            print(f"  [termplot] plotext unavailable ({e}); using ascii sparklines "
                  f"for the rest of the run.")

    best_f1 = max(vf1) if vf1 else 0.0
    print(f"  loss  {_sparkline(vloss)}  {vloss[-1]:.3f}")
    print(f"  F1    {_sparkline(vf1, 0.0, 1.0)}  {vf1[-1]:.3f} (best {best_f1:.3f})")
    print(f"  acc   {_sparkline(vacc, 0.0, 1.0)}  {vacc[-1]:.3f}")
