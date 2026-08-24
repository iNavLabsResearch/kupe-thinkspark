"""Realtime training curves drawn *in the terminal* (Kaggle / Colab / SSH).

Kaggle shows cell stdout live, so we redraw compact ASCII/plotext charts after
every epoch — you watch loss fall and macro-F1 climb without opening a PNG.

Uses `plotext` when available (crisp braille plots); otherwise falls back to a
dependency-free ASCII sparkline so it never crashes a run.
"""

from __future__ import annotations

try:
    import plotext as _plt  # type: ignore
    _HAS_PLOTEXT = True
except Exception:  # pragma: no cover - optional dep
    _HAS_PLOTEXT = False

_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(ys: list[float], lo: float | None = None, hi: float | None = None) -> str:
    if not ys:
        return ""
    lo = min(ys) if lo is None else lo
    hi = max(ys) if hi is None else hi
    rng = (hi - lo) or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((y - lo) / rng * (len(_BLOCKS) - 1)))]
                   for y in ys)


def render(history: dict, *, epochs_total: int | None = None) -> None:
    """Draw val-loss and val-macro-F1 (headline metric) as live terminal charts."""
    ve = history.get("val_epoch") or []
    vloss = history.get("val_loss") or []
    vf1 = history.get("val_macro_f1") or []
    vacc = history.get("val_acc_intent") or []
    if not ve:
        return

    if _HAS_PLOTEXT:
        _plt.clf()
        _plt.subplots(1, 2)
        _plt.subplot(1, 1)
        _plt.plot(ve, vloss, marker="braille", color="red+")
        _plt.title("val loss")
        _plt.xlabel("epoch")
        _plt.subplot(1, 2)
        if vf1:
            _plt.plot(ve, vf1, marker="braille", color="green+", label="macro-F1")
        if vacc:
            _plt.plot(ve, vacc, marker="braille", color="cyan", label="intent acc")
        _plt.ylim(0, 1)
        _plt.title("val intent — macro-F1 / acc")
        _plt.xlabel("epoch")
        _plt.plotsize(100, 18)
        _plt.theme("clear")
        _plt.show()
    else:
        best_f1 = max(vf1) if vf1 else 0.0
        print(f"  loss  {_sparkline(vloss)}  {vloss[-1]:.3f}")
        print(f"  F1    {_sparkline(vf1, 0.0, 1.0)}  {vf1[-1]:.3f} (best {best_f1:.3f})")
        print(f"  acc   {_sparkline(vacc, 0.0, 1.0)}  {vacc[-1]:.3f}")
