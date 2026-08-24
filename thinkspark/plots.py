"""Matplotlib (Agg) plots: training curves + confusion matrix. No display needed."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_curves(history: dict, out_path: str | Path) -> None:
    """history keys: step, train_loss, train_acc_intent, and per-epoch
    val_* series keyed by 'val_epoch','val_loss','val_acc_intent',... """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(history["step"], history["train_loss"], lw=1, color="#4c72b0")
    ax[0, 0].set_title("train loss"); ax[0, 0].set_xlabel("step")

    ax[0, 1].plot(history["step"], history["train_acc_intent"], lw=1, color="#55a868")
    ax[0, 1].set_title("train intent accuracy"); ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_xlabel("step")

    if history.get("val_epoch"):
        ve = history["val_epoch"]
        ax[1, 0].plot(ve, history["val_loss"], "-o", label="val", color="#c44e52")
        ax[1, 0].set_title("val loss"); ax[1, 0].set_xlabel("epoch")
        for key, lab, col in [
            ("val_acc_intent", "intent", "#c44e52"),
            ("val_acc_language", "language", "#8172b3"),
            ("val_acc_filler_type", "filler_type", "#ccb974"),
            ("val_acc_emotion", "emotion", "#64b5cd"),
        ]:
            if history.get(key):
                ax[1, 1].plot(ve, history[key], "-o", label=lab, color=col)
        ax[1, 1].set_title("val accuracy by head"); ax[1, 1].set_ylim(0, 1)
        ax[1, 1].set_xlabel("epoch"); ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: str | Path,
                   title: str = "intent confusion (test)") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    with np.errstate(all="ignore"):
        norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.6), max(6, n * 0.6)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
