"""Accuracy + confusion helpers (numpy/torch, device-agnostic)."""

from __future__ import annotations

import numpy as np
import torch


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(-1)
    return (pred == target).float().mean().item()


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, k: int = 2) -> float:
    k = min(k, logits.size(-1))
    topk = logits.topk(k, dim=-1).indices
    return (topk == target.unsqueeze(-1)).any(-1).float().mean().item()


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    return 2 * prec * rec / np.clip(prec + rec, 1e-9, None)


def macro_f1(cm: np.ndarray) -> float:
    return float(np.mean(per_class_f1(cm)))
