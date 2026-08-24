"""Class-imbalance-aware losses for ThinkSpark.

Why this file exists
--------------------
The training intents are *wildly* imbalanced (≈47:1 — `thinking` is 30.8 % of the
data, `polite_interrupt` 0.7 %, `encouraging` 1.1 %). A plain
``nn.CrossEntropyLoss`` just learns to shout the majority class: accuracy looks
okay while macro-F1 flatlines and the rare intents score 0.000. That is exactly
the failure the earlier run showed.

Two standard, well-tested remedies, both provided here and switchable from YAML:

* **Class-balanced weighting** (Cui et al., CVPR 2019, "Class-Balanced Loss Based
  on Effective Number of Samples"): weight class ``c`` by
  ``(1 - β) / (1 - β^{n_c})``. As β→1 this approaches inverse-frequency; β=0 is
  uniform. β≈0.999 is the paper's default and works well here.

* **Focal loss** (Lin et al., 2017): down-weight easy, already-correct examples by
  ``(1 - p_t)^γ`` so the optimiser keeps spending gradient on the hard/rare ones.

The two compose: class weights fix the *prior*, focal fixes the *per-example*
gradient. Label smoothing is folded in for calibration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(counts, beta: float = 0.999, num_classes: int | None = None) -> torch.Tensor:
    """Class-balanced weights from per-class counts (Cui et al. 2019).

    Returns a tensor of length ``num_classes`` normalised to mean 1.0 so the loss
    scale stays comparable to unweighted CE (keeps LR/other-head weights sane).
    """
    n = num_classes or len(counts)
    counts = np.asarray([counts.get(i, 0) if isinstance(counts, dict) else counts[i]
                         for i in range(n)], dtype=np.float64)
    counts = np.clip(counts, 1.0, None)          # unseen class -> weight as if 1 sample
    if beta <= 0.0:
        w = np.ones(n, dtype=np.float64)
    else:
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff
    w = w / w.mean()                              # normalise to mean 1
    return torch.tensor(w, dtype=torch.float32)


class HeadLoss(nn.Module):
    """One classification head's loss: weighted CE or focal, + label smoothing.

    * ``kind="ce"``   -> weighted cross-entropy (class weights via `weight`)
    * ``kind="focal"``-> class-weighted focal loss with `gamma`
    """

    def __init__(self, kind: str = "ce", weight: torch.Tensor | None = None,
                 gamma: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.kind = kind
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        # registered as a buffer so it follows .to(device)/AMP and is checkpoint-visible
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.kind == "focal":
            # focal needs per-sample CE; keep label smoothing for calibration
            ce = F.cross_entropy(logits, target, weight=w,
                                 label_smoothing=self.label_smoothing, reduction="none")
            with torch.no_grad():
                pt = torch.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
            return ((1.0 - pt).clamp_min(1e-6) ** self.gamma * ce).mean()
        return F.cross_entropy(logits, target, weight=w,
                               label_smoothing=self.label_smoothing)


def build_head_losses(kind_map: dict[str, str], weight_map: dict[str, torch.Tensor | None],
                      gamma: float, label_smoothing: float) -> dict[str, HeadLoss]:
    """Assemble one HeadLoss per head from per-head kind + optional class weights."""
    return {
        h: HeadLoss(kind=kind_map.get(h, "ce"), weight=weight_map.get(h),
                    gamma=gamma, label_smoothing=label_smoothing)
        for h in kind_map
    }
