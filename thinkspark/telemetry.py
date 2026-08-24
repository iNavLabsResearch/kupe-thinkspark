"""Lightweight throughput + device stats for a Mac (MPS/CPU) — no CUDA/wandb."""

from __future__ import annotations

import time

import torch


def pick_device(pref: str = "auto") -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Throughput:
    def __init__(self):
        self.t0 = time.time()
        self.samples = 0
        self.tokens = 0

    def update(self, n_samples: int, n_tokens: int):
        self.samples += n_samples
        self.tokens += n_tokens

    def rate(self) -> dict:
        dt = max(time.time() - self.t0, 1e-6)
        return {"samples_per_s": self.samples / dt, "tokens_per_s": self.tokens / dt}


def device_stats(device: torch.device) -> dict:
    out = {}
    try:
        if device.type == "mps":
            out["mps_alloc_mb"] = torch.mps.current_allocated_memory() / 1e6
        elif device.type == "cuda":
            out["cuda_alloc_mb"] = torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    try:
        import os
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux kilobytes
        out["rss_mb"] = rss / (1e6 if os.uname().sysname == "Darwin" else 1e3)
    except Exception:
        pass
    return out
