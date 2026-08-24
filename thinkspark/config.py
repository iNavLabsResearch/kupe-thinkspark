"""Typed training config loaded from YAML (every field maps 1:1 to the yaml)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataCfg:
    train_jsonl: str = "data/splits/train.jsonl"
    val_jsonl: str = "data/splits/val.jsonl"
    test_jsonl: str = "data/splits/test.jsonl"
    label_maps: str = "data/vocab/label_maps.json"
    max_input_len: int = 96      # bytes of the user utterance (primary)
    max_context_len: int = 384   # bytes of the PAST conversation transcript
                                 # (multi-turn, possibly multilingual — needs room)
    hf_repo: str = "anuj-inavlabs/kupe-thinkspark"
    hf_fetch: bool = True        # download from HF if local splits are missing
    hf_refresh: bool = False     # re-download even if local files exist
    # Label scheme applied AT LOAD (no data regeneration):
    #   "super" -> 9 clean agent-reactions (default; higher accuracy)
    #   "fine"  -> the original 17 intents
    intent_scheme: str = "super"


@dataclass
class ModelCfg:
    # Wider + slightly deeper than the original 128/4L: a byte-level model has to
    # compose bytes -> subwords -> meaning itself, so it needs the capacity. Still
    # tiny (~8-12M) and fast on CPU/MPS for these short sequences — latency stays low.
    d_model: int = 192
    n_heads: int = 6
    input_layers: int = 5        # depth of the (primary) input encoder
    context_layers: int = 3      # multi-turn multilingual context needs capacity
    fusion_layers: int = 2       # cross-attention fusion (input queries context)
    ffn_mult: int = 3
    dropout: float = 0.1
    # Encoder backend. "byte" = the tiny from-scratch dual-encoder (default,
    # zero-OOV, lowest latency). "hf" = a pretrained multilingual encoder
    # (e.g. IndicBERT / MuRIL / XLM-R) — higher ceiling, higher latency; opt-in.
    encoder_backend: str = "byte"
    hf_model_name: str = "ai4bharat/indic-bert"


@dataclass
class OptimCfg:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 5.0e-4           # a touch higher; warmup + cosine keep it stable
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01   # applied to matmul weights only (not norm/bias/embed)
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    # multi-task loss weights (intent is the headline task)
    w_intent: float = 1.0
    w_lang: float = 0.5
    w_register: float = 0.3
    w_emotion: float = 0.5
    w_fillertype: float = 0.5
    label_smoothing: float = 0.05

    # --- imbalance handling (the fix for the flat macro-F1) --------------------
    # Per-head loss kind: "ce" (weighted cross-entropy) or "focal".
    # Intent is 47:1 imbalanced -> focal + class-balancing helps most.
    loss_intent: str = "focal"
    loss_emotion: str = "ce"
    loss_fillertype: str = "ce"
    loss_lang: str = "ce"
    loss_register: str = "ce"
    focal_gamma: float = 1.5     # 0 = plain CE; 1-2 focuses on hard/rare examples
    # Class-balanced weighting (Cui et al. 2019). beta in [0,1); 0 disables,
    # ->1 approaches inverse-frequency. Applied to the listed heads.
    class_balance_beta: float = 0.999
    balance_heads: tuple = ("intent", "emotion", "filler_type")


@dataclass
class RunCfg:
    device: str = "auto"         # auto -> cuda | mps | cpu
    seed: int = 42
    num_workers: int = 0         # 0 is safest on macOS; linux CUDA can use 2
    output_dir: str = "artifacts/thinkspark"
    log_every: int = 10
    eval_every_epochs: int = 1
    plot_every_epochs: int = 1
    keep_last_checkpoints: int = 3
    gpus: int = 0                # 0 = auto (use all CUDA GPUs)
    distributed: str = "auto"    # auto | on | off  (auto DDP when 2+ CUDA GPUs)
    amp: bool = True             # mixed precision on CUDA (T4 / Colab / PC)
    term_plot: bool = True       # draw live loss/F1 curves in the terminal (Kaggle)
    # Early stopping (item 4): stop when val macro-F1 hasn't improved for
    # `early_stop_patience` evals — best checkpoint is always kept.
    early_stop: bool = True
    early_stop_patience: int = 4
    early_stop_min_delta: float = 0.002


@dataclass
class TrainConfig:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    run: RunCfg = field(default_factory=RunCfg)

    @staticmethod
    def load(path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return TrainConfig(
            data=DataCfg(**(raw.get("data") or {})),
            model=ModelCfg(**(raw.get("model") or {})),
            optim=OptimCfg(**(raw.get("optim") or {})),
            run=RunCfg(**(raw.get("run") or {})),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
