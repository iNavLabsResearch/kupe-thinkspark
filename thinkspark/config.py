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


@dataclass
class ModelCfg:
    d_model: int = 128
    n_heads: int = 4
    input_layers: int = 4        # depth of the (primary) input encoder
    context_layers: int = 3      # multi-turn multilingual context needs capacity
    fusion_layers: int = 2       # cross-attention fusion (input queries context)
    ffn_mult: int = 2
    dropout: float = 0.1


@dataclass
class OptimCfg:
    epochs: int = 12
    batch_size: int = 64
    lr: float = 3.0e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    # multi-task loss weights (intent is the headline task)
    w_intent: float = 1.0
    w_lang: float = 0.5
    w_register: float = 0.3
    w_emotion: float = 0.5
    w_fillertype: float = 0.5
    label_smoothing: float = 0.05


@dataclass
class RunCfg:
    device: str = "auto"         # auto -> mps | cuda | cpu
    seed: int = 42
    num_workers: int = 0         # 0 is safest/ fastest on macOS for small data
    output_dir: str = "artifacts/thinkspark"
    log_every: int = 10
    eval_every_epochs: int = 1
    plot_every_epochs: int = 1
    keep_last_checkpoints: int = 3


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
