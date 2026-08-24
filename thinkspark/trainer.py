"""ThinkSpark training loop — Mac MPS, Colab 1-GPU, Kaggle T4x2 DDP.

Fetches splits from Hugging Face if they are not already on disk, then trains.
Two-or-more CUDA GPUs auto-launch DistributedDataParallel.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config.taxonomy import (
    EMOTION2ID, FILLERTYPE2ID, INTENT2ID, INTENTS, LANG2ID, REGISTER2ID,
)
from .config import TrainConfig
from .dataset import ThinkSparkDataset, collate, read_jsonl
from .losses import build_head_losses, effective_number_weights
from . import termplot
from .distributed import (
    barrier,
    destroy_process_group,
    ddp_worker_setup,
    init_process_group,
    is_dist,
    is_main,
    local_rank,
    rank,
    should_spawn_ddp,
    spawn_ddp,
    unwrap,
    world_size,
)
from .hf_data import ensure_training_data, resolve_data_path
from .metrics import accuracy, confusion_matrix, macro_f1, per_class_f1
from .model import ThinkSpark, count_params
from .plots import plot_confusion, plot_curves
from .telemetry import Throughput, device_stats, pick_device

HEADS = ["intent", "language", "register", "emotion", "filler_type"]


def _loss_weights(cfg) -> dict:
    o = cfg.optim
    return {
        "intent": o.w_intent, "language": o.w_lang, "register": o.w_register,
        "emotion": o.w_emotion, "filler_type": o.w_fillertype,
    }


_HEAD_ROW_KEY = {  # head -> (row field, default value, str->id map)
    "intent": ("intent", None, INTENT2ID),
    "language": ("language", None, LANG2ID),
    "register": ("register", "casual", REGISTER2ID),
    "emotion": ("emotion", "neutral", EMOTION2ID),
    "filler_type": ("filler_type", "none", FILLERTYPE2ID),
}


def _class_counts(rows: list[dict], head: str, n: int) -> dict:
    """Per-class sample counts for one head, straight from the train rows."""
    field, default, id_map = _HEAD_ROW_KEY[head]
    counts = {i: 0 for i in range(n)}
    for r in rows:
        val = r.get(field, default) if default is not None else r[field]
        idx = id_map.get(val, id_map.get(default) if default is not None else None)
        if idx is not None:
            counts[idx] += 1
    return counts


def _build_losses(cfg: TrainConfig, rows: list[dict], n_classes: dict, device):
    """One HeadLoss per head: focal/CE + optional class-balanced weights.

    Class weighting (effective-number, Cui 2019) is the fix for the 47:1 intent
    imbalance that pinned macro-F1 near 0.23 with rare intents at 0.000.
    """
    o = cfg.optim
    kind_map = {
        "intent": o.loss_intent, "language": o.loss_lang, "register": o.loss_register,
        "emotion": o.loss_emotion, "filler_type": o.loss_fillertype,
    }
    balance = set(o.balance_heads or ())
    weight_map = {}
    for h in HEADS:
        if h in balance and o.class_balance_beta > 0.0:
            counts = _class_counts(rows, h, n_classes[h])
            weight_map[h] = effective_number_weights(
                counts, beta=o.class_balance_beta, num_classes=n_classes[h]).to(device)
        else:
            weight_map[h] = None
    losses = build_head_losses(kind_map, weight_map, o.focal_gamma, o.label_smoothing)
    return {h: l.to(device) for h, l in losses.items()}, weight_map


def _use_amp(cfg: TrainConfig, device: torch.device) -> bool:
    return bool(cfg.run.amp) and device.type == "cuda"


def _param_groups(model: nn.Module, weight_decay: float):
    """AdamW hygiene: decay only 2-D matmul weights. LayerNorm/bias/embeddings
    are scale/shift params — decaying them just fights the norm and hurts."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "embed" in name or "pos" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _build_loaders(cfg: TrainConfig):
    tr = read_jsonl(resolve_data_path(cfg.data.train_jsonl))
    va = read_jsonl(resolve_data_path(cfg.data.val_jsonl))
    te = read_jsonl(resolve_data_path(cfg.data.test_jsonl))

    def mk(rows, *, shuffle: bool, sampler=None):
        ds = ThinkSparkDataset(rows, cfg.data.max_input_len, cfg.data.max_context_len)
        return DataLoader(
            ds,
            batch_size=cfg.optim.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=cfg.run.num_workers,
            collate_fn=collate,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    train_ds = ThinkSparkDataset(tr, cfg.data.max_input_len, cfg.data.max_context_len)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.run.num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl, mk(va, shuffle=False), mk(te, shuffle=False), (len(tr), len(va), len(te)), train_sampler


def _to_device(batch, device):
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, ce, weights, n_classes, collect_cm_for="intent",
             use_amp: bool = False):
    model.eval()
    tot_loss = 0.0
    correct = {h: 0 for h in HEADS}
    seen = 0
    cm_preds, cm_targets = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(batch)
            bs = batch["intent"].size(0)
            loss = 0.0
            for h in HEADS:
                loss = loss + weights[h] * ce[h](out[h], batch[h])
                correct[h] += (out[h].argmax(-1) == batch[h]).sum().item()
        seen += bs
        tot_loss += float(loss) * bs
        cm_preds.append(out[collect_cm_for].argmax(-1).detach().cpu().numpy())
        cm_targets.append(batch[collect_cm_for].detach().cpu().numpy())
    seen = max(seen, 1)
    metrics = {"loss": tot_loss / seen}
    for h in HEADS:
        metrics[f"acc_{h}"] = correct[h] / seen
    cm = confusion_matrix(np.concatenate(cm_preds), np.concatenate(cm_targets),
                          n_classes[collect_cm_for])
    metrics["macro_f1_intent"] = macro_f1(cm)
    return metrics, cm


def train(cfg: TrainConfig) -> None:
    """Entry point: fetch data, then single-GPU or auto DDP."""
    if should_spawn_ddp(cfg):
        ensure_training_data(
            cfg.data.train_jsonl, cfg.data.val_jsonl, cfg.data.test_jsonl, cfg.data.label_maps,
            repo=cfg.data.hf_repo, fetch=cfg.data.hf_fetch, refresh=cfg.data.hf_refresh,
        )
        spawn_ddp(cfg, _ddp_worker)
        return
    init_process_group()
    try:
        if is_main():
            ensure_training_data(
                cfg.data.train_jsonl, cfg.data.val_jsonl, cfg.data.test_jsonl, cfg.data.label_maps,
                repo=cfg.data.hf_repo, fetch=cfg.data.hf_fetch, refresh=cfg.data.hf_refresh,
            )
        barrier()
        _train_loop(cfg)
    finally:
        destroy_process_group()


def _ddp_worker(rank_i: int, world: int, cfg: TrainConfig) -> None:
    ddp_worker_setup(rank_i, world)
    try:
        _train_loop(cfg)
    finally:
        destroy_process_group()


def _train_loop(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.run.seed + rank())
    np.random.seed(cfg.run.seed + rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.run.seed + rank())

    device = pick_device(cfg.run.device)
    use_amp = _use_amp(cfg, device)
    if is_main():
        extra = f"  DDP world={world_size()}" if is_dist() else ""
        amp = "  amp=fp16" if use_amp else ""
        print(f"[device] {device}{extra}{amp}")

    label_maps = json.loads(resolve_data_path(cfg.data.label_maps).read_text(encoding="utf-8"))
    n_classes = {
        "intent": len(label_maps["intents"]),
        "language": len(label_maps["lang_list"]),
        "register": len(label_maps["registers"]),
        "emotion": len(label_maps["emotions"]),
        "filler_type": len(label_maps["filler_types"]),
    }

    train_dl, val_dl, test_dl, sizes, train_sampler = _build_loaders(cfg)
    if is_main():
        print(f"[data] train={sizes[0]} val={sizes[1]} test={sizes[2]}  "
              f"batch={cfg.optim.batch_size} x {world_size()} GPU(s)")

    model = ThinkSpark(
        cfg.model, n_classes["intent"], n_classes["language"], n_classes["register"],
        n_classes["emotion"], n_classes["filler_type"],
        cfg.data.max_input_len, cfg.data.max_context_len,
    ).to(device)
    if is_dist() and device.type == "cuda":
        model = DDP(
            model,
            device_ids=[device.index if device.index is not None else local_rank()],
            output_device=device.index if device.index is not None else local_rank(),
            find_unused_parameters=False,
        )
    if is_main():
        trainable, total = count_params(unwrap(model))
        print(f"[model] {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    weights = _loss_weights(cfg)
    ce, weight_map = _build_losses(cfg, train_dl.dataset.rows, n_classes, device)
    if is_main():
        iw = weight_map.get("intent")
        kinds = ", ".join(f"{h}:{ce[h].kind}" for h in HEADS)
        print(f"[loss] {kinds} | focal_gamma={cfg.optim.focal_gamma} "
              f"| class_balance_beta={cfg.optim.class_balance_beta}")
        if iw is not None:
            print(f"[loss] intent class-weights range "
                  f"[{iw.min():.2f}, {iw.max():.2f}] (rare intents up-weighted)")

    opt = torch.optim.AdamW(_param_groups(unwrap(model), cfg.optim.weight_decay),
                            lr=cfg.optim.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * cfg.optim.epochs
    warmup = max(int(total_steps * cfg.optim.warmup_ratio), 1)
    min_lr = cfg.optim.lr * cfg.optim.min_lr_ratio

    def lr_at(step):
        if step < warmup:
            return cfg.optim.lr * step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_lr + (cfg.optim.lr - min_lr) * cos

    out_dir = Path(cfg.run.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(out_dir / "effective_config.yaml")
    curves_png = out_dir / "training_curves.png"
    confusion_png = out_dir / "confusion_intent.png"

    history = {"step": [], "train_loss": [], "train_acc_intent": [],
               "val_epoch": [], "val_loss": [], "val_acc_intent": [],
               "val_acc_language": [], "val_acc_filler_type": [],
               "val_acc_emotion": [], "val_macro_f1": []}
    best_f1 = -1.0
    global_step = 0
    tp = Throughput()

    for epoch in range(1, cfg.optim.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        run_loss = run_acc = 0.0
        seen = 0
        iterator = train_dl
        if is_main():
            iterator = tqdm(train_dl, desc=f"epoch {epoch}/{cfg.optim.epochs}",
                            unit="batch", dynamic_ncols=True)
        for batch in iterator:
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step)
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(batch)
                loss = sum(weights[h] * ce[h](out[h], batch[h]) for h in HEADS)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = batch["intent"].size(0)
            acc_i = accuracy(out["intent"], batch["intent"])
            run_loss += float(loss.detach()) * bs
            run_acc += acc_i * bs
            seen += bs
            tp.update(bs, int(batch["input_mask"].sum().item()))
            global_step += 1

            if is_main() and global_step % cfg.run.log_every == 0:
                history["step"].append(global_step)
                history["train_loss"].append(run_loss / seen)
                history["train_acc_intent"].append(run_acc / seen)
            if is_main() and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(
                    f"loss {run_loss/seen:.3f} | intent_acc {run_acc/seen:.3f} "
                    f"| lr {opt.param_groups[0]['lr']:.2e}", refresh=False)
        if is_main() and hasattr(iterator, "close"):
            iterator.close()

        barrier()
        if epoch % cfg.run.eval_every_epochs == 0 or epoch == cfg.optim.epochs:
            if is_main():
                vm, _ = evaluate(unwrap(model), val_dl, device, ce, weights, n_classes,
                                 use_amp=use_amp)
                history["val_epoch"].append(epoch)
                history["val_loss"].append(vm["loss"])
                history["val_acc_intent"].append(vm["acc_intent"])
                history["val_acc_language"].append(vm["acc_language"])
                history["val_acc_filler_type"].append(vm["acc_filler_type"])
                history["val_acc_emotion"].append(vm["acc_emotion"])
                history["val_macro_f1"].append(vm["macro_f1_intent"])
                rate = tp.rate(); ds = device_stats(device)
                print(
                    f"  [val] epoch {epoch}: loss {vm['loss']:.3f} | "
                    f"intent {vm['acc_intent']:.3f} (F1 {vm['macro_f1_intent']:.3f}) | "
                    f"lang {vm['acc_language']:.3f} | reg {vm['acc_register']:.3f} | "
                    f"emo {vm['acc_emotion']:.3f} | ftype {vm['acc_filler_type']:.3f} | "
                    f"{rate['samples_per_s']:.0f} smp/s"
                    + (f" | rss {ds.get('rss_mb',0):.0f}MB" if ds.get("rss_mb") else "")
                )
                star = ""
                if vm["macro_f1_intent"] > best_f1:   # strict: keep the first-best on ties
                    best_f1 = vm["macro_f1_intent"]
                    _save(unwrap(model), cfg, label_maps, out_dir / "best", epoch, vm)
                    star = "  <- best macro-F1 (checkpointed)"
                if star:
                    print(star)
                # realtime terminal curves (Kaggle/Colab/SSH friendly)
                if cfg.run.term_plot:
                    termplot.render(history, epochs_total=cfg.optim.epochs)
        if is_main() and epoch % cfg.run.plot_every_epochs == 0:
            plot_curves(history, curves_png)
        if is_main():
            _save(unwrap(model), cfg, label_maps, out_dir / f"epoch-{epoch}", epoch, None)
            _rotate(out_dir, cfg.run.keep_last_checkpoints)
        barrier()

    if not is_main():
        return

    print("\n[test] evaluating best checkpoint on held-out test set ...")
    best = out_dir / "best"
    raw_model = unwrap(model)
    if (best / "model.pt").exists():
        raw_model.load_state_dict(torch.load(best / "model.pt", map_location=device))
    tm, cm = evaluate(raw_model, test_dl, device, ce, weights, n_classes, use_amp=use_amp)
    print(f"[test] loss {tm['loss']:.3f} | intent {tm['acc_intent']:.3f} "
          f"(macro-F1 {tm['macro_f1_intent']:.3f}) | lang {tm['acc_language']:.3f} "
          f"| emo {tm['acc_emotion']:.3f} | ftype {tm['acc_filler_type']:.3f}")

    f1s = per_class_f1(cm)
    print("\n  per-intent F1 (test):")
    for name, f in sorted(zip(INTENTS, f1s), key=lambda x: -x[1]):
        print(f"    {name:22s} {f:.3f}")

    plot_confusion(cm, INTENTS, confusion_png)
    plot_curves(history, curves_png)
    (out_dir / "test_metrics.json").write_text(
        json.dumps({**tm, "per_intent_f1": dict(zip(INTENTS, f1s.tolist()))},
                   indent=2), encoding="utf-8")
    print(f"\n[done] plots: {curves_png}\n       {confusion_png}")
    print(f"[done] best checkpoint: {best}")


def _save(model, cfg: TrainConfig, label_maps, path: Path, epoch, metrics):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    meta = {"epoch": epoch, "model_cfg": cfg.model.__dict__,
            "data_cfg": cfg.data.__dict__, "metrics": metrics}
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (path / "label_maps.json").write_text(
        json.dumps(label_maps, ensure_ascii=False, indent=1), encoding="utf-8")


def _rotate(out_dir: Path, keep: int):
    if keep <= 0:
        return
    ckpts = sorted([p for p in out_dir.glob("epoch-*") if p.is_dir()],
                   key=lambda p: int(p.name.split("-")[1]))
    for p in ckpts[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
