"""ThinkSpark training loop for Mac M1 (MPS/CPU).

Shows EVERYTHING live in the terminal: a tqdm bar per epoch with running loss and
per-head accuracy, a printed metrics table after every validation, macro-F1, and
PNG plots (loss/accuracy curves + confusion matrix) refreshed each epoch. No
CUDA, no wandb — pure local.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.taxonomy import INTENTS
from .config import TrainConfig
from .dataset import ThinkSparkDataset, collate, read_jsonl
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


def _build_loaders(cfg: TrainConfig):
    tr = read_jsonl(cfg.data.train_jsonl)
    va = read_jsonl(cfg.data.val_jsonl)
    te = read_jsonl(cfg.data.test_jsonl)

    def mk(rows, shuffle):
        ds = ThinkSparkDataset(rows, cfg.data.max_input_len, cfg.data.max_context_len)
        return DataLoader(ds, batch_size=cfg.optim.batch_size, shuffle=shuffle,
                          num_workers=cfg.run.num_workers, collate_fn=collate,
                          drop_last=False)

    return mk(tr, True), mk(va, False), mk(te, False), (len(tr), len(va), len(te))


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, ce, weights, n_classes, collect_cm_for="intent"):
    model.eval()
    tot_loss = 0.0
    correct = {h: 0 for h in HEADS}
    seen = 0
    cm_preds, cm_targets = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch)
        bs = batch["intent"].size(0)
        seen += bs
        loss = 0.0
        for h in HEADS:
            loss = loss + weights[h] * ce[h](out[h], batch[h])
            correct[h] += (out[h].argmax(-1) == batch[h]).sum().item()
        tot_loss += float(loss) * bs
        cm_preds.append(out[collect_cm_for].argmax(-1).cpu().numpy())
        cm_targets.append(batch[collect_cm_for].cpu().numpy())
    seen = max(seen, 1)
    metrics = {"loss": tot_loss / seen}
    for h in HEADS:
        metrics[f"acc_{h}"] = correct[h] / seen
    cm = confusion_matrix(np.concatenate(cm_preds), np.concatenate(cm_targets),
                          n_classes[collect_cm_for])
    metrics["macro_f1_intent"] = macro_f1(cm)
    return metrics, cm


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.run.seed)
    np.random.seed(cfg.run.seed)
    device = pick_device(cfg.run.device)
    print(f"[device] {device}")

    label_maps = json.loads(Path(cfg.data.label_maps).read_text(encoding="utf-8"))
    n_classes = {
        "intent": len(label_maps["intents"]),
        "language": len(label_maps["lang_list"]),
        "register": len(label_maps["registers"]),
        "emotion": len(label_maps["emotions"]),
        "filler_type": len(label_maps["filler_types"]),
    }

    train_dl, val_dl, test_dl, sizes = _build_loaders(cfg)
    print(f"[data] train={sizes[0]} val={sizes[1]} test={sizes[2]}")

    model = ThinkSpark(
        cfg.model, n_classes["intent"], n_classes["language"], n_classes["register"],
        n_classes["emotion"], n_classes["filler_type"],
        cfg.data.max_input_len, cfg.data.max_context_len,
    ).to(device)
    trainable, total = count_params(model)
    print(f"[model] {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    weights = _loss_weights(cfg)
    ce = {h: nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing) for h in HEADS}

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                            weight_decay=cfg.optim.weight_decay)
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
        run_loss = run_acc = 0.0
        seen = 0
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{cfg.optim.epochs}",
                   unit="batch", dynamic_ncols=True)
        for it, batch in enumerate(bar):
            for g in opt.param_groups:
                g["lr"] = lr_at(global_step)
            batch = _to_device(batch, device)
            out = model(batch)
            loss = sum(weights[h] * ce[h](out[h], batch[h]) for h in HEADS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()

            bs = batch["intent"].size(0)
            acc_i = accuracy(out["intent"], batch["intent"])
            run_loss += float(loss.detach()) * bs
            run_acc += acc_i * bs
            seen += bs
            tp.update(bs, int(batch["input_mask"].sum().item()))
            global_step += 1

            if global_step % cfg.run.log_every == 0:
                history["step"].append(global_step)
                history["train_loss"].append(run_loss / seen)
                history["train_acc_intent"].append(run_acc / seen)
            bar.set_postfix_str(
                f"loss {run_loss/seen:.3f} | intent_acc {run_acc/seen:.3f} "
                f"| lr {opt.param_groups[0]['lr']:.2e}", refresh=False)
        bar.close()

        # ---- validation ----
        if epoch % cfg.run.eval_every_epochs == 0 or epoch == cfg.optim.epochs:
            vm, _ = evaluate(model, val_dl, device, ce, weights, n_classes)
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

            if vm["macro_f1_intent"] >= best_f1:
                best_f1 = vm["macro_f1_intent"]
                _save(model, cfg, label_maps, out_dir / "best", epoch, vm)

        if epoch % cfg.run.plot_every_epochs == 0:
            plot_curves(history, curves_png)
        _save(model, cfg, label_maps, out_dir / f"epoch-{epoch}", epoch, None)
        _rotate(out_dir, cfg.run.keep_last_checkpoints)

    # ---- final test ----
    print("\n[test] evaluating best checkpoint on held-out test set ...")
    best = out_dir / "best"
    if (best / "model.pt").exists():
        model.load_state_dict(torch.load(best / "model.pt", map_location=device))
    tm, cm = evaluate(model, test_dl, device, ce, weights, n_classes)
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
