#!/usr/bin/env python3
"""Export a trained ThinkSpark checkpoint to ONNX for CPU inference in the voice
worker (kupe-agents runs it via onnxruntime — no torch dependency there).

    python scripts/export_onnx.py --ckpt artifacts/thinkspark/best --out artifacts/onnx

Produces in --out:
    thinkspark.onnx            (5 output heads; dynamic batch + seq length)
    label_maps.json            (copied from the checkpoint)
    filler_dictionary.json     (copied from data/vocab)
    onnx_meta.json             (input/output names, max lens, byte-tokenizer info)

These four files are what the agent worker downloads from HF at warmup.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from thinkspark.model import ThinkSpark
from thinkspark.tokenizer import PAD_ID, BOS_ID, EOS_ID, VOCAB_SIZE


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


class _OnnxWrap(nn.Module):
    """Flatten the dict I/O into positional tensors ONNX can trace, and accept
    int masks (cast to bool inside) since ONNX prefers int tensors."""

    def __init__(self, model: ThinkSpark):
        super().__init__()
        self.m = model

    def forward(self, input_ids, input_mask, context_ids, context_mask):
        out = self.m.forward_onnx({
            "input_ids": input_ids,
            "input_mask": input_mask.bool(),
            "context_ids": context_ids,
            "context_mask": context_mask.bool(),
        })
        return (out["intent"], out["language"], out["register"],
                out["emotion"], out["filler_type"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/thinkspark/best")
    ap.add_argument("--out", default="artifacts/onnx")
    ap.add_argument("--filler", default="data/vocab/filler_dictionary.json")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    labels = json.loads((ckpt / "label_maps.json").read_text(encoding="utf-8"))
    mcfg = _Cfg(meta["model_cfg"])
    dcfg = meta["data_cfg"]
    max_in, max_ctx = dcfg["max_input_len"], dcfg["max_context_len"]

    model = ThinkSpark(
        mcfg, len(labels["intents"]), len(labels["lang_list"]),
        len(labels["registers"]), len(labels["emotions"]),
        len(labels["filler_types"]), max_in, max_ctx,
    )
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.eval()
    wrap = _OnnxWrap(model).eval()

    # dummy inputs (batch=1); dynamic axes let the worker use any length ≤ max
    di = torch.randint(0, 255, (1, min(16, max_in)), dtype=torch.long)
    dim = torch.ones_like(di, dtype=torch.long)
    dc = torch.randint(0, 255, (1, min(24, max_ctx)), dtype=torch.long)
    dcm = torch.ones_like(dc, dtype=torch.long)

    onnx_path = out / "thinkspark.onnx"
    out_names = ["intent", "language", "register", "emotion", "filler_type"]
    dynamic = {
        "input_ids": {0: "batch", 1: "in_len"},
        "input_mask": {0: "batch", 1: "in_len"},
        "context_ids": {0: "batch", 1: "ctx_len"},
        "context_mask": {0: "batch", 1: "ctx_len"},
        **{n: {0: "batch"} for n in out_names},
    }
    export_kwargs = dict(
        input_names=["input_ids", "input_mask", "context_ids", "context_mask"],
        output_names=out_names, dynamic_axes=dynamic, opset_version=args.opset,
        do_constant_folding=True,
    )
    try:
        # torch>=2.x defaults to the dynamo exporter (needs onnxscript); the
        # stable TorchScript path has no extra deps and handles this graph fine.
        torch.onnx.export(wrap, (di, dim, dc, dcm), str(onnx_path),
                          dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrap, (di, dim, dc, dcm), str(onnx_path), **export_kwargs)
    print(f"[onnx] wrote {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # copy the artifacts the worker needs alongside the graph
    shutil.copy(ckpt / "label_maps.json", out / "label_maps.json")
    filler = Path(args.filler)
    if filler.exists():
        shutil.copy(filler, out / "filler_dictionary.json")
    else:
        print(f"[warn] filler dict not found at {filler}")

    (out / "onnx_meta.json").write_text(json.dumps({
        "input_names": ["input_ids", "input_mask", "context_ids", "context_mask"],
        "output_names": out_names,
        "max_input_len": max_in, "max_context_len": max_ctx,
        "tokenizer": {"kind": "byte", "pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID,
                      "vocab_size": VOCAB_SIZE},
        "intent_scheme": labels.get("intent_scheme", "super"),
    }, indent=2), encoding="utf-8")

    # numerical parity check torch vs onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        feeds = {"input_ids": di.numpy(), "input_mask": dim.numpy(),
                 "context_ids": dc.numpy(), "context_mask": dcm.numpy()}
        onnx_out = sess.run(out_names, feeds)
        with torch.no_grad():
            torch_out = wrap(di, dim, dc, dcm)
        diffs = [float(np.abs(o - t.numpy()).max()) for o, t in zip(onnx_out, torch_out)]
        print(f"[verify] max abs diff per head (torch vs onnx): "
              f"{[f'{d:.2e}' for d in diffs]}")
        assert max(diffs) < 1e-3, "ONNX/torch mismatch too large"
        print("[verify] parity OK")
    except ImportError:
        print("[verify] onnxruntime not installed here — skipped parity check")

    print(f"[done] export dir: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
