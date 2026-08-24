"""Single-GPU + multi-GPU (DDP) helpers.

Kaggle T4 x2, Colab 1x T4/L4, a local NVIDIA PC, and Mac MPS all go through
the same train() entry. Two-or-more CUDA devices auto-spawn DDP unless the
process was already launched with torchrun.
"""

from __future__ import annotations

import os
import socket
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .config import TrainConfig


def cuda_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def already_distributed() -> bool:
    return "RANK" in os.environ or "LOCAL_RANK" in os.environ


def is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def rank() -> int:
    return torch.distributed.get_rank() if is_dist() else 0


def world_size() -> int:
    return torch.distributed.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return rank()


def broadcast_bool(value: bool) -> bool:
    """Broadcast a bool from rank 0 to all ranks (so every rank breaks the epoch
    loop together on early stop). No-op returning `value` when not distributed."""
    if not is_dist():
        return value
    dev = torch.device(f"cuda:{local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    t = torch.tensor([1 if value else 0], dtype=torch.int, device=dev)
    torch.distributed.broadcast(t, src=0)
    return bool(t.item())


def barrier() -> None:
    if not is_dist():
        return
    # On NCCL a device-unbound barrier picks a device "under the current context"
    # and can DEADLOCK on Kaggle T4x2 (P2P/IB disabled). Bind it to this rank's
    # GPU so both ranks agree on the collective's device.
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[local_rank()])
    else:
        torch.distributed.barrier()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _wanted_gpus(cfg: TrainConfig) -> int:
    n = cuda_gpu_count()
    wanted = int(getattr(cfg.run, "gpus", 0) or 0)
    if wanted > 0:
        return min(wanted, n)
    return n


def should_spawn_ddp(cfg: TrainConfig) -> bool:
    mode = str(getattr(cfg.run, "distributed", "auto") or "auto").lower()
    if mode in {"off", "false", "0", "no"}:
        return False
    if already_distributed() or is_dist():
        return False
    n = _wanted_gpus(cfg)
    if n < 2:
        return False
    if mode in {"on", "true", "1", "yes"}:
        return True
    return True  # auto + 2+ CUDA GPUs


def _backend() -> str:
    if torch.cuda.is_available() and sys.platform != "win32":
        return "nccl"
    return "gloo"


def init_process_group() -> None:
    if is_dist() or not already_distributed():
        return
    # Kaggle T4 x2 often has no P2P/IB between cards — NCCL hangs without this.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    backend = _backend()
    if torch.cuda.is_available():
        dev = torch.device("cuda", local_rank())
        torch.cuda.set_device(dev)
        # device_id binds the whole process group to this rank's GPU: it enables
        # eager NCCL init and makes barrier()/collectives device-correct, which is
        # what stops the T4x2 hang after the first epoch's eval.
        try:
            torch.distributed.init_process_group(backend=backend, device_id=dev)
        except TypeError:  # older torch without device_id kwarg
            torch.distributed.init_process_group(backend=backend)
    else:
        torch.distributed.init_process_group(backend=backend)


def destroy_process_group() -> None:
    if is_dist():
        torch.distributed.destroy_process_group()


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def spawn_ddp(cfg: TrainConfig, worker) -> None:
    import torch.multiprocessing as mp

    n = _wanted_gpus(cfg)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    print(f"[dist] auto-launch DDP on {n} CUDA GPUs (Kaggle T4x2 / multi-GPU PC)")
    mp.spawn(worker, args=(n, cfg), nprocs=n, join=True)


def ddp_worker_setup(rank_i: int, world: int) -> None:
    os.environ["RANK"] = str(rank_i)
    os.environ["LOCAL_RANK"] = str(rank_i)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    init_process_group()
