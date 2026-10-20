"""Optimization stack for running strands-decider on CPU.

Four optimizations, each verified to leave routing decisions bit-identical on an
r8i.32xlarge (Intel Xeon 6975P, oneDNN + AVX-512/AMX):

  1. LoRA merge   -- fold the adapter into the base weights (one matmul per layer
                     instead of three). ~1.1-1.2x.
  2. matmul prec  -- torch.set_float32_matmul_precision("high") lets oneDNN use the
                     AMX bf16-accumulate path for fp32 matmuls. up to ~1.25x.
  3. torch.compile-- fuses the many small norm/elementwise ops (not the matmuls,
                     which oneDNN already handles). ~1.14-1.39x. One-time compile cost.
  4. core pinning -- see pin_to_physical_cores() / the numactl wrapper in scripts.
                     One thread per physical core on a single NUMA node removes
                     hyperthread contention and cross-socket memory traffic, which
                     collapses p95 onto p50.

FLA (flash-linear-attention) is intentionally NOT used: transformers'
is_flash_linear_attention_available() requires CUDA/XPU/MLU, so it is a no-op on CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class OptConfig:
    threads: int = 32
    merge_lora: bool = True
    matmul_precision: str = "high"   # "high" | "medium" | "highest"(off)
    compile: bool = False            # one-time ~150s compile; worth it for a server
    compile_dynamic: bool = True
    checkpoint: str = "StrandsAgents/strands-decider-2B-hobson-v19"
    use_prefix_cache: bool = True


def pin_to_physical_cores(cores: list[int]) -> None:
    """Pin the current process to a set of CPU cores (best-effort).

    For a single engine, prefer one thread per *physical* core on one NUMA node.
    From a shell, `numactl --physcpubind=0-31 --membind=0 -- python ...` is equivalent
    and also binds memory, which matters on multi-socket hosts.
    """
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        pass


def build_engine(cfg: OptConfig):
    """Load a strands-decider engine with the optimization stack applied.

    Returns the SystemOneEngine. Call this once and reuse it; loading + merge is a
    few seconds and compile (if enabled) is a one-time cost paid on the first calls.
    """
    import torch

    torch.set_num_threads(cfg.threads)
    if cfg.matmul_precision:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    from strands_decider.infer import load_engine

    engine = load_engine(cfg.checkpoint, device="cpu", use_prefix_cache=cfg.use_prefix_cache)

    if cfg.merge_lora:
        merge = getattr(engine.model.torso, "merge_and_unload", None)
        if merge is not None:
            with torch.inference_mode():
                engine.model.torso = merge()

    if cfg.compile:
        engine.model.torso = torch.compile(
            engine.model.torso, dynamic=cfg.compile_dynamic or None
        )

    return engine


def warmup(engine, question, states, n: int = 3) -> None:
    """Run a few decisions to warm caches and (if enabled) trigger compilation.

    Pass one state per distinct sequence length you care about so torch.compile
    specialises each shape before you start timing.
    """
    for i in range(max(n, len(states))):
        engine.ask(states[i % len(states)], question)
