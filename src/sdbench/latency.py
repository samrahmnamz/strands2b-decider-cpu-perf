#!/usr/bin/env python3
"""Per-tier latency + single-stream throughput for the routing workload.

Examples:
  # baseline (no optimizations), mimics an untuned default
  python -m sdbench.latency --threads 64 --no-merge --no-precision --json-out results/baseline.json

  # full optimization stack (pin externally with numactl for best p95)
  numactl --physcpubind=0-31 --membind=0 -- \
    python -m sdbench.latency --threads 32 --compile --json-out results/optimized.json
"""
from __future__ import annotations

import argparse
import json
import time

from .optimize import OptConfig, build_engine, warmup
from .stats import summarize
from .workload import PROMPTS, TIERS, route_question


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-merge", action="store_true", help="disable LoRA merge")
    ap.add_argument("--no-precision", action="store_true", help="disable matmul precision=high")
    ap.add_argument("--compile", action="store_true", help="enable torch.compile (one-time cost)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = OptConfig(
        threads=args.threads,
        merge_lora=not args.no_merge,
        matmul_precision="highest" if args.no_precision else "high",
        compile=args.compile,
    )

    t0 = time.perf_counter()
    engine = build_engine(cfg)
    load_s = time.perf_counter() - t0

    import torch
    q = route_question()
    warmup(engine, q, [PROMPTS[t] for t in TIERS], n=args.warmup)

    print(f"threads={torch.get_num_threads()} merge={cfg.merge_lora} "
          f"precision={cfg.matmul_precision} compile={cfg.compile} load={load_s:.1f}s")

    results = []
    for tier in TIERS:
        state = PROMPTS[tier]
        ntok = len(engine.tok(state, add_special_tokens=True)["input_ids"])
        lat = []
        choice = conf = None
        for _ in range(args.iters):
            s = time.perf_counter()
            resp = engine.ask(state, q)
            lat.append((time.perf_counter() - s) * 1000)
            a = resp.answers["route"]
            choice, conf = a.choice, a.confidence
        row = {"tier": tier, "state_tokens": ntok, **summarize(lat),
               "route": choice, "confidence": conf}
        results.append(row)
        print(f"  {tier:7s} tok={ntok:5d} p50={row['p50_ms']:8.1f}ms "
              f"p95={row['p95_ms']:8.1f}ms {row['throughput_dec_per_s']:6.2f} dec/s "
              f"-> {choice} ({conf})")

    out = {
        "checkpoint": cfg.checkpoint,
        "config": {"threads": cfg.threads, "merge_lora": cfg.merge_lora,
                   "matmul_precision": cfg.matmul_precision, "compile": cfg.compile},
        "load_s": round(load_s, 2), "iters": args.iters, "results": results,
    }
    if args.json_out:
        import os
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
