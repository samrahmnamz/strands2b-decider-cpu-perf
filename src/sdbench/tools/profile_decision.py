#!/usr/bin/env python3
"""Profile a single routing decision to see where CPU time goes.

Reproduces the bottleneck analysis in docs/OPTIMIZATION.md: the dominant cost is
`torch._C._nn.linear` (the matmuls), which is why the merge + precision + compile
stack targets them. Run on any tier.

Usage:
  python -m sdbench.tools.profile_decision [--tier medium] [--threads 32] [--iters 20]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tiny")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--merge", action="store_true", help="profile the merged model")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")

    from sdbench.optimize import OptConfig, build_engine
    from sdbench.workload import PROMPTS, route_question

    cfg = OptConfig(threads=args.threads, merge_lora=args.merge, matmul_precision="high")
    engine = build_engine(cfg)
    q = route_question()
    state = PROMPTS[args.tier]

    for _ in range(5):
        engine.ask(state, q)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.iters):
        engine.ask(state, q)
    pr.disable()

    print(f"=== tier={args.tier} threads={args.threads} merge={args.merge} "
          f"iters={args.iters} ===")
    print("--- top cumulative ---")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(20)
    print("--- top self-time ---")
    pstats.Stats(pr).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
