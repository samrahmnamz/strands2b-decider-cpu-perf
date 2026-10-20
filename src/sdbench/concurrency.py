#!/usr/bin/env python3
"""Concurrency scaling: N worker processes, each a pinned optimized engine replica.

Models a CPU router running a pool of worker processes. Each worker pins to a
disjoint group of physical cores, loads an optimized engine, and serves routing
decisions from a rotating prompt mix for a fixed duration. Reports aggregate
throughput and the latency distribution under load.

Keep workers * threads_per <= 64 (physical cores on r8i.32xlarge).

Examples:
  python -m sdbench.concurrency --workers 8 --threads-per 8 --mix short
  python -m sdbench.concurrency --workers 16 --threads-per 4 --duration 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from .stats import pct
import statistics

PHYSICAL_CORES = list(range(64))


def _worker(wid, cores, threads, duration, warmup_n, mix_name, result_q):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        pass

    from .optimize import OptConfig, build_engine
    from .workload import PROMPTS, TRAFFIC_MIX, route_question

    cfg = OptConfig(threads=threads, merge_lora=True, matmul_precision="high", compile=False)
    engine = build_engine(cfg)
    q = route_question()
    states = [PROMPTS[m] for m in TRAFFIC_MIX[mix_name]]

    for i in range(warmup_n):
        engine.ask(states[i % len(states)], q)

    lat = []
    count = 0
    i = 0
    t_end = time.perf_counter() + duration
    while time.perf_counter() < t_end:
        s = time.perf_counter()
        engine.ask(states[i % len(states)], q)
        lat.append((time.perf_counter() - s) * 1000)
        count += 1
        i += 1
    result_q.put({"wid": wid, "count": count, "lat": lat})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads-per", type=int, required=True)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mix", choices=["default", "short"], default="short")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    need = args.workers * args.threads_per
    if need > len(PHYSICAL_CORES):
        print(f"WARNING: {need} cores requested > {len(PHYSICAL_CORES)} physical; oversubscribed.")

    groups = []
    for w in range(args.workers):
        start = (w * args.threads_per) % len(PHYSICAL_CORES)
        groups.append([PHYSICAL_CORES[(start + j) % len(PHYSICAL_CORES)] for j in range(args.threads_per)])

    ctx = mp.get_context("spawn")
    rq = ctx.Queue()
    procs = []
    for w in range(args.workers):
        p = ctx.Process(target=_worker, args=(w, groups[w], args.threads_per,
                                              args.duration, args.warmup, args.mix, rq))
        p.start()
        procs.append(p)
    results = [rq.get() for _ in range(args.workers)]
    for p in procs:
        p.join()

    total = sum(r["count"] for r in results)
    all_lat = [x for r in results for x in r["lat"]]
    agg = total / args.duration

    print(f"workers={args.workers} threads_per={args.threads_per} cores={need} "
          f"mix={args.mix} duration={args.duration}s")
    print(f"  aggregate throughput: {agg:.1f} decisions/sec  (total {total})")
    print(f"  latency under load: p50={pct(all_lat,0.5):.1f}ms p95={pct(all_lat,0.95):.1f}ms "
          f"p99={pct(all_lat,0.99):.1f}ms mean={statistics.fmean(all_lat):.1f}ms")

    out = {
        "workers": args.workers, "threads_per": args.threads_per, "cores_used": need,
        "mix": args.mix, "duration_s": args.duration, "total_decisions": total,
        "aggregate_throughput_dec_per_s": round(agg, 1),
        "latency_under_load_ms": {"p50": round(pct(all_lat, 0.5), 1),
                                  "p95": round(pct(all_lat, 0.95), 1),
                                  "p99": round(pct(all_lat, 0.99), 1),
                                  "mean": round(statistics.fmean(all_lat), 1)},
        "per_worker_counts": [r["count"] for r in results],
    }
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
