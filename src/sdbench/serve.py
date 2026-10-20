#!/usr/bin/env python3
"""Deployable optimized System One server for CPU.

Serves POST /v1/systemone (identical schema to `strands-decider serve`). It reuses
strands-decider's own FastAPI app -- so request/response handling is exactly the
upstream behaviour -- and then applies the CPU optimization stack in place to the
loaded engine: LoRA merged, matmul precision=high, optional torch.compile.

CPU inference is thread-safe for the forward pass, so one process overlaps requests.
For higher throughput, run several replicas behind a load balancer (docs/DEPLOYMENT.md):
throughput on this model is memory-bandwidth bound, so ~8 replicas x 8 threads
saturate an r8i.32xlarge.

Examples:
  numactl --physcpubind=0-31 --membind=0 -- \
    python -m sdbench.serve --threads 32 --compile --port 8000

  curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
    "state": "Payouts failing for 3 days!",
    "questions": {"route": {"type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": null, "technical_support": null, "sales": null}}}
  }'
"""
from __future__ import annotations

import argparse
import time


def optimize_engine_in_place(engine, *, matmul_precision="high", merge=True, compile_model=False):
    """Apply the optimization stack to an already-loaded SystemOneEngine."""
    import torch

    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if merge:
        m = getattr(engine.model.torso, "merge_and_unload", None)
        if m is not None:
            with torch.inference_mode():
                engine.model.torso = m()
    if compile_model:
        engine.model.torso = torch.compile(engine.model.torso, dynamic=True)
    return engine


def build_app(threads: int, compile_model: bool, model_name: str | None):
    import torch

    torch.set_num_threads(threads)

    import strands_decider.server as sd_server
    from .workload import PROMPTS, TIERS, route_question

    # Let strands build and load the app/engine (correct upstream request handling).
    app = sd_server.create_app(
        "StrandsAgents/strands-decider-2B-hobson-v19",
        device="cpu", use_prefix_cache=True, model_name=model_name,
    )
    engine = sd_server.get_engine()

    t0 = time.perf_counter()
    optimize_engine_in_place(engine, merge=True, matmul_precision="high", compile_model=compile_model)
    # Warm (and compile, if enabled) across tier shapes before taking traffic.
    q = route_question()
    for t in TIERS:
        engine.ask(PROMPTS[t], q)
    ready_s = time.perf_counter() - t0

    @app.get("/opt")
    def opt_status() -> dict:
        return {"threads": torch.get_num_threads(), "merge_lora": True,
                "matmul_precision": "high", "compile": compile_model,
                "optimize_ready_s": round(ready_s, 2)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--compile", action="store_true", help="torch.compile (one-time ~150s cost)")
    ap.add_argument("--model-name", default=None)
    args = ap.parse_args()

    import uvicorn

    app = build_app(args.threads, args.compile, args.model_name)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
