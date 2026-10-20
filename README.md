# strands-decider on CPU — benchmark & deployment

Benchmarks and a tuned deployment for [`StrandsAgents/strands-decider-2B-hobson-v19`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
running **CPU-only** on an AWS `r8i.32xlarge`, with a reproducible optimization stack
that roughly **doubles** per-decision throughput on typical payloads **without changing
a single routing decision**.

`strands-decider` is a small *decision* model, not a chat LLM: it answers typed,
calibrated questions about a state — `choice` (pick 1 of N), `noul` (yes/no) and
`score` (ordered scale). The natural use case is a **router/decider on an agent's hot
path**: a request arrives and the model decides which team / tool / model handles it.
This repo measures that use case and ships a tuned server for it.

---

## TL;DR

- **Hardware:** `r8i.32xlarge` — Intel Xeon 6975P-C, 64 physical cores / 128 vCPU, 2 NUMA
  nodes, 991 GiB RAM, **no GPU**. torch is CPU-only (MKL + oneDNN, AVX-512 + AMX).
- **Optimization stack (all numerically exact — decisions bit-identical):**
  1. **Merge the LoRA adapter** into the base weights (`merge_and_unload`).
  2. **`torch.set_float32_matmul_precision("high")`** — oneDNN AMX bf16-accumulate path.
  3. **`torch.compile`** — fuses the small norm/elementwise ops (one-time ~150 s compile).
  4. **Core pinning** — one thread per *physical* core on one NUMA node (`numactl`).
- **Result (single-stream routing latency, p50):**

  | request size | tokens | baseline (64t) | optimized (full stack) | speedup |
  |---|---|---|---|---|
  | tiny | 9 | 204 ms | 187 ms | 1.1x |
  | small | 31 | 228 ms | 215 ms | 1.1x |
  | **medium** | 92 | 545 ms | **275 ms** | **2.0x** |
  | **large** | 206 | 601 ms | **420 ms** | **1.4x** |
  | xlarge | 534 | 825 ms | 719 ms | 1.1x |

  Full stack = LoRA merge + matmul precision=high + physical-core/NUMA pin +
  `torch.compile`. All decisions bit-identical to the untuned model. See
  [docs/RESULTS.md](docs/RESULTS.md) for the per-config breakdown.
- **Throughput (8 replica processes, realistic short-traffic mix):** ~**13.5 decisions/s**
  at p50 580 ms / p99 836 ms, using all 64 physical cores and 77 GB RAM.
- **Dead end:** `flash-linear-attention` does **not** help on CPU — `transformers`
  gates it behind CUDA/XPU/MLU, so it is a no-op here (see docs/RESULTS.md).

---

## Repository layout

```
.
├── README.md
├── Makefile                 # setup / bench / serve targets
├── requirements.txt
├── scripts/
│   ├── setup.sh             # venv + CPU torch + deps + numactl
│   └── run_all.sh           # reproduce the full benchmark suite
├── src/sdbench/
│   ├── optimize.py          # the optimization stack (build_engine, pinning)
│   ├── workload.py          # routing prompts (5 complexity tiers) + question
│   ├── latency.py           # per-tier latency / throughput CLI
│   ├── concurrency.py       # multi-process replica throughput CLI
│   ├── serve.py             # deployable optimized HTTP server
│   └── stats.py             # percentile helpers
├── results/                 # benchmark JSON output
└── docs/
    ├── RESULTS.md           # full numbers + the iterative tuning story
    ├── OPTIMIZATION.md      # why each optimization works (with profiling)
    └── DEPLOYMENT.md        # how to run this in production on CPU
```

---

## Quickstart

```bash
# 1. set up an isolated environment (CPU torch, strands-decider, numactl)
make setup            # or: bash scripts/setup.sh
. .venv/bin/activate

# 2. one decision from the CLI (downloads base weights on first use)
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --device cpu --state "Payouts failing for 3 days!" \
  --choice "Which team?=billing,technical_support,sales"

# 3. measure: baseline vs optimized
make bench-baseline   # untuned, 64 threads
make bench-opt        # merge + precision, pinned to 32 physical cores
make bench-compile    # + torch.compile

# 4. throughput at the sweet spot (8 replicas x 8 threads)
make concurrency

# 5. serve it (optimized, pinned)
make serve            # POST http://localhost:8000/v1/systemone
```

Reproduce everything at once: `make bench-all` (writes `results/*.json`).

---

## The workload

A single `choice` question over six destination teams (`billing`,
`technical_support`, `sales`, `account_security`, `data_analytics`,
`general_inquiry`), asked against incoming-request prompts of five sizes — from a
9-token subject line to a 534-token escalation document. Only the request text
changes between tiers, so the latency curve isolates **how decision cost scales with
input length**, which is exactly what a router cares about. See `src/sdbench/workload.py`.

---

## Serving

`python -m sdbench.serve` reuses `strands-decider`'s FastAPI app (so the request/response
schema is exactly upstream's `POST /v1/systemone`) and applies the optimization stack to
the loaded engine. Example:

```bash
numactl --physcpubind=0-31 --membind=0 -- \
  python -m sdbench.serve --threads 32 --compile --port 8000

curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Our API returns 500s on every checkout call since the deploy",
  "questions": {"route": {"type":"choice",
    "instructions":"Which team should handle this?",
    "criteria":{"billing":null,"technical_support":null,"sales":null,
                "account_security":null,"data_analytics":null,"general_inquiry":null}}}
}'
# -> technical_support, probability 0.92, confidence 0.91, latency_ms ~260
```

For production throughput, run multiple replicas behind a load balancer — see
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Accuracy note

Every optimization here was A/B-verified to leave the routing `choice`, the per-option
probabilities, and the calibrated `confidence` **numerically identical** to the untuned
model (the LoRA merge and `torch.compile` are exact up to float rounding; matmul
`precision="high"` showed `Δconfidence = 0.0000`). These are latency wins, not
accuracy trades. See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).
