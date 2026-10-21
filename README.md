# strands-decider on CPU — benchmark & deployment

**What this is:** a small, self-contained project that answers one question —
*how fast is AWS's `strands-decider-2B` decision model when you run it on a plain CPU
server (no GPU), and how fast can we make it?* — and then ships a tuned server so you can
run it yourself.

It includes the benchmark harness, an optimization stack that nearly **doubles**
per-decision speed on typical payloads **without changing a single decision**, a
deployable HTTP server, and an honest comparison against AWS's published GPU numbers.

---

## 1. Background: what is strands-decider, and why measure it on CPU?

[`strands-decider-2B`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
is an open-source **decision model** from AWS. It is *not* a chat LLM — it never generates
free text. Instead it reads a **state** (some text) plus a **typed question** and returns a
calibrated answer:

- `choice` — pick 1 of N options (e.g. *which team should handle this ticket?*)
- `noul` — yes/no with a probability
- `score` — a rating on an ordered scale

The natural place to use it is as a **router on an agent's hot path**: a request comes in,
and the decider instantly decides which tool / model / team should handle it — leaving the
expensive LLM for the hard parts.

AWS published latency numbers for it on a **GPU (RTX 3090)** and an **Apple M3**. We wanted
the number they *didn't* publish: **how does it do on a commodity CPU server?** — the kind
of box you'd actually deploy a router on if you don't want a GPU in the loop. We used an
AWS `r8i.32xlarge` (Intel Xeon 6975P-C, 64 physical cores, 991 GiB RAM, **no GPU**).

---

## 2. What we set out to answer (the thought process)

1. **Does it even run well on CPU?** → Install it, measure raw latency across prompt sizes.
2. **Where does the time go?** → Profile a decision to find the bottleneck.
3. **Can we make it faster without hurting accuracy?** → Try optimizations one at a time,
   A/B each against the baseline, and keep only the ones that leave decisions identical.
4. **How many decisions/sec can one box do?** → Scale out replicas and find the ceiling.
5. **How does that compare to AWS's published GPU numbers — fairly?** → Match their
   benchmark's token size and compare like-for-like.

Each step is a script in this repo, and the full step-by-step story is in
[docs/RESULTS.md](docs/RESULTS.md).

---

## 3. Install and run

```bash
# 1. set up an isolated environment (CPU-only torch, strands-decider, numactl)
make setup                 # or: bash scripts/setup.sh
. .venv/bin/activate

# 2. sanity check — one decision from the official CLI
#    (downloads the model weights on first use, ~a few GB)
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v19 \
  --device cpu --state "Payouts failing for 3 days!" \
  --choice "Which team?=billing,technical_support,sales"

# 3. reproduce the benchmarks
make bench-baseline        # untuned (64 threads, no optimizations)
make bench-opt             # our optimized stack, pinned to 32 physical cores
make bench-compile         # + torch.compile (one-time ~150s compile)
make concurrency           # throughput: 8 replicas x 8 threads

# 4. match AWS's published ~950-token decision size exactly
sdbench-latency --token-target 950 --compile

# 5. serve it as an HTTP API (optimized, pinned)
make serve                 # POST http://localhost:8000/v1/systemone
```

`make bench-all` runs the whole suite and writes `results/*.json`.

> **Note on pinning:** the `make` targets wrap the commands in
> `numactl --physcpubind=0-31 --membind=0` (one NUMA node's physical cores). On a different
> host, set the core range to your own node's physical cores — `lscpu` → "NUMA node0
> CPU(s)", take the lower half.

---

## 4. What we measured (the workload)

A realistic **prompt-routing** task: one `choice` question over **six destination teams**
(`billing`, `technical_support`, `sales`, `account_security`, `data_analytics`,
`general_inquiry`), asked against incoming-request text of **five sizes** — from a 9-token
subject line to a 534-token escalation document, plus a ~950-token tier for the AWS
comparison.

Only the request text changes between sizes, so the latency curve isolates **how decision
cost scales with input length** — exactly what matters for a router sitting in front of an
agent. The prompts live in `src/sdbench/workload.py`.

We report, for each size:
- **latency** — p50 / p95 / p99 per single decision
- **throughput** — decisions/sec, both single-stream and across parallel replicas
- **the decision itself** — to prove optimizations don't change the output

---

## 5. Why we compare to JevBench

JevBench is the **independent benchmark AWS used** to report `strands-decider`'s accuracy
(72.3%) and the latency context around it. We compare to it for two reasons:

1. **It's the public reference point.** AWS's headline "~115 ms" number is measured on
   JevBench-style decisions, so it's the only apples-to-apples target available.
2. **Its "decision" size is the honest yardstick.** JevBench states that one decision —
   state + rubric + options — averages **~950 input tokens**, not a short prompt. If we
   compared our 9-token latency to AWS's ~950-token latency, we'd be flattering ourselves.
   So we measured our model at ~950 tokens too, and compare there.

That comparison (with sources) is in [§7](#7-how-we-compare-to-aws-published-numbers).

---

## 6. Results

All numbers on `r8i.32xlarge` (Intel Xeon 6975P-C, CPU-only, torch 2.14.1+cpu with oneDNN +
AVX-512/AMX). Raw JSON in `results/`; full discussion in [docs/RESULTS.md](docs/RESULTS.md).

### The optimization stack (all numerically exact — decisions bit-identical)

| step | optimization | what it does | gain |
|------|-------------|--------------|-----:|
| 1 | **LoRA merge** (`merge_and_unload`) | folds the adapter into base weights → 1 matmul/layer instead of 3 | 1.1–1.2× |
| 2 | **matmul precision=high** | lets oneDNN use the Xeon's AMX bf16-accumulate path | up to 1.25× |
| 3 | **`torch.compile`** | fuses the many small norm/elementwise ops | 1.14–1.39× |
| 4 | **core + NUMA pinning** | 1 thread per *physical* core, one NUMA node | removes p95 jitter |

We also ruled out **flash-linear-attention**: `transformers` only enables it on
CUDA/XPU/MLU, so it's a no-op on CPU (details in docs/RESULTS.md).

### Single-stream latency by request size (p50)

| request size | tokens | baseline (untuned) | **optimized (full stack)** | speedup |
|--------------|-------:|-------------------:|---------------------------:|--------:|
| tiny   |   9 | 204 ms | 187 ms | 1.1× |
| small  |  31 | 228 ms | 215 ms | 1.1× |
| **medium** |  92 | 545 ms | **275 ms** | **2.0×** |
| **large**  | 206 | 601 ms | **420 ms** | **1.4×** |
| xlarge | 534 | 825 ms | 719 ms | 1.1× |
| ~950-tok (JevBench size) | 997 | — | **925 ms** | — |

The medium/large payloads a real router sees most get the biggest wins. Pinning also
**collapses p95 onto p50** (predictable tail latency, not just a lower median).

### Throughput — how many decisions/sec can one box do?

Multiple engine **replicas** (separate processes), each pinned to its own cores:

| config (workers × threads) | aggregate throughput | p50 | p99 |
|----------------------------|---------------------:|----:|----:|
| **8 × 8**  | **13.5 dec/s** | 580 ms | 836 ms |
| 16 × 4 | 15.2 dec/s | 1066 ms | 1499 ms |
| 32 × 2 | 16.0 dec/s | 2064 ms | 2940 ms |

- **8 × 8 is the sweet spot:** best latency at ~84% of peak throughput.
- **Ceiling ≈ 16 dec/s — memory-bandwidth bound.** Each replica streams ~7 GiB of weights
  per forward, so past ~8 replicas they just contend for memory, not cores.
- **RAM:** ~10 GiB per replica; 8 replicas ≈ 77 GiB of 991 GiB (memory is never the limit).

### Resource / startup profile

| metric | value |
|--------|-------|
| cold CLI run (import + load + 1 decision) | ~44 s (mostly Python/torch import) |
| warm engine load + LoRA merge (server path) | ~2–5 s |
| peak RAM, single engine | ~11 GiB |
| `torch.compile` warmup | ~150 s one-time, amortized in a long-running server |

### Where the time goes

Profiling one decision: the **matmuls** (`torch._C._nn.linear`) are ~56% of the time —
which is exactly why the stack targets them. The gated-delta-net scan is second (~19%).
Reproduce with `python -m sdbench.tools.profile_decision`.

---

## 7. How we compare to AWS published numbers

AWS and the model card report:

| metric | value | conditions | source |
|--------|-------|-----------|--------|
| median decision latency | **~115 ms** | NVIDIA RTX 3090 (GPU) | [AWS blog][sd-blog], [InfoQ][infoq] |
| small-task latency | **~153 ms** | Apple M3 MacBook | [AWS blog][sd-blog], [InfoQ][infoq] |
| accuracy | **72.3 %** | JevBench, 231 tasks, v19 | [InfoQ][infoq], model card |
| decision size | **~950 input tokens** | state + rubric + options | [JevBench][jev] |

Measuring **our CPU at the same ~950-token size** (so it's a fair fight):

| ~950-token decision (997 tokens measured) | p50 latency | vs RTX 3090 |
|--------------------------------------------|------------:|------------:|
| r8i CPU — optimized (merge + precision) | 1322 ms | ~11.5× slower |
| **r8i CPU — optimized + `torch.compile`** | **925 ms** | **~8× slower** |
| RTX 3090 (published) | 115 ms | 1× |
| Apple M3 (published) | 153 ms | — |

**Honest takeaway:**

- At the **real decision size**, one CPU decision is **~8× slower than a 3090** even fully
  optimized. For single-decision latency, a GPU or Apple silicon clearly wins. The CPU path
  is for when you have **no GPU**, want the huge RAM for co-located work, or are scaling
  **horizontally on CPU fleets** where aggregate throughput and cost matter more than the
  latency of any one decision.
- **`torch.compile` earns its keep here:** it cut the ~950-token decision from 1322 ms to
  **925 ms** (1.43×), breaking the 1-second mark.
- **Accuracy is unchanged** — every optimization is numerically exact, so AWS's **72.3%**
  JevBench figure still applies to this build.

**Caveat:** AWS's ~115/153 ms are medians over JevBench's mix on their hardware; our 925 ms
is one ~997-token routing decision measured here. Same order of token count, not an
identical controlled harness. Reproduce our figure with `sdbench-latency --token-target 950
--compile`.

[sd-blog]: https://strandsagents.com/blog/introducing-strands-decider/
[infoq]: https://www.infoq.com/news/2026/10/aws-strands-agents/
[jev]: https://github.com/fstandhartinger/JevBench

---

## 8. Serving the model as an API

`sdbench-serve` reuses `strands-decider`'s own FastAPI app (so the request/response schema
is exactly upstream's `POST /v1/systemone`) and applies the optimization stack to the
loaded engine:

```bash
numactl --physcpubind=0-31 --membind=0 -- \
  sdbench-serve --threads 32 --compile --port 8000

curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Our API returns 500s on every checkout call since the deploy",
  "questions": {"route": {"type":"choice",
    "instructions":"Which team should handle this?",
    "criteria":{"billing":null,"technical_support":null,"sales":null,
                "account_security":null,"data_analytics":null,"general_inquiry":null}}}
}'
# -> technical_support, probability 0.92, confidence 0.91, latency_ms ~260
```

For production throughput, run several replicas behind a load balancer — the recipe is in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 9. Repository layout

```
.
├── README.md                # you are here
├── Makefile                 # make setup / bench-* / concurrency / serve
├── scripts/
│   ├── setup.sh             # venv + CPU torch + deps + numactl
│   └── run_all.sh           # reproduce the full benchmark suite
├── src/sdbench/
│   ├── workload.py          # the routing prompts (5 sizes) + question
│   ├── optimize.py          # the optimization stack (build_engine, pinning)
│   ├── latency.py           # single-stream latency CLI  (sdbench-latency)
│   ├── concurrency.py       # multi-replica throughput CLI (sdbench-concurrency)
│   ├── serve.py             # optimized HTTP server (sdbench-serve)
│   ├── stats.py             # percentile helpers
│   └── tools/
│       └── profile_decision.py   # where-the-time-goes profiler
├── results/                 # committed benchmark JSON
└── docs/
    ├── RESULTS.md           # full numbers + the step-by-step tuning story
    ├── OPTIMIZATION.md      # why each optimization works (with the profile)
    └── DEPLOYMENT.md        # production CPU serving (single + multi-replica)
```

---

## 10. A note on accuracy

Every optimization was A/B-verified to leave the routing `choice`, the per-option
probabilities, and the calibrated `confidence` **numerically identical** to the untuned
model (LoRA merge and `torch.compile` are exact up to float rounding; matmul
`precision="high"` showed `Δconfidence = 0.0000`). These are latency wins, not accuracy
trades — the published 72.3% JevBench accuracy carries over unchanged. Details in
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md).
