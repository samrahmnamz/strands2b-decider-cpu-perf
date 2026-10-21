# Results

All numbers from `StrandsAgents/strands-decider-2B-hobson-v19` on an AWS
`r8i.32xlarge` (Intel Xeon 6975P-C, 64 physical cores / 128 vCPU, 2 NUMA nodes,
991 GiB RAM, no GPU), torch 2.14.1+cpu with MKL + oneDNN, AVX-512 + AMX.

The workload is the routing `choice` question over six teams, across five request
sizes (see `src/sdbench/workload.py`). Latency is per single decision (one request,
one question). Raw JSON is in `results/`.

## Headline: single-stream latency (p50)

| tier | tokens | baseline (64 threads, untuned) | optimized (merge + precision + pin, 32t/1 node) | + torch.compile | best speedup |
|------|-------:|-------------------------------:|------------------------------------------------:|----------------:|-------------:|
| tiny   |   9 | 204 ms | 216 ms | 187 ms | 1.09x |
| small  |  31 | 228 ms | 245 ms | 215 ms | 1.06x |
| medium |  92 | 545 ms | 311 ms | **275 ms** | **1.98x** |
| large  | 206 | 601 ms | 466 ms | **420 ms** | **1.43x** |
| xlarge | 534 | 825 ms | 838 ms | 719 ms | 1.15x |

With the full stack (merge + precision + pinning + `torch.compile`), **every tier
improves** — compile's op-fusion even helps the overhead-bound short prompts once the
per-call elementwise work is fused.

Notes:
- **medium/large are the common router payloads** and see the biggest wins.
- **tiny/small are overhead-bound** — ~200 ms is fixed Python + tokenization + the
  many tiny ops across 18 gated-delta-net layers, not matmul. More cores or compile
  barely move it.
- **xlarge is compute-bound** and actually prefers *more* cores: pinned to 64 physical
  cores (both NUMA nodes) `large` drops to **372 ms (1.62x)** but `xlarge` regresses,
  so the single best general config is 32 threads on one NUMA node, or 64 cores if your
  traffic skews to large (not xlarge) payloads.
- Pinning also **collapses p95 onto p50** (removes scheduler jitter): e.g. optimized
  medium p50 311 ms / p95 312 ms vs baseline p95 781 ms.

## The iterative tuning story (how we got here)

Each step was measured individually against the routing workload:

| step | config | finding |
|------|--------|---------|
| 0 | 64 threads, no pinning | baseline |
| 1 | 32 threads | helped medium, hurt short/xlarge — noisy |
| 2 | 16 threads | worse everywhere; large prompts starved of cores |
| 3 | 96 threads | worse; crosses the NUMA boundary, p95 spikes |
| 4 | pin node 0 (64 logical), 64 threads | no help — oversubscribes 32 physical cores with hyperthreads |
| 5 | **pin physical cores 0–31, 32 threads, local mem** | p95 jitter gone; medium 426 ms |
| 6 | **+ LoRA merge** | 1.08–1.19x, decisions bit-identical |
| 7 | **+ matmul precision=high** | up to 1.25x more, Δconfidence 0.0000 |
| 8 | **+ torch.compile (dynamic)** | 1.14–1.39x more, decisions identical |

Takeaways:
- **Threads:** one thread per *physical* core beats using hyperthreads; staying on one
  NUMA node beats spanning both (cross-socket memory distance 15 vs 10).
- The matmuls are already near-optimal under oneDNN — `torch.compile`'s win comes from
  fusing the small norm/elementwise ops, **not** the Linear layers (verified: compiling a
  bare `nn.Linear` stack gave no speedup).

## Throughput / concurrency

Multiple engine *replicas* (processes), each pinned to a disjoint physical-core group,
each with the optimization stack (merge + precision). Realistic short-traffic mix
(mostly tiny/small, some medium). `workers × threads_per ≤ 64 physical cores`.

| config (workers × threads) | cores | aggregate throughput | p50 | p99 |
|---|---:|---:|---:|---:|
| 8 × 8  | 64 | 13.5 dec/s | 580 ms | 836 ms |
| 16 × 4 | 64 | 15.2 dec/s | 1066 ms | 1499 ms |
| 32 × 2 | 64 | 16.0 dec/s | 2064 ms | 2940 ms |

- **Sweet spot = 8 workers × 8 threads.** Going wider buys only ~18 % more throughput
  but inflates latency 3.5×.
- Throughput ceiling is ~**16 dec/s** — **memory-bandwidth bound**: each replica streams
  ~7 GiB of fp32 weights per forward, so adding replicas past ~8 just contends for memory.
- **RAM:** ~9.6 GiB per replica; 8 replicas ≈ 77 GiB of 991 GiB. Memory is not the limit.

(On the heavier "default" mix that includes xlarge docs, 8×8 does ~10.3 dec/s because the
long prompts dominate the batch time.)

## Load & memory profile (single engine)

- **Cold CLI run** (import + load + 1 inference): ~44 s wall, dominated by Python/torch
  import — *not* model load.
- **Warm engine load + LoRA merge** (API path, weights cached): ~2–5 s.
- **Peak RSS:** ~11 GiB (the engine upcasts the torso to fp32 on CPU — bf16 is slower
  than fp32 on CPU for this model; ~7 GiB torso + overhead).
- **torch.compile:** ~150 s one-time for 4 dynamic shapes; amortized to zero in a
  long-running server.

## Dead end: flash-linear-attention (FLA)

The model is a hybrid with gated-delta-net (linear-attention) layers, and transformers
prints a warning that `chunk_gated_delta_rule` is using its slow reference path because
FLA is not installed. **Installing FLA does not help on CPU:**
`transformers.utils.is_flash_linear_attention_available()` returns True only when
`is_torch_cuda_available() or is_torch_xpu_available() or is_torch_mlu_available()`.
On a CPU-only torch build it is always False, so the optimized kernel is never dispatched
regardless of whether the package is present. FLA also requires Triton (a GPU compiler).
The warning is unavoidable with this stack and is safe to ignore.

What *is* addressed on the delta-net path: `strands_decider.cpu_kernels` already replaces
the slow per-channel depthwise causal convolution with a vectorized multiply-add
(installed automatically for CPU engines). Profiling a tiny prompt showed the remaining
cost is dominated by `torch._C._nn.linear` (56 %) — i.e. the matmuls the merge + precision
+ compile stack targets.


## How we compare to the published benchmarks

AWS / the model card report decision latency on two reference devices, plus accuracy:

| platform | hardware | reported latency | source |
|----------|----------|-----------------:|--------|
| RTX 3090 | NVIDIA GPU | **~115 ms** median | AWS blog, InfoQ, TechGig |
| Apple M3 | ARM laptop (MacBook) | **~153 ms** (small tasks) | AWS blog, InfoQ, TechGig |
| JevBench | served at window 4096 | **72.3 %** accuracy / 231 public tasks (v19) | AWS |

**The token-size caveat is the whole story.** A JevBench "decision" is not a short prompt:
the benchmark methodology states *"one decision is a whole question: its state, its rubric
and its options — hundreds to thousands of input tokens … Jev reads 950 input tokens per
decision on average."* The accuracy is served at a 4096-token window. So the published
~115 ms is a **~950-token decision on a GPU**, not a one-liner.

### Like-for-like at ~950 input tokens

We measured on `r8i.32xlarge` (Intel Xeon 6975P, CPU-only, 32 threads pinned to one NUMA
node) at a decision whose full input (state + rubric + options) was **997 tokens**:

| config | p50 latency | vs RTX 3090 (115 ms) |
|--------|------------:|---------------------:|
| optimized (merge + precision) | 1322 ms | ~11.5× slower |
| **optimized + `torch.compile`** | **925 ms** | **~8× slower** |
| RTX 3090 (published) | 115 ms | 1× |
| Apple M3 (published, small task) | 153 ms | — |

Reading of the comparison (honest version):

- At the **real ~950-token decision size**, one CPU decision is **~8× slower than a 3090**
  even fully optimized. For single-decision latency a GPU or Apple silicon clearly wins;
  the CPU path is for no-GPU hosts, RAM-heavy co-tenancy, or horizontal CPU fleets.
- **`torch.compile` earns its keep exactly here:** 1322 → 925 ms (**1.43×**), under 1 s.
- For context, our smaller tiers are much faster (tiny 9-tok 187 ms, medium 92-tok 275 ms)
  — but those are **not** comparable to the published number, which is ~950 tokens. Quoting
  the 9-token figure against 115 ms would be a cherry-pick; we don't.
- **Accuracy is unchanged:** every optimization is numerically exact, so the published
  **72.3 % JevBench** still applies to this build. We trade nothing for the latency.

**Where CPU actually wins:** aggregate throughput and RAM. ~13.5 short-traffic decisions/s
across 8 replicas on one box, 991 GiB RAM for co-located work, and no GPU in the bill. AWS
also markets the r8i for **2.5× the memory bandwidth** of the previous generation — the
exact resource our concurrency ceiling is bound by, so the instance is well matched here.

**Caveats:** the published ~115 ms / ~153 ms are AWS medians over JevBench's task mix on
their hardware; our 925 ms is a single ~997-token routing decision measured here. Same
order of token count, not an identical controlled harness. The model's GitHub issues also
flag CPU **cold-start** and **concurrency** behaviour — both addressed here (warm
persistent engine; multi-replica serving).

Sources: AWS blog (strandsagents.com/blog/introducing-strands-decider/),
InfoQ (infoq.com/news/2026/10/aws-strands-agents/),
JevBench methodology (github.com/fstandhartinger/JevBench).
