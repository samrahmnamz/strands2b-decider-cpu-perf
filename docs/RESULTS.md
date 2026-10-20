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
