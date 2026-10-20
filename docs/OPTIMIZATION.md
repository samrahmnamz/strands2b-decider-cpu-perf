# Optimization stack — why each one works

Everything here is implemented in `src/sdbench/optimize.py` (`build_engine`) and, for
the server, `src/sdbench/serve.py` (`optimize_engine_in_place`). All four were
A/B-verified to leave routing decisions, probabilities and calibrated confidence
**numerically identical** to the untuned model.

## Where the time goes (profile)

Profiling a tiny (9-token) routing decision at 32 threads, by cumulative time:

```
torch._C._nn.linear          2.08 s / 3.73 s   (56%)   11,200 calls
  (LoRA path contributes)    2.47 s cumulative
torch_chunk_gated_delta_rule 0.70 s            (19%)   incl. linalg_solve_triangular 0.20 s
causal_conv1d_cpu            0.09 s                     (already optimized by strands)
```

The dominant cost is **matmuls**, and the LoRA adapter *multiplies* the number of them.
That points straight at optimizations 1–3. The gated-delta-net scan is second; it is pure
fp32 PyTorch and not easily improved on CPU (FLA is CUDA-only — see RESULTS.md).

## 1. Merge the LoRA adapter — `merge_and_unload()`

The checkpoint is a rank-16 LoRA adapter on `Qwen/Qwen3.5-2B-Base`. At inference each
adapted Linear computes `base(x) + B(A(x))` — three matmuls where a plain layer needs
one. PEFT's `merge_and_unload()` folds `A·B·(alpha/r)` into the base weight, so every
layer becomes a single matmul again and the torso reverts from a `PeftModel` wrapper to a
plain `Qwen3_5TextModel`.

- **Win:** 1.08–1.19x (bigger prompts benefit more — they are more matmul-bound).
- **Exactness:** the merge is the same linear algebra, up to float rounding; measured
  routing choice, probabilities and confidence were unchanged.
- **Cost:** sub-second, one-time at load.

## 2. `torch.set_float32_matmul_precision("high")`

On CPU the engine upcasts the torso to fp32 (bf16 is *slower* than fp32 on CPU for this
model). With `precision="high"`, PyTorch lets oneDNN use the Xeon's **AMX bf16-accumulate**
path for fp32 matmuls — same inputs/outputs in fp32, reduced-precision internal
accumulation on the AMX tiles.

- **Win:** up to 1.25x (medium), stacks on the merge.
- **Exactness:** measured `Δconfidence = 0.0000` across all tiers — decisions identical.
- **`"medium"`** was tested too and is *not* worth it: noisier and it perturbs
  confidences (Δ up to 0.005) for no reliable speed gain.

## 3. `torch.compile` (dynamic)

oneDNN already makes the big matmuls near-optimal, so Inductor can't beat them — verified
by compiling a bare `Linear→SiLU→Linear` stack, which showed **no** speedup. The real win
is **fusing the many small ops**: RMSNorm, the elementwise steps of the delta-rule, gating.

- **Win:** 1.14–1.39x on top of merge + precision (medium 1.39x, large 1.34x).
- **Exactness:** routing decisions and confidence unchanged.
- **Cost:** ~150 s one-time compile for the dynamic shapes (use `dynamic=True` so each
  sequence length does not trigger a fresh recompile). Irrelevant for a long-running
  server; **not** worth it for one-shot CLI use.

Enable with `--compile` on `sdbench.latency` / `sdbench.serve`.

## 4. Core & NUMA pinning

The box has 64 physical cores split across 2 NUMA nodes, each core exposing 2
hyperthreads (128 logical CPUs total). Two effects dominate:

- **Hyperthread contention:** two threads on one physical core share its AVX-512/AMX FP
  units — the bottleneck for matmul-heavy inference. One thread per *physical* core is
  faster than filling all logical CPUs.
- **Cross-socket memory:** threads on node 1 reading weights allocated on node 0 pay the
  remote-access penalty (NUMA distance 15 vs 10). Spanning both nodes (e.g. 96 threads)
  *regressed* latency and spiked p95.

**Recommended:** pin one thread per physical core on a single NUMA node and bind memory
there:

```bash
numactl --physcpubind=0-31 --membind=0 -- python -m sdbench.latency --threads 32 ...
```

(Find your node's physical core ids with `lscpu` → "NUMA node0 CPU(s)"; use the lower
half — the upper ids are the hyperthread siblings.)

Pinning's most visible effect is **predictability**: it collapses p95 onto p50 by removing
scheduler migration and NUMA jitter.

## What we deliberately did NOT do

- **flash-linear-attention:** CUDA/XPU/MLU-gated in transformers; a no-op on CPU. See
  RESULTS.md.
- **bf16 weights:** slower than fp32 on CPU for this model (the engine already upcasts to
  fp32 for exactly this reason).
- **Quantization (int8 / dynamic):** a real lever for the memory-bandwidth ceiling that
  caps concurrency, and the model has community quantized variants — but it trades
  exactness, so it is out of scope for this "identical-decisions" stack. It is the first
  thing to try if you need higher aggregate throughput and can tolerate tiny numeric drift.
