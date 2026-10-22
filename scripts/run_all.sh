#!/usr/bin/env bash
# Reproduce the full benchmark suite. Run after scripts/setup.sh.
#   - baseline latency (untuned, 64 threads)
#   - optimized latency (merge + precision, pinned)
#   - optimized + compile latency (pinned)
#   - ~950-token (JevBench-size) latency: baseline + optimized + compile
#   - concurrency sweep (sweet spot + ceiling)
# Results land in results/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
# shellcheck disable=SC1091
. .venv/bin/activate
export PYTHONPATH=src
mkdir -p results

# Physical cores of one NUMA node on r8i.32xlarge. Adjust for other hosts
# (lscpu -> "NUMA node0 CPU(s)"; use the first-half physical ids).
PIN1="numactl --physcpubind=0-31 --membind=0 --"
PINALL="numactl --physcpubind=0-63 --interleave=0,1 --"

echo "== 1/6 baseline latency (64 threads, no optimizations) =="
python -m sdbench.latency --threads 64 --no-merge --no-precision \
  --iters 30 --json-out results/latency_baseline.json

echo "== 2/6 optimized latency (merge + precision, pinned 32t/1 node) =="
$PIN1 python -m sdbench.latency --threads 32 \
  --iters 30 --json-out results/latency_optimized.json

echo "== 3/6 optimized + compile (pinned 32t/1 node) =="
$PIN1 python -m sdbench.latency --threads 32 --compile \
  --iters 30 --json-out results/latency_optimized_compile.json

echo "== 4/6 ~950-token (JevBench-size) latency: baseline + optimized + compile =="
python -m sdbench.latency --threads 64 --no-merge --no-precision --token-target 950 \
  --iters 30 --json-out results/latency_jev_baseline.json
$PIN1 python -m sdbench.latency --threads 32 --token-target 950 \
  --iters 30 --json-out results/latency_jev_optimized.json
$PIN1 python -m sdbench.latency --threads 32 --compile --token-target 950 \
  --iters 30 --json-out results/latency_jev_optimized_compile.json

echo "== 5/6 concurrency sweet spot (8 workers x 8 threads, short mix) =="
python -m sdbench.concurrency --workers 8 --threads-per 8 --mix short \
  --duration 20 --json-out results/conc_8x8_short.json

echo "== 6/6 concurrency ceiling probes =="
python -m sdbench.concurrency --workers 16 --threads-per 4 --mix short \
  --duration 20 --json-out results/conc_16x4_short.json
python -m sdbench.concurrency --workers 32 --threads-per 2 --mix short \
  --duration 20 --json-out results/conc_32x2_short.json

echo "== done. See results/ and docs/RESULTS.md =="
