# strands-decider CPU benchmark & deploy — convenience targets
# Most targets expect the venv at .venv (run `make setup` first).

PY := .venv/bin/python
PIN1 := numactl --physcpubind=0-31 --membind=0 --
export PYTHONPATH := src

TOKEN_TARGET := 950

.PHONY: help setup bench-baseline bench-opt bench-compile bench-jev bench-all concurrency serve serve-compile clean

help:
	@echo "targets:"
	@echo "  setup           create venv, install CPU torch + deps + numactl"
	@echo "  bench-baseline  untuned latency (64 threads, no optimizations)"
	@echo "  bench-opt       optimized latency (merge + precision, pinned)"
	@echo "  bench-compile   optimized + torch.compile (pinned)"
	@echo "  bench-jev       ~950-token (JevBench size) baseline + optimized + compile"
	@echo "  bench-all       run the full suite via scripts/run_all.sh"
	@echo "  concurrency     8x8 short-traffic throughput (the sweet spot)"
	@echo "  serve           start the optimized server on :8000 (pinned)"
	@echo "  serve-compile   same, with torch.compile"
	@echo "  clean           remove venv and results"

setup:
	bash scripts/setup.sh

bench-baseline:
	$(PY) -m sdbench.latency --threads 64 --no-merge --no-precision --json-out results/latency_baseline.json

bench-opt:
	$(PIN1) $(PY) -m sdbench.latency --threads 32 --json-out results/latency_optimized.json

bench-compile:
	$(PIN1) $(PY) -m sdbench.latency --threads 32 --compile --json-out results/latency_optimized_compile.json

# ~950-token (JevBench-size) decision: baseline + optimized + optimized/compile,
# so the JevBench-size row has a real CPU baseline->optimized speedup.
bench-jev:
	$(PY) -m sdbench.latency --threads 64 --no-merge --no-precision --token-target $(TOKEN_TARGET) --json-out results/latency_jev_baseline.json
	$(PIN1) $(PY) -m sdbench.latency --threads 32 --token-target $(TOKEN_TARGET) --json-out results/latency_jev_optimized.json
	$(PIN1) $(PY) -m sdbench.latency --threads 32 --compile --token-target $(TOKEN_TARGET) --json-out results/latency_jev_optimized_compile.json

bench-all:
	bash scripts/run_all.sh

concurrency:
	$(PY) -m sdbench.concurrency --workers 8 --threads-per 8 --mix short --json-out results/conc_8x8_short.json

serve:
	$(PIN1) $(PY) -m sdbench.serve --threads 32 --host 0.0.0.0 --port 8000

serve-compile:
	$(PIN1) $(PY) -m sdbench.serve --threads 32 --compile --host 0.0.0.0 --port 8000

clean:
	rm -rf .venv results/*.json
