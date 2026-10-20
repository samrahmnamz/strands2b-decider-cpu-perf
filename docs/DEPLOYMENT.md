# Deployment

How to run `strands-decider` as a CPU decision service, tuned for an `r8i.32xlarge`
(and adaptable to any multi-core CPU host).

## Sizing

- **Memory:** ~9.6 GiB RSS per engine replica (fp32 torso ~7 GiB + overhead). Plan
  ~10 GiB/replica.
- **CPU:** throughput is **memory-bandwidth bound**, not core-count bound, past ~8
  replicas on this host. The practical ceiling was ~16 decisions/s regardless of how the
  64 physical cores were divided.
- **Sweet spot on r8i.32xlarge:** **8 replicas × 8 threads** — best latency at ~84 % of
  peak throughput (13.5 dec/s short-traffic, p50 580 ms / p99 836 ms).

Match replica count to your latency target, not just throughput: fewer, fatter replicas
(8×8) give far better tail latency than many thin ones (32×2).

## Single replica (simplest)

```bash
. .venv/bin/activate
numactl --physcpubind=0-31 --membind=0 -- \
  python -m sdbench.serve --threads 32 --compile --host 0.0.0.0 --port 8000
```

- `--compile` pays a one-time ~150 s warmup (done before the server accepts traffic; the
  `/opt` endpoint reports `optimize_ready_s`). Drop it if you need fast startup.
- `--threads 32` + single-NUMA pin is the low-latency config. CPU inference is thread-safe,
  so one uvicorn process overlaps concurrent requests across its threads.

Health / status:
```bash
curl -s localhost:8000/health     # upstream engine info
curl -s localhost:8000/opt        # which optimizations are active
```

## Multiple replicas (production throughput)

Run N single-NUMA-pinned replicas on different ports, each bound to a disjoint
physical-core group, behind a load balancer (nginx, an ALB, or a simple round-robin).
Example: 8 replicas × 8 cores across the 64 physical cores.

```bash
#!/usr/bin/env bash
. .venv/bin/activate
for i in $(seq 0 7); do
  lo=$(( i * 8 )); hi=$(( lo + 7 ))
  node=$(( lo < 32 ? 0 : 1 ))     # cores 0-31 -> node 0, 32-63 -> node 1
  numactl --physcpubind=${lo}-${hi} --membind=${node} -- \
    python -m sdbench.serve --threads 8 --port $(( 8000 + i )) \
    > "/tmp/decider_${i}.log" 2>&1 &
done
wait
```

Then point a load balancer at `127.0.0.1:8000..8007`. A minimal nginx upstream:

```nginx
upstream decider {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    # ... through 8007
}
server {
    listen 80;
    location /v1/systemone { proxy_pass http://decider; }
}
```

Notes:
- Give each replica its own NUMA node's memory (`--membind`) to avoid cross-socket
  traffic — it is the single biggest latency/jitter factor.
- `--compile` per replica means 8 × ~150 s of warmup; stagger startup or skip compile if
  boot time matters more than the extra ~1.2–1.4x.

## The API

Identical to upstream `strands-decider serve` — `POST /v1/systemone`:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Payouts failing for 3 days!",
  "questions": {
    "route":   {"type":"choice","instructions":"Which team?","criteria":{"billing":null,"technical_support":null,"sales":null}},
    "urgent":  {"type":"noul","instructions":"Does this convey urgency?"},
    "mood":    {"type":"score","instructions":"How frustrated is the writer?","criteria":["calm","frustrated","depressed"]}
  }
}'
```

Response carries the typed answer, per-option probabilities, calibrated `confidence`, a
`usage` block, and `latency_ms`. Asking several questions about one `state` in a single
request is cheaper than separate calls — the engine shares the state's prefix across them
(`use_prefix_cache=True`).

## Operational tips

- **Warm before cutover.** The server warms all tier shapes (and compiles, if enabled)
  before binding, so the first real request is already fast.
- **Pin, always.** Even without multiple replicas, `numactl --membind` removes the NUMA
  jitter that otherwise shows up as a fat p95/p99 tail.
- **Watch RSS, not CPU%,** for capacity: you will hit the memory-bandwidth ceiling before
  you run out of cores.
- **Keep the HF cache warm** (`~/.cache/huggingface`) in your image or on an attached
  volume so replicas don't re-download the ~37 weight files on cold boot.
- **Scaling beyond one box:** because throughput is memory-bandwidth bound per host,
  horizontal scale-out (more hosts behind the LB) is more effective than bigger hosts.
  If you can tolerate tiny numeric drift, int8/dynamic quantization is the lever that
  raises the per-host ceiling (out of scope for this exact-decisions build).
