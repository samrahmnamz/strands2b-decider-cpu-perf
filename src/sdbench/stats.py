"""Small latency-statistics helpers shared by the benchmark tools."""
from __future__ import annotations

import statistics


def pct(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 1]."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (k - lo) * (xs[hi] - xs[lo])


def summarize(lat_ms: list[float]) -> dict:
    return {
        "n": len(lat_ms),
        "mean_ms": round(statistics.fmean(lat_ms), 1),
        "p50_ms": round(pct(lat_ms, 0.50), 1),
        "p95_ms": round(pct(lat_ms, 0.95), 1),
        "p99_ms": round(pct(lat_ms, 0.99), 1),
        "min_ms": round(min(lat_ms), 1),
        "max_ms": round(max(lat_ms), 1),
        "throughput_dec_per_s": round(1000.0 / statistics.fmean(lat_ms), 2),
    }
