"""Per-request metrics collection and aggregation for KVSwitch evaluation."""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Metrics collected for a single request during an experiment."""

    request_id: int
    baseline: str  # "l4_rr", "l7", "kvswitch"
    e2e_latency_ms: float  # Client-measured RTT
    ttft_ms: float  # Client-side TTFT estimate
    simulated_ttft_ms: float  # From mock worker response
    simulated_tpot_ms: float  # From mock worker response
    simulated_e2e_ms: float  # From mock worker response
    routing_overhead_ms: float  # L7: routing_ms; KVSwitch/L4: 0
    matched_blocks: int
    worker_id: str
    prompt_tokens: int
    output_tokens: int
    prefix_group: str
    scheduled_time_s: float
    actual_send_time_s: float


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_v):
        return sorted_v[f]
    return sorted_v[f] + (k - f) * (sorted_v[c] - sorted_v[f])


def compute_summary(metrics: list[RequestMetric]) -> dict:
    """Compute aggregate statistics from a list of request metrics.

    Returns a dict with P50/P95/P99 for client E2E, estimated client TTFT,
    and simulated TPOT; mean cache hit rate; mean routing overhead; and throughput.
    """
    if not metrics:
        return {}

    e2e = [m.e2e_latency_ms for m in metrics]
    ttft = [m.ttft_ms for m in metrics]
    tpot = [m.simulated_tpot_ms for m in metrics]
    routing = [m.routing_overhead_ms for m in metrics]
    hit_rates = [
        m.matched_blocks / max(m.prompt_tokens // 16, 1)  # blocks = tokens / block_size
        for m in metrics
    ]

    # Throughput: requests / wall-clock span.
    send_times = [m.actual_send_time_s for m in metrics]
    span_s = max(send_times) - min(send_times) if len(send_times) > 1 else 1.0
    throughput = len(metrics) / span_s if span_s > 0 else 0.0

    return {
        "n_requests": len(metrics),
        "e2e_mean_ms": statistics.mean(e2e),
        "e2e_p50_ms": _percentile(e2e, 50),
        "e2e_p95_ms": _percentile(e2e, 95),
        "e2e_p99_ms": _percentile(e2e, 99),
        "ttft_mean_ms": statistics.mean(ttft),
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "tpot_mean_ms": statistics.mean(tpot),
        "tpot_p50_ms": _percentile(tpot, 50),
        "tpot_p95_ms": _percentile(tpot, 95),
        "tpot_p99_ms": _percentile(tpot, 99),
        "routing_overhead_mean_ms": statistics.mean(routing),
        "cache_hit_rate_mean": statistics.mean(hit_rates),
        "throughput_rps": throughput,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_experiment_results(
    experiment: str,
    config: dict,
    results: dict[str, list[RequestMetric]],
    path: Path,
) -> None:
    """Save per-baseline metrics + summaries to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    output: dict = {
        "experiment": experiment,
        "config": config,
        "results": {},
    }
    for baseline, metrics in results.items():
        output["results"][baseline] = {
            "per_request": [asdict(m) for m in metrics],
            "summary": compute_summary(metrics),
        }

    path.write_text(json.dumps(output, indent=2))
    logger.info("Saved experiment results to %s", path)


def load_experiment_results(path: Path) -> dict:
    """Load experiment results from a JSON file."""
    return json.loads(path.read_text())
