# ruff: noqa: E402
"""Mininet experiment: Direct vs. L7 Router end-to-end latency.

Sweeps prompt lengths and prefix ratios from the profiling traces,
using the measured TTFT as the mock worker's simulated delay.

Must run as root.  Results are saved to a JSON file that the analysis
notebook can load without needing sudo.

Usage (via shell wrapper):
    bash exp/run_mininet.sh
"""

import json
import logging
import shlex
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import Node, OVSBridge

from kvswitch.network.topology import SpineLeafTopo
from kvswitch.utils.logger import setup_logging
from kvswitch.vllm.profiling import ProfilingResult, load_results_csv

logger = logging.getLogger(__name__)

WORKER_PORT = 8000
ROUTER_PORT = 9000
DEFAULT_TRACES = Path("results/profiling/ttft_traces.csv")
DEFAULT_OUTPUT = Path("results/mininet/comparison.json")

PYTHON = sys.executable
HEALTHCHECK_MODULE = "kvswitch.network.cli.healthcheck"
MEASURE_CLIENT_MODULE = "kvswitch.network.cli.measure_client"


def _module_cmd(module: str, **kwargs: str | int | float) -> str:
    args = " ".join(
        f"--{key.replace('_', '-')} {shlex.quote(str(value))}"
        for key, value in kwargs.items()
    )
    return f"{shlex.quote(PYTHON)} -m {module} {args}"


# ---------------------------------------------------------------------------
# TTFT lookup from profiling traces
# ---------------------------------------------------------------------------


def build_ttft_table(
    results: list[ProfilingResult],
) -> dict[tuple[int, float], float]:
    """Return {(prompt_tokens, prefix_ratio): mean_ttft_ms} from traces."""
    groups: dict[tuple[int, float], list[float]] = defaultdict(list)
    for r in results:
        groups[(r.prompt_tokens, r.prefix_ratio)].append(r.ttft * 1000)
    return {k: statistics.mean(v) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# Mininet helpers
# ---------------------------------------------------------------------------


def _start_bg(host: Node, cmd: str, settle: float = 1.0) -> None:
    """Start a command in the background on a Mininet host."""
    host.cmd(f"{cmd} &")
    time.sleep(settle)


def _wait_for_service(
    host: Node,
    target_ip: str,
    target_port: int,
    timeout: float = 60.0,
    interval: float = 1.0,
) -> None:
    """Poll a UDP health endpoint from *host* until it responds."""
    healthcheck_cmd = _module_cmd(
        HEALTHCHECK_MODULE,
        host=target_ip,
        port=target_port,
        timeout=2.0,
    )
    deadline = time.monotonic() + timeout
    out = "EMPTY"
    while time.monotonic() < deadline:
        out = host.cmd(healthcheck_cmd)
        assert isinstance(out, str)
        if "ok" in out:
            logger.info("Service at %s:%d is ready", target_ip, target_port)
            return
        time.sleep(interval)
    raise TimeoutError(
        f"Service at {target_ip}:{target_port} not ready after {timeout}s. "
        f"Last output: {out!r}"
    )


def _kill_bg(host: Node) -> None:
    """Kill all background Python processes on a Mininet host."""
    host.cmd("kill %% 2>/dev/null; wait 2>/dev/null")


def _measure_latencies(
    host: Node,
    target_ip: str,
    target_port: int,
    n: int,
    prompt_tokens: int = 256,
) -> list[float]:
    """Send *n* UDP requests from *host*, return per-request RTTs (ms)."""
    measure_cmd = _module_cmd(
        MEASURE_CLIENT_MODULE,
        host=target_ip,
        port=target_port,
        n=n,
        prompt_tokens=prompt_tokens,
        timeout=10.0,
    )
    out = host.cmd(measure_cmd)
    assert isinstance(out, str)
    try:
        lines = [ln.lstrip("> ").strip() for ln in out.strip().split("\n")]
        last = next(ln for ln in reversed(lines) if ln.startswith("["))
        return json.loads(last)
    except (json.JSONDecodeError, StopIteration):
        logger.error("Failed to parse client output:\n%s", out)
        raise


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_direct(
    net: Mininet,
    n_requests: int,
    ttft_ms: float,
    prompt_tokens: int,
) -> list[float]:
    """Run one direct (no router) measurement. Network must already be started."""
    worker = cast(Node, net.get("worker0"))
    client = cast(Node, net.get("client"))

    _start_bg(
        worker,
        f"{PYTHON} -m kvswitch.mock.worker --port {WORKER_PORT} --ttft-ms {ttft_ms}",
    )
    _wait_for_service(client, "10.0.0.1", WORKER_PORT)

    latencies = _measure_latencies(
        client, "10.0.0.1", WORKER_PORT, n_requests, prompt_tokens
    )

    _kill_bg(worker)
    return latencies


def run_l7_router(
    net: Mininet,
    n_requests: int,
    ttft_ms: float,
    prompt_tokens: int,
    model: str,
) -> list[float]:
    """Run one L7 router measurement. Network must already be started."""
    worker = cast(Node, net.get("worker0"))
    router_host = cast(Node, net.get("router"))
    client = cast(Node, net.get("client"))

    _start_bg(
        worker,
        f"{PYTHON} -m kvswitch.mock.worker --port {WORKER_PORT} --ttft-ms {ttft_ms}",
    )
    _wait_for_service(client, "10.0.0.1", WORKER_PORT)

    proxy_log = "/tmp/l7_proxy.log"
    _start_bg(
        router_host,
        "HF_HOME=/root/.cache/huggingface "
        "TRANSFORMERS_OFFLINE=1 "
        "HF_HUB_OFFLINE=1 "
        f"{PYTHON} -m kvswitch.router.l7_proxy"
        f" --model {model} --port {ROUTER_PORT}"
        f" --workers 10.0.0.1:{WORKER_PORT}"
        f" > {proxy_log} 2>&1",
    )
    try:
        _wait_for_service(client, "10.0.0.200", ROUTER_PORT, timeout=120.0)
    except TimeoutError:
        log_out = router_host.cmd(f"cat {proxy_log}")
        logger.error("L7 proxy log:\n%s", log_out)
        raise

    latencies = _measure_latencies(
        client, "10.0.0.200", ROUTER_PORT, n_requests, prompt_tokens
    )

    _kill_bg(worker)
    _kill_bg(router_host)
    return latencies


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Mininet experiment: Direct vs. L7 Router (sweep)"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default=str(DEFAULT_TRACES),
        help="Path to TTFT profiling CSV",
    )
    parser.add_argument("--n-requests", type=int, default=10)
    parser.add_argument(
        "--prefix-ratios",
        type=str,
        default="1.0",
        help="Comma-separated prefix sharing ratios to sweep",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--delay",
        type=str,
        default=None,
        help="Per-link latency, e.g. '0.1ms', '1ms'",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Using Python: %s", PYTHON)

    # Load profiling traces and build TTFT lookup table.
    traces_path = Path(args.traces)
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces not found: {traces_path}")
    profiling_results = load_results_csv(traces_path)
    ttft_table = build_ttft_table(profiling_results)
    logger.info("Loaded %d trace configs from %s", len(ttft_table), traces_path)

    prefix_ratios = [float(x) for x in args.prefix_ratios.split(",")]
    prompt_lengths = sorted({k[0] for k in ttft_table.keys()})

    logger.info("Prompt lengths: %s", prompt_lengths)
    logger.info("Prefix ratios: %s", prefix_ratios)
    logger.info("Link delay: %s", args.delay or "none")

    # --- Direct topology (start once, sweep inside) ---
    logger.info("=== Direct topology (no router) ===")
    topo_direct = SpineLeafTopo(n_workers=1, with_router=False, delay=args.delay)
    net_direct = Mininet(
        topo=topo_direct, switch=OVSBridge, link=TCLink, controller=None
    )
    net_direct.start()
    net_direct.pingAll()

    direct_results: dict[str, list[float]] = {}
    for pl in prompt_lengths:
        for ratio in prefix_ratios:
            ttft_ms = ttft_table.get((pl, ratio))
            if ttft_ms is None:
                logger.warning("No trace for prompt=%d ratio=%.2f, skipping", pl, ratio)
                continue
            key = f"{pl}_{ratio}"
            logger.info(
                "  Direct: prompt=%d ratio=%.2f ttft=%.2fms", pl, ratio, ttft_ms
            )
            direct_results[key] = run_direct(net_direct, args.n_requests, ttft_ms, pl)

    net_direct.stop()

    # --- L7 Router topology (start once, sweep inside) ---
    logger.info("=== L7 Router topology ===")
    topo_l7 = SpineLeafTopo(n_workers=1, with_router=True, delay=args.delay)
    net_l7 = Mininet(topo=topo_l7, switch=OVSBridge, link=TCLink, controller=None)
    net_l7.start()
    net_l7.pingAll()

    l7_results: dict[str, list[float]] = {}
    for pl in prompt_lengths:
        for ratio in prefix_ratios:
            ttft_ms = ttft_table.get((pl, ratio))
            if ttft_ms is None:
                continue
            key = f"{pl}_{ratio}"
            logger.info("  L7: prompt=%d ratio=%.2f ttft=%.2fms", pl, ratio, ttft_ms)
            l7_results[key] = run_l7_router(
                net_l7, args.n_requests, ttft_ms, pl, args.model
            )

    net_l7.stop()

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "params": {
            "n_requests": args.n_requests,
            "prefix_ratios": prefix_ratios,
            "prompt_lengths": prompt_lengths,
            "model": args.model,
            "traces": str(traces_path),
            "delay": args.delay,
        },
        "direct": direct_results,
        "l7_router": l7_results,
    }
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # --- Summary ---
    for key in sorted(direct_results.keys()):
        d_mean = statistics.mean(direct_results[key])
        l7_mean = statistics.mean(l7_results[key])
        overhead = l7_mean - d_mean
        pl, ratio = key.split("_")
        logger.info(
            "  prompt=%s ratio=%s: direct=%.2fms l7=%.2fms overhead=%.2fms",
            pl,
            ratio,
            d_mean,
            l7_mean,
            overhead,
        )


if __name__ == "__main__":
    main()
