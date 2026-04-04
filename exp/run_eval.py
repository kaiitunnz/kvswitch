# ruff: noqa: E402
"""Unified evaluation experiment runner for KVSwitch.

Runs selected baselines (L4 RR, L7, KVSwitch) on a BMv2 Mininet topology
using ShareGPT-derived workloads with Poisson arrivals.

Must run as root inside Docker.  Use ``exp/run_eval.sh`` as the entry point.
"""

import argparse
import json
import logging
import shlex
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import Node

from kvswitch.eval.metrics import (
    RequestMetric,
    compute_summary,
    save_experiment_results,
)
from kvswitch.eval.workload import (
    WorkloadConfig,
    WorkloadGenerator,
    save_workload,
)
from kvswitch.network.bmv2 import BMv2Switch, reset_device_ids
from kvswitch.network.topology import (
    CONTROLLER_IP,
    ROUTER_IP,
    BMv2SpineLeafTopo,
    worker_ip,
)
from kvswitch.utils.logger import setup_logging
from kvswitch.vllm.profiling import load_results_csv

logger = logging.getLogger(__name__)

PYTHON = sys.executable
WORKER_PORT = 8000
ROUTER_PORT = 9000
CONTROLLER_PORT = 9100

HEALTHCHECK_MODULE = "kvswitch.network.cli.healthcheck"
WORKLOAD_CLIENT_MODULE = "kvswitch.network.cli.workload_client"


def _module_cmd(module: str, **kwargs: str | int | float) -> str:
    args = " ".join(
        f"--{key.replace('_', '-')} {shlex.quote(str(value))}"
        for key, value in kwargs.items()
    )
    return f"{shlex.quote(PYTHON)} -m {module} {args}"


# ---------------------------------------------------------------------------
# TTFT / TPOT lookup from profiling traces
# ---------------------------------------------------------------------------


def build_latency_tables(
    traces_path: Path,
) -> tuple[dict[tuple[int, float], float], dict[tuple[int, float], float]]:
    """Build {(prompt_tokens, prefix_ratio): mean_ms} tables for TTFT and TPOT."""
    results = load_results_csv(traces_path)
    ttft_groups: dict[tuple[int, float], list[float]] = defaultdict(list)
    tpot_groups: dict[tuple[int, float], list[float]] = defaultdict(list)
    for r in results:
        key = (r.prompt_tokens, r.prefix_ratio)
        ttft_groups[key].append(r.ttft * 1000)
        if r.tpot is not None:
            tpot_groups[key].append(r.tpot * 1000)
    ttft_table = {k: statistics.mean(v) for k, v in ttft_groups.items()}
    tpot_table = {k: statistics.mean(v) for k, v in tpot_groups.items()}
    return ttft_table, tpot_table


# ---------------------------------------------------------------------------
# Mininet helpers
# ---------------------------------------------------------------------------


def _populate_ipv4_routes(net: Mininet) -> None:
    """Populate ipv4_lpm tables on all BMv2 switches and static ARP on hosts.

    Each switch gets LPM rules that map host IP prefixes to the correct
    egress port.  Each host gets static ARP entries for all other hosts
    (since ARP broadcast is not handled by the P4 program).
    """
    # Collect host IP → MAC mappings and figure out which switch port
    # each host is connected to.
    host_info: dict[str, tuple[str, str]] = {}  # name -> (ip, mac)
    for h in net.hosts:
        ip = h.IP()
        mac = h.MAC()
        if ip and mac:
            host_info[h.name] = (ip, mac)

    # For each switch, find which port leads to which host (or to the other switch).
    for sw in net.switches:
        if not hasattr(sw, "table_add"):
            continue
        for port_num, intf in enumerate(sw.intfList()):
            if intf.name == "lo":
                continue
            link = intf.link
            if link is None:
                continue
            # The other end of this link.
            other_intf = link.intf1 if link.intf2 == intf else link.intf2
            other_node = other_intf.node

            if other_node in net.hosts:
                # Direct host connection — add /32 LPM rule.
                ip = other_node.IP()
                if ip:
                    sw.table_add(
                        "ipv4_lpm",
                        "forward",
                        [f"{ip}/32"],
                        [str(port_num)],
                    )
            else:
                # Connection to another switch — add default route via
                # this port for any IP not matched by a /32 rule.
                # We add /8 as a catch-all since all hosts are in 10.0.0.0/24.
                sw.table_add(
                    "ipv4_lpm",
                    "forward",
                    ["10.0.0.0/8"],
                    [str(port_num)],
                )

    # Static ARP on every host so they don't need broadcast.
    for h in net.hosts:
        for other_name, (other_ip, other_mac) in host_info.items():
            if other_name != h.name:
                h.cmd(f"arp -s {other_ip} {other_mac}")

    logger.info("Populated ipv4_lpm routes and static ARP on %d hosts", len(net.hosts))


def _start_bg(host, cmd: str, settle: float = 1.0) -> None:
    host.cmd(f"{cmd} &")
    time.sleep(settle)


def _disable_offloads(net: Mininet) -> None:
    """Disable NIC offloads that can break BMv2 packet handling in Mininet."""
    offloads = "rx off tx off sg off tso off ufo off gso off gro off lro off"
    for node in [*net.hosts, *net.switches]:
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            node.cmd(f"ethtool -K {intf.name} {offloads} >/dev/null 2>&1 || true")


def _wait_for_service(
    host,
    target_ip: str,
    target_port: int,
    timeout: float = 60.0,
    interval: float = 1.0,
) -> None:
    cmd = _module_cmd(HEALTHCHECK_MODULE, host=target_ip, port=target_port, timeout=2)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = host.cmd(cmd)
        if "ok" in out:
            logger.info("Service at %s:%d is ready", target_ip, target_port)
            return
        time.sleep(interval)
    raise TimeoutError(
        f"Service at {target_ip}:{target_port} not ready after {timeout}s"
    )


def _kill_bg(host) -> None:
    host.cmd("kill %% 2>/dev/null; wait 2>/dev/null")


def _collect_results(
    host: Node,
    workload_path: str,
    target_ip: str,
    target_port: int,
    timeout: float,
    kvswitch: bool = False,
) -> list[dict]:
    """Run the workload client on a Mininet host and parse JSON output."""
    cmd = _module_cmd(
        WORKLOAD_CLIENT_MODULE,
        workload=workload_path,
        host=target_ip,
        port=target_port,
        timeout=timeout,
    )
    if kvswitch:
        cmd += " --kvswitch"
    out = host.cmd(cmd)
    assert isinstance(out, str)
    try:
        lines = [ln.lstrip("> ").strip() for ln in out.strip().split("\n")]
        last = next(ln for ln in reversed(lines) if ln.startswith("["))
        return json.loads(last)
    except (json.JSONDecodeError, StopIteration):
        logger.error("Failed to parse workload client output:\n%s", out)
        raise


def _to_request_metrics(raw: list[dict], baseline: str) -> list[RequestMetric]:
    """Convert raw workload client output to RequestMetric list."""
    metrics = []
    for r in raw:
        routing = r.get("routing") or {}
        metrics.append(
            RequestMetric(
                request_id=r.get("request_id", 0),
                baseline=baseline,
                e2e_latency_ms=r.get("e2e_latency_ms", 0.0),
                simulated_ttft_ms=r.get("simulated_ttft_ms", 0.0) or 0.0,
                simulated_tpot_ms=r.get("simulated_tpot_ms", 0.0) or 0.0,
                simulated_e2e_ms=r.get("simulated_e2e_ms", 0.0) or 0.0,
                routing_overhead_ms=routing.get("routing_ms", 0.0) or 0.0,
                matched_blocks=r.get("matched_blocks", 0) or 0,
                worker_id=r.get("worker_id", ""),
                prompt_tokens=r.get("prompt_tokens", 0),
                output_tokens=r.get("output_tokens", 0) or 0,
                prefix_group=r.get("prefix_group", "none"),
                scheduled_time_s=r.get("scheduled_time_s", 0.0),
                actual_send_time_s=r.get("actual_send_time_s", 0.0),
            )
        )
    return metrics


def _get_node(net: Mininet, name: str) -> Node:
    return cast(Node, net.get(name))


# ---------------------------------------------------------------------------
# Per-baseline experiment functions
# ---------------------------------------------------------------------------


def _start_workers(
    net: Mininet,
    n_workers: int,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    controller_host: str | None = None,
    controller_port: int | None = None,
    kvswitch_port: int | None = None,
) -> None:
    """Start mock workers on all worker hosts."""
    for i in range(n_workers):
        host = net.get(f"worker{i}")
        cmd = (
            f"{PYTHON} -m kvswitch.mock.worker"
            f" --host 0.0.0.0 --port {WORKER_PORT}"
            f" --ttft-ms {ttft_ms} --tpot-ms {tpot_ms}"
            f" --worker-id worker{i}"
        )
        if controller_host and controller_port:
            cmd += f" --controller-host {controller_host} --controller-port {controller_port}"
        if kvswitch_port is not None:
            cmd += f" --kvswitch-port {kvswitch_port}"
        _start_bg(host, cmd)

    # Wait for all workers to be ready.
    client = net.get("client")
    for i in range(n_workers):
        _wait_for_service(client, worker_ip(i), WORKER_PORT)


def _controller_workers_arg(net: Mininet, n_workers: int) -> str:
    """Build the controller worker-placement argument from the live Mininet topology."""
    spine = _get_node(net, "s1")
    leaf = _get_node(net, "s2")

    spine_port_to_leaf = None
    for port_num, intf in enumerate(spine.intfList()):
        if intf.name == "lo" or intf.link is None:
            continue
        other_intf = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
        if other_intf.node == leaf:
            spine_port_to_leaf = port_num
            break
    if spine_port_to_leaf is None:
        raise RuntimeError("failed to determine spine port connected to leaf switch")

    worker_specs: list[str] = []
    for i in range(n_workers):
        worker = _get_node(net, f"worker{i}")
        leaf_port = None
        for port_num, intf in enumerate(leaf.intfList()):
            if intf.name == "lo" or intf.link is None:
                continue
            other_intf = intf.link.intf1 if intf.link.intf2 == intf else intf.link.intf2
            if other_intf.node == worker:
                leaf_port = port_num
                break
        if leaf_port is None:
            raise RuntimeError(f"failed to determine leaf port for worker{i}")

        worker_specs.append(
            f"worker{i},s2,{worker_ip(i)},{spine_port_to_leaf},{leaf_port}"
        )

    return ";".join(worker_specs)


def _stop_workers(net: Mininet, n_workers: int) -> None:
    for i in range(n_workers):
        _kill_bg(net.get(f"worker{i}"))


def run_baseline_l4_rr(
    net: Mininet,
    n_workers: int,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    client_timeout: float,
) -> list[RequestMetric]:
    """L4 Round-Robin: TCAM empty → ECMP distributes traffic."""
    logger.info("--- Baseline: L4 Round-Robin ---")
    _start_workers(net, n_workers, ttft_ms, tpot_ms, max_output_tokens)

    # Pre-populate ECMP tables with uniform weights for all workers.
    # For now, workers are reachable directly; ECMP hashing distributes.
    # The client sends to a well-known worker IP (we round-robin in client).
    # TODO: With BMv2, program ECMP tables via simple_switch_CLI.

    # For the initial implementation, the client round-robins across worker IPs.
    # The workload_client sends to a single target, so we use worker0.
    # In a full implementation, the BMv2 ECMP table would handle distribution.
    client = _get_node(net, "client")
    raw = _collect_results(
        client, workload_path, worker_ip(0), WORKER_PORT, client_timeout
    )

    _stop_workers(net, n_workers)
    return _to_request_metrics(raw, "l4_rr")


def run_baseline_l7(
    net: Mininet,
    n_workers: int,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    model: str,
    client_timeout: float,
) -> list[RequestMetric]:
    """L7 Prefix-Aware: Client → L7 proxy → best worker."""
    logger.info("--- Baseline: L7 Prefix-Aware Router ---")
    _start_workers(net, n_workers, ttft_ms, tpot_ms, max_output_tokens)

    # Start L7 proxy on the router host.
    router_host = _get_node(net, "router")
    workers_arg = ",".join(f"{worker_ip(i)}:{WORKER_PORT}" for i in range(n_workers))
    proxy_log = "/tmp/l7_proxy.log"
    _start_bg(
        router_host,
        "HF_HOME=/root/.cache/huggingface "
        "TRANSFORMERS_OFFLINE=1 "
        "HF_HUB_OFFLINE=1 "
        f"{PYTHON} -m kvswitch.router.l7_proxy"
        f" --model {model} --port {ROUTER_PORT}"
        f" --workers {workers_arg}"
        f" > {proxy_log} 2>&1",
    )

    client = _get_node(net, "client")
    try:
        _wait_for_service(client, ROUTER_IP, ROUTER_PORT, timeout=120.0)
    except TimeoutError:
        log_out = router_host.cmd(f"cat {proxy_log}")
        logger.error("L7 proxy log:\n%s", log_out)
        raise

    raw = _collect_results(
        client, workload_path, ROUTER_IP, ROUTER_PORT, client_timeout
    )

    _kill_bg(router_host)
    _stop_workers(net, n_workers)
    return _to_request_metrics(raw, "l7")


def run_baseline_kvswitch(
    net: Mininet,
    n_workers: int,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    client_timeout: float,
) -> list[RequestMetric]:
    """KVSwitch: SDN controller populates TCAM; switches route shim-header traffic."""
    logger.info("--- Baseline: KVSwitch ---")

    # Start SDN controller on the controller host.
    ctrl_host = _get_node(net, "controller")
    ctrl_log = "/tmp/sdn_controller.log"
    workers_arg = _controller_workers_arg(net, n_workers)
    _start_bg(
        ctrl_host,
        f"{PYTHON} -m kvswitch.controller.sdn_controller"
        f" --host 0.0.0.0 --port {CONTROLLER_PORT}"
        f" --workers {shlex.quote(workers_arg)}"
        f" > {ctrl_log} 2>&1",
    )

    client = _get_node(net, "client")
    try:
        _wait_for_service(client, CONTROLLER_IP, CONTROLLER_PORT, timeout=30.0)
    except TimeoutError:
        log_out = ctrl_host.cmd(f"cat {ctrl_log}")
        logger.error("SDN controller log:\n%s", log_out)
        raise

    # Start workers with controller connection and KVSwitch listener on
    # port 4789 so they can receive shim-header traffic from the switches.
    from kvswitch.sdk.client import KVSWITCH_UDP_PORT

    _start_workers(
        net,
        n_workers,
        ttft_ms,
        tpot_ms,
        max_output_tokens,
        controller_host=CONTROLLER_IP,
        controller_port=CONTROLLER_PORT,
        kvswitch_port=KVSWITCH_UDP_PORT,
    )

    # Client sends with KVSwitch shim header on port 4789.  The BMv2
    # switches parse the header, match h0/h1/h2 against TCAM rules
    # (populated by the controller from worker cache events), and
    # forward to the matched worker's egress port.  On TCAM miss,
    # the ECMP fallback distributes traffic.
    #
    # The destination IP is worker0 but the switch overrides the
    # egress port based on the shim header match, so the packet may
    # land on any worker.
    raw = _collect_results(
        client,
        workload_path,
        worker_ip(0),
        KVSWITCH_UDP_PORT,
        client_timeout,
        kvswitch=True,
    )

    _kill_bg(ctrl_host)
    _stop_workers(net, n_workers)
    return _to_request_metrics(raw, "kvswitch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


BASELINE_REGISTRY = {
    "l4_rr": {"needs_router": False, "needs_controller": False},
    "l7": {"needs_router": True, "needs_controller": False},
    "kvswitch": {"needs_router": False, "needs_controller": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KVSwitch evaluation experiment runner"
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="l4_rr,l7,kvswitch",
        help="Comma-separated baselines: l4_rr, l7, kvswitch",
    )
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--prefix-sharing-ratio", type=float, default=0.5)
    parser.add_argument("--num-prefix-groups", type=int, default=3)
    parser.add_argument("--system-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument(
        "--traces",
        type=str,
        default=str(Path("results/profiling/inference_traces.csv")),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path("data/ShareGPT_V3_unfiltered_cleaned_split.json")),
    )
    parser.add_argument(
        "--p4-json",
        type=str,
        default="build/p4/kvswitch.json/kvswitch.json",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--delay", type=str, default=None)
    parser.add_argument(
        "--ttft-ms",
        type=float,
        default=None,
        help="Override TTFT (else from traces, prefix_ratio=0.0, 1024 tokens)",
    )
    parser.add_argument(
        "--tpot-ms", type=float, default=None, help="Override TPOT (else from traces)"
    )
    parser.add_argument("--client-timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=str, default="results/eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Using Python: %s", PYTHON)

    baselines = [b.strip() for b in args.baselines.split(",")]
    for b in baselines:
        if b not in BASELINE_REGISTRY:
            raise ValueError(
                f"Unknown baseline: {b}. Choose from: {list(BASELINE_REGISTRY)}"
            )

    # --- Load TTFT/TPOT from traces ---
    traces_path = Path(args.traces)
    if traces_path.exists():
        ttft_table, tpot_table = build_latency_tables(traces_path)
        logger.info("Loaded %d trace configs from %s", len(ttft_table), traces_path)
    else:
        ttft_table, tpot_table = {}, {}
        logger.warning("Traces not found: %s", traces_path)

    # Default TTFT/TPOT: use 1024-token, 0% prefix ratio as representative.
    ttft_ms = args.ttft_ms or ttft_table.get((1024, 0.0), 15.0)
    tpot_ms = args.tpot_ms or tpot_table.get((1024, 0.0), 3.0)
    logger.info("Using TTFT=%.2fms, TPOT=%.2fms", ttft_ms, tpot_ms)

    # --- Generate workload ---
    logger.info("Generating workload...")
    wl_config = WorkloadConfig(
        dataset_path=Path(args.dataset),
        num_requests=args.num_requests,
        request_rate=args.request_rate,
        prefix_sharing_ratio=args.prefix_sharing_ratio,
        num_prefix_groups=args.num_prefix_groups,
        system_prompt_tokens=args.system_prompt_tokens,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        model=args.model,
    )
    generator = WorkloadGenerator(wl_config)
    requests = generator.generate()

    # Write workload to a temp file accessible inside Mininet.
    workload_path = "/tmp/eval_workload.json"
    save_workload(requests, Path(workload_path))

    # --- Determine topology needs ---
    needs_router = any(BASELINE_REGISTRY[b]["needs_router"] for b in baselines)
    needs_controller = any(BASELINE_REGISTRY[b]["needs_controller"] for b in baselines)

    # --- Build and start topology ---
    logger.info(
        "Building BMv2 topology: %d workers, router=%s, controller=%s",
        args.n_workers,
        needs_router,
        needs_controller,
    )
    reset_device_ids()
    topo = BMv2SpineLeafTopo(
        n_workers=args.n_workers,
        with_router=needs_router,
        with_controller=needs_controller,
        delay=args.delay,
    )
    net = Mininet(
        topo=topo,
        switch=lambda name, **kw: BMv2Switch(name, json_path=args.p4_json, **kw),  # type: ignore
        link=TCLink,
        controller=None,
    )
    net.start()
    _disable_offloads(net)
    _populate_ipv4_routes(net)

    # Dump table entries for debugging connectivity issues.
    for sw in net.switches:
        if hasattr(sw, "cli_cmd"):
            dump = sw.cmd(f'echo "table_dump ipv4_lpm" | {sw.cli_cmd}')
            logger.info("Table dump for %s:\n%s", sw.name, dump)

    net.pingAll()

    # --- Run each baseline ---
    all_results: dict[str, list[RequestMetric]] = {}
    for baseline in baselines:
        try:
            if baseline == "l4_rr":
                metrics = run_baseline_l4_rr(
                    net,
                    args.n_workers,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.client_timeout,
                )
            elif baseline == "l7":
                metrics = run_baseline_l7(
                    net,
                    args.n_workers,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.model,
                    args.client_timeout,
                )
            elif baseline == "kvswitch":
                metrics = run_baseline_kvswitch(
                    net,
                    args.n_workers,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.client_timeout,
                )
            else:
                continue

            all_results[baseline] = metrics
            summary = compute_summary(metrics)
            logger.info(
                "%s: e2e_p50=%.1fms e2e_p95=%.1fms cache_hit=%.2f",
                baseline,
                summary.get("e2e_p50_ms", 0),
                summary.get("e2e_p95_ms", 0),
                summary.get("cache_hit_rate_mean", 0),
            )
        except Exception:
            logger.exception("Baseline %s failed", baseline)

    net.stop()

    # --- Save results ---
    output_dir = Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    config = {
        "baselines": baselines,
        "n_workers": args.n_workers,
        "num_requests": args.num_requests,
        "request_rate": args.request_rate,
        "prefix_sharing_ratio": args.prefix_sharing_ratio,
        "num_prefix_groups": args.num_prefix_groups,
        "system_prompt_tokens": args.system_prompt_tokens,
        "max_output_tokens": args.max_output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "delay": args.delay,
        "model": args.model,
        "p4_json": args.p4_json,
        "seed": args.seed,
    }
    save_experiment_results("eval", config, all_results, output_path)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
