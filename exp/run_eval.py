# ruff: noqa: E402
"""Unified evaluation experiment runner for KVSwitch.

Runs selected baselines (L4 RR, L7, KVSwitch) on a BMv2 Mininet topology
using ShareGPT-derived workloads with Poisson arrivals.

Must run as root inside Docker.  Use ``exp/run_eval.sh`` as the entry point.
"""

import argparse
import asyncio
import json
import logging
import shlex
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Coroutine, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mininet.link import Intf, Link, TCLink
from mininet.net import Mininet
from mininet.node import Node, Switch

from kvswitch.controller.finsy_adapter import FinsyAdapter
from kvswitch.controller.sdn_controller import (
    SDNController,
    parse_worker_placements,
)
from kvswitch.controller.switch_adapter import TableAddOp
from kvswitch.eval.metrics import (
    RequestMetric,
    compute_summary,
    save_experiment_results,
)
from kvswitch.eval.workload import (
    WorkloadConfig,
    WorkloadGenerator,
    WorkloadRequest,
    save_workload,
)
from kvswitch.network.bmv2 import BMv2Switch, reset_device_ids
from kvswitch.network.topology import (
    CONTROLLER_IP,
    KVSWITCH_SERVICE_IP,
    KVSWITCH_SERVICE_MAC,
    ROUTER_IP,
    BMv2SpineLeafTopo,
    worker_ip,
)
from kvswitch.sdk.client import KVSWITCH_UDP_PORT
from kvswitch.sdk.header import HEADER_SIZE
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


def _resolve_p4_artifacts(p4_path: str) -> tuple[str, str, str]:
    """Resolve BMv2 JSON and P4Runtime artifacts from a file or directory path."""
    candidate = Path(p4_path)
    if candidate.is_dir():
        json_path = candidate / "kvswitch.json"
        p4info_path = candidate / "kvswitch.p4info.txtpb"
    else:
        json_path = candidate
        p4info_path = candidate.with_suffix(".p4info.txtpb")

    if not json_path.exists() or not p4info_path.exists():
        raise FileNotFoundError(
            "Missing compiled P4 artifacts. Expected "
            f"JSON={json_path} and P4Info={p4info_path}. "
            "Run `bash scripts/compile_p4.sh` first."
        )

    return str(json_path), str(p4info_path), str(json_path)


class AsyncLoopThread:
    """Run an asyncio event loop in a background thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> None:
        self._thread.start()

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5.0)


def _switch_runtime_config(net: Mininet) -> tuple[dict[str, str], dict[str, int]]:
    """Build gRPC addresses and device IDs for the live BMv2 switches."""
    switch_grpc: dict[str, str] = {}
    device_ids: dict[str, int] = {}
    for sw in net.switches:
        if hasattr(sw, "grpc_addr") and hasattr(sw, "device_id"):
            switch_grpc[sw.name] = sw.grpc_addr
            device_ids[sw.name] = sw.device_id
    return switch_grpc, device_ids


def _populate_ipv4_routes(net: Mininet, adapter: FinsyAdapter) -> None:
    """Populate ipv4_lpm tables on all BMv2 switches and static ARP on hosts.

    Each switch gets LPM rules that map host IP prefixes to the correct
    egress port. Each host gets static ARP entries for all other hosts
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
    sw: Switch
    intf: Intf
    for sw in net.switches:
        ops: list[TableAddOp] = []
        for port_num, intf in enumerate(sw.intfList()):
            if intf.name == "lo":
                continue
            link = cast(Link | None, intf.link)
            if link is None:
                continue
            # The other end of this link.
            other_intf = link.intf1 if link.intf2 == intf else link.intf2
            assert other_intf is not None
            other_node = cast(Node, other_intf.node)

            if other_node in net.hosts:
                # Direct host connection — add /32 LPM rule.
                ip = other_node.IP()
                if ip:
                    ops.append(
                        TableAddOp(
                            switch=sw.name,  # type: ignore[arg-type]
                            table="ipv4_lpm",
                            action="forward",
                            match={"hdr.ipv4.dstAddr": f"{ip}/32"},
                            action_params={"port": port_num},
                        )
                    )
            else:
                # Connection to another switch — add default route via
                # this port for any IP not matched by a /32 rule.
                # We add /8 as a catch-all since all hosts are in 10.0.0.0/24.
                ops.append(
                    TableAddOp(
                        switch=sw.name,  # type: ignore[arg-type]
                        table="ipv4_lpm",
                        action="forward",
                        match={"hdr.ipv4.dstAddr": "10.0.0.0/8"},
                        action_params={"port": port_num},
                    )
                )

        if ops:
            adapter.apply_ops(ops)

    for h in net.hosts:
        for other_name, (other_ip, other_mac) in host_info.items():
            if other_name != h.name:
                h.cmd(f"arp -s {other_ip} {other_mac}")

    logger.info("Populated ipv4_lpm routes and static ARP on %d hosts", len(net.hosts))


def _configure_kvswitch_service(net: Mininet, n_workers: int) -> None:
    """Expose a shared KVSwitch service IP on all workers and client ARP."""
    client = _get_node(net, "client")
    client.cmd(f"arp -s {KVSWITCH_SERVICE_IP} {KVSWITCH_SERVICE_MAC} || true")

    for i in range(n_workers):
        worker = _get_node(net, f"worker{i}")
        worker_intf = worker.defaultIntf()
        if worker_intf is None:
            raise RuntimeError(f"worker{i} has no default interface")
        worker.cmd(
            f"ip addr add {KVSWITCH_SERVICE_IP}/32 dev {worker_intf.name} 2>/dev/null || true"
        )

    logger.info(
        "Configured KVSwitch virtual service %s (%s) on %d workers",
        KVSWITCH_SERVICE_IP,
        KVSWITCH_SERVICE_MAC,
        n_workers,
    )


def _log_kvswitch_packet_sizes(requests: Iterable[WorkloadRequest]) -> None:
    """Log KVSwitch packet sizes and warn if they exceed standard MTU."""
    if not requests:
        return

    packet_sizes: list[int] = []
    oversized: list[tuple[int, int, int]] = []
    for req in requests:
        payload = {
            "endpoint": "generate",
            "prompt_token_ids": req.prompt_token_ids,
            "max_tokens": req.max_tokens,
        }
        packet_size = HEADER_SIZE + len(json.dumps(payload).encode("utf-8"))
        packet_sizes.append(packet_size)
        if packet_size > 1500:
            oversized.append((req.request_id, len(req.prompt_token_ids), packet_size))

    logger.info(
        "KVSwitch UDP payload sizes: min=%dB p50=%dB max=%dB",
        min(packet_sizes),
        int(statistics.median(packet_sizes)),
        max(packet_sizes),
    )
    if oversized:
        examples = ", ".join(
            f"req={request_id}/tokens={tokens}/bytes={size}"
            for request_id, tokens, size in oversized[:5]
        )
        logger.warning(
            "Detected %d KVSwitch packets larger than 1500B; fragmented UDP payloads "
            "are not handled reliably by the BMv2 KVSwitch dataplane. Examples: %s",
            len(oversized),
            examples,
        )


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


def _set_interface_mtu(net: Mininet, mtu: int) -> None:
    """Raise MTU across hosts and switches to avoid BMv2 UDP fragmentation."""
    node: Node
    intf: Intf
    for node in [*net.hosts, *net.switches]:
        for intf in node.intfList():
            if intf.name == "lo":
                continue
            node.cmd(f"ip link set dev {intf.name} mtu {mtu} >/dev/null 2>&1 || true")

    logger.info("Set Mininet interface MTU to %d on hosts and switches", mtu)


def _wait_for_service(
    host: Node,
    target_ip: str,
    target_port: int,
    timeout: float = 60.0,
    interval: float = 1.0,
) -> None:
    cmd = _module_cmd(HEALTHCHECK_MODULE, host=target_ip, port=target_port, timeout=2)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = host.cmd(cmd)
        assert isinstance(out, str)
        if "ok" in out:
            logger.info("Service at %s:%d is ready", target_ip, target_port)
            return
        time.sleep(interval)
    raise TimeoutError(
        f"Service at {target_ip}:{target_port} not ready after {timeout}s"
    )


def _kill_bg(host: Node) -> None:
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
        results: list[dict[str, Any]] = json.loads(last)
    except (json.JSONDecodeError, StopIteration):
        logger.error("Failed to parse workload client output:\n%s", out)
        raise

    errors = [
        (item.get("request_id"), item.get("error"))
        for item in results
        if item.get("error")
    ]
    if errors:
        examples = ", ".join(
            f"req={request_id}: {error}" for request_id, error in errors[:5]
        )
        logger.warning(
            "Workload client reported %d request errors for %s:%d. Examples: %s",
            len(errors),
            target_ip,
            target_port,
            examples,
        )
    return results


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
    client = _get_node(net, "client")
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
            ",".join(
                [
                    f"worker{i}",
                    "s2",
                    worker_ip(i),
                    worker.MAC(),
                    str(spine_port_to_leaf),
                    str(leaf_port),
                ]
            )
        )

    return ";".join(worker_specs)


def _stop_workers(net: Mininet, n_workers: int) -> None:
    for i in range(n_workers):
        _kill_bg(_get_node(net, f"worker{i}"))


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


def _setup_root_ns_controller_ip(net: Mininet, controller_ip: str) -> str | None:
    """Make ``controller_ip`` reachable from Mininet hosts by assigning it
    to the controller host's veth interface in root namespace.

    The controller Mininet host's default interface is a veth pair.  The
    host-side lives in the host namespace; the switch-side lives in root
    namespace.  We add the IP as a secondary address on the switch-side
    veth so that packets destined for ``controller_ip`` arrive in root
    namespace (where the SDN controller process runs).

    Returns the root-namespace interface name, or None on failure.
    """
    ctrl_host = _get_node(net, "controller")
    if ctrl_host is None:
        return None

    ctrl_intf = ctrl_host.defaultIntf()
    if ctrl_intf is None or ctrl_intf.link is None:
        return None

    # Find the switch-side interface of the controller's link.
    link = ctrl_intf.link
    switch_intf = link.intf1 if link.intf2 == ctrl_intf else link.intf2
    switch_intf_name = switch_intf.name

    # Add the controller IP to the switch-side interface in root namespace.
    subprocess.run(
        ["ip", "addr", "add", f"{controller_ip}/32", "dev", switch_intf_name],
        capture_output=True,
    )
    logger.info(
        "Assigned %s to root-ns interface %s for controller reachability",
        controller_ip,
        switch_intf_name,
    )
    return switch_intf_name


def run_baseline_kvswitch(
    net: Mininet,
    n_workers: int,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    client_timeout: float,
    adapter: FinsyAdapter,
    controller_runtime: AsyncLoopThread,
) -> list[RequestMetric]:
    """KVSwitch: SDN controller populates TCAM; switches route shim-header traffic."""
    logger.info("--- Baseline: KVSwitch ---")

    # Assign the controller IP to a root-namespace interface so that
    # Mininet workers can send cache events to the controller, which
    # runs in-process (root namespace) with a FinsyWriter.
    _setup_root_ns_controller_ip(net, CONTROLLER_IP)
    _configure_kvswitch_service(net, n_workers)

    # Build worker placements from the live topology.
    workers_arg = _controller_workers_arg(net, n_workers)
    worker_placements = parse_worker_placements(workers_arg)

    # Create and start the controller in-process with the FinsyWriter.
    controller = SDNController(
        workers=worker_placements,
        host=CONTROLLER_IP,
        port=CONTROLLER_PORT,
        adapter=adapter,
        spine_switch="s1",
    )
    controller_runtime.run(controller.start())
    logger.info(
        "SDN controller running in root namespace on %s:%d",
        CONTROLLER_IP,
        CONTROLLER_PORT,
    )

    # Start workers with controller connection and KVSwitch listener.
    client = _get_node(net, "client")
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

    raw = _collect_results(
        client,
        workload_path,
        KVSWITCH_SERVICE_IP,
        KVSWITCH_UDP_PORT,
        client_timeout,
        kvswitch=True,
    )

    controller.close()
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
        default="build/p4/kvswitch",
        help="Path to the BMv2 JSON file or compile_p4.sh output directory",
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
    _log_kvswitch_packet_sizes(requests)

    # Write workload to a temp file accessible inside Mininet.
    workload_path = "/tmp/eval_workload.json"
    save_workload(requests, Path(workload_path))

    # Resolve compiled P4 artifacts up front so both BMv2 and Finsy use
    # the same inputs regardless of whether the caller passed a JSON file
    # path or the compile output directory.
    p4_json_path, p4info_path, p4blob_path = _resolve_p4_artifacts(args.p4_json)

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
        switch=lambda name, **kw: BMv2Switch(name, json_path=p4_json_path, **kw),  # type: ignore
        link=TCLink,
        controller=None,
    )
    net.start()
    _set_interface_mtu(net, 9000)
    _disable_offloads(net)

    controller_runtime = AsyncLoopThread()
    controller_runtime.start()
    switch_grpc, device_ids = _switch_runtime_config(net)
    finsy_adapter = FinsyAdapter(
        switches=switch_grpc,
        p4info_path=Path(p4info_path),
        p4blob_path=Path(p4blob_path),
        device_ids=device_ids,
    )
    controller_runtime.run(finsy_adapter.start())
    _populate_ipv4_routes(net, finsy_adapter)

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
                    adapter=finsy_adapter,
                    controller_runtime=controller_runtime,
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

    finsy_adapter.close()
    controller_runtime.stop()
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
