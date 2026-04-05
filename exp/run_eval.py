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
import sys
from dataclasses import dataclass
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
    ECMP_BUCKETS,
    SDNController,
    WorkerPlacement,
)
from kvswitch.controller.switch_adapter import SwitchOp, TableAddOp, TableClearOp
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
from kvswitch.network.control_plane import ControlPlane
from kvswitch.network.topology import (
    CLOS_ROUTER_IP,
    KVSWITCH_SERVICE_IP,
    KVSWITCH_SERVICE_MAC,
    ClosTopology,
    clos_worker_ip,
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


@dataclass
class TopoConfig:
    """Derived topology metadata for passing to baselines."""

    n_workers: int
    workers_per_leaf: int
    n_worker_leaves: int
    spine_names: list[str]
    placements: list[WorkerPlacement]


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


def fit_linear_ttft_model(
    traces_path: Path,
) -> tuple[float, float]:
    """Fit TTFT_ms = base + per_token * uncached_tokens from profiling traces.

    Returns ``(base_ttft_ms, per_token_ttft_ms)``.
    """
    results = load_results_csv(traces_path)
    xs: list[float] = []  # uncached_tokens
    ys: list[float] = []  # ttft_ms
    for r in results:
        uncached = r.prompt_tokens - r.num_cached_tokens
        xs.append(float(uncached))
        ys.append(r.ttft * 1000)

    n = len(xs)
    if n < 2:
        return 12.0, 0.014  # sensible defaults

    # From closed-form solution to linear regression
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)

    if ss_xx == 0:
        return mean_y, 0.0

    per_token = ss_xy / ss_xx
    base = mean_y - per_token * mean_x
    return max(base, 0.0), max(per_token, 0.0)


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

    For each switch port, discovers hosts reachable within one hop
    (directly connected or on the immediate neighbor switch) and installs
    /32 LPM routes.  This is correct for two-tier Clos topologies where
    any host is at most 2 switch hops from any switch.
    """
    host_info: dict[str, tuple[str, str]] = {}
    for h in net.hosts:
        ip = h.IP()
        mac = h.MAC()
        if ip and mac:
            host_info[h.name] = (ip, mac)

    sw: Switch
    for sw in net.switches:
        ops: list[TableAddOp] = []
        installed_ips: set[str] = set()
        for port_num, intf in enumerate(sw.intfList()):
            if intf.name == "lo":
                continue
            link = cast(Link | None, intf.link)
            if link is None:
                continue
            other_intf = link.intf1 if link.intf2 == intf else link.intf2
            other_node = cast(Node, other_intf.node)

            if other_node in net.hosts:
                # Directly connected host.
                ip = other_node.IP()
                if ip and ip not in installed_ips:
                    installed_ips.add(ip)
                    ops.append(
                        TableAddOp(
                            switch=sw.name,
                            table="ipv4_lpm",
                            action="forward",
                            match={"hdr.ipv4.dstAddr": f"{ip}/32"},
                            action_params={"port": port_num},
                        )
                    )
            else:
                # Neighbor switch — add routes for hosts on it and on
                # its own neighbor switches (2-hop reach).  In a 2-tier
                # Clos this covers ingress→spine→leaf→worker paths.
                neighbor = cast(Switch, other_node)
                hop2_switches: list[Switch] = [neighbor]
                for _, n_intf in enumerate(neighbor.intfList()):
                    if n_intf.name == "lo":
                        continue
                    n_link = cast(Link | None, n_intf.link)
                    if n_link is None:
                        continue
                    n_peer = n_link.intf1 if n_link.intf2 == n_intf else n_link.intf2
                    n_node = cast(Node, n_peer.node)
                    if n_node not in net.hosts and n_node != sw:
                        hop2_switches.append(cast(Switch, n_node))
                for hop_sw in hop2_switches:
                    for _, h_intf in enumerate(hop_sw.intfList()):
                        if h_intf.name == "lo":
                            continue
                        h_link = cast(Link | None, h_intf.link)
                        if h_link is None:
                            continue
                        h_peer = h_link.intf1 if h_link.intf2 == h_intf else h_link.intf2
                        h_node = cast(Node, h_peer.node)
                        if h_node in net.hosts:
                            ip = h_node.IP()
                            if ip and ip not in installed_ips:
                                installed_ips.add(ip)
                                ops.append(
                                    TableAddOp(
                                        switch=sw.name,
                                        table="ipv4_lpm",
                                        action="forward",
                                        match={"hdr.ipv4.dstAddr": f"{ip}/32"},
                                        action_params={"port": port_num},
                                    )
                                )
        if ops:
            adapter.apply_ops(ops)

    for h in net.hosts:
        default_intf = h.defaultIntf()
        intf_name = default_intf.name if default_intf is not None else "lo"
        for other_name, (other_ip, other_mac) in host_info.items():
            if other_name != h.name:
                # ip neigh works across subnets; arp -s may silently fail.
                h.cmd(
                    f"ip neigh replace {other_ip} lladdr {other_mac}"
                    f" dev {intf_name} nud permanent"
                )
        # Default route for cross-subnet traffic in per-leaf topologies.
        if default_intf is not None:
            h.cmd(f"ip route add default dev {intf_name} 2>/dev/null || true")

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


def _kvswitch_packet_size(req: WorkloadRequest) -> int:
    """Compute the KVSwitch UDP packet size for a workload request."""
    payload = {
        "endpoint": "generate",
        "prompt_token_ids": req.prompt_token_ids,
        "max_tokens": req.max_tokens,
    }
    return HEADER_SIZE + len(json.dumps(payload).encode("utf-8"))


def _filter_oversized_requests(
    requests: list[WorkloadRequest],
    max_payload_bytes: int,
) -> list[WorkloadRequest]:
    """Remove requests whose KVSwitch packet exceeds the payload limit.

    Returns the filtered list; logs which requests were dropped and why.
    """
    kept: list[WorkloadRequest] = []
    dropped: list[tuple[int, int, int]] = []
    for req in requests:
        size = _kvswitch_packet_size(req)
        if size > max_payload_bytes:
            dropped.append((req.request_id, len(req.prompt_token_ids), size))
        else:
            kept.append(req)
    if dropped:
        examples = ", ".join(
            f"req={rid}/tokens={tok}/bytes={sz}" for rid, tok, sz in dropped[:5]
        )
        logger.warning(
            "Dropped %d/%d KVSwitch requests exceeding %dB payload limit. Examples: %s",
            len(dropped),
            len(requests),
            max_payload_bytes,
            examples,
        )
    return kept


def _log_kvswitch_packet_sizes(requests: Iterable[WorkloadRequest]) -> None:
    """Log KVSwitch packet sizes and warn if they exceed standard MTU."""
    if not requests:
        return

    packet_sizes: list[int] = []
    oversized: list[tuple[int, int, int]] = []
    for req in requests:
        size = _kvswitch_packet_size(req)
        packet_sizes.append(size)
        if size > 1500:
            oversized.append((req.request_id, len(req.prompt_token_ids), size))

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
                ttft_ms=r.get("ttft_ms", 0.0) or 0.0,
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
    workers_per_leaf: int,
    tpot_ms: float,
    max_output_tokens: int,
    controller_host: str | None = None,
    controller_port: int | None = None,
    kvswitch_port: int | None = None,
    base_ttft_ms: float | None = None,
    per_token_ttft_ms: float | None = None,
    ttft_ms: float = 10.0,
    load_update_interval_ms: float = 500.0,
    load_update_delta: int = 500,
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 4,
) -> None:
    """Start mock workers on all worker hosts."""
    for i in range(n_workers):
        host = net.get(f"worker{i}")
        cmd = (
            f"{PYTHON} -m kvswitch.mock.worker"
            f" --host 0.0.0.0 --port {WORKER_PORT}"
            f" --ttft-ms {ttft_ms} --tpot-ms {tpot_ms}"
            f" --worker-id worker{i}"
            f" --load-update-interval-ms {load_update_interval_ms}"
            f" --load-update-delta {load_update_delta}"
            f" --max-num-batched-tokens {max_num_batched_tokens}"
            f" --max-num-seqs {max_num_seqs}"
        )
        if base_ttft_ms is not None and per_token_ttft_ms is not None:
            cmd += f" --base-ttft-ms {base_ttft_ms} --per-token-ttft-ms {per_token_ttft_ms}"
        if controller_host and controller_port:
            cmd += f" --controller-host {controller_host} --controller-port {controller_port}"
        if kvswitch_port is not None:
            cmd += f" --kvswitch-port {kvswitch_port}"
        _start_bg(host, cmd)

    # Wait for all workers to be ready.
    client = _get_node(net, "client")
    for i in range(n_workers):
        leaf_idx = i // workers_per_leaf
        worker_idx = i % workers_per_leaf
        _wait_for_service(client, clos_worker_ip(leaf_idx, worker_idx), WORKER_PORT)


def _find_port(sw: Node, target: Node) -> int:
    """Return the port number on *sw* that connects to *target*."""
    for port_num, intf in enumerate(sw.intfList()):
        if intf.name == "lo":
            continue
        link = cast(Link | None, intf.link)
        if link is None:
            continue
        other = link.intf1 if link.intf2 == intf else link.intf2
        if cast(Node, other.node) == target:
            return port_num
    raise RuntimeError(f"no link from {sw.name} to {target.name}")


def _build_worker_placements(
    net: Mininet,
    n_workers: int,
    workers_per_leaf: int,
    n_worker_leaves: int,
    spine_names: list[str],
) -> list[WorkerPlacement]:
    """Discover per-worker placement from the live Clos topology."""

    placements: list[WorkerPlacement] = []
    for i in range(n_workers):
        leaf_idx = i // workers_per_leaf
        worker_idx = i % workers_per_leaf
        leaf_name = f"leaf{leaf_idx}"
        leaf_node = _get_node(net, leaf_name)
        worker_node = _get_node(net, f"worker{i}")
        leaf_port = _find_port(leaf_node, worker_node)

        spine_ports: dict[str, int] = {}
        for sp_name in spine_names:
            sp_node = _get_node(net, sp_name)
            spine_ports[sp_name] = _find_port(sp_node, leaf_node)

        placements.append(
            WorkerPlacement(
                worker_id=f"worker{i}",
                leaf_switch=leaf_name,
                worker_ip=clos_worker_ip(leaf_idx, worker_idx),
                worker_mac=worker_node.MAC(),
                leaf_port=leaf_port,
                spine_ports=spine_ports,
            )
        )
    return placements


def _stop_workers(net: Mininet, n_workers: int) -> None:
    for i in range(n_workers):
        _kill_bg(_get_node(net, f"worker{i}"))


def _program_uniform_ecmp(
    net: Mininet,
    placements: list[WorkerPlacement],
    spine_names: list[str],
    adapter: FinsyAdapter,
) -> None:
    """Program uniform ECMP tables on all spines and leaves for L4 round-robin."""

    # Clear per-prefix ECMP tables (may have stale entries from a previous run).
    for sp_name in spine_names:
        adapter.apply_ops([TableClearOp(switch=sp_name, table="spine_prefix_ecmp")])

    # --- Spine ECMP: distribute buckets across worker leaves ---
    leaf_names = sorted({p.leaf_switch for p in placements})
    for sp_name in spine_names:
        ops: list[SwitchOp] = [
            TableClearOp(switch=sp_name, table="spine_ecmp_select")
        ]
        for bucket in range(ECMP_BUCKETS):
            leaf_name = leaf_names[bucket % len(leaf_names)]
            # Find the spine port leading to this leaf.
            port = next(p.spine_ports[sp_name] for p in placements if p.leaf_switch == leaf_name)
            ops.append(
                TableAddOp(
                    switch=sp_name,
                    table="spine_ecmp_select",
                    action="route_to_pod",
                    match={"meta.ecmp_bucket": bucket},
                    action_params={"port": port},
                )
            )
        adapter.apply_ops(ops)

    # --- Leaf ECMP: distribute buckets across local workers ---
    workers_by_leaf: dict[str, list[WorkerPlacement]] = defaultdict(list)
    for p in placements:
        workers_by_leaf[p.leaf_switch].append(p)
    n_leaves = len(workers_by_leaf)
    for leaf_name, leaf_workers in sorted(workers_by_leaf.items()):
        ops = [TableClearOp(switch=leaf_name, table="leaf_ecmp_select")]
        n_local = len(leaf_workers)
        for bucket in range(ECMP_BUCKETS):
            # Use bucket // n_leaves for worker selection so the leaf-level
            # decision is decorrelated from the spine-level decision (which
            # uses bucket % n_leaves).  The P4 hash is identical at every
            # hop, so using the same bucket modulus would always select the
            # same worker-index on every leaf.
            w = leaf_workers[(bucket // n_leaves) % n_local]
            ops.append(
                TableAddOp(
                    switch=leaf_name,
                    table="leaf_ecmp_select",
                    action="route_to_worker",
                    match={"meta.ecmp_bucket": bucket},
                    action_params={
                        "port": w.leaf_port,
                        "dst_mac": int(w.worker_mac.replace(":", ""), 16),
                    },
                )
            )
        adapter.apply_ops(ops)

    # --- Ingress ECMP: distribute buckets across spines ---
    ingress_switches = [sw for sw in net.switches if sw.name.startswith("ingress")]
    for ing in ingress_switches:
        ops = [TableClearOp(switch=ing.name, table="spine_ecmp_select")]
        for bucket in range(ECMP_BUCKETS):
            sp_name = spine_names[bucket % len(spine_names)]
            sp_port = _find_port(ing, _get_node(net, sp_name))
            ops.append(
                TableAddOp(
                    switch=ing.name,
                    table="spine_ecmp_select",
                    action="route_to_pod",
                    match={"meta.ecmp_bucket": bucket},
                    action_params={"port": sp_port},
                )
            )
        adapter.apply_ops(ops)

    logger.info(
        "Programmed uniform ECMP: %d spines, %d leaves, %d workers",
        len(spine_names),
        len(leaf_names),
        len(placements),
    )


def _program_ingress_ecmp(
    net: Mininet,
    spine_names: list[str],
    adapter: FinsyAdapter,
) -> None:
    """Program ingress-leaf ECMP to distribute across spines."""
    ingress_switches = [sw for sw in net.switches if sw.name.startswith("ingress")]
    for ing in ingress_switches:
        ops: list[SwitchOp] = [
            TableClearOp(switch=ing.name, table="spine_ecmp_select")
        ]
        for bucket in range(ECMP_BUCKETS):
            sp_name = spine_names[bucket % len(spine_names)]
            sp_port = _find_port(ing, _get_node(net, sp_name))
            ops.append(
                TableAddOp(
                    switch=ing.name,
                    table="spine_ecmp_select",
                    action="route_to_pod",
                    match={"meta.ecmp_bucket": bucket},
                    action_params={"port": sp_port},
                )
            )
        adapter.apply_ops(ops)
    logger.info("Programmed ingress ECMP across %d spines", len(spine_names))


def run_baseline_l4_rr(
    net: Mininet,
    topo: TopoConfig,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    client_timeout: float,
    adapter: FinsyAdapter,
    base_ttft_ms: float | None = None,
    per_token_ttft_ms: float | None = None,
) -> list[RequestMetric]:
    """L4 Round-Robin: uniform ECMP distributes traffic via KVSwitch VIP."""
    logger.info("--- Baseline: L4 Round-Robin ---")

    _configure_kvswitch_service(net, topo.n_workers)
    _program_uniform_ecmp(net, topo.placements, topo.spine_names, adapter)

    _start_workers(
        net,
        topo.n_workers,
        topo.workers_per_leaf,
        tpot_ms,
        max_output_tokens,
        kvswitch_port=KVSWITCH_UDP_PORT,
        base_ttft_ms=base_ttft_ms,
        per_token_ttft_ms=per_token_ttft_ms,
        ttft_ms=ttft_ms,
    )

    client = _get_node(net, "client")

    # Warm up worker caches so the timed run reflects steady-state.
    warmup_workload = _build_warmup_workload(workload_path)
    if warmup_workload:
        warmup_path = "/tmp/eval_warmup.json"
        with open(warmup_path, "w") as f:
            json.dump(warmup_workload, f)
        logger.info("L4 RR warm-up: %d requests", len(warmup_workload))
        _collect_results(
            client,
            warmup_path,
            KVSWITCH_SERVICE_IP,
            KVSWITCH_UDP_PORT,
            client_timeout,
            kvswitch=True,
        )

    raw = _collect_results(
        client,
        workload_path,
        KVSWITCH_SERVICE_IP,
        KVSWITCH_UDP_PORT,
        client_timeout,
        kvswitch=True,
    )

    _stop_workers(net, topo.n_workers)
    return _to_request_metrics(raw, "l4_rr")


def run_baseline_l7(
    net: Mininet,
    topo: TopoConfig,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    model: str,
    client_timeout: float,
    base_ttft_ms: float | None = None,
    per_token_ttft_ms: float | None = None,
) -> list[RequestMetric]:
    """L7 Prefix-Aware: Client → L7 proxy → best worker."""
    logger.info("--- Baseline: L7 Prefix-Aware Router ---")
    _start_workers(
        net,
        topo.n_workers,
        topo.workers_per_leaf,
        tpot_ms,
        max_output_tokens,
        base_ttft_ms=base_ttft_ms,
        per_token_ttft_ms=per_token_ttft_ms,
        ttft_ms=ttft_ms,
    )

    # Start L7 proxy on the router host.
    router_host = _get_node(net, "router")
    workers_arg = ",".join(f"{p.worker_ip}:{WORKER_PORT}" for p in topo.placements)
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
        _wait_for_service(client, CLOS_ROUTER_IP, ROUTER_PORT, timeout=120.0)
    except TimeoutError:
        log_out = router_host.cmd(f"cat {proxy_log}")
        logger.error("L7 proxy log:\n%s", log_out)
        raise

    # Warm up worker caches and router prefix table.
    warmup_workload = _build_warmup_workload(workload_path)
    if warmup_workload:
        warmup_path = "/tmp/eval_warmup.json"
        with open(warmup_path, "w") as f:
            json.dump(warmup_workload, f)
        logger.info("L7 warm-up: %d requests", len(warmup_workload))
        _collect_results(
            client,
            warmup_path,
            CLOS_ROUTER_IP,
            ROUTER_PORT,
            client_timeout,
        )

    raw = _collect_results(
        client, workload_path, CLOS_ROUTER_IP, ROUTER_PORT, client_timeout
    )

    _kill_bg(router_host)
    _stop_workers(net, topo.n_workers)
    return _to_request_metrics(raw, "l7")


def _build_warmup_workload(workload_path: str, n_per_group: int = 2) -> list[dict]:
    """Pick representative requests per prefix group for warm-up.

    Selects up to *n_per_group* requests from each prefix group so that:

    - the controller sees enough alloc events to pass ``admission_threshold``
    - the workers' local caches are seeded for the group's system-prompt prefix
    - deeper prefix-chain entries get at least partial TCAM coverage

    All warm-up requests fire immediately (``scheduled_time=0``).
    """
    with open(workload_path) as f:
        requests: list[dict] = json.load(f)
    group_counts: dict[str, int] = {}
    warmup: list[dict] = []
    for req in requests:
        group = req.get("prefix_group", "none")
        if group == "none":
            continue
        count = group_counts.get(group, 0)
        if count >= n_per_group:
            continue
        group_counts[group] = count + 1
        warmup.append(
            {
                **req,
                "request_id": 90000 + len(warmup),
                "scheduled_time": 0.0,
            }
        )
    return warmup


def run_baseline_kvswitch(
    net: Mininet,
    topo: TopoConfig,
    ttft_ms: float,
    tpot_ms: float,
    max_output_tokens: int,
    workload_path: str,
    client_timeout: float,
    adapter: FinsyAdapter,
    controller_runtime: AsyncLoopThread,
    base_ttft_ms: float | None = None,
    per_token_ttft_ms: float | None = None,
    admission_threshold: int = 2,
) -> list[RequestMetric]:
    """KVSwitch: SDN controller populates TCAM; switches route shim-header traffic."""
    logger.info("--- Baseline: KVSwitch ---")
    _configure_kvswitch_service(net, topo.n_workers)

    with ControlPlane(net, topo.n_workers) as cp:
        controller_ip = cp.controller_ip
        controller = SDNController(
            workers=topo.placements,
            host=controller_ip,
            port=CONTROLLER_PORT,
            adapter=adapter,
            spine_switches=topo.spine_names,
            coalesce_interval_s=2.0,
            admission_threshold=admission_threshold,
        )
        controller_runtime.run(controller.start())
        logger.info(
            "SDN controller running on control-plane network %s:%d",
            controller_ip,
            CONTROLLER_PORT,
        )

        # Program ingress leaf ECMP to distribute across spines.
        # The controller manages spine and worker-leaf ECMP but not
        # the ingress tier — that's static infrastructure.
        _program_ingress_ecmp(net, topo.spine_names, adapter)

        client = _get_node(net, "client")
        _start_workers(
            net,
            topo.n_workers,
            topo.workers_per_leaf,
            tpot_ms,
            max_output_tokens,
            controller_host=controller_ip,
            controller_port=CONTROLLER_PORT,
            kvswitch_port=KVSWITCH_UDP_PORT,
            base_ttft_ms=base_ttft_ms,
            per_token_ttft_ms=per_token_ttft_ms,
            ttft_ms=ttft_ms,
        )

        # Warm-up: send requests to populate both TCAM prefix rules and worker
        # caches.  Workers stay running so switch and cache state are consistent.
        warmup_workload = _build_warmup_workload(
            workload_path, n_per_group=max(admission_threshold, 20)
        )
        if warmup_workload:
            warmup_path = "/tmp/eval_warmup.json"
            with open(warmup_path, "w") as f:
                json.dump(warmup_workload, f)
            logger.info(
                "Running %d warm-up requests to populate TCAM rules and caches...",
                len(warmup_workload),
            )
            _collect_results(
                client,
                warmup_path,
                KVSWITCH_SERVICE_IP,
                KVSWITCH_UDP_PORT,
                client_timeout,
                kvswitch=True,
            )
            time.sleep(2.0)
            logger.info(
                "Warm-up complete. Controller snapshot: spine_rules=%d leaf_rules=%d",
                len(controller.spine_tcam.snapshot()),
                sum(len(t.snapshot()) for t in controller.leaf_tcams.values()),
            )

        try:
            raw = _collect_results(
                client,
                workload_path,
                KVSWITCH_SERVICE_IP,
                KVSWITCH_UDP_PORT,
                client_timeout,
                kvswitch=True,
            )
        finally:
            controller.close()
            _stop_workers(net, topo.n_workers)

    return _to_request_metrics(raw, "kvswitch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


BASELINE_REGISTRY = {
    "l4_rr": {"needs_router": False},
    "l7": {"needs_router": True},
    "kvswitch": {"needs_router": False},
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
    parser.add_argument("--n-spines", type=int, default=2)
    parser.add_argument("--n-worker-leaves", type=int, default=1)
    parser.add_argument("--workers-per-leaf", type=int, default=4)
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
    parser.add_argument(
        "--base-ttft-ms",
        type=float,
        default=None,
        help="Override base TTFT for linear model (else fitted from traces)",
    )
    parser.add_argument(
        "--per-token-ttft-ms",
        type=float,
        default=None,
        help="Override per-token TTFT for linear model (else fitted from traces)",
    )
    parser.add_argument("--client-timeout", type=float, default=60.0)
    parser.add_argument(
        "--max-kvswitch-payload",
        type=int,
        default=9000,
        help="Drop KVSwitch requests whose UDP payload exceeds this size in bytes",
    )
    parser.add_argument("--admission-threshold", type=int, default=2)
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

    # Fit the linear TTFT model from traces (or use CLI overrides).
    if args.base_ttft_ms is not None and args.per_token_ttft_ms is not None:
        base_ttft_ms = args.base_ttft_ms
        per_token_ttft_ms = args.per_token_ttft_ms
    elif traces_path.exists():
        base_ttft_ms, per_token_ttft_ms = fit_linear_ttft_model(traces_path)
    else:
        base_ttft_ms, per_token_ttft_ms = 12.9, 0.01406  # sensible defaults

    logger.info(
        "Using TTFT model: base=%.2fms + %.5fms/uncached_token (fallback fixed=%.2fms), TPOT=%.2fms",
        base_ttft_ms,
        per_token_ttft_ms,
        ttft_ms,
        tpot_ms,
    )

    # --- Generate workload ---
    # Over-generate to compensate for oversized-packet filtering, so the
    # final workload has exactly num_requests entries.
    target = args.num_requests
    overshoot = int(target * 1.5)
    logger.info("Generating workload (target=%d, generating=%d)...", target, overshoot)
    wl_config = WorkloadConfig(
        dataset_path=Path(args.dataset),
        num_requests=overshoot,
        request_rate=args.request_rate,
        prefix_sharing_ratio=args.prefix_sharing_ratio,
        num_prefix_groups=args.num_prefix_groups,
        system_prompt_tokens=args.system_prompt_tokens,
        max_output_tokens=args.max_output_tokens,
        seed=args.seed,
        model=args.model,
    )
    generator = WorkloadGenerator(wl_config)
    raw_requests = generator.generate()
    _log_kvswitch_packet_sizes(raw_requests)

    # Filter oversized packets, then trim to exact target count.
    requests = _filter_oversized_requests(raw_requests, args.max_kvswitch_payload)
    if len(requests) < target:
        logger.warning(
            "Only %d requests remain after filtering (target %d); using all",
            len(requests),
            target,
        )
    else:
        requests = requests[:target]

    # Re-number request IDs and recalculate scheduled times so arrival
    # times are contiguous after the oversized gaps are removed.
    t = 0.0
    rng = __import__("random").Random(args.seed + 1)
    for i, req in enumerate(requests):
        req.request_id = i
        req.scheduled_time = t
        if args.request_rate > 0:
            t += rng.expovariate(args.request_rate)

    workload_path = "/tmp/eval_workload.json"
    save_workload(requests, Path(workload_path))
    logger.info("Final workload: %d requests", len(requests))

    # Resolve compiled P4 artifacts up front so both BMv2 and Finsy use
    # the same inputs regardless of whether the caller passed a JSON file
    # path or the compile output directory.
    p4_json_path, p4info_path, p4blob_path = _resolve_p4_artifacts(args.p4_json)

    # --- Topology parameters ---
    n_spines = args.n_spines
    n_worker_leaves = args.n_worker_leaves
    workers_per_leaf = args.workers_per_leaf
    n_workers = n_worker_leaves * workers_per_leaf
    needs_router = any(BASELINE_REGISTRY[b]["needs_router"] for b in baselines)
    spine_names = [f"spine{i}" for i in range(n_spines)]

    # --- Build and start Clos topology ---
    logger.info(
        "Building Clos topology: %d spines, %d worker leaves × %d workers, router=%s",
        n_spines,
        n_worker_leaves,
        workers_per_leaf,
        needs_router,
    )
    reset_device_ids()
    clos = ClosTopology(
        n_spines=n_spines,
        n_worker_leaves=n_worker_leaves,
        workers_per_leaf=workers_per_leaf,
        with_router=needs_router,
        delay=args.delay,
    )
    net = Mininet(
        topo=clos,
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

    placements = _build_worker_placements(
        net, n_workers, workers_per_leaf, n_worker_leaves, spine_names
    )
    topo_config = TopoConfig(
        n_workers=n_workers,
        workers_per_leaf=workers_per_leaf,
        n_worker_leaves=n_worker_leaves,
        spine_names=spine_names,
        placements=placements,
    )

    net.pingAll()

    # --- Run each baseline ---
    all_results: dict[str, list[RequestMetric]] = {}
    for baseline in baselines:
        try:
            if baseline == "l4_rr":
                metrics = run_baseline_l4_rr(
                    net,
                    topo_config,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.client_timeout,
                    adapter=finsy_adapter,
                    base_ttft_ms=base_ttft_ms,
                    per_token_ttft_ms=per_token_ttft_ms,
                )
            elif baseline == "l7":
                metrics = run_baseline_l7(
                    net,
                    topo_config,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.model,
                    args.client_timeout,
                    base_ttft_ms=base_ttft_ms,
                    per_token_ttft_ms=per_token_ttft_ms,
                )
            elif baseline == "kvswitch":
                metrics = run_baseline_kvswitch(
                    net,
                    topo_config,
                    ttft_ms,
                    tpot_ms,
                    args.max_output_tokens,
                    workload_path,
                    args.client_timeout,
                    adapter=finsy_adapter,
                    controller_runtime=controller_runtime,
                    base_ttft_ms=base_ttft_ms,
                    per_token_ttft_ms=per_token_ttft_ms,
                    admission_threshold=args.admission_threshold,
                )
            else:
                continue

            all_results[baseline] = metrics
            summary = compute_summary(metrics)
            logger.info(
                "%s: e2e_p50=%.1fms e2e_p95=%.1fms ttft_p50=%.1fms ttft_p95=%.1fms cache_hit=%.2f",
                baseline,
                summary.get("e2e_p50_ms", 0),
                summary.get("e2e_p95_ms", 0),
                summary.get("ttft_p50_ms", 0),
                summary.get("ttft_p95_ms", 0),
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
        "n_spines": n_spines,
        "n_worker_leaves": n_worker_leaves,
        "workers_per_leaf": workers_per_leaf,
        "n_workers": n_workers,
        "num_requests": args.num_requests,
        "request_rate": args.request_rate,
        "prefix_sharing_ratio": args.prefix_sharing_ratio,
        "num_prefix_groups": args.num_prefix_groups,
        "system_prompt_tokens": args.system_prompt_tokens,
        "max_output_tokens": args.max_output_tokens,
        "ttft_ms": ttft_ms,
        "base_ttft_ms": base_ttft_ms,
        "per_token_ttft_ms": per_token_ttft_ms,
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
