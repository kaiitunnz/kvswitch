"""Prototype SDN controller for KVSwitch cache synchronization.

The current repository uses the existing UDP substrate for cache-event delivery.
This preserves the planned alloc/evict/queue_update protocol while remaining
lightweight and testable inside the current Python-only prototype.
"""

import argparse
import asyncio
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kvswitch.controller.tcam_manager import PrefixKey, TcamManager
from kvswitch.utils.logger import setup_logging
from kvswitch.utils.prefix import (
    format_prefix_key,
    leaf_prefix_key,
    normalize_prefix_hashes,
    spine_prefix_key,
)
from kvswitch.utils.udp import UDPRequest, UDPResponse, UDPServer

logger = logging.getLogger(__name__)

EventType = Literal["alloc", "evict", "queue_update"]
ECMP_BUCKETS = 32


def parse_worker_placement(spec: str) -> "WorkerPlacement":
    """Parse ``worker_id,leaf_switch,worker_ip,spine_port,leaf_port``."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 5:
        raise ValueError(
            "worker spec must be worker_id,leaf_switch,worker_ip,spine_port,leaf_port"
        )
    worker_id, leaf_switch, worker_ip, raw_spine_port, raw_leaf_port = parts
    return WorkerPlacement(
        worker_id=worker_id,
        leaf_switch=leaf_switch,
        worker_ip=worker_ip,
        spine_port=int(raw_spine_port),
        leaf_port=int(raw_leaf_port),
    )


def parse_worker_placements(specs: str) -> list["WorkerPlacement"]:
    """Parse ``;``-separated worker placement specs."""
    placements = [
        parse_worker_placement(spec) for spec in specs.split(";") if spec.strip()
    ]
    if not placements:
        raise ValueError("at least one worker placement is required")
    return placements


@dataclass(frozen=True)
class WorkerPlacement:
    """Physical placement and forwarding metadata for a worker."""

    worker_id: str
    leaf_switch: str
    worker_ip: str
    spine_port: int
    leaf_port: int


@dataclass(frozen=True)
class CacheSyncEvent:
    """Cache or load event emitted by a worker."""

    event_type: EventType
    worker_id: str
    prefix_hashes: tuple[int, ...] = ()
    queue_depth: int | None = None
    timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "endpoint": "cache_event",
            "event_type": self.event_type,
            "worker_id": self.worker_id,
        }
        if self.prefix_hashes:
            payload["prefix_hashes"] = normalize_prefix_hashes(self.prefix_hashes)
        if self.queue_depth is not None:
            payload["queue_depth"] = self.queue_depth
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CacheSyncEvent":
        prefix_hashes = tuple(normalize_prefix_hashes(payload.get("prefix_hashes", [])))
        raw_queue_depth = payload.get("queue_depth")
        raw_timestamp = payload.get("timestamp")
        return cls(
            event_type=str(payload.get("event_type", "alloc")),  # type: ignore
            worker_id=str(payload["worker_id"]),
            prefix_hashes=prefix_hashes,
            queue_depth=(int(raw_queue_depth) if raw_queue_depth is not None else None),
            timestamp=(float(raw_timestamp) if raw_timestamp is not None else None),
        )


class SwitchWriter(Protocol):
    """Applies control-plane commands to a logical switch target."""

    def apply(self, switch_name: str, commands: list[str]) -> None: ...


@dataclass
class InMemorySwitchWriter:
    """Test writer that records the commands sent to each switch."""

    commands_by_switch: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def apply(self, switch_name: str, commands: list[str]) -> None:
        self.commands_by_switch[switch_name].extend(commands)


@dataclass
class BMv2CLIWriter:
    """Optional BMv2 CLI writer for prototype deployments."""

    cli_by_switch: dict[str, str]
    history: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def apply(self, switch_name: str, commands: list[str]) -> None:
        self.history[switch_name].extend(commands)
        cli = self.cli_by_switch.get(switch_name)
        if cli is None or not commands:
            return
        subprocess.run(
            [cli],
            input="\n".join(commands) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )


class SDNController:
    """KVSwitch control plane with cache-event-driven table synchronization."""

    def __init__(
        self,
        workers: list[WorkerPlacement],
        host: str = "0.0.0.0",
        port: int = 9100,
        admission_threshold: int = 2,
        window_s: float = 60.0,
        max_spine_entries: int = 1024,
        max_leaf_entries: int = 1024,
        alpha: float = 1.0,
        beta: float = 1.0,
        writer: SwitchWriter | None = None,
    ) -> None:
        self._workers = {worker.worker_id: worker for worker in workers}
        self._server = UDPServer(host=host, port=port, handler=self._handle)
        self._writer = writer or InMemorySwitchWriter()
        self.alpha = alpha
        self.beta = beta

        self.spine_tcam = TcamManager(
            admission_threshold=admission_threshold,
            window_s=window_s,
            max_entries=max_spine_entries,
        )
        self.leaf_tcam = TcamManager(
            admission_threshold=admission_threshold,
            window_s=window_s,
            max_entries=max_leaf_entries,
        )

        # Keep the observed cache holders separate from the currently installed rules.
        self.queue_depths: dict[str, int] = {worker.worker_id: 0 for worker in workers}
        self._spine_locations: dict[tuple[int, ...], set[str]] = defaultdict(set)
        self._leaf_locations: dict[PrefixKey, set[str]] = defaultdict(set)

    @property
    def port(self) -> int:
        return self._server.port

    async def start(self) -> None:
        await self._server.start()
        self._refresh_ecmp_weights()
        logger.info("SDN controller listening on %s:%d", self._server.host, self.port)

    def close(self) -> None:
        self._server.close()

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue_depths": dict(self.queue_depths),
            "spine_rules": self.spine_tcam.snapshot(),
            "leaf_rules": self.leaf_tcam.snapshot(),
            "spine_locations": {
                format_prefix_key(prefix): sorted(workers)
                for prefix, workers in self._spine_locations.items()
            },
            "leaf_locations": {
                format_prefix_key(prefix): sorted(workers)
                for prefix, workers in self._leaf_locations.items()
            },
        }

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        endpoint = request.data.get("endpoint", "cache_event")
        if endpoint == "health":
            return UDPResponse(data={"status": "ok", "port": self.port})
        if endpoint == "snapshot":
            return UDPResponse(data={"status": "ok", "snapshot": self.snapshot()})
        if endpoint != "cache_event":
            return UDPResponse(data={"error": f"unknown endpoint: {endpoint}"})

        event = CacheSyncEvent.from_payload(request.data)
        commands = self.handle_event(event)
        return UDPResponse(data={"status": "ok", "commands": commands})

    def handle_event(self, event: CacheSyncEvent) -> list[str]:
        now = time.time() if event.timestamp is None else event.timestamp
        if event.worker_id not in self._workers:
            raise ValueError(f"unknown worker: {event.worker_id}")

        commands: list[str] = []
        if event.event_type == "queue_update":
            if event.queue_depth is None:
                raise ValueError("queue_update requires queue_depth")
            self.queue_depths[event.worker_id] = event.queue_depth
            # Queue updates can change both installed rule targets and ECMP weights.
            commands.extend(self._reconcile_all_rules(now))
            commands.extend(self._refresh_ecmp_weights())
            return commands

        if not event.prefix_hashes:
            raise ValueError(f"{event.event_type} requires prefix_hashes")

        spine_key = spine_prefix_key(event.prefix_hashes)
        leaf_key = leaf_prefix_key(event.prefix_hashes)
        if event.event_type == "alloc":
            # Observe cache ownership first, then admit popular prefixes into TCAM.
            self._spine_locations[spine_key].add(event.worker_id)
            self._leaf_locations[leaf_key].add(event.worker_id)
            spine_hits = self.spine_tcam.record_observation(spine_key, now)
            leaf_hits = self.leaf_tcam.record_observation(leaf_key, now)
            commands.extend(self._maybe_install_spine_rule(spine_key, spine_hits, now))
            commands.extend(self._maybe_install_leaf_rule(leaf_key, leaf_hits, now))
            return commands

        if event.event_type == "evict":
            # Drop stale placements and reinstall a fallback only if replicas remain.
            commands.extend(self._apply_spine_eviction(spine_key, event.worker_id, now))
            commands.extend(self._apply_leaf_eviction(leaf_key, event.worker_id, now))
            commands.extend(self._refresh_ecmp_weights())
            return commands

        raise ValueError(f"unsupported event type: {event.event_type}")

    def _worker_score(self, worker_id: str) -> float:
        return self.alpha - self.beta * float(self.queue_depths.get(worker_id, 0))

    def _select_worker(self, candidates: set[str]) -> str:
        if not candidates:
            raise ValueError("no worker candidates available")
        return max(
            candidates, key=lambda worker_id: (self._worker_score(worker_id), worker_id)
        )

    def _spine_command_for_worker(self, prefix: tuple[int, ...], worker_id: str) -> str:
        worker = self._workers[worker_id]
        h0 = prefix[0]
        return (
            "table_add spine_prefix_route route_to_pod "
            f"0x{h0:08x}&&&0xffffffff => {worker.spine_port}"
        )

    def _leaf_command_for_worker(
        self, prefix: PrefixKey, worker_id: str
    ) -> tuple[str, str]:
        worker = self._workers[worker_id]
        values = [0, 0, 0]
        masks = [0, 0, 0]
        for idx, value in enumerate(prefix):
            values[idx] = value
            masks[idx] = 0xFFFFFFFF
        match = " ".join(
            f"0x{value:08x}&&&0x{mask:08x}" for value, mask in zip(values, masks)
        )
        return (
            worker.leaf_switch,
            f"table_add leaf_prefix_route route_to_worker {match} => {worker.leaf_port}",
        )

    def _delete_spine_command(self, prefix: tuple[int, ...]) -> str:
        return f"table_delete spine_prefix_route 0x{prefix[0]:08x}"

    def _delete_leaf_command(
        self, prefix: PrefixKey, worker_id: str
    ) -> tuple[str, str]:
        switch = self._workers[worker_id].leaf_switch
        match = ".".join(f"{value:08x}" for value in prefix)
        return switch, f"table_delete leaf_prefix_route {match}"

    def _leaf_switch_for_prefix(self, prefix: PrefixKey) -> str:
        candidates = self._leaf_locations.get(prefix)
        if not candidates:
            return "leaf0"
        worker_id = self._select_worker(candidates)
        return self._workers[worker_id].leaf_switch

    def _maybe_install_spine_rule(
        self,
        prefix: tuple[int, ...],
        hit_count: int,
        now: float,
    ) -> list[str]:
        # Spine rules coarse-route by the first exported hash only.
        if not self.spine_tcam.admitted(prefix, now):
            return []
        candidates = self._spine_locations.get(prefix, set())
        if not candidates:
            return []
        worker_id = self._select_worker(candidates)
        _, evicted = self.spine_tcam.install(
            prefix, worker_id, hit_count=hit_count, now=now
        )
        commands: list[str] = []
        if evicted is not None:
            commands.append(self._delete_spine_command(evicted[0]))
        commands.append(self._spine_command_for_worker(prefix, worker_id))
        self._writer.apply("spine", commands)
        return commands

    def _maybe_install_leaf_rule(
        self,
        prefix: PrefixKey,
        hit_count: int,
        now: float,
    ) -> list[str]:
        # Leaf rules refine the decision with as many exported hashes as we have.
        if not self.leaf_tcam.admitted(prefix, now):
            return []
        candidates = self._leaf_locations.get(prefix, set())
        if not candidates:
            return []
        worker_id = self._select_worker(candidates)
        rule, evicted = self.leaf_tcam.install(
            prefix, worker_id, hit_count=hit_count, now=now
        )
        switch_name, command = self._leaf_command_for_worker(prefix, worker_id)
        commands: list[str] = []
        if evicted is not None:
            evicted_switch, evicted_cmd = self._delete_leaf_command(
                evicted[0], evicted[1].target_id
            )
            self._writer.apply(evicted_switch, [evicted_cmd])
            commands.append(evicted_cmd)
        commands.append(command)
        self._writer.apply(switch_name, [command])
        return commands

    def _apply_spine_eviction(
        self,
        prefix: tuple[int, ...],
        worker_id: str,
        now: float,
    ) -> list[str]:
        candidates = self._spine_locations.get(prefix)
        if not candidates:
            return []
        candidates.discard(worker_id)
        if candidates:
            hit_count = self.spine_tcam.observation_count(prefix, now)
            return self._maybe_install_spine_rule(prefix, hit_count, now)
        self._spine_locations.pop(prefix, None)
        if self.spine_tcam.remove(prefix) is None:
            return []
        commands = [self._delete_spine_command(prefix)]
        self._writer.apply("spine", commands)
        return commands

    def _apply_leaf_eviction(
        self,
        prefix: PrefixKey,
        worker_id: str,
        now: float,
    ) -> list[str]:
        candidates = self._leaf_locations.get(prefix)
        if not candidates:
            return []
        candidates.discard(worker_id)
        if candidates:
            hit_count = self.leaf_tcam.observation_count(prefix, now)
            return self._maybe_install_leaf_rule(prefix, hit_count, now)
        self._leaf_locations.pop(prefix, None)
        removed = self.leaf_tcam.remove(prefix)
        if removed is None:
            return []
        switch_name, command = self._delete_leaf_command(prefix, removed.target_id)
        self._writer.apply(switch_name, [command])
        return [command]

    def _reconcile_all_rules(self, now: float) -> list[str]:
        commands: list[str] = []
        # Queue shifts can change the best worker for already-admitted prefixes.
        for prefix in list(self.spine_tcam._installed.keys()):
            candidates = self._spine_locations.get(prefix, set())
            if not candidates:
                continue
            hit_count = self.spine_tcam.observation_count(prefix, now)
            commands.extend(self._maybe_install_spine_rule(prefix, hit_count, now))

        for prefix in list(self.leaf_tcam._installed.keys()):
            candidates = self._leaf_locations.get(prefix, set())
            if not candidates:
                continue
            hit_count = self.leaf_tcam.observation_count(prefix, now)
            commands.extend(self._maybe_install_leaf_rule(prefix, hit_count, now))
        return commands

    def _refresh_ecmp_weights(self) -> list[str]:
        commands: list[str] = []
        commands.extend(self._program_spine_ecmp())
        commands.extend(self._program_leaf_ecmp())
        return commands

    def _inverse_queue_weight(self, worker_id: str) -> float:
        return 1.0 / (1.0 + float(self.queue_depths.get(worker_id, 0)))

    def _allocate_buckets(self, weights: dict[str, float]) -> dict[int, str]:
        if not weights:
            return {}
        total = sum(weights.values())
        if total <= 0:
            equal = 1.0 / len(weights)
            weights = {key: equal for key in weights}
            total = 1.0

        raw_counts = {
            key: max(1, int(round((value / total) * ECMP_BUCKETS)))
            for key, value in weights.items()
        }
        while sum(raw_counts.values()) > ECMP_BUCKETS:
            key = max(raw_counts, key=raw_counts.__getitem__)
            if raw_counts[key] > 1:
                raw_counts[key] -= 1
            else:
                break
        while sum(raw_counts.values()) < ECMP_BUCKETS:
            key = max(weights, key=weights.__getitem__)
            raw_counts[key] += 1

        mapping: dict[int, str] = {}
        bucket = 0
        for key in sorted(raw_counts):
            for _ in range(raw_counts[key]):
                if bucket >= ECMP_BUCKETS:
                    break
                mapping[bucket] = key
                bucket += 1
        return mapping

    def _program_spine_ecmp(self) -> list[str]:
        # Misses are balanced at the pod level before leaf-level selection.
        per_leaf: dict[str, float] = defaultdict(float)
        for worker_id, placement in self._workers.items():
            per_leaf[placement.leaf_switch] += self._inverse_queue_weight(worker_id)
        bucket_map = self._allocate_buckets(per_leaf)
        commands = ["table_clear spine_ecmp_select"]
        leaves = {
            placement.leaf_switch: placement.spine_port
            for placement in self._workers.values()
        }
        for bucket, leaf_switch in bucket_map.items():
            commands.append(
                f"table_add spine_ecmp_select route_to_pod {bucket} => {leaves[leaf_switch]}"
            )
        self._writer.apply("spine", commands)
        return commands

    def _program_leaf_ecmp(self) -> list[str]:
        commands: list[str] = []
        # Each leaf rebalances only across its directly attached workers.
        workers_by_leaf: dict[str, dict[str, float]] = defaultdict(dict)
        for worker_id, placement in self._workers.items():
            workers_by_leaf[placement.leaf_switch][worker_id] = (
                self._inverse_queue_weight(worker_id)
            )

        for leaf_switch, weights in workers_by_leaf.items():
            bucket_map = self._allocate_buckets(weights)
            leaf_commands = ["table_clear leaf_ecmp_select"]
            for bucket, worker_id in bucket_map.items():
                leaf_commands.append(
                    f"table_add leaf_ecmp_select route_to_worker {bucket} => {self._workers[worker_id].leaf_port}"
                )
            self._writer.apply(leaf_switch, leaf_commands)
            commands.extend(leaf_commands)
        return commands

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        finally:
            self.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="KVSwitch SDN controller")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--workers", type=str, required=True)
    parser.add_argument("--admission-threshold", type=int, default=2)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--max-spine-entries", type=int, default=1024)
    parser.add_argument("--max-leaf-entries", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    controller = SDNController(
        workers=parse_worker_placements(args.workers),
        host=args.host,
        port=args.port,
        admission_threshold=args.admission_threshold,
        window_s=args.window_s,
        max_spine_entries=args.max_spine_entries,
        max_leaf_entries=args.max_leaf_entries,
        alpha=args.alpha,
        beta=args.beta,
    )
    asyncio.run(controller.run_forever())


if __name__ == "__main__":
    main()
