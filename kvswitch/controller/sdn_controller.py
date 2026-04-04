"""Prototype SDN controller for KVSwitch cache synchronization.

The current repository uses the existing UDP substrate for cache-event delivery.
This preserves the planned alloc/evict/queue_update protocol while remaining
lightweight and testable inside the current Python-only prototype.
"""

import argparse
import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from kvswitch.controller.switch_adapter import (
    InMemorySwitchAdapter,
    SwitchAdapter,
    SwitchOp,
    TableAddOp,
    TableClearOp,
    TableDeleteOp,
)
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
    """Parse ``worker_id,leaf_switch,worker_ip,worker_mac,spine_port,leaf_port``."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 6:
        raise ValueError(
            "worker spec must be "
            "worker_id,leaf_switch,worker_ip,worker_mac,spine_port,leaf_port"
        )
    worker_id, leaf_switch, worker_ip, worker_mac, raw_spine_port, raw_leaf_port = parts
    return WorkerPlacement(
        worker_id=worker_id,
        leaf_switch=leaf_switch,
        worker_ip=worker_ip,
        worker_mac=worker_mac,
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
    worker_mac: str
    spine_port: int
    leaf_port: int


@dataclass(frozen=True)
class CacheSyncEvent:
    """Cache or load event emitted by a worker."""

    event_type: EventType
    worker_id: str
    prefix_hashes: tuple[int, ...] = ()
    load: int | None = None
    active_requests: int | None = None
    active_batched_tokens: int | None = None
    queued_requests: int | None = None
    queued_batched_tokens: int | None = None
    timestamp: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "endpoint": "cache_event",
            "event_type": self.event_type,
            "worker_id": self.worker_id,
        }
        if self.prefix_hashes:
            payload["prefix_hashes"] = normalize_prefix_hashes(self.prefix_hashes)
        if self.load is not None:
            payload["load"] = self.load
        if self.active_requests is not None:
            payload["active_requests"] = self.active_requests
        if self.active_batched_tokens is not None:
            payload["active_batched_tokens"] = self.active_batched_tokens
        if self.queued_requests is not None:
            payload["queued_requests"] = self.queued_requests
        if self.queued_batched_tokens is not None:
            payload["queued_batched_tokens"] = self.queued_batched_tokens
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CacheSyncEvent":
        prefix_hashes = tuple(normalize_prefix_hashes(payload.get("prefix_hashes", [])))
        raw_load = payload.get("load")
        raw_active_requests = payload.get("active_requests")
        raw_active_batched_tokens = payload.get("active_batched_tokens")
        raw_queued_requests = payload.get("queued_requests")
        raw_queued_batched_tokens = payload.get("queued_batched_tokens")
        raw_timestamp = payload.get("timestamp")
        return cls(
            event_type=str(payload.get("event_type", "alloc")),  # type: ignore
            worker_id=str(payload["worker_id"]),
            prefix_hashes=prefix_hashes,
            load=(int(raw_load) if raw_load is not None else None),
            active_requests=(
                int(raw_active_requests) if raw_active_requests is not None else None
            ),
            active_batched_tokens=(
                int(raw_active_batched_tokens)
                if raw_active_batched_tokens is not None
                else None
            ),
            queued_requests=(
                int(raw_queued_requests) if raw_queued_requests is not None else None
            ),
            queued_batched_tokens=(
                int(raw_queued_batched_tokens)
                if raw_queued_batched_tokens is not None
                else None
            ),
            timestamp=(float(raw_timestamp) if raw_timestamp is not None else None),
        )


@dataclass(slots=True)
class WorkerLoadState:
    active_requests: int = 0
    active_batched_tokens: int = 0
    queued_requests: int = 0
    queued_batched_tokens: int = 0
    load: int = 0


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
        adapter: SwitchAdapter | None = None,
        spine_switch: str = "s1",
    ) -> None:
        self._workers = {worker.worker_id: worker for worker in workers}
        self._server = UDPServer(host=host, port=port, handler=self._handle)
        self._adapter: SwitchAdapter = adapter or InMemorySwitchAdapter()
        self._spine_switch = spine_switch
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
        self.worker_loads: dict[str, WorkerLoadState] = {
            worker.worker_id: WorkerLoadState() for worker in workers
        }
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
            "worker_loads": {
                worker_id: asdict(state)
                for worker_id, state in self.worker_loads.items()
            },
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
        logger.info("Received cache event: %s", self._describe_event(event))
        ops = self.handle_event(event)
        return UDPResponse(data={"status": "ok", "n_ops": len(ops)})

    @staticmethod
    def _describe_event(event: CacheSyncEvent) -> str:
        details = [f"type={event.event_type}", f"worker={event.worker_id}"]
        if event.prefix_hashes:
            details.append(f"prefix={format_prefix_key(event.prefix_hashes)}")
        if event.load is not None:
            details.append(f"load={event.load}")
        if event.active_requests is not None:
            details.append(f"active_requests={event.active_requests}")
        if event.active_batched_tokens is not None:
            details.append(f"active_batched_tokens={event.active_batched_tokens}")
        if event.queued_requests is not None:
            details.append(f"queued_requests={event.queued_requests}")
        if event.queued_batched_tokens is not None:
            details.append(f"queued_batched_tokens={event.queued_batched_tokens}")
        return ", ".join(details)

    def handle_event(self, event: CacheSyncEvent) -> list[SwitchOp]:
        now = time.time() if event.timestamp is None else event.timestamp
        if event.worker_id not in self._workers:
            raise ValueError(f"unknown worker: {event.worker_id}")

        ops: list[SwitchOp] = []
        if event.event_type == "queue_update":
            if event.load is None:
                raise ValueError("queue_update requires load")
            self.worker_loads[event.worker_id] = WorkerLoadState(
                active_requests=event.active_requests or 0,
                active_batched_tokens=event.active_batched_tokens or 0,
                queued_requests=event.queued_requests or 0,
                queued_batched_tokens=event.queued_batched_tokens or 0,
                load=event.load,
            )
            # Load updates can change both installed rule targets and ECMP weights.
            logger.info(
                "Updated worker load for %s to load=%d active_requests=%d queued_requests=%d active_batched_tokens=%d queued_batched_tokens=%d",
                event.worker_id,
                event.load,
                event.active_requests or 0,
                event.queued_requests or 0,
                event.active_batched_tokens or 0,
                event.queued_batched_tokens or 0,
            )
            ops.extend(self._reconcile_all_rules(now))
            ops.extend(self._refresh_ecmp_weights())
            return ops

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
            ops.extend(self._maybe_install_spine_rule(spine_key, spine_hits, now))
            ops.extend(self._maybe_install_leaf_rule(leaf_key, leaf_hits, now))
            return ops

        if event.event_type == "evict":
            # Drop stale placements and reinstall a fallback only if replicas remain.
            ops.extend(self._apply_spine_eviction(spine_key, event.worker_id, now))
            ops.extend(self._apply_leaf_eviction(leaf_key, event.worker_id, now))
            ops.extend(self._refresh_ecmp_weights())
            return ops

        raise ValueError(f"unsupported event type: {event.event_type}")

    def _worker_score(self, worker_id: str) -> float:
        return self.alpha - self.beta * float(self.worker_loads[worker_id].load)

    def _select_worker(self, candidates: set[str]) -> str:
        if not candidates:
            raise ValueError("no worker candidates available")
        return max(
            candidates, key=lambda worker_id: (self._worker_score(worker_id), worker_id)
        )

    def _spine_add_op(self, prefix: tuple[int, ...], worker_id: str) -> TableAddOp:
        worker = self._workers[worker_id]
        h0 = prefix[0]
        return TableAddOp(
            switch=self._spine_switch,
            table="spine_prefix_route",
            action="route_to_pod",
            match={"hdr.kvswitch.h0": f"0x{h0:08x}&&&0xffffffff"},
            action_params={"port": worker.spine_port},
            priority=1,
        )

    @staticmethod
    def _mac_int(mac: str) -> int:
        return int(mac.replace(":", ""), 16)

    def _leaf_add_op(self, prefix: PrefixKey, worker_id: str) -> tuple[str, TableAddOp]:
        worker = self._workers[worker_id]
        match: dict[str, str | int] = {}
        for idx, field_name in enumerate(
            ("hdr.kvswitch.h0", "hdr.kvswitch.h1", "hdr.kvswitch.h2")
        ):
            value = prefix[idx] if idx < len(prefix) else 0
            mask = 0xFFFFFFFF if idx < len(prefix) else 0
            match[field_name] = f"0x{value:08x}&&&0x{mask:08x}"
        return (
            worker.leaf_switch,
            TableAddOp(
                switch=worker.leaf_switch,
                table="leaf_prefix_route",
                action="route_to_worker",
                match=match,
                action_params={
                    "port": worker.leaf_port,
                    "dst_mac": self._mac_int(worker.worker_mac),
                },
                priority=1,
            ),
        )

    def _spine_delete_op(self, prefix: tuple[int, ...]) -> TableDeleteOp:
        h0 = prefix[0]
        return TableDeleteOp(
            switch=self._spine_switch,
            table="spine_prefix_route",
            match={"hdr.kvswitch.h0": f"0x{h0:08x}&&&0xffffffff"},
            priority=1,
        )

    def _leaf_delete_op(
        self, prefix: PrefixKey, worker_id: str
    ) -> tuple[str, TableDeleteOp]:
        worker = self._workers[worker_id]
        match: dict[str, str | int] = {}
        for idx, field_name in enumerate(
            ("hdr.kvswitch.h0", "hdr.kvswitch.h1", "hdr.kvswitch.h2")
        ):
            value = prefix[idx] if idx < len(prefix) else 0
            mask = 0xFFFFFFFF if idx < len(prefix) else 0
            match[field_name] = f"0x{value:08x}&&&0x{mask:08x}"
        return (
            worker.leaf_switch,
            TableDeleteOp(
                switch=worker.leaf_switch,
                table="leaf_prefix_route",
                match=match,
                priority=1,
            ),
        )

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
    ) -> list[SwitchOp]:
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
        ops: list[SwitchOp] = []
        if evicted is not None:
            ops.append(self._spine_delete_op(evicted[0]))
            logger.info(
                "Evicting spine prefix rule prefix=%s previous_worker=%s",
                format_prefix_key(evicted[0]),
                evicted[1].target_id,
            )
        add_op = self._spine_add_op(prefix, worker_id)
        ops.append(add_op)
        logger.info(
            "Installing spine prefix rule prefix=%s worker=%s switch=%s",
            format_prefix_key(prefix),
            worker_id,
            self._spine_switch,
        )
        self._adapter.apply_ops(ops)
        return ops

    def _maybe_install_leaf_rule(
        self,
        prefix: PrefixKey,
        hit_count: int,
        now: float,
    ) -> list[SwitchOp]:
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
        switch_name, add_op = self._leaf_add_op(prefix, worker_id)
        ops: list[SwitchOp] = []
        if evicted is not None:
            evicted_switch, evict_op = self._leaf_delete_op(
                evicted[0], evicted[1].target_id
            )
            self._adapter.apply_ops([evict_op])
            logger.info(
                "Evicting leaf prefix rule prefix=%s previous_worker=%s switch=%s",
                format_prefix_key(evicted[0]),
                evicted[1].target_id,
                evicted_switch,
            )
            ops.append(evict_op)
        ops.append(add_op)
        logger.info(
            "Installing leaf prefix rule prefix=%s worker=%s switch=%s",
            format_prefix_key(prefix),
            worker_id,
            switch_name,
        )
        self._adapter.apply_ops([add_op])
        return ops

    def _apply_spine_eviction(
        self,
        prefix: tuple[int, ...],
        worker_id: str,
        now: float,
    ) -> list[SwitchOp]:
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
        op = self._spine_delete_op(prefix)
        logger.info(
            "Removing spine prefix rule prefix=%s switch=%s",
            format_prefix_key(prefix),
            self._spine_switch,
        )
        self._adapter.apply_ops([op])
        return [op]

    def _apply_leaf_eviction(
        self,
        prefix: PrefixKey,
        worker_id: str,
        now: float,
    ) -> list[SwitchOp]:
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
        switch_name, op = self._leaf_delete_op(prefix, removed.target_id)
        logger.info(
            "Removing leaf prefix rule prefix=%s worker=%s switch=%s",
            format_prefix_key(prefix),
            removed.target_id,
            switch_name,
        )
        self._adapter.apply_ops([op])
        return [op]

    def _reconcile_all_rules(self, now: float) -> list[SwitchOp]:
        ops: list[SwitchOp] = []
        # Queue shifts can change the best worker for already-admitted prefixes.
        for prefix in list(self.spine_tcam._installed.keys()):
            candidates = self._spine_locations.get(prefix, set())
            if not candidates:
                continue
            hit_count = self.spine_tcam.observation_count(prefix, now)
            ops.extend(self._maybe_install_spine_rule(prefix, hit_count, now))

        for prefix in list(self.leaf_tcam._installed.keys()):
            candidates = self._leaf_locations.get(prefix, set())
            if not candidates:
                continue
            hit_count = self.leaf_tcam.observation_count(prefix, now)
            ops.extend(self._maybe_install_leaf_rule(prefix, hit_count, now))
        return ops

    def _refresh_ecmp_weights(self) -> list[SwitchOp]:
        ops: list[SwitchOp] = []
        ops.extend(self._program_spine_ecmp())
        ops.extend(self._program_leaf_ecmp())
        return ops

    def _inverse_queue_weight(self, worker_id: str) -> float:
        return 1.0 / (1.0 + float(self.worker_loads[worker_id].load))

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

    def _program_spine_ecmp(self) -> list[SwitchOp]:
        # Misses are balanced at the pod level before leaf-level selection.
        per_leaf: dict[str, float] = defaultdict(float)
        for worker_id, placement in self._workers.items():
            per_leaf[placement.leaf_switch] += self._inverse_queue_weight(worker_id)
        bucket_map = self._allocate_buckets(per_leaf)
        leaves = {
            placement.leaf_switch: placement.spine_port
            for placement in self._workers.values()
        }
        ops: list[SwitchOp] = [
            TableClearOp(switch=self._spine_switch, table="spine_ecmp_select")
        ]
        for bucket, leaf_switch in bucket_map.items():
            ops.append(
                TableAddOp(
                    switch=self._spine_switch,
                    table="spine_ecmp_select",
                    action="route_to_pod",
                    match={"meta.ecmp_bucket": bucket},
                    action_params={"port": leaves[leaf_switch]},
                )
            )
        self._adapter.apply_ops(ops)
        return ops

    def _program_leaf_ecmp(self) -> list[SwitchOp]:
        all_ops: list[SwitchOp] = []
        # Each leaf rebalances only across its directly attached workers.
        workers_by_leaf: dict[str, dict[str, float]] = defaultdict(dict)
        for worker_id, placement in self._workers.items():
            workers_by_leaf[placement.leaf_switch][worker_id] = (
                self._inverse_queue_weight(worker_id)
            )

        for leaf_switch, weights in workers_by_leaf.items():
            bucket_map = self._allocate_buckets(weights)
            ops: list[SwitchOp] = [
                TableClearOp(switch=leaf_switch, table="leaf_ecmp_select")
            ]
            for bucket, worker_id in bucket_map.items():
                worker = self._workers[worker_id]
                ops.append(
                    TableAddOp(
                        switch=leaf_switch,
                        table="leaf_ecmp_select",
                        action="route_to_worker",
                        match={"meta.ecmp_bucket": bucket},
                        action_params={
                            "port": worker.leaf_port,
                            "dst_mac": self._mac_int(worker.worker_mac),
                        },
                    )
                )
            self._adapter.apply_ops(ops)
            all_ops.extend(ops)
        return all_ops

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
