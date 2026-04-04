"""Tests for KVSwitch controller logic and worker/controller integration."""

import asyncio
import time

from kvswitch.controller import (
    CacheSyncEvent,
    InMemorySwitchAdapter,
    SDNController,
    TableAddOp,
    TcamManager,
)
from kvswitch.controller.sdn_controller import WorkerPlacement
from kvswitch.mock.worker import MockWorker
from kvswitch.sdk.client import KVSwitchUDPClient
from kvswitch.sdk.hashing import compute_truncated_hashes


def test_tcam_manager_admits_and_evicts_lru_prefix() -> None:
    manager = TcamManager(admission_threshold=2, window_s=10.0, max_entries=1)

    assert manager.record_observation((1,), now=1.0) == 1
    assert not manager.admitted((1,), now=1.0)
    assert manager.record_observation((1,), now=2.0) == 2
    rule, evicted = manager.install((1,), "worker0", hit_count=2, now=2.0)
    assert evicted is None
    assert rule.target_id == "worker0"

    assert manager.record_observation((2,), now=3.0) == 1
    assert manager.record_observation((2,), now=4.0) == 2
    _, evicted = manager.install((2,), "worker1", hit_count=2, now=4.0)

    assert evicted is not None
    assert evicted[0] == (1,)
    assert evicted[1].target_id == "worker0"


def test_controller_installs_reroutes_and_deletes_leaf_rules_on_correct_switch() -> (
    None
):
    adapter = InMemorySwitchAdapter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.0.0.1",
                worker_mac="02:00:00:00:00:01",
                spine_port=10,
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_port=11,
                leaf_port=2,
            ),
        ],
        admission_threshold=2,
        max_spine_entries=2,
        max_leaf_entries=1,
        beta=1.0,
        adapter=adapter,
        spine_switch="s1",
    )

    prefix = (0x1, 0x2, 0x3)
    assert (
        controller.handle_event(
            CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
        )
        == []
    )

    install_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=2.0)
    )
    spine_adds = [
        op
        for op in install_ops
        if isinstance(op, TableAddOp) and op.table == "spine_prefix_route"
    ]
    leaf_adds = [
        op
        for op in install_ops
        if isinstance(op, TableAddOp) and op.table == "leaf_prefix_route"
    ]
    assert len(spine_adds) > 0
    assert len(leaf_adds) > 0
    assert adapter.table_adds(switch="s1", table="spine_prefix_route")
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker0"

    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", queue_depth=8, timestamp=2.5)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", queue_depth=0, timestamp=2.5)
    )
    reroute_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=3.0)
    )
    spine_reroutes = [
        op
        for op in reroute_ops
        if isinstance(op, TableAddOp) and op.table == "spine_prefix_route"
    ]
    leaf_reroutes = [
        op
        for op in reroute_ops
        if isinstance(op, TableAddOp) and op.table == "leaf_prefix_route"
    ]
    assert any(op.action_params["port"] == 11 for op in spine_reroutes)
    assert any(op.action_params["port"] == 2 for op in leaf_reroutes)
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker1"

    other_prefix = (0x9, 0x8, 0x7)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=other_prefix, timestamp=4.0)
    )
    eviction_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=other_prefix, timestamp=5.0)
    )
    # Old leaf rule for prefix should have been deleted.
    leaf_deletes = adapter.table_deletes(table="leaf_prefix_route")
    assert any(
        op.match.get("hdr.kvswitch.h0") == "0x00000001&&&0xffffffff"
        for op in leaf_deletes
    )
    # New leaf rule for other_prefix should be added.
    new_leaf_adds = [
        op
        for op in eviction_ops
        if isinstance(op, TableAddOp) and op.table == "leaf_prefix_route"
    ]
    assert any(
        op.action_params["port"] == 1
        and op.match.get("hdr.kvswitch.h0") == "0x00000009&&&0xffffffff"
        for op in new_leaf_adds
    )

    fallback_ops = controller.handle_event(
        CacheSyncEvent("evict", "worker1", prefix_hashes=prefix, timestamp=6.0)
    )
    spine_fallbacks = [
        op
        for op in fallback_ops
        if isinstance(op, TableAddOp) and op.table == "spine_prefix_route"
    ]
    assert any(op.action_params["port"] == 10 for op in spine_fallbacks)
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached before timeout")


def test_mock_worker_emits_cache_events_and_controller_tracks_prefixes() -> None:
    async def _run() -> None:
        adapter = InMemorySwitchAdapter()
        controller = SDNController(
            workers=[
                WorkerPlacement(
                    worker_id="worker0",
                    leaf_switch="leaf0",
                    worker_ip="10.0.0.1",
                    worker_mac="02:00:00:00:00:01",
                    spine_port=10,
                    leaf_port=1,
                )
            ],
            host="127.0.0.1",
            port=0,
            admission_threshold=1,
            adapter=adapter,
        )
        await controller.start()

        worker = MockWorker(
            host="127.0.0.1",
            port=0,
            kvswitch_port=0,
            ttft_ms=1.0,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller.port,
            max_cached_prefixes=8,
        )
        await worker.start()
        assert worker.kvswitch_port is not None

        prefix_hashes = compute_truncated_hashes([0] * 256, b"test-key")
        client = KVSwitchUDPClient(
            host="127.0.0.1", port=worker.kvswitch_port, timeout=5.0
        )
        request = {
            "endpoint": "generate",
            "prompt_token_ids": [0] * 256,
            "max_tokens": 1,
        }

        first = await client.send(request, prefix_hashes=prefix_hashes, req_id=7)
        assert first["matched_blocks"] == 0
        assert first["num_cached_tokens"] == 0

        expected_prefix = tuple(prefix_hashes)
        await _wait_until(
            lambda: controller.leaf_tcam.installed_rule(expected_prefix) is not None,
            timeout=2.0,
        )

        second = await client.send(request, prefix_hashes=prefix_hashes, req_id=8)
        assert second["matched_blocks"] == 16
        assert second["num_cached_tokens"] == 256

        snapshot = controller.snapshot()
        assert len(snapshot["spine_rules"]) > 0
        assert len(snapshot["leaf_rules"]) > 0

        worker.close()
        controller.close()

    asyncio.run(_run())
