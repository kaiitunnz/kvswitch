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
from kvswitch.utils.udp import UDPClient


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
        CacheSyncEvent("queue_update", "worker0", load=8, timestamp=2.5)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=2.5)
    )
    # Sticky: alloc from worker1 does NOT reroute — worker0 keeps the rule.
    sticky_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=3.0)
    )
    assert sticky_ops == []
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker0"

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

    # Evict from worker1 — worker0 still owns the prefix.  worker1 is
    # removed as a candidate but the rule stays on worker0 (sticky).
    controller.handle_event(
        CacheSyncEvent("evict", "worker1", prefix_hashes=prefix, timestamp=6.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"


def test_sticky_prefix_rules_resist_alloc_from_lighter_worker() -> None:
    """Once a prefix rule is installed, a subsequent alloc from a lighter
    worker does NOT reroute — the existing owner keeps the rule."""
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
        admission_threshold=1,
        adapter=adapter,
        spine_switch="s1",
    )

    controller.handle_event(
        CacheSyncEvent(
            "queue_update",
            "worker0",
            load=12,
            active_requests=1,
            active_batched_tokens=12,
            timestamp=1.0,
        )
    )
    controller.handle_event(
        CacheSyncEvent(
            "queue_update",
            "worker1",
            load=1,
            active_requests=10,
            active_batched_tokens=1,
            timestamp=1.0,
        )
    )

    prefix = (0x1, 0x2, 0x3)
    # First alloc installs on worker1 (lower load → selected as best).
    controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=2.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"

    # Second alloc from worker0 — worker1 still owns it (sticky).
    sticky_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=3.0)
    )
    assert sticky_ops == []
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker1"


def test_controller_skips_prefix_reinstall_when_target_is_unchanged() -> None:
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
        admission_threshold=1,
        adapter=adapter,
        spine_switch="s1",
    )

    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=0, timestamp=0.5)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=0.5)
    )

    prefix = (0x53, 0x9D, 0x89)
    install_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=1.0)
    )
    assert install_ops
    op_count = len(adapter.ops)

    stable_ops = controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=2.0)
    )
    assert stable_ops == []
    assert len(adapter.ops) == op_count


def test_controller_skips_noop_ecmp_refresh() -> None:
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
        adapter=adapter,
        spine_switch="s1",
    )

    first_ops = controller._refresh_ecmp_weights()
    assert first_ops
    first_op_count = len(adapter.ops)

    second_ops = controller._refresh_ecmp_weights()
    assert second_ops == []
    assert len(adapter.ops) == first_op_count

    changed_ops = controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=8, timestamp=1.0)
    )
    assert changed_ops
    assert len(adapter.ops) > first_op_count

    stable_again = controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=8, timestamp=2.0)
    )
    assert stable_again == []


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached before timeout")


def test_mock_worker_emits_queue_backlog_metrics_to_controller() -> None:
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
            ttft_ms=80.0,
            tpot_ms=10.0,
            max_num_seqs=1,
            max_num_batched_tokens=10,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller.port,
        )
        await worker.start()
        port = worker.port

        first_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
        second_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
        first_task = asyncio.create_task(
            first_client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": [1, 2, 3],
                    "max_tokens": 2,
                }
            )
        )
        await asyncio.sleep(0.02)
        second_task = asyncio.create_task(
            second_client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": [4, 5, 6],
                    "max_tokens": 2,
                }
            )
        )

        await _wait_until(
            lambda: controller.snapshot()["worker_loads"]["worker0"]["queued_requests"]
            == 1,
            timeout=2.0,
        )
        snapshot = controller.snapshot()
        assert snapshot["worker_loads"]["worker0"] == {
            "active_requests": 1,
            "active_batched_tokens": 5,
            "queued_requests": 1,
            "queued_batched_tokens": 5,
            "load": 10,
        }

        await asyncio.gather(first_task, second_task)
        await _wait_until(
            lambda: controller.snapshot()["worker_loads"]["worker0"]["load"] == 0,
            timeout=2.0,
        )

        worker.close()
        controller.close()

    asyncio.run(_run())


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


def test_coalesced_refresh_reduces_ecmp_reprogramming() -> None:
    """With coalescing enabled, rapid queue_updates should trigger only one
    deferred reconciliation instead of one per event."""

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
            adapter=adapter,
            spine_switch="s1",
            coalesce_interval_s=0.05,
        )
        # Initial ECMP programming from start().
        await controller.start()
        initial_op_count = len(adapter.ops)

        # Fire 10 rapid queue_update events — with coalescing, these should
        # NOT each trigger a full reconciliation.
        for load in range(10):
            controller.handle_event(
                CacheSyncEvent("queue_update", "worker0", load=load, timestamp=1.0)
            )

        # Immediately after: no ops should have been flushed yet (still coalescing).
        assert len(adapter.ops) == initial_op_count

        # Wait for the coalesce interval to pass.
        await asyncio.sleep(0.1)

        # Now exactly one deferred reconciliation should have happened.
        post_coalesce_count = len(adapter.ops)
        assert post_coalesce_count > initial_op_count

        # A second burst should again coalesce into one refresh.
        for load in range(10, 20):
            controller.handle_event(
                CacheSyncEvent("queue_update", "worker1", load=load, timestamp=2.0)
            )
        assert len(adapter.ops) == post_coalesce_count
        await asyncio.sleep(0.1)
        assert len(adapter.ops) > post_coalesce_count

        controller.close()

    asyncio.run(_run())


def test_worker_load_update_throttling() -> None:
    """Workers with load_update_interval_s > 0 should suppress rapid emissions."""

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
            ttft_ms=1.0,
            tpot_ms=0.0,
            max_num_seqs=8,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller.port,
            load_update_interval_s=0.5,
        )
        await worker.start()

        client = UDPClient(host="127.0.0.1", port=worker.port, timeout=5.0)

        # Send 5 rapid requests — with 0.5s throttle, at most ~2 load updates
        # should reach the controller instead of 10+ (2 per request).
        tasks = []
        for i in range(5):
            tasks.append(
                asyncio.create_task(
                    client.send(
                        {
                            "endpoint": "generate",
                            "prompt_token_ids": [i],
                            "max_tokens": 1,
                        }
                    )
                )
            )
        await asyncio.gather(*tasks)

        # Wait for everything to settle.
        await asyncio.sleep(0.1)

        # All requests completed — load should be 0.
        snapshot = controller.snapshot()
        load_state = snapshot["worker_loads"]["worker0"]
        assert load_state["load"] == 0

        worker.close()
        controller.close()

    asyncio.run(_run())


def test_queue_update_never_reroutes_prefix_rules() -> None:
    """queue_update events should only affect ECMP weights, never reroute
    installed prefix rules — the current target has the prefix cached."""
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
        admission_threshold=1,
        adapter=adapter,
        spine_switch="s1",
    )

    # Make worker0 the clear winner so allocs install the rule on worker0.
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=10, timestamp=0.5)
    )

    # Install a prefix rule on worker0 (higher score due to worker1's load).
    prefix = (0x1, 0x2, 0x3)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
    )
    controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=2.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker0"

    # Massive load imbalance: worker0=1000, worker1=0.
    # Prefix rule must NOT move — worker0 has the cache.
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=1000, timestamp=3.0)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=3.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker0"


def test_sticky_rule_transfers_on_owner_eviction() -> None:
    """When the current owner evicts a prefix, the rule transfers to the
    remaining candidate."""
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
        admission_threshold=1,
        adapter=adapter,
        spine_switch="s1",
    )

    prefix = (0x1, 0x2, 0x3)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
    )
    controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=2.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"

    # worker0 evicts → rule should transfer to worker1.
    evict_ops = controller.handle_event(
        CacheSyncEvent("evict", "worker0", prefix_hashes=prefix, timestamp=3.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"
    assert any(
        isinstance(op, TableAddOp) and op.action_params["port"] == 11
        for op in evict_ops
    )


def test_warmup_populates_both_tcam_and_worker_cache() -> None:
    """After warm-up requests the controller TCAM and worker caches should
    both contain the prefix-group entries."""

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
            tpot_ms=0.0,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller.port,
            max_cached_prefixes=8,
            base_ttft_ms=10.0,
            per_token_ttft_ms=0.5,
        )
        await worker.start()
        assert worker.kvswitch_port is not None

        prefix_hashes = compute_truncated_hashes([0] * 256, b"test-key")
        kv_client = KVSwitchUDPClient(
            host="127.0.0.1", port=worker.kvswitch_port, timeout=5.0
        )
        request = {
            "endpoint": "generate",
            "prompt_token_ids": [0] * 256,
            "max_tokens": 1,
        }

        # First request: cold cache.
        first = await kv_client.send(request, prefix_hashes=prefix_hashes, req_id=1)
        assert first["matched_blocks"] == 0
        cold_ttft = first["simulated_ttft_ms"]

        await _wait_until(
            lambda: controller.spine_tcam.installed_rule(tuple(prefix_hashes[:1]))
            is not None,
            timeout=2.0,
        )

        # Second request: worker cache warm → cache hits and lower TTFT.
        second = await kv_client.send(request, prefix_hashes=prefix_hashes, req_id=2)
        assert second["matched_blocks"] > 0
        warm_ttft = second["simulated_ttft_ms"]
        assert warm_ttft < cold_ttft

        # TCAM rules installed.
        assert len(controller.spine_tcam.snapshot()) > 0
        assert len(controller.leaf_tcam.snapshot()) > 0

        # Worker cache populated (via health endpoint).
        health_client = UDPClient(host="127.0.0.1", port=worker.port, timeout=5.0)
        health = await health_client.send({"endpoint": "health"})
        assert health["cached_prefixes"] > 0

        worker.close()
        controller.close()

    asyncio.run(_run())
