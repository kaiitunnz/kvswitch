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
from kvswitch.controller.sdn_controller import ECMP_BUCKETS, WorkerPlacement
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
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        admission_threshold=2,
        max_spine_entries=2,
        max_leaf_entries=1,
        beta=1.0,
        adapter=adapter,
        spine_switches=["s1"],
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
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix).target_id == "worker0"

    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=8, timestamp=2.5)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=2.5)
    )
    # Alloc from worker1 triggers multi-leaf expansion: leaf1 gets a rule
    # and the spine ECMP group now includes both leaves.
    expand_ops = controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=3.0)
    )
    assert len(expand_ops) > 0
    assert controller.spine_tcam.installed_rule((0x1,)) is not None
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix).target_id == "worker0"
    assert controller.leaf_tcams["leaf1"].installed_rule(prefix).target_id == "worker1"

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
    # New leaf rule for other_prefix should be added (set_leaf_prefix_ecmp_group).
    new_leaf_adds = [
        op
        for op in eviction_ops
        if isinstance(op, TableAddOp) and op.table == "leaf_prefix_route"
    ]
    assert any(
        "group_id" in op.action_params
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
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        admission_threshold=1,
        adapter=adapter,
        spine_switches=["s1"],
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

    # Second alloc from worker0 — expands to leaf0 but leaf1 rule stays.
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=3.0)
    )
    # Multi-leaf expansion: leaf0 gets a rule, spine ECMP reweighted.
    assert controller.leaf_tcams["leaf1"].installed_rule(prefix).target_id == "worker1"
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix).target_id == "worker0"


def test_controller_skips_prefix_reinstall_when_target_is_unchanged() -> None:
    adapter = InMemorySwitchAdapter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.0.0.1",
                worker_mac="02:00:00:00:00:01",
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        admission_threshold=1,
        adapter=adapter,
        spine_switches=["s1"],
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
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        adapter=adapter,
        spine_switches=["s1"],
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
                    spine_ports={"s1": 10},
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
                    spine_ports={"s1": 10},
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
            lambda: any(tcam.installed_rule(expected_prefix) is not None for tcam in controller.leaf_tcams.values()),
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
                    spine_ports={"s1": 10},
                    leaf_port=1,
                ),
                WorkerPlacement(
                    worker_id="worker1",
                    leaf_switch="leaf1",
                    worker_ip="10.0.1.1",
                    worker_mac="02:00:00:00:00:02",
                    spine_ports={"s1": 11},
                    leaf_port=2,
                ),
            ],
            adapter=adapter,
            spine_switches=["s1"],
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
                    spine_ports={"s1": 10},
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


def test_score_conditional_reroute_on_queue_update() -> None:
    """queue_update reroutes prefix rules when a cached candidate has a
    significantly better score, but holds steady for small deltas."""
    adapter = InMemorySwitchAdapter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.0.0.1",
                worker_mac="02:00:00:00:00:01",
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        admission_threshold=1,
        adapter=adapter,
        spine_switches=["s1"],
    )

    # Install prefix on both workers (cross-leaf).
    prefix = (0x1, 0x2, 0x3)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
    )
    controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=2.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)) is not None
    # Both leaves should have leaf rules.
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix) is not None
    assert controller.leaf_tcams["leaf1"].installed_rule(prefix) is not None

    # Small load delta (20 < threshold 50): leaf rules stay.
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=20, timestamp=3.0)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", load=0, timestamp=3.0)
    )
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix).target_id == "worker0"

    # Large load delta (1000 >> threshold 50): spine ECMP shifts weight
    # toward leaf1.  Leaf rules unchanged (each leaf's local best stays).
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", load=1000, timestamp=4.0)
    )
    # The spine ECMP group should now heavily favor leaf1.
    spine_key = (0x1,)
    group_id = controller._prefix_group_ids.get(spine_key)
    assert group_id is not None
    bucket_map = controller._prefix_ecmp_bucket_maps["s1"].get(spine_key, {})
    leaf1_buckets = sum(1 for leaf in bucket_map.values() if leaf == "leaf1")
    assert leaf1_buckets > ECMP_BUCKETS // 2  # Majority goes to leaf1


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
                spine_ports={"s1": 10},
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                worker_mac="02:00:00:00:00:02",
                spine_ports={"s1": 11},
                leaf_port=2,
            ),
        ],
        admission_threshold=1,
        adapter=adapter,
        spine_switches=["s1"],
    )

    prefix = (0x1, 0x2, 0x3)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
    )
    controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=2.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)) is not None

    # worker0 evicts → ECMP group shrinks to leaf1 only.
    controller.handle_event(
        CacheSyncEvent("evict", "worker0", prefix_hashes=prefix, timestamp=3.0)
    )
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"
    # After eviction, leaf0 rule should be gone.
    assert controller.leaf_tcams["leaf0"].installed_rule(prefix) is None
    assert controller.leaf_tcams["leaf1"].installed_rule(prefix) is not None


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
                    spine_ports={"s1": 10},
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
        assert any(len(tcam.snapshot()) > 0 for tcam in controller.leaf_tcams.values())

        # Worker cache populated (via health endpoint).
        health_client = UDPClient(host="127.0.0.1", port=worker.port, timeout=5.0)
        health = await health_client.send({"endpoint": "health"})
        assert health["cached_prefixes"] > 0

        worker.close()
        controller.close()

    asyncio.run(_run())


def test_oob_controller_receives_events_on_alternate_bind_address() -> None:
    """The controller can bind to any IP (e.g. an OOB management address)
    and workers can send events there.  This validates the OOB path without
    requiring actual network namespaces."""

    async def _run() -> None:
        adapter = InMemorySwitchAdapter()
        # Bind to 127.0.0.1 (simulates the control-plane IP in production).
        controller = SDNController(
            workers=[
                WorkerPlacement(
                    worker_id="worker0",
                    leaf_switch="leaf0",
                    worker_ip="10.0.0.1",
                    worker_mac="02:00:00:00:00:01",
                    spine_ports={"s1": 10},
                    leaf_port=1,
                )
            ],
            host="127.0.0.1",
            port=0,
            admission_threshold=1,
            adapter=adapter,
        )
        await controller.start()
        controller_port = controller.port

        # Worker sends events directly to the controller (no relay).
        worker = MockWorker(
            host="127.0.0.1",
            port=0,
            kvswitch_port=0,
            ttft_ms=1.0,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller_port,
            max_cached_prefixes=8,
        )
        await worker.start()
        assert worker.kvswitch_port is not None

        prefix_hashes = compute_truncated_hashes([0] * 256, b"oob-key")
        kv_client = KVSwitchUDPClient(
            host="127.0.0.1", port=worker.kvswitch_port, timeout=5.0
        )
        await kv_client.send(
            {"endpoint": "generate", "prompt_token_ids": [0] * 256, "max_tokens": 1},
            prefix_hashes=prefix_hashes,
            req_id=1,
        )

        # Events should reach the controller via the direct path.
        await _wait_until(
            lambda: controller.spine_tcam.installed_rule(tuple(prefix_hashes[:1]))
            is not None,
            timeout=2.0,
        )
        assert len(controller.spine_tcam.snapshot()) > 0

        # Verify load state was updated (queue_update events received).
        await _wait_until(
            lambda: controller.worker_loads["worker0"].load == 0,
            timeout=2.0,
        )

        worker.close()
        controller.close()

    asyncio.run(_run())


def test_multi_spine_prefix_rule_replicated_to_all_spines() -> None:
    """Prefix rules are installed on every spine switch."""
    adapter = InMemorySwitchAdapter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.2.0.1",
                worker_mac="02:00:00:00:00:01",
                leaf_port=3,
                spine_ports={"spine0": 2, "spine1": 2},
            ),
        ],
        admission_threshold=1,
        adapter=adapter,
        spine_switches=["spine0", "spine1"],
    )

    prefix = (0x1, 0x2, 0x3)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
    )

    # Both spines should have the prefix route (set_prefix_ecmp_group)
    # and per-prefix ECMP entries (route_to_pod).
    spine0_route = adapter.table_adds(switch="spine0", table="spine_prefix_route")
    spine1_route = adapter.table_adds(switch="spine1", table="spine_prefix_route")
    assert len(spine0_route) == 1
    assert len(spine1_route) == 1
    assert "group_id" in spine0_route[0].action_params
    assert "group_id" in spine1_route[0].action_params

    spine0_ecmp = adapter.table_adds(switch="spine0", table="spine_prefix_ecmp")
    spine1_ecmp = adapter.table_adds(switch="spine1", table="spine_prefix_ecmp")
    assert len(spine0_ecmp) > 0
    assert len(spine1_ecmp) > 0
    # All ECMP entries should route to port 2 (single leaf).
    assert all(op.action_params["port"] == 2 for op in spine0_ecmp)
    assert all(op.action_params["port"] == 2 for op in spine1_ecmp)


def test_multi_spine_ecmp_programmed_per_spine() -> None:
    """Each spine gets its own ECMP table with correct per-spine port mappings."""
    adapter = InMemorySwitchAdapter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.2.0.1",
                worker_mac="02:00:00:00:00:01",
                leaf_port=3,
                spine_ports={"spine0": 2, "spine1": 3},
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.2.1.1",
                worker_mac="02:00:00:00:00:02",
                leaf_port=3,
                spine_ports={"spine0": 4, "spine1": 5},
            ),
        ],
        adapter=adapter,
        spine_switches=["spine0", "spine1"],
    )

    controller._refresh_ecmp_weights()

    # Each spine should have ECMP entries.
    s0_adds = adapter.table_adds(switch="spine0", table="spine_ecmp_select")
    s1_adds = adapter.table_adds(switch="spine1", table="spine_ecmp_select")
    assert len(s0_adds) > 0
    assert len(s1_adds) > 0

    # Port numbers should differ between spines.
    s0_ports = {op.action_params["port"] for op in s0_adds}
    s1_ports = {op.action_params["port"] for op in s1_adds}
    assert s0_ports == {2, 4}
    assert s1_ports == {3, 5}
