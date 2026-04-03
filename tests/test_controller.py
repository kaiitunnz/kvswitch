"""Tests for KVSwitch controller logic and worker/controller integration."""

import asyncio
import time

from kvswitch.controller import (
    CacheSyncEvent,
    InMemorySwitchWriter,
    SDNController,
    TcamManager,
)
from kvswitch.controller.sdn_controller import WorkerPlacement
from kvswitch.mock.worker import MockWorker
from kvswitch.sdk.header import KVSwitchShimHeader
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
    writer = InMemorySwitchWriter()
    controller = SDNController(
        workers=[
            WorkerPlacement(
                worker_id="worker0",
                leaf_switch="leaf0",
                worker_ip="10.0.0.1",
                spine_port=10,
                leaf_port=1,
            ),
            WorkerPlacement(
                worker_id="worker1",
                leaf_switch="leaf1",
                worker_ip="10.0.1.1",
                spine_port=11,
                leaf_port=2,
            ),
        ],
        admission_threshold=2,
        max_spine_entries=2,
        max_leaf_entries=1,
        beta=1.0,
        writer=writer,
    )

    prefix = (0x1, 0x2, 0x3)
    assert (
        controller.handle_event(
            CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=1.0)
        )
        == []
    )

    install_commands = controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=prefix, timestamp=2.0)
    )
    assert any("spine_prefix_route" in command for command in install_commands)
    assert any("leaf_prefix_route" in command for command in install_commands)
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker0"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker0"

    controller.handle_event(
        CacheSyncEvent("queue_update", "worker0", queue_depth=8, timestamp=2.5)
    )
    controller.handle_event(
        CacheSyncEvent("queue_update", "worker1", queue_depth=0, timestamp=2.5)
    )
    reroute_commands = controller.handle_event(
        CacheSyncEvent("alloc", "worker1", prefix_hashes=prefix, timestamp=3.0)
    )
    assert any(command.endswith("=> 11") for command in reroute_commands)
    assert any(command.endswith("=> 2") for command in reroute_commands)
    assert controller.spine_tcam.installed_rule((0x1,)).target_id == "worker1"
    assert controller.leaf_tcam.installed_rule(prefix).target_id == "worker1"

    other_prefix = (0x9, 0x8, 0x7)
    controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=other_prefix, timestamp=4.0)
    )
    eviction_commands = controller.handle_event(
        CacheSyncEvent("alloc", "worker0", prefix_hashes=other_prefix, timestamp=5.0)
    )
    assert any(
        command == "table_delete leaf_prefix_route 00000001.00000002.00000003"
        for command in writer.commands_by_switch["leaf1"]
    )
    assert any(
        command
        == "table_add leaf_prefix_route route_to_worker 0x00000009&&&0xffffffff 0x00000008&&&0xffffffff 0x00000007&&&0xffffffff => 1"
        for command in eviction_commands
    )

    fallback_commands = controller.handle_event(
        CacheSyncEvent("evict", "worker1", prefix_hashes=prefix, timestamp=6.0)
    )
    assert any(command.endswith("=> 10") for command in fallback_commands)
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
        writer = InMemorySwitchWriter()
        controller = SDNController(
            workers=[
                WorkerPlacement(
                    worker_id="worker0",
                    leaf_switch="leaf0",
                    worker_ip="10.0.0.1",
                    spine_port=10,
                    leaf_port=1,
                )
            ],
            host="127.0.0.1",
            port=0,
            admission_threshold=1,
            writer=writer,
        )
        await controller.start()

        worker = MockWorker(
            host="127.0.0.1",
            port=0,
            ttft_ms=1.0,
            worker_id="worker0",
            controller_host="127.0.0.1",
            controller_port=controller.port,
            max_cached_prefixes=8,
        )
        await worker.start()

        client = UDPClient(host="127.0.0.1", port=worker.port, timeout=5.0)
        header = KVSwitchShimHeader.from_hashes(
            [0x11111111, 0x22222222, 0x33333333], req_id=7
        ).to_dict()
        request = {
            "endpoint": "generate",
            "prompt_token_ids": [0] * 256,
            "max_tokens": 1,
            "kvswitch_header": header,
        }

        first = await client.send(request)
        assert first["matched_blocks"] == 0
        assert first["num_cached_tokens"] == 0

        await _wait_until(
            lambda: controller.leaf_tcam.installed_rule(
                (0x11111111, 0x22222222, 0x33333333)
            )
            is not None
        )

        second = await client.send(request)
        assert second["matched_blocks"] == 16
        assert second["num_cached_tokens"] == 256

        snapshot = controller.snapshot()
        assert "11111111" in snapshot["spine_rules"]
        assert "11111111.22222222.33333333" in snapshot["leaf_rules"]

        worker.close()
        controller.close()

    asyncio.run(_run())
