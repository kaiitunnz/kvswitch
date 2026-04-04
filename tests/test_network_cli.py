"""Tests for network CLI helpers used by Mininet experiments."""

import asyncio
import json
import time

from kvswitch.eval.metrics import RequestMetric, compute_summary
from kvswitch.mock.worker import MockWorker
from kvswitch.network.cli.healthcheck import check_health
from kvswitch.network.cli.healthcheck import main as healthcheck_main
from kvswitch.network.cli.measure_client import main as measure_client_main
from kvswitch.network.cli.measure_client import (
    measure_latencies,
)
from kvswitch.network.cli.udp_relay import UDPRelay
from kvswitch.network.cli.workload_client import _estimate_ttft_ms, _send_one
from kvswitch.utils.udp import UDPClient, UDPRequest, UDPResponse, UDPServer


def _run_worker(coro):
    async def _run() -> None:
        worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
        await worker.start()
        try:
            await coro(worker.port)
        finally:
            worker.close()

    asyncio.run(_run())


class TestHealthcheckCli:
    def test_check_health(self) -> None:
        async def _assert(port: int) -> None:
            assert await check_health("127.0.0.1", port, timeout=5.0) == "ok"

        _run_worker(_assert)

    def test_main_prints_status(self, capsys, monkeypatch) -> None:
        async def _fake_check_health(host: str, port: int, timeout: float) -> str:
            assert host == "127.0.0.1"
            assert port == 8000
            assert timeout == 2.0
            return "ok"

        monkeypatch.setattr(
            "kvswitch.network.cli.healthcheck.check_health", _fake_check_health
        )

        healthcheck_main(["--host", "127.0.0.1", "--port", "8000"])
        assert capsys.readouterr().out.strip() == "ok"


class TestUdpRelay:
    def test_udp_relay_forwards_request_and_response(self) -> None:
        async def _assert() -> None:
            async def _upstream_handler(request: UDPRequest) -> UDPResponse:
                return UDPResponse(
                    data={
                        "status": "ok",
                        "endpoint": request.data.get("endpoint"),
                        "payload": request.data.get("value"),
                    }
                )

            upstream = UDPServer(host="127.0.0.1", port=0, handler=_upstream_handler)
            await upstream.start()
            relay = UDPRelay(
                host="127.0.0.1",
                port=0,
                upstream_host="127.0.0.1",
                upstream_port=upstream.port,
                timeout=2.0,
            )
            await relay.start()
            try:
                client = UDPClient(host="127.0.0.1", port=relay.bound_port, timeout=2.0)
                response = await client.send({"endpoint": "health", "value": 7})
            finally:
                relay.close()
                upstream.close()

            assert response == {"status": "ok", "endpoint": "health", "payload": 7}

        asyncio.run(_assert())


class TestMeasureClientCli:
    def test_measure_latencies(self) -> None:
        async def _assert(port: int) -> None:
            results = await measure_latencies(
                host="127.0.0.1",
                port=port,
                n=3,
                prompt_tokens=8,
                timeout=5.0,
            )
            assert len(results) == 3
            assert all(latency >= 0.0 for latency in results)

        _run_worker(_assert)

    def test_send_one_kvswitch_uses_shim_not_json_prefix_hashes(
        self, monkeypatch
    ) -> None:
        seen: dict[str, object] = {}

        async def _fake_send(
            self,
            data: dict,
            prefix_hashes: list[int] | None = None,
            req_id: int = 0,
        ) -> dict:
            seen["data"] = data
            seen["prefix_hashes"] = prefix_hashes
            seen["req_id"] = req_id
            return {
                "worker_id": "worker0",
                "matched_blocks": 0,
                "simulated_ttft_ms": 5.0,
                "simulated_e2e_ms": 4.0,
            }

        monkeypatch.setattr("kvswitch.sdk.client.KVSwitchUDPClient.send", _fake_send)

        result = asyncio.run(
            _send_one(
                {
                    "request_id": 11,
                    "prompt_token_ids": [1, 2, 3],
                    "max_tokens": 4,
                    "prefix_hashes": [10, 20],
                    "scheduled_time": 0.0,
                    "prefix_group": "group_0",
                },
                host="127.0.0.1",
                port=4789,
                timeout=5.0,
                t0=time.perf_counter(),
                kvswitch=True,
            )
        )

        assert seen["data"] == {
            "endpoint": "generate",
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 4,
        }
        assert seen["prefix_hashes"] == [10, 20]
        assert seen["req_id"] == 11
        assert result["worker_id"] == "worker0"
        assert result["ttft_ms"] >= 5.0

    def test_estimate_ttft_adds_non_negative_residual(self) -> None:
        assert (
            _estimate_ttft_ms(
                12.0,
                {"simulated_ttft_ms": 5.0, "simulated_e2e_ms": 9.0},
            )
            == 8.0
        )
        assert (
            _estimate_ttft_ms(
                7.0,
                {"simulated_ttft_ms": 5.0, "simulated_e2e_ms": 9.0},
            )
            == 5.0
        )
        assert _estimate_ttft_ms(7.0, {"error": "timeout"}) is None

    def test_compute_summary_uses_client_ttft(self) -> None:
        metrics = [
            RequestMetric(
                request_id=1,
                baseline="kvswitch",
                e2e_latency_ms=12.0,
                ttft_ms=8.0,
                simulated_ttft_ms=5.0,
                simulated_tpot_ms=2.0,
                simulated_e2e_ms=9.0,
                routing_overhead_ms=0.0,
                matched_blocks=1,
                worker_id="worker0",
                prompt_tokens=16,
                output_tokens=2,
                prefix_group="group_0",
                scheduled_time_s=0.0,
                actual_send_time_s=0.0,
            ),
            RequestMetric(
                request_id=2,
                baseline="kvswitch",
                e2e_latency_ms=20.0,
                ttft_ms=11.0,
                simulated_ttft_ms=6.0,
                simulated_tpot_ms=3.0,
                simulated_e2e_ms=15.0,
                routing_overhead_ms=0.0,
                matched_blocks=0,
                worker_id="worker1",
                prompt_tokens=16,
                output_tokens=2,
                prefix_group="group_1",
                scheduled_time_s=0.0,
                actual_send_time_s=1.0,
            ),
        ]

        summary = compute_summary(metrics)

        assert summary["ttft_mean_ms"] == 9.5
        assert summary["ttft_p50_ms"] == 9.5

    def test_main_prints_json(self, capsys, monkeypatch) -> None:
        async def _fake_measure_latencies(
            host: str,
            port: int,
            n: int,
            prompt_tokens: int,
            timeout: float,
        ) -> list[float]:
            assert host == "127.0.0.1"
            assert port == 8000
            assert n == 2
            assert prompt_tokens == 4
            assert timeout == 10.0
            return [1.0, 2.0]

        monkeypatch.setattr(
            "kvswitch.network.cli.measure_client.measure_latencies",
            _fake_measure_latencies,
        )

        measure_client_main(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--n",
                "2",
                "--prompt-tokens",
                "4",
            ]
        )
        output = capsys.readouterr().out.strip()
        results = json.loads(output)
        assert results == [1.0, 2.0]
