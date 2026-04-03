"""Tests for network CLI helpers used by Mininet experiments."""

import asyncio
import json

from kvswitch.mock.worker import MockWorker
from kvswitch.network.cli.healthcheck import check_health
from kvswitch.network.cli.healthcheck import main as healthcheck_main
from kvswitch.network.cli.measure_client import main as measure_client_main
from kvswitch.network.cli.measure_client import (
    measure_latencies,
)


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
