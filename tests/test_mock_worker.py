"""Tests for kvswitch.mock.worker."""

import asyncio
import time

import pytest

from kvswitch.mock.worker import DEFAULT_MAX_NUM_SEQS, MockWorker
from kvswitch.utils.udp import UDPClient


def _get_port(worker: MockWorker) -> int:
    """Get the actual bound port from the worker's UDP server."""
    return worker._server.bound_port()


# ---------------------------------------------------------------------------
# Basic request handling
# ---------------------------------------------------------------------------


class TestMockWorkerEndpoints:
    def test_health(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"endpoint": "health"})
            assert resp["status"] == "ok"
            assert resp["active"] == 0

            worker.close()

        asyncio.run(_run())

    def test_generate(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": [1, 2, 3],
                }
            )
            assert resp["text"] == ["<mock output>"]
            assert resp["prompt_tokens"] == 3
            assert resp["simulated_ttft_ms"] == pytest.approx(1.0)
            assert resp["worker_port"] == port

            worker.close()

        asyncio.run(_run())

    def test_unknown_endpoint(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"endpoint": "unknown"})
            assert "error" in resp

            worker.close()

        asyncio.run(_run())

    def test_generate_simulates_delay(self) -> None:
        async def _run() -> None:
            ttft_ms = 50.0
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=ttft_ms)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            t0 = time.perf_counter()
            await client.send({"endpoint": "generate", "prompt_token_ids": []})
            elapsed_ms = (time.perf_counter() - t0) * 1000

            assert elapsed_ms >= ttft_ms * 0.8  # Allow some slack.
            worker.close()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Batch capacity (max_num_seqs)
# ---------------------------------------------------------------------------


class TestBatchCapacity:
    def test_default_max_num_seqs(self) -> None:
        worker = MockWorker(host="127.0.0.1", port=0)
        assert worker.max_num_seqs == DEFAULT_MAX_NUM_SEQS

    def test_concurrent_requests_within_limit(self) -> None:
        """Requests within the batch limit run concurrently."""

        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=50.0,
                max_num_seqs=5,
            )
            await worker.start()
            port = _get_port(worker)

            # Fire 5 requests concurrently; all should fit in the batch.
            async def _send() -> float:
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                t0 = time.perf_counter()
                await client.send({"endpoint": "generate", "prompt_token_ids": []})
                return (time.perf_counter() - t0) * 1000

            tasks = [asyncio.create_task(_send()) for _ in range(5)]
            latencies = await asyncio.gather(*tasks)

            # All 5 should complete in roughly one TTFT period (concurrent).
            for lat in latencies:
                assert lat < 200.0  # Well under 2x TTFT if truly concurrent.

            worker.close()

        asyncio.run(_run())

    def test_requests_beyond_limit_queue(self) -> None:
        """Requests exceeding the batch limit experience queuing delay."""

        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=50.0,
                max_num_seqs=1,
            )
            await worker.start()
            port = _get_port(worker)

            # Fire 2 requests concurrently with batch size 1.
            # Second request must wait for the first to finish.
            async def _send() -> float:
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                t0 = time.perf_counter()
                await client.send({"endpoint": "generate", "prompt_token_ids": []})
                return (time.perf_counter() - t0) * 1000

            tasks = [asyncio.create_task(_send()) for _ in range(2)]
            latencies = sorted(await asyncio.gather(*tasks))

            # First finishes in ~50ms, second in ~100ms (queued behind first).
            assert latencies[0] < 100.0
            assert latencies[1] >= 80.0  # At least ~1.5x TTFT.

            worker.close()

        asyncio.run(_run())

    def test_active_count(self) -> None:
        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=100.0,
                max_num_seqs=10,
            )
            await worker.start()
            port = _get_port(worker)

            # Fire a request and check active count while it's in-flight.
            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)

            # Start a generate request in the background.
            gen_task = asyncio.create_task(
                client.send({"endpoint": "generate", "prompt_token_ids": []})
            )
            await asyncio.sleep(0.02)  # Let it start processing.

            health_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await health_client.send({"endpoint": "health"})
            assert resp["active"] >= 1

            await gen_task
            worker.close()

        asyncio.run(_run())
