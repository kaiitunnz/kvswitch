"""Tests for kvswitch.mock.worker."""

import asyncio
import time

import pytest

from kvswitch.mock.worker import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_NUM_SEQS,
    MockWorker,
)
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
            assert resp["cached_prefixes"] == 0
            assert resp["exported_prefixes"] == 0

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

    def test_default_block_size_matches_vllm_cuda_default(self) -> None:
        worker = MockWorker(host="127.0.0.1", port=0)
        assert worker.block_size == DEFAULT_BLOCK_SIZE == 16

    def test_generate_uses_full_cacheable_blocks(self) -> None:
        async def _run() -> None:
            prompt_token_ids = list(range(40))
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": prompt_token_ids,
                }
            )
            second = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": prompt_token_ids,
                }
            )

            assert first["matched_blocks"] == 0
            assert first["num_cached_tokens"] == 0
            assert first["cached_prefixes"] == 2
            assert second["matched_blocks"] == 2
            assert second["num_cached_tokens"] == 32
            assert second["cached_prefixes"] == 2

            worker.close()

        asyncio.run(_run())

    def test_generate_simulates_ttft_plus_tpot_delay(self) -> None:
        async def _run() -> None:
            ttft_ms = 30.0
            tpot_ms = 15.0
            max_tokens = 4
            worker = MockWorker(
                host="127.0.0.1", port=0, ttft_ms=ttft_ms, tpot_ms=tpot_ms
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            t0 = time.perf_counter()
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": [],
                    "max_tokens": max_tokens,
                }
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            expected_ms = ttft_ms + (max_tokens - 1) * tpot_ms
            assert resp["output_tokens"] == max_tokens
            assert resp["simulated_ttft_ms"] == pytest.approx(ttft_ms)
            assert resp["simulated_tpot_ms"] == pytest.approx(tpot_ms)
            assert resp["simulated_e2e_ms"] == pytest.approx(expected_ms)
            assert elapsed_ms >= expected_ms * 0.8  # Allow some slack.
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

    def test_queued_request_observes_cache_updates_from_prior_request(self) -> None:
        async def _run() -> None:
            prompt_token_ids = list(range(32))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=40.0,
                max_num_seqs=1,
            )
            await worker.start()
            port = _get_port(worker)

            async def _send() -> dict:
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                return await client.send(
                    {
                        "endpoint": "generate",
                        "prompt_token_ids": prompt_token_ids,
                    }
                )

            first_task = asyncio.create_task(_send())
            await asyncio.sleep(0.01)
            second_task = asyncio.create_task(_send())
            first, second = await asyncio.gather(first_task, second_task)

            assert first["matched_blocks"] == 0
            assert first["num_cached_tokens"] == 0
            assert second["matched_blocks"] == 2
            assert second["num_cached_tokens"] == 32

            worker.close()

        asyncio.run(_run())

    def test_token_budget_limits_concurrency(self) -> None:
        """Requests queue when max_num_batched_tokens would be exceeded."""

        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=40.0,
                tpot_ms=10.0,
                max_num_seqs=10,
                max_num_batched_tokens=6,
            )
            await worker.start()
            port = _get_port(worker)

            async def _send(prompt_token_ids: list[int], max_tokens: int) -> float:
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                t0 = time.perf_counter()
                await client.send(
                    {
                        "endpoint": "generate",
                        "prompt_token_ids": prompt_token_ids,
                        "max_tokens": max_tokens,
                    }
                )
                return (time.perf_counter() - t0) * 1000

            tasks = [
                asyncio.create_task(_send([1, 2], 2)),
                asyncio.create_task(_send([3, 4], 2)),
            ]
            latencies = sorted(await asyncio.gather(*tasks))

            assert latencies[0] < 100.0
            assert latencies[1] >= 70.0
            worker.close()

        asyncio.run(_run())

    def test_rejects_request_exceeding_token_budget(self) -> None:
        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=1.0,
                max_num_batched_tokens=3,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": [1, 2, 3],
                    "max_tokens": 2,
                }
            )

            assert "error" in resp
            assert "max_num_batched_tokens" in resp["error"]
            worker.close()

        asyncio.run(_run())

    def test_health_reports_active_batched_tokens(self) -> None:
        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=80.0,
                tpot_ms=10.0,
                max_num_batched_tokens=10,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            gen_task = asyncio.create_task(
                client.send(
                    {
                        "endpoint": "generate",
                        "prompt_token_ids": [1, 2, 3],
                        "max_tokens": 2,
                    }
                )
            )
            await asyncio.sleep(0.02)

            health_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await health_client.send({"endpoint": "health"})
            assert resp["active"] >= 1
            assert resp["active_batched_tokens"] >= 5

            await gen_task
            worker.close()

        asyncio.run(_run())
