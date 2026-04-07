"""Tests for kvswitch.mock.worker."""

import asyncio
import time

import pytest

from kvswitch.mock.worker import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_NUM_SEQS,
    MockWorker,
)
from kvswitch.sdk.client import KVSwitchUDPClient
from kvswitch.sdk.hashing import compute_truncated_hashes
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
            assert resp["active_batched_tokens"] == 0
            assert resp["queued_requests"] == 0
            assert resp["queued_batched_tokens"] == 0
            assert resp["load"] == 0
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

    def test_generate_via_kvswitch_shim_uses_header_hashes(self) -> None:
        async def _run() -> None:
            prompt_token_ids = list(range(512))
            prefix_hashes = compute_truncated_hashes(prompt_token_ids, b"kvswitch-eval")
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                kvswitch_port=0,
                ttft_ms=1.0,
            )
            await worker.start()
            assert worker.kvswitch_port is not None

            client = KVSwitchUDPClient(
                host="127.0.0.1",
                port=worker.kvswitch_port,
                timeout=5.0,
            )
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": prompt_token_ids,
                },
                prefix_hashes=prefix_hashes,
                req_id=7,
            )

            assert resp["prompt_tokens"] == 512
            assert resp["exported_prefixes"] == len(prefix_hashes)
            assert list(worker._export_prefix_cache.keys()) == [
                tuple(prefix_hashes[:1]),
                tuple(prefix_hashes[:2]),
            ]

            worker.close()

        asyncio.run(_run())

    def test_plain_udp_prefix_hashes_are_ignored_on_receive_side(self) -> None:
        async def _run() -> None:
            prompt_token_ids = list(range(512))
            prefix_hashes = compute_truncated_hashes(prompt_token_ids, b"kvswitch-eval")
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": prompt_token_ids,
                    "prefix_hashes": prefix_hashes,
                }
            )

            assert resp["prompt_tokens"] == 512
            assert resp["exported_prefixes"] == 0
            assert list(worker._export_prefix_cache.keys()) == []

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

    def test_sequential_requests_observe_cache_updates(self) -> None:
        """A request sent after a prior one completes sees the cached prefix."""

        async def _run() -> None:
            prompt_token_ids = list(range(32))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=1.0,
                max_num_seqs=1,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )
            second = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )

            assert first["matched_blocks"] == 0
            assert first["num_cached_tokens"] == 0
            assert second["matched_blocks"] == 2
            assert second["num_cached_tokens"] == 32

            worker.close()

        asyncio.run(_run())

    def test_concurrent_requests_both_miss_cache(self) -> None:
        """Two requests arriving simultaneously both miss the cache."""

        async def _run() -> None:
            prompt_token_ids = list(range(32))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=40.0,
                max_num_seqs=2,
            )
            await worker.start()
            port = _get_port(worker)

            async def _send() -> dict:
                client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                return await client.send(
                    {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
                )

            first_task = asyncio.create_task(_send())
            await asyncio.sleep(0.01)
            second_task = asyncio.create_task(_send())
            first, second = await asyncio.gather(first_task, second_task)

            # Both arrive before either completes — both miss the cache.
            assert first["matched_blocks"] == 0
            assert second["matched_blocks"] == 0

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

    def test_health_reports_active_and_queued_load_metrics(self) -> None:
        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=80.0,
                tpot_ms=10.0,
                max_num_seqs=1,
                max_num_batched_tokens=10,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first_task = asyncio.create_task(
                client.send(
                    {
                        "endpoint": "generate",
                        "prompt_token_ids": [1, 2, 3],
                        "max_tokens": 2,
                    }
                )
            )
            await asyncio.sleep(0.02)

            second_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            second_task = asyncio.create_task(
                second_client.send(
                    {
                        "endpoint": "generate",
                        "prompt_token_ids": [4, 5, 6],
                        "max_tokens": 2,
                    }
                )
            )

            async def _health() -> dict:
                health_client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                return await health_client.send({"endpoint": "health"})

            health = None
            for _ in range(40):
                health = await _health()
                if health["queued_requests"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert health is not None
            assert health["active"] == 1
            assert health["active_batched_tokens"] == 5
            assert health["queued_requests"] == 1
            assert health["queued_batched_tokens"] == 5
            assert health["load"] == 10

            await asyncio.gather(first_task, second_task)
            worker.close()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Cache-aware TTFT and token budget
# ---------------------------------------------------------------------------


class TestCacheAwareTTFT:
    def test_cache_hit_reduces_simulated_ttft(self) -> None:
        """Second request with cached prefix has lower TTFT."""

        async def _run() -> None:
            prompt_token_ids = list(range(64))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                base_ttft_ms=10.0,
                per_uncached_token_ttft_ms=0.5,
                tpot_ms=0.0,
                max_num_seqs=1,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )
            second = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )

            # First: uncached=64, TTFT = 10 + 0.5*64 = 42 ms.
            assert first["simulated_ttft_ms"] == pytest.approx(42.0)
            assert first["uncached_tokens"] == 64

            # Second: all 4 blocks cached, uncached=0, TTFT = 10 ms.
            assert second["simulated_ttft_ms"] == pytest.approx(10.0)
            assert second["uncached_tokens"] == 0

            worker.close()

        asyncio.run(_run())

    def test_partial_cache_hit(self) -> None:
        """Request sharing a prefix gets reduced TTFT for the uncached portion."""

        async def _run() -> None:
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                base_ttft_ms=10.0,
                per_uncached_token_ttft_ms=0.5,
                tpot_ms=0.0,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            await client.send(
                {"endpoint": "generate", "prompt_token_ids": list(range(64))}
            )
            resp = await client.send(
                {"endpoint": "generate", "prompt_token_ids": list(range(80))}
            )

            assert resp["matched_blocks"] == 4
            assert resp["uncached_tokens"] == 16
            assert resp["simulated_ttft_ms"] == pytest.approx(10.0 + 0.5 * 16)

            worker.close()

        asyncio.run(_run())

    def test_token_budget_uses_uncached_tokens(self) -> None:
        """Cached tokens don't count against the token budget."""

        async def _run() -> None:
            prompt = list(range(48))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                base_ttft_ms=5.0,
                per_uncached_token_ttft_ms=0.01,
                tpot_ms=0.0,
                max_num_batched_tokens=50,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt}
            )
            assert first["batched_tokens"] == 49

            new_prompt = list(range(100, 148))

            async def _send(tokens: list[int]) -> dict:
                c = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
                return await c.send(
                    {"endpoint": "generate", "prompt_token_ids": tokens}
                )

            cached_task = asyncio.create_task(_send(prompt))
            new_task = asyncio.create_task(_send(new_prompt))
            cached_resp, new_resp = await asyncio.gather(cached_task, new_task)

            assert cached_resp["batched_tokens"] == 1
            assert new_resp["batched_tokens"] == 49

            worker.close()

        asyncio.run(_run())

    def test_fallback_to_fixed_ttft(self) -> None:
        """Without linear params, TTFT stays fixed regardless of cache state."""

        async def _run() -> None:
            prompt_token_ids = list(range(64))
            worker = MockWorker(
                host="127.0.0.1",
                port=0,
                ttft_ms=25.0,
                tpot_ms=0.0,
            )
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            first = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )
            second = await client.send(
                {"endpoint": "generate", "prompt_token_ids": prompt_token_ids}
            )

            assert first["simulated_ttft_ms"] == pytest.approx(25.0)
            assert second["simulated_ttft_ms"] == pytest.approx(25.0)

            worker.close()

        asyncio.run(_run())

    def test_response_includes_uncached_tokens(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            port = _get_port(worker)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send(
                {"endpoint": "generate", "prompt_token_ids": list(range(32))}
            )
            assert resp["uncached_tokens"] == 32
            assert resp["prompt_tokens"] == 32

            worker.close()

        asyncio.run(_run())
