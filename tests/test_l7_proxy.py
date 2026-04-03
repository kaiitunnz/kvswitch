"""Tests for kvswitch.router.l7_proxy — integration with mock worker."""

import asyncio

from kvswitch.mock.worker import MockWorker
from kvswitch.router.l7_proxy import L7Proxy, _parse_workers
from kvswitch.utils.udp import UDPClient

MODEL = "Qwen/Qwen2.5-1.5B"


def _get_port(server_obj) -> int:
    """Get the actual bound port from a UDPServer-backed object."""
    return server_obj._server.bound_port()


# ---------------------------------------------------------------------------
# _parse_workers helper
# ---------------------------------------------------------------------------


class TestParseWorkers:
    def test_single(self) -> None:
        assert _parse_workers("10.0.0.1:8000") == [("10.0.0.1", 8000)]

    def test_multiple(self) -> None:
        result = _parse_workers("10.0.0.1:8000,10.0.0.2:9000")
        assert result == [("10.0.0.1", 8000), ("10.0.0.2", 9000)]

    def test_with_spaces(self) -> None:
        result = _parse_workers("  10.0.0.1:8000 , 10.0.0.2:9000  ")
        assert result == [("10.0.0.1", 8000), ("10.0.0.2", 9000)]


# ---------------------------------------------------------------------------
# L7Proxy health endpoint
# ---------------------------------------------------------------------------


class TestL7ProxyHealth:
    def test_health(self) -> None:
        async def _run() -> None:
            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", 19999)],
            )
            await proxy.start()
            port = _get_port(proxy)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"endpoint": "health"})
            assert resp["status"] == "ok"

            proxy.close()

        asyncio.run(_run())

    def test_unknown_endpoint(self) -> None:
        async def _run() -> None:
            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", 19999)],
            )
            await proxy.start()
            port = _get_port(proxy)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"endpoint": "foobar"})
            assert "error" in resp

            proxy.close()

        asyncio.run(_run())

    def test_missing_prompt(self) -> None:
        async def _run() -> None:
            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", 19999)],
            )
            await proxy.start()
            port = _get_port(proxy)

            client = UDPClient(host="127.0.0.1", port=port, timeout=5.0)
            resp = await client.send({"endpoint": "generate"})
            assert "error" in resp

            proxy.close()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# End-to-end: proxy → mock worker
# ---------------------------------------------------------------------------


class TestL7ProxyE2E:
    def test_forward_with_token_ids(self) -> None:
        async def _run() -> None:
            # Start mock worker.
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            worker_port = _get_port(worker)

            # Start proxy pointing at the worker.
            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", worker_port)],
            )
            await proxy.start()
            proxy_port = _get_port(proxy)

            # Send request through the proxy.
            client = UDPClient(host="127.0.0.1", port=proxy_port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": list(range(64)),
                }
            )

            assert resp["text"] == ["<mock output>"]
            assert "routing" in resp
            assert resp["routing"]["worker_idx"] == 0
            assert resp["routing"]["tokenize_ms"] == 0.0  # token_ids skips tokenize
            assert resp["routing"]["proxy_total_ms"] > 0

            proxy.close()
            worker.close()

        asyncio.run(_run())

    def test_forward_with_text_prompt(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            worker_port = _get_port(worker)

            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", worker_port)],
            )
            await proxy.start()
            proxy_port = _get_port(proxy)

            client = UDPClient(host="127.0.0.1", port=proxy_port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt": "Hello, world!",
                }
            )

            assert resp["text"] == ["<mock output>"]
            assert resp["routing"]["tokenize_ms"] > 0.0  # text prompt triggers tokenize

            proxy.close()
            worker.close()

        asyncio.run(_run())

    def test_routing_metadata_fields(self) -> None:
        async def _run() -> None:
            worker = MockWorker(host="127.0.0.1", port=0, ttft_ms=1.0)
            await worker.start()
            worker_port = _get_port(worker)

            proxy = L7Proxy(
                model=MODEL,
                host="127.0.0.1",
                port=0,
                workers=[("127.0.0.1", worker_port)],
            )
            await proxy.start()
            proxy_port = _get_port(proxy)

            client = UDPClient(host="127.0.0.1", port=proxy_port, timeout=5.0)
            resp = await client.send(
                {
                    "endpoint": "generate",
                    "prompt_token_ids": list(range(32)),
                }
            )

            routing = resp["routing"]
            expected_keys = {
                "worker_idx",
                "matched_blocks",
                "total_blocks",
                "tokenize_ms",
                "hash_ms",
                "lookup_ms",
                "routing_ms",
                "proxy_total_ms",
            }
            assert set(routing.keys()) == expected_keys

            proxy.close()
            worker.close()

        asyncio.run(_run())
