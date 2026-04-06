"""L7 prefix-aware routing proxy.

Sits between client and workers as a UDP forwarding service:

1. Receive request from client
2. Tokenize prompt, hash blocks, lookup longest prefix match (L7Router)
3. Forward request to the selected worker
4. Return worker's response to client

Usage::

    python -m kvswitch.router.l7_proxy \\
        --model Qwen/Qwen2.5-1.5B \\
        --port 9000 \\
        --workers 10.0.0.1:8000,10.0.0.2:8000
"""

import asyncio
import logging
import time

from kvswitch.router.l7_router import L7Router, RoutingResult
from kvswitch.utils.udp import UDPClient, UDPRequest, UDPResponse, UDPServer

logger = logging.getLogger(__name__)


class L7Proxy:
    """UDP proxy that performs L7 routing (prefix-aware or round-robin).

    Parameters
    ----------
    model:
        HuggingFace model name for the tokenizer.
    host:
        Bind address for the proxy.
    port:
        Bind port for the proxy.
    workers:
        List of `(host, port)` tuples for backend workers.
    block_size:
        Token block size for prefix hashing.
    client_timeout:
        Timeout in seconds for forwarding requests to workers.
    round_robin:
        When True, skip prefix-aware routing and cycle through workers.
    """

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 9000,
        workers: list[tuple[str, int]] | None = None,
        block_size: int = 16,
        client_timeout: float = 30.0,
        round_robin: bool = False,
    ) -> None:
        workers = workers or [("127.0.0.1", 8000)]
        self.router = L7Router(
            model=model,
            n_workers=len(workers),
            block_size=block_size,
        )
        self.workers = workers
        self.client_timeout = client_timeout
        self._round_robin = round_robin
        self._rr_counter = 0
        self._server = UDPServer(host=host, port=port, handler=self._handle)

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        endpoint = request.data.get("endpoint", "generate")

        if endpoint == "health":
            return UDPResponse(data={"status": "ok"})

        if endpoint != "generate":
            return UDPResponse(data={"error": f"unknown endpoint: {endpoint}"})

        # --- L7 routing pipeline ---
        t_start = time.perf_counter()

        routing_result: RoutingResult | None = None
        if self._round_robin:
            worker_idx = self._rr_counter % len(self.workers)
            self._rr_counter += 1
        else:
            prompt = request.data.get("prompt")
            prompt_token_ids = request.data.get("prompt_token_ids")

            if prompt is not None and prompt_token_ids is not None:
                # Both fields present: tokenize the raw prompt to simulate
                # realistic L7 tokenization delay, but use the original
                # prompt_token_ids for the actual routing decision and
                # forwarding to workers.
                routing_result = self.router.route(prompt, prompt_token_ids)
            elif prompt is not None:
                routing_result = self.router.route(prompt)
            elif prompt_token_ids is not None:
                routing_result = self.router.route_token_ids(prompt_token_ids)
            else:
                return UDPResponse(data={"error": "missing prompt or prompt_token_ids"})

            worker_idx = routing_result.worker_idx

        worker_host, worker_port = self.workers[worker_idx]

        # Forward request to the selected worker.
        fwd_client = UDPClient(
            host=worker_host,
            port=worker_port,
            timeout=self.client_timeout,
        )
        worker_resp = await fwd_client.send(request.data)

        total_ms = (time.perf_counter() - t_start) * 1000

        # Annotate response with routing metadata.
        if self._round_robin:
            worker_resp["routing"] = {
                "worker_idx": worker_idx,
                "matched_blocks": 0,
                "proxy_total_ms": total_ms,
            }
        else:
            assert routing_result is not None
            self.router.update_cache(worker_idx, routing_result.block_hashes)
            worker_resp["routing"] = {
                "worker_idx": worker_idx,
                "matched_blocks": routing_result.matched_blocks,
                "total_blocks": routing_result.total_blocks,
                "tokenize_ms": routing_result.tokenize_ms,
                "hash_ms": routing_result.hash_ms,
                "lookup_ms": routing_result.lookup_ms,
                "routing_ms": routing_result.total_ms,
                "proxy_total_ms": total_ms,
            }

        return UDPResponse(data=worker_resp)

    async def start(self) -> None:
        await self._server.start()
        logger.info(
            "L7 proxy listening on port %d, routing to %d worker(s)",
            self._server.port,
            len(self.workers),
        )

    def close(self) -> None:
        self._server.close()

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        finally:
            self.close()


def _parse_workers(s: str) -> list[tuple[str, int]]:
    """Parse 'host:port,host:port,...' into a list of tuples."""
    workers = []
    for entry in s.split(","):
        host, port_str = entry.strip().rsplit(":", 1)
        workers.append((host, int(port_str)))
    return workers


if __name__ == "__main__":
    import argparse

    from kvswitch.utils.logger import setup_logging

    parser = argparse.ArgumentParser(description="L7 prefix-aware routing proxy")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--workers",
        type=str,
        default="127.0.0.1:8000",
        help="Comma-separated host:port pairs for backend workers",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--round-robin", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    asyncio.run(
        L7Proxy(
            model=args.model,
            host=args.host,
            port=args.port,
            workers=_parse_workers(args.workers),
            block_size=args.block_size,
            round_robin=args.round_robin,
        ).run_forever()
    )
