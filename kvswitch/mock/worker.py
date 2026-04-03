"""Mock LLM worker that simulates TTFT from profiling traces.

Listens on UDP, receives generate requests, sleeps for the profiled
TTFT duration, and returns a canned response.  No GPU required.

A semaphore limits the number of concurrent requests to `max_num_seqs`
(default 256, matching vLLM's default for API server usage).  Requests
that arrive while the batch is full queue behind the semaphore, adding
realistic queuing delay.

Usage::

    python -m kvswitch.mock.worker --port 8000 --ttft-ms 15.0 --max-num-seqs 256
"""

import asyncio
import logging

from kvswitch.utils.udp import UDPRequest, UDPResponse, UDPServer

logger = logging.getLogger(__name__)

# vLLM default max_num_seqs for UsageContext.OPENAI_API_SERVER on GPU.
DEFAULT_MAX_NUM_SEQS = 256


class MockWorker:
    """Trace-driven mock LLM worker.

    Parameters
    ----------
    host:
        Bind address.
    port:
        Bind port.
    ttft_ms:
        Simulated TTFT in milliseconds.  Every generate request will
        sleep this long before responding.
    max_num_seqs:
        Maximum number of concurrent in-flight requests (batch capacity).
        When full, new requests queue until a slot opens.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        ttft_ms: float = 10.0,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
    ) -> None:
        self.host = host
        self.port = port
        self.ttft_s = ttft_ms / 1000.0
        self.max_num_seqs = max_num_seqs
        self._batch_sem = asyncio.Semaphore(max_num_seqs)
        self._active: int = 0
        self._server = UDPServer(host=host, port=port, handler=self._handle)

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        endpoint = request.data.get("endpoint", "generate")

        if endpoint == "health":
            return UDPResponse(data={"status": "ok", "active": self._active})

        if endpoint == "generate":
            async with self._batch_sem:
                self._active += 1
                try:
                    # Simulate prefill latency.
                    await asyncio.sleep(self.ttft_s)
                finally:
                    self._active -= 1

            prompt_token_ids = request.data.get("prompt_token_ids", [])
            n_tokens = len(prompt_token_ids) if prompt_token_ids else 0
            return UDPResponse(
                data={
                    "text": ["<mock output>"],
                    "num_cached_tokens": 0,
                    "worker_port": self.port,
                    "simulated_ttft_ms": self.ttft_s * 1000,
                    "prompt_tokens": n_tokens,
                }
            )

        return UDPResponse(data={"error": f"unknown endpoint: {endpoint}"})

    async def start(self) -> None:
        await self._server.start()
        self.port = self._server.bound_port()
        logger.info(
            "Mock worker listening on %s:%d (ttft=%.1fms, max_num_seqs=%d)",
            self.host,
            self.port,
            self.ttft_s * 1000,
            self.max_num_seqs,
        )

    def close(self) -> None:
        self._server.close()

    async def run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        finally:
            self.close()


if __name__ == "__main__":
    import argparse

    from kvswitch.utils.logger import setup_logging

    parser = argparse.ArgumentParser(description="Mock LLM worker")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ttft-ms", type=float, default=10.0)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="Max concurrent requests (batch capacity)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    asyncio.run(
        MockWorker(
            host=args.host,
            port=args.port,
            ttft_ms=args.ttft_ms,
            max_num_seqs=args.max_num_seqs,
        ).run_forever()
    )
