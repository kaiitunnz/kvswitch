"""Mock LLM worker that simulates TTFT/TPOT, cache state, and sync events.

Listens on UDP, receives generate requests, sleeps for a synthetic end-to-end
latency, and returns a canned response. No GPU required.

Concurrency is limited by both:
- `max_num_seqs`: maximum number of in-flight requests
- `max_num_batched_tokens`: optional total token budget across in-flight requests

The worker maintains two related but distinct views of prefix state:
- a vLLM-like local cache keyed by all cumulative 16-token block hashes
- a KVSwitch export cache keyed by the coarse shim-header prefixes sent to the SDN
  controller (`alloc`, `evict`, `queue_update`)
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Final

from kvswitch.sdk.header import KVSwitchShimHeader
from kvswitch.utils.prefix import (
    cumulative_sha256_chain,
    normalize_prefix_hashes,
    prefix_chain,
)
from kvswitch.utils.udp import UDPClient, UDPRequest, UDPResponse, UDPServer

logger = logging.getLogger(__name__)

# vLLM default max_num_seqs for UsageContext.OPENAI_API_SERVER on GPU.
DEFAULT_MAX_NUM_SEQS: Final[int] = 256
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 1
DEFAULT_BLOCK_SIZE: Final[int] = 16
MAX_KVSWITCH_HASHES: Final[int] = 4


class MockWorker:
    """Trace-driven mock LLM worker."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        ttft_ms: float = 10.0,
        tpot_ms: float = 0.0,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        worker_id: str | None = None,
        controller_host: str | None = None,
        controller_port: int | None = None,
        max_cached_prefixes: int | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.ttft_s = ttft_ms / 1000.0
        self.tpot_s = tpot_ms / 1000.0
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.worker_id = worker_id or f"{host}:{port}"
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.max_cached_prefixes = max_cached_prefixes

        self._capacity_cond = asyncio.Condition()
        self._active: int = 0
        self._active_batched_tokens: int = 0
        self._local_block_cache: OrderedDict[bytes, None] = OrderedDict()
        self._export_prefix_cache: OrderedDict[tuple[int, ...], None] = OrderedDict()
        self._server = UDPServer(host=host, port=port, handler=self._handle)

    @staticmethod
    def _request_output_tokens(request: UDPRequest) -> int:
        raw = request.data.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        try:
            return max(int(raw), 1)
        except (TypeError, ValueError):
            return DEFAULT_MAX_OUTPUT_TOKENS

    @classmethod
    def _request_batched_tokens(cls, request: UDPRequest) -> int:
        prompt_token_ids = request.data.get("prompt_token_ids", [])
        prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0
        output_tokens = cls._request_output_tokens(request)
        return max(prompt_tokens + output_tokens, 1)

    def _local_block_hashes(self, prompt_token_ids: list[int]) -> list[bytes]:
        return cumulative_sha256_chain(prompt_token_ids, self.block_size)

    @staticmethod
    def _extract_export_prefix_hashes(request: UDPRequest) -> list[int]:
        header_payload = request.data.get("kvswitch_header")
        if isinstance(header_payload, dict):
            return normalize_prefix_hashes(
                KVSwitchShimHeader.from_dict(header_payload).active_hashes(),
                max_hashes=MAX_KVSWITCH_HASHES,
            )

        raw_hashes = request.data.get("prefix_hashes", [])
        if not isinstance(raw_hashes, list):
            return []
        return normalize_prefix_hashes(raw_hashes, max_hashes=MAX_KVSWITCH_HASHES)

    def _matched_local_blocks(self, block_hashes: list[bytes]) -> int:
        matched = 0
        for block_hash in block_hashes:
            if block_hash not in self._local_block_cache:
                break
            self._local_block_cache.move_to_end(block_hash)
            matched += 1
        return matched

    async def _emit_event(
        self,
        event_type: str,
        prefix_hashes: tuple[int, ...] = (),
        queue_depth: int | None = None,
    ) -> None:
        if self.controller_host is None or self.controller_port is None:
            return
        payload: dict[str, Any] = {
            "endpoint": "cache_event",
            "event_type": event_type,
            "worker_id": self.worker_id,
        }
        if prefix_hashes:
            payload["prefix_hashes"] = list(prefix_hashes)
        if queue_depth is not None:
            payload["queue_depth"] = queue_depth

        client = UDPClient(
            host=self.controller_host,
            port=self.controller_port,
            timeout=2.0,
        )
        try:
            await client.send(payload)
        except Exception:
            logger.warning(
                "Failed to emit %s event from worker %s",
                event_type,
                self.worker_id,
                exc_info=True,
            )

    async def _refresh_queue_depth(self) -> None:
        await self._emit_event("queue_update", queue_depth=self._active)

    async def _acquire_capacity(self, batched_tokens: int) -> bool:
        async with self._capacity_cond:
            if (
                self.max_num_batched_tokens is not None
                and batched_tokens > self.max_num_batched_tokens
            ):
                return False

            while self._active >= self.max_num_seqs or (
                self.max_num_batched_tokens is not None
                and self._active_batched_tokens + batched_tokens
                > self.max_num_batched_tokens
            ):
                await self._capacity_cond.wait()

            self._active += 1
            self._active_batched_tokens += batched_tokens

        await self._refresh_queue_depth()
        return True

    async def _release_capacity(self, batched_tokens: int) -> None:
        async with self._capacity_cond:
            self._active -= 1
            self._active_batched_tokens -= batched_tokens
            self._capacity_cond.notify_all()

        await self._refresh_queue_depth()

    def _update_local_cache(self, block_hashes: list[bytes]) -> None:
        for block_hash in block_hashes:
            if block_hash in self._local_block_cache:
                self._local_block_cache.move_to_end(block_hash)
                continue
            self._local_block_cache[block_hash] = None

    async def _update_export_prefix_cache(self, prefix_hashes: list[int]) -> None:
        for prefix in prefix_chain(prefix_hashes):
            if prefix in self._export_prefix_cache:
                self._export_prefix_cache.move_to_end(prefix)
                continue

            self._export_prefix_cache[prefix] = None
            await self._emit_event("alloc", prefix_hashes=prefix)

            while (
                self.max_cached_prefixes is not None
                and len(self._export_prefix_cache) > self.max_cached_prefixes
            ):
                evicted_prefix, _ = self._export_prefix_cache.popitem(last=False)
                await self._emit_event("evict", prefix_hashes=evicted_prefix)

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        endpoint = request.data.get("endpoint", "generate")

        if endpoint == "health":
            return UDPResponse(
                data={
                    "status": "ok",
                    "worker_id": self.worker_id,
                    "active": self._active,
                    "active_batched_tokens": self._active_batched_tokens,
                    "cached_prefixes": len(self._local_block_cache),
                    "exported_prefixes": len(self._export_prefix_cache),
                }
            )

        if endpoint == "generate":
            prompt_token_ids = request.data.get("prompt_token_ids", [])
            prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0
            output_tokens = self._request_output_tokens(request)
            batched_tokens = self._request_batched_tokens(request)
            local_block_hashes = self._local_block_hashes(prompt_token_ids)
            export_prefix_hashes = self._extract_export_prefix_hashes(request)

            if not await self._acquire_capacity(batched_tokens):
                return UDPResponse(
                    data={
                        "error": (
                            "request exceeds max_num_batched_tokens: "
                            f"required={batched_tokens} "
                            f"limit={self.max_num_batched_tokens}"
                        )
                    }
                )

            matched_blocks = self._matched_local_blocks(local_block_hashes)
            num_cached_tokens = matched_blocks * self.block_size

            try:
                simulated_e2e_s = self.ttft_s + max(output_tokens - 1, 0) * self.tpot_s
                await asyncio.sleep(simulated_e2e_s)
            finally:
                await self._release_capacity(batched_tokens)

            self._update_local_cache(local_block_hashes)
            await self._update_export_prefix_cache(export_prefix_hashes)
            return UDPResponse(
                data={
                    "text": ["<mock output>"],
                    "num_cached_tokens": num_cached_tokens,
                    "matched_blocks": matched_blocks,
                    "worker_id": self.worker_id,
                    "worker_port": self.port,
                    "simulated_ttft_ms": self.ttft_s * 1000,
                    "simulated_tpot_ms": self.tpot_s * 1000,
                    "simulated_e2e_ms": simulated_e2e_s * 1000,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "batched_tokens": batched_tokens,
                    "cached_prefixes": len(self._local_block_cache),
                    "exported_prefixes": len(self._export_prefix_cache),
                }
            )

        return UDPResponse(data={"error": f"unknown endpoint: {endpoint}"})

    async def start(self) -> None:
        await self._server.start()
        self.port = self._server.bound_port()
        if self.worker_id == f"{self.host}:0":
            self.worker_id = f"{self.host}:{self.port}"
        await self._refresh_queue_depth()
        logger.info(
            "Mock worker listening on %s:%d (ttft=%.1fms, tpot=%.1fms, max_num_seqs=%d, max_num_batched_tokens=%s, controller=%s:%s)",
            self.host,
            self.port,
            self.ttft_s * 1000,
            self.tpot_s * 1000,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.controller_host,
            self.controller_port,
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
    parser.add_argument("--tpot-ms", type=float, default=0.0)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help="Max concurrent requests (batch capacity)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional total token budget across in-flight requests",
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--worker-id", type=str, default=None)
    parser.add_argument("--controller-host", type=str, default=None)
    parser.add_argument("--controller-port", type=int, default=None)
    parser.add_argument("--max-cached-prefixes", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    asyncio.run(
        MockWorker(
            host=args.host,
            port=args.port,
            ttft_ms=args.ttft_ms,
            tpot_ms=args.tpot_ms,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            block_size=args.block_size,
            worker_id=args.worker_id,
            controller_host=args.controller_host,
            controller_port=args.controller_port,
            max_cached_prefixes=args.max_cached_prefixes,
        ).run_forever()
    )
