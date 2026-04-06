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
import json
import logging
import socket
import time
from collections import OrderedDict
from typing import Any, Final

from kvswitch.sdk.client import KVSWITCH_UDP_PORT
from kvswitch.sdk.header import KVSwitchShimHeader
from kvswitch.sdk.server import KVSwitchUDPServerProtocol
from kvswitch.utils.prefix import (
    cumulative_sha256_chain,
    normalize_prefix_hashes,
    prefix_chain,
)
from kvswitch.utils.udp import UDPRequest, UDPResponse, UDPServer

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
        kvswitch_port: int | None = None,
        base_ttft_ms: float | None = None,
        per_token_ttft_ms: float | None = None,
        load_update_interval_s: float = 0.0,
        load_update_delta: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self.ttft_s = ttft_ms / 1000.0
        self.tpot_s = tpot_ms / 1000.0
        self._base_ttft_s = base_ttft_ms / 1000.0 if base_ttft_ms is not None else None
        self._per_token_ttft_s = (
            per_token_ttft_ms / 1000.0 if per_token_ttft_ms is not None else None
        )
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size
        self.worker_id = worker_id or f"{host}:{port}"
        self.controller_host = controller_host
        self.controller_port = controller_port
        self.max_cached_prefixes = max_cached_prefixes
        self.kvswitch_port = kvswitch_port
        self._load_update_interval_s = load_update_interval_s
        self._load_update_delta = load_update_delta
        self._last_load_update_time: float = 0.0
        self._last_emitted_load: int = 0

        self._capacity_cond = asyncio.Condition()
        self._active: int = 0
        self._active_batched_tokens: int = 0
        self._queued_requests: int = 0
        self._queued_batched_tokens: int = 0
        self._local_block_cache: OrderedDict[bytes, None] = OrderedDict()
        self._export_prefix_cache: OrderedDict[tuple[int, ...], None] = OrderedDict()
        self._server = UDPServer(host=host, port=port, handler=self._handle)
        self._kvswitch_transport: asyncio.DatagramTransport | None = None
        self._event_sock: socket.socket | None = None

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
        header_payload = request.data.get("_kvswitch_shim")
        if not isinstance(header_payload, dict):
            return []
        return normalize_prefix_hashes(
            KVSwitchShimHeader.from_dict(header_payload).active_hashes(),
            max_hashes=MAX_KVSWITCH_HASHES,
        )

    def _matched_local_blocks(self, block_hashes: list[bytes]) -> int:
        matched = 0
        for block_hash in block_hashes:
            if block_hash not in self._local_block_cache:
                break
            self._local_block_cache.move_to_end(block_hash)
            matched += 1
        return matched

    def _compute_ttft_s(self, uncached_tokens: int) -> float:
        """Compute TTFT based on uncached token count.

        If ``base_ttft_ms`` and ``per_token_ttft_ms`` were provided at
        construction, TTFT is calculated as

        ``ttft = base_ttft + per_token_ttft * uncached_tokens``

        Otherwise, the fixed ``ttft_ms`` is used.
        """
        if self._base_ttft_s is not None and self._per_token_ttft_s is not None:
            return self._base_ttft_s + self._per_token_ttft_s * uncached_tokens
        return self.ttft_s

    def _ensure_event_sock(self) -> socket.socket | None:
        """Lazily create a persistent non-blocking UDP socket for events."""
        if self.controller_host is None or self.controller_port is None:
            return None
        if self._event_sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            self._event_sock = sock
        return self._event_sock

    def _emit_event(
        self,
        event_type: str,
        prefix_hashes: tuple[int, ...] = (),
        load: int | None = None,
        active_requests: int | None = None,
        active_batched_tokens: int | None = None,
        queued_requests: int | None = None,
        queued_batched_tokens: int | None = None,
    ) -> None:
        """Fire-and-forget a cache event to the SDN controller.

        Uses a persistent non-blocking UDP socket with no response wait.
        This avoids per-event socket creation and asyncio contention.
        """
        sock = self._ensure_event_sock()
        if sock is None:
            return
        payload: dict[str, Any] = {
            "endpoint": "cache_event",
            "event_type": event_type,
            "worker_id": self.worker_id,
        }
        if prefix_hashes:
            payload["prefix_hashes"] = list(prefix_hashes)
        if load is not None:
            payload["load"] = load
        if active_requests is not None:
            payload["active_requests"] = active_requests
        if active_batched_tokens is not None:
            payload["active_batched_tokens"] = active_batched_tokens
        if queued_requests is not None:
            payload["queued_requests"] = queued_requests
        if queued_batched_tokens is not None:
            payload["queued_batched_tokens"] = queued_batched_tokens

        data = json.dumps(payload).encode("utf-8")
        try:
            sock.sendto(data, (self.controller_host, self.controller_port))
        except OSError:
            logger.debug(
                "Failed to emit %s event from worker %s",
                event_type,
                self.worker_id,
                exc_info=True,
            )

    def _load_metric(self) -> int:
        return self._active_batched_tokens + self._queued_batched_tokens

    def _emit_load_update(self, force: bool = False) -> None:
        now = time.monotonic()
        current_load = self._load_metric()
        if not force:
            if (now - self._last_load_update_time) < self._load_update_interval_s:
                return
            if abs(current_load - self._last_emitted_load) < self._load_update_delta:
                return
        self._last_load_update_time = now
        self._last_emitted_load = current_load
        self._emit_event(
            "queue_update",
            load=self._load_metric(),
            active_requests=self._active,
            active_batched_tokens=self._active_batched_tokens,
            queued_requests=self._queued_requests,
            queued_batched_tokens=self._queued_batched_tokens,
        )

    async def _acquire_capacity(self, batched_tokens: int) -> bool:
        queued = False
        acquired = False
        try:
            async with self._capacity_cond:
                if (
                    self.max_num_batched_tokens is not None
                    and batched_tokens > self.max_num_batched_tokens
                ):
                    return False

                while self._active >= self.max_num_seqs or (
                    self.max_num_batched_tokens is not None
                    and (
                        self._active_batched_tokens + batched_tokens
                        > self.max_num_batched_tokens
                    )
                ):
                    if not queued:
                        self._queued_requests += 1
                        self._queued_batched_tokens += batched_tokens
                        queued = True
                        self._emit_load_update()
                    await self._capacity_cond.wait()

                if queued:
                    self._queued_requests -= 1
                    self._queued_batched_tokens -= batched_tokens
                self._active += 1
                self._active_batched_tokens += batched_tokens
                acquired = True

            self._emit_load_update()
            return True
        except Exception:
            if queued and not acquired:
                async with self._capacity_cond:
                    self._queued_requests -= 1
                    self._queued_batched_tokens -= batched_tokens
                self._emit_load_update()
            raise

    async def _release_capacity(self, batched_tokens: int) -> None:
        async with self._capacity_cond:
            self._active -= 1
            self._active_batched_tokens -= batched_tokens
            self._capacity_cond.notify_all()

        self._emit_load_update()

    def _update_local_cache(self, block_hashes: list[bytes]) -> None:
        for block_hash in block_hashes:
            if block_hash in self._local_block_cache:
                self._local_block_cache.move_to_end(block_hash)
                continue
            self._local_block_cache[block_hash] = None

    def _update_export_prefix_cache(self, prefix_hashes: list[int]) -> None:
        for prefix in prefix_chain(prefix_hashes, max_hashes=3):
            if prefix in self._export_prefix_cache:
                self._export_prefix_cache.move_to_end(prefix)
                continue

            self._export_prefix_cache[prefix] = None
            self._emit_event("alloc", prefix_hashes=prefix)

            while (
                self.max_cached_prefixes is not None
                and len(self._export_prefix_cache) > self.max_cached_prefixes
            ):
                evicted_prefix, _ = self._export_prefix_cache.popitem(last=False)
                self._emit_event("evict", prefix_hashes=evicted_prefix)

    async def _handle(self, request: UDPRequest) -> UDPResponse:
        endpoint = request.data.get("endpoint", "generate")

        if endpoint == "health":
            return UDPResponse(
                data={
                    "status": "ok",
                    "worker_id": self.worker_id,
                    "active": self._active,
                    "active_batched_tokens": self._active_batched_tokens,
                    "queued_requests": self._queued_requests,
                    "queued_batched_tokens": self._queued_batched_tokens,
                    "load": self._load_metric(),
                    "cached_prefixes": len(self._local_block_cache),
                    "exported_prefixes": len(self._export_prefix_cache),
                }
            )

        if endpoint == "generate":
            prompt_token_ids = request.data.get("prompt_token_ids", [])
            prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0
            output_tokens = self._request_output_tokens(request)
            local_block_hashes = self._local_block_hashes(prompt_token_ids)
            export_prefix_hashes = self._extract_export_prefix_hashes(request)

            # Check cache before acquiring capacity
            matched_blocks = self._matched_local_blocks(local_block_hashes)
            num_cached_tokens = matched_blocks * self.block_size
            uncached_tokens = max(prompt_tokens - num_cached_tokens, 0)

            # Only uncached prompt tokens are charged as prefill work here,
            # so repeated prefixes lower the worker's simulated batching load.
            # This is intentionally a simple admission-control approximation:
            # it treats each request as contributing its own uncached prefill
            # tokens plus output tokens, without trying to model exact shared
            # live KV/cache residency across concurrent requests.
            effective_batched_tokens = max(uncached_tokens + output_tokens, 1)

            if not await self._acquire_capacity(effective_batched_tokens):
                return UDPResponse(
                    data={
                        "error": (
                            "request exceeds max_num_batched_tokens: "
                            f"required={effective_batched_tokens} "
                            f"limit={self.max_num_batched_tokens}"
                        )
                    }
                )

            try:
                simulated_ttft_s = self._compute_ttft_s(uncached_tokens)
                tpot_s = max(output_tokens - 1, 0) * self.tpot_s
                simulated_e2e_s = simulated_ttft_s + tpot_s

                # Prefill
                await asyncio.sleep(simulated_ttft_s)
                self._update_local_cache(local_block_hashes)
                self._update_export_prefix_cache(export_prefix_hashes)

                # Decode
                if tpot_s > 0:
                    await asyncio.sleep(tpot_s)
            finally:
                await self._release_capacity(effective_batched_tokens)

            return UDPResponse(
                data={
                    "text": ["<mock output>"],
                    "num_cached_tokens": num_cached_tokens,
                    "matched_blocks": matched_blocks,
                    "uncached_tokens": uncached_tokens,
                    "worker_id": self.worker_id,
                    "worker_port": self.port,
                    "simulated_ttft_ms": simulated_ttft_s * 1000,
                    "simulated_tpot_ms": self.tpot_s * 1000,
                    "simulated_e2e_ms": simulated_e2e_s * 1000,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "batched_tokens": effective_batched_tokens,
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

        # Start a second listener on the KVSwitch UDP port (4789) that
        # strips the binary shim header before dispatching to _handle.
        if self.kvswitch_port is not None:
            loop = asyncio.get_running_loop()
            self._kvswitch_transport, _ = await loop.create_datagram_endpoint(
                lambda: KVSwitchUDPServerProtocol(self._handle),
                local_addr=(self.host, self.kvswitch_port),
            )
            sockname = self._kvswitch_transport.get_extra_info("sockname")
            if isinstance(sockname, tuple) and len(sockname) >= 2:
                self.kvswitch_port = int(sockname[1])
            logger.info(
                "Mock worker KVSwitch listener on %s:%d",
                self.host,
                self.kvswitch_port,
            )

        self._emit_load_update(force=True)
        if self._base_ttft_s is not None and self._per_token_ttft_s is not None:
            ttft_desc = f"base_ttft={self._base_ttft_s*1000:.1f}ms + {self._per_token_ttft_s*1000:.5f}ms/token"
        else:
            ttft_desc = f"ttft={self.ttft_s*1000:.1f}ms"
        logger.info(
            "Mock worker listening on %s:%d (%s, tpot=%.1fms, max_num_seqs=%d, max_num_batched_tokens=%s, controller=%s:%s)",
            self.host,
            self.port,
            ttft_desc,
            self.tpot_s * 1000,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.controller_host,
            self.controller_port,
        )

    def close(self) -> None:
        self._server.close()
        if self._kvswitch_transport is not None:
            self._kvswitch_transport.close()
        if self._event_sock is not None:
            self._event_sock.close()
            self._event_sock = None

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
    parser.add_argument(
        "--kvswitch-port",
        type=int,
        default=None,
        help=f"Also listen on this port for KVSwitch shim-header traffic (default: {KVSWITCH_UDP_PORT})",
    )
    parser.add_argument(
        "--base-ttft-ms",
        type=float,
        default=None,
        help="Baseline TTFT when fully cached (enables linear TTFT model)",
    )
    parser.add_argument(
        "--per-token-ttft-ms",
        type=float,
        default=None,
        help="Marginal TTFT per uncached token (enables linear TTFT model)",
    )
    parser.add_argument(
        "--load-update-interval-ms",
        type=float,
        default=0.0,
        help="Minimum interval between load-update emissions (ms)",
    )
    parser.add_argument(
        "--load-update-delta",
        type=int,
        default=0,
        help="Minimum load change (tokens) to trigger an update",
    )
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
            kvswitch_port=args.kvswitch_port,
            base_ttft_ms=args.base_ttft_ms,
            per_token_ttft_ms=args.per_token_ttft_ms,
            load_update_interval_s=args.load_update_interval_ms / 1000.0,
            load_update_delta=args.load_update_delta,
        ).run_forever()
    )
