"""Software L7 prefix-aware router baseline.

Simulates the full SGLang-style routing pipeline:
  receive request → tokenize prompt → hash token blocks →
  lookup prefix cache (longest match) → select worker

Each step is individually timed so we can quantify L7 overhead.
"""

import logging
import time
from dataclasses import dataclass

from vllm.tokenizers import get_tokenizer

from kvswitch.utils.prefix import cumulative_sha256_chain

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of a single routing decision with per-step timing."""

    worker_idx: int
    matched_blocks: int
    total_blocks: int
    tokenize_ms: float
    hash_ms: float
    lookup_ms: float
    total_ms: float


class L7Router:
    """Software L7 prefix-aware router.

    Maintains a per-worker prefix cache directory and routes incoming
    requests to the worker holding the longest matching prefix.

    Parameters
    ----------
    model:
        HuggingFace model name used to load the tokenizer.
    n_workers:
        Number of backend workers.
    block_size:
        Token block size for prefix hashing (must match the vLLM engine).
    """

    def __init__(
        self,
        model: str,
        n_workers: int = 1,
        block_size: int = 16,
    ) -> None:
        self.tokenizer = get_tokenizer(model)
        self.n_workers = n_workers
        self.block_size = block_size

        # block_hash → set of worker indices that hold this block cached.
        self.cache_table: dict[bytes, set[int]] = {}

        # Round-robin counter for fallback when no prefix match.
        self._rr_counter = 0

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def hash_blocks(self, token_ids: list[int]) -> list[bytes]:
        """Compute cumulative SHA-256 block hashes for a token sequence.

        Mirrors vLLM's APC semantics: each block hash depends on the
        parent hash plus the current block's tokens.
        """
        return cumulative_sha256_chain(token_ids, self.block_size)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, prompt: str) -> RoutingResult:
        """Full L7 routing pipeline with per-step timing."""
        t_total = time.perf_counter()

        # Step 1 — Tokenize.
        t0 = time.perf_counter()
        token_ids = self.tokenizer.encode(prompt)
        tokenize_ms = (time.perf_counter() - t0) * 1000

        # Step 2 — Hash blocks.
        t0 = time.perf_counter()
        block_hashes = self.hash_blocks(token_ids)
        hash_ms = (time.perf_counter() - t0) * 1000

        # Step 3 — Lookup longest prefix match per worker.
        t0 = time.perf_counter()
        worker_idx, matched = self._lookup(block_hashes)
        lookup_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return RoutingResult(
            worker_idx=worker_idx,
            matched_blocks=matched,
            total_blocks=len(block_hashes),
            tokenize_ms=tokenize_ms,
            hash_ms=hash_ms,
            lookup_ms=lookup_ms,
            total_ms=total_ms,
        )

    def route_token_ids(self, token_ids: list[int]) -> RoutingResult:
        """Route using pre-tokenized input (skip tokenization step)."""
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        block_hashes = self.hash_blocks(token_ids)
        hash_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        worker_idx, matched = self._lookup(block_hashes)
        lookup_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        return RoutingResult(
            worker_idx=worker_idx,
            matched_blocks=matched,
            total_blocks=len(block_hashes),
            tokenize_ms=0.0,
            hash_ms=hash_ms,
            lookup_ms=lookup_ms,
            total_ms=total_ms,
        )

    def _lookup(self, block_hashes: list[bytes]) -> tuple[int, int]:
        """Find the worker with the longest matching prefix.

        Returns (worker_idx, matched_block_count).  Falls back to
        round-robin if no cache match.
        """
        # Score each worker: number of consecutive matching blocks from the start.
        best_worker = -1
        best_match = 0

        # For each worker, count how many leading blocks are cached.
        scores: dict[int, int] = {}
        for block_idx, bh in enumerate(block_hashes):
            workers_with_block = self.cache_table.get(bh)
            if workers_with_block is None:
                break
            for w in workers_with_block:
                if w not in scores:
                    scores[w] = 0
                # Only count if this extends the worker's consecutive run.
                if scores[w] == block_idx:
                    scores[w] = block_idx + 1

        if scores:
            best_worker = max(scores, key=lambda w: scores[w])
            best_match = scores[best_worker]

        if best_worker < 0:
            # Fallback: round-robin.
            best_worker = self._rr_counter % self.n_workers
            self._rr_counter += 1

        return best_worker, best_match

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def update_cache(self, worker_idx: int, block_hashes: list[bytes]) -> None:
        """Register that *worker_idx* has cached the given blocks."""
        for bh in block_hashes:
            self.cache_table.setdefault(bh, set()).add(worker_idx)

    def evict_cache(self, worker_idx: int, block_hashes: list[bytes]) -> None:
        """Remove cache entries for *worker_idx*."""
        for bh in block_hashes:
            workers = self.cache_table.get(bh)
            if workers is not None:
                workers.discard(worker_idx)
                if not workers:
                    del self.cache_table[bh]

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache_table.clear()
