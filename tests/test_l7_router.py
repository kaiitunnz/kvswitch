"""Tests for kvswitch.router.l7_router."""

from kvswitch.router.l7_router import L7Router, RoutingResult

MODEL = "Qwen/Qwen2.5-1.5B"


def _make_router(n_workers: int = 2, block_size: int = 16) -> L7Router:
    return L7Router(model=MODEL, n_workers=n_workers, block_size=block_size)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestHashBlocks:
    def test_deterministic(self) -> None:
        router = _make_router()
        tokens = list(range(64))
        assert router.hash_blocks(tokens) == router.hash_blocks(tokens)

    def test_block_count(self) -> None:
        router = _make_router(block_size=16)
        # 64 tokens / 16 block_size = 4 full blocks.
        assert len(router.hash_blocks(list(range(64)))) == 4

    def test_partial_block_ignored(self) -> None:
        router = _make_router(block_size=16)
        # 20 tokens → only 1 full block (tokens 0-15), remainder ignored.
        assert len(router.hash_blocks(list(range(20)))) == 1

    def test_fewer_than_block_size(self) -> None:
        router = _make_router(block_size=16)
        assert router.hash_blocks(list(range(10))) == []

    def test_cumulative_chain(self) -> None:
        """Changing an early token changes all subsequent block hashes."""
        router = _make_router(block_size=16)
        tokens_a = list(range(48))
        tokens_b = list(range(48))
        tokens_b[0] = 9999  # Change first token.
        hashes_a = router.hash_blocks(tokens_a)
        hashes_b = router.hash_blocks(tokens_b)
        # All 3 blocks should differ.
        assert len(hashes_a) == 3
        for ha, hb in zip(hashes_a, hashes_b):
            assert ha != hb

    def test_shared_prefix_same_hashes(self) -> None:
        """Two sequences sharing a prefix produce the same leading hashes."""
        router = _make_router(block_size=16)
        prefix = list(range(32))
        seq_a = prefix + list(range(100, 116))
        seq_b = prefix + list(range(200, 216))
        hashes_a = router.hash_blocks(seq_a)
        hashes_b = router.hash_blocks(seq_b)
        # First 2 blocks are identical (shared prefix).
        assert hashes_a[:2] == hashes_b[:2]
        # Third block differs (different suffix).
        assert hashes_a[2] != hashes_b[2]


# ---------------------------------------------------------------------------
# Routing — no cache (round-robin fallback)
# ---------------------------------------------------------------------------


class TestRoutingNoCache:
    def test_round_robin(self) -> None:
        router = _make_router(n_workers=3)
        tokens = list(range(64))
        workers = [router.route_token_ids(tokens).worker_idx for _ in range(6)]
        assert workers == [0, 1, 2, 0, 1, 2]

    def test_zero_matched_blocks(self) -> None:
        router = _make_router()
        result = router.route_token_ids(list(range(32)))
        assert result.matched_blocks == 0

    def test_result_fields(self) -> None:
        router = _make_router()
        result = router.route_token_ids(list(range(32)))
        assert isinstance(result, RoutingResult)
        assert result.tokenize_ms == 0.0  # token_ids path skips tokenization
        assert result.hash_ms >= 0.0
        assert result.lookup_ms >= 0.0
        assert result.total_ms >= 0.0
        assert result.total_blocks == 2  # 32 tokens / 16 block_size


# ---------------------------------------------------------------------------
# Routing — with cache
# ---------------------------------------------------------------------------


class TestRoutingWithCache:
    def test_routes_to_cached_worker(self) -> None:
        router = _make_router(n_workers=3)
        tokens = list(range(48))
        hashes = router.hash_blocks(tokens)
        router.update_cache(2, hashes)

        result = router.route_token_ids(tokens)
        assert result.worker_idx == 2
        assert result.matched_blocks == 3

    def test_longest_prefix_wins(self) -> None:
        router = _make_router(n_workers=3)
        tokens = list(range(64))
        hashes = router.hash_blocks(tokens)

        # Worker 0 has first 2 blocks, worker 1 has all 4.
        router.update_cache(0, hashes[:2])
        router.update_cache(1, hashes[:4])

        result = router.route_token_ids(tokens)
        assert result.worker_idx == 1
        assert result.matched_blocks == 4

    def test_non_consecutive_blocks_not_counted(self) -> None:
        """Only consecutive blocks from the start count."""
        router = _make_router(n_workers=2)
        tokens = list(range(64))
        hashes = router.hash_blocks(tokens)

        # Worker 0 has block 0 and block 2, but NOT block 1.
        router.update_cache(0, [hashes[0], hashes[2]])

        result = router.route_token_ids(tokens)
        # Only block 0 is consecutive from the start.
        assert result.matched_blocks == 1


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    def test_update_and_evict(self) -> None:
        router = _make_router(n_workers=2)
        tokens = list(range(32))
        hashes = router.hash_blocks(tokens)

        router.update_cache(0, hashes)
        assert router.route_token_ids(tokens).worker_idx == 0

        router.evict_cache(0, hashes)
        # Should fall back to round-robin after eviction.
        assert router.route_token_ids(tokens).matched_blocks == 0

    def test_clear_cache(self) -> None:
        router = _make_router(n_workers=2)
        tokens = list(range(32))
        router.update_cache(0, router.hash_blocks(tokens))
        router.clear_cache()
        assert router.cache_table == {}

    def test_evict_nonexistent_is_safe(self) -> None:
        router = _make_router()
        router.evict_cache(0, [b"fake_hash"])  # Should not raise.

    def test_multiple_workers_same_block(self) -> None:
        router = _make_router(n_workers=3)
        tokens = list(range(32))
        hashes = router.hash_blocks(tokens)
        router.update_cache(0, hashes)
        router.update_cache(1, hashes)

        # Both workers have the same prefix; router picks one.
        result = router.route_token_ids(tokens)
        assert result.worker_idx in (0, 1)
        assert result.matched_blocks == 2

        # Evict from worker 0; should route to worker 1.
        router.evict_cache(0, hashes)
        result = router.route_token_ids(tokens)
        assert result.worker_idx == 1


# ---------------------------------------------------------------------------
# route() with text prompt (uses real tokenizer)
# ---------------------------------------------------------------------------


class TestRouteText:
    def test_route_returns_result(self) -> None:
        router = _make_router()
        result = router.route("Hello, world! This is a test prompt.")
        assert isinstance(result, RoutingResult)
        assert result.tokenize_ms > 0.0
        assert result.total_blocks >= 0

    def test_timing_positive(self) -> None:
        router = _make_router()
        result = router.route("A " * 200)
        assert result.tokenize_ms >= 0.0
        assert result.hash_ms >= 0.0
        assert result.lookup_ms >= 0.0
        assert result.total_ms >= result.tokenize_ms

    def test_route_with_override_token_ids_uses_provided_ids(self) -> None:
        """When token_ids are provided, route() uses them for hashing/routing
        instead of the tokenization result, while still paying tokenize cost."""
        router = _make_router(n_workers=2)
        # Use specific token IDs that produce known block hashes.
        override_ids = list(range(32))
        hashes = router.hash_blocks(override_ids)
        router.update_cache(1, hashes)

        # Route with a prompt whose tokenization differs from override_ids.
        result = router.route("Hello, world!", token_ids=override_ids)
        assert result.worker_idx == 1
        assert result.matched_blocks == 2
        assert result.tokenize_ms > 0.0  # tokenization still happened

    def test_route_with_override_tokenize_cost_included(self) -> None:
        """route(prompt, token_ids) includes tokenize_ms in total_ms."""
        router = _make_router()
        result = router.route("A " * 200, token_ids=list(range(32)))
        assert result.tokenize_ms > 0.0
        assert result.total_ms >= result.tokenize_ms

    def test_route_without_override_uses_tokenized(self) -> None:
        """Without token_ids, route() uses the tokenized prompt for routing."""
        router = _make_router(n_workers=2)
        prompt = "Hello, world! This is a test prompt with enough tokens."
        # Route twice — should produce identical routing decisions.
        r1 = router.route(prompt)
        # Reset round-robin counter by clearing cache context.
        r2 = router.route(prompt)
        assert r1.total_blocks == r2.total_blocks
        assert r1.block_hashes == r2.block_hashes
