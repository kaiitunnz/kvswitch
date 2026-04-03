"""Tests for kvswitch.vllm.profiling — no GPU required."""

import asyncio
import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kvswitch.vllm.profiling import (
    CSV_COLUMNS,
    ProfilingConfig,
    ProfilingResult,
    build_prompt_pair,
    generate_token_ids,
    load_results_csv,
    measure_ttft,
    run_profiling,
    save_results_csv,
    start_server_process,
    stop_server_process,
    wait_for_server,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_client(num_cached_tokens: int = 0) -> AsyncMock:
    """Return a mock VLLMClient that returns canned responses."""
    client = AsyncMock()
    client.health = AsyncMock(return_value={"status": "ok"})
    client.reset = AsyncMock(return_value={"status": "ok"})
    client.generate_tokens = AsyncMock(
        return_value={
            "text": ["output"],
            "num_cached_tokens": num_cached_tokens,
            "metrics": {"first_token_latency": 0.005},
        }
    )
    return client


# ---------------------------------------------------------------------------
# ProfilingConfig
# ---------------------------------------------------------------------------


class TestProfilingConfig:
    def test_defaults(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.num_trials == 5
        assert cfg.max_output_tokens == 1
        assert 128 in cfg.prompt_lengths
        assert 0.0 in cfg.prefix_ratios

    def test_custom(self) -> None:
        cfg = ProfilingConfig(prompt_lengths=[64], prefix_ratios=[0.5], num_trials=2)
        assert cfg.prompt_lengths == [64]
        assert cfg.prefix_ratios == [0.5]
        assert cfg.num_trials == 2


# ---------------------------------------------------------------------------
# generate_token_ids
# ---------------------------------------------------------------------------


class TestGenerateTokenIds:
    def test_length(self) -> None:
        ids = generate_token_ids(100, seed=0)
        assert len(ids) == 100

    def test_deterministic(self) -> None:
        a = generate_token_ids(50, seed=42)
        b = generate_token_ids(50, seed=42)
        assert a == b

    def test_different_seeds(self) -> None:
        a = generate_token_ids(50, seed=1)
        b = generate_token_ids(50, seed=2)
        assert a != b

    def test_range(self) -> None:
        ids = generate_token_ids(200, seed=0, token_id_min=100, token_id_max=200)
        assert all(100 <= x <= 200 for x in ids)

    def test_empty(self) -> None:
        assert generate_token_ids(0, seed=0) == []


# ---------------------------------------------------------------------------
# build_prompt_pair
# ---------------------------------------------------------------------------


class TestBuildPromptPair:
    def test_no_prefix(self) -> None:
        cfg = ProfilingConfig(seed=42)
        prime, measure = build_prompt_pair(100, 0.0, cfg, trial=0)
        assert len(prime) == 100
        assert len(measure) == 100
        # With 0% prefix, no shared prefix — prompts should differ.
        assert prime != measure

    def test_full_prefix(self) -> None:
        cfg = ProfilingConfig(seed=42)
        prime, measure = build_prompt_pair(100, 1.0, cfg, trial=0)
        assert len(prime) == 100
        assert len(measure) == 100
        # 100% prefix — both should be identical (no suffix).
        assert prime == measure

    def test_half_prefix(self) -> None:
        cfg = ProfilingConfig(seed=42)
        prime, measure = build_prompt_pair(100, 0.5, cfg, trial=0)
        assert len(prime) == 100
        assert len(measure) == 100
        # First 50 tokens should match.
        assert prime[:50] == measure[:50]
        # Suffixes should differ.
        assert prime[50:] != measure[50:]

    def test_length_preserved(self) -> None:
        cfg = ProfilingConfig(seed=0)
        for length in [128, 256, 1024]:
            for ratio in [0.0, 0.25, 0.75, 1.0]:
                prime, measure = build_prompt_pair(length, ratio, cfg, trial=0)
                assert len(prime) == length
                assert len(measure) == length


# ---------------------------------------------------------------------------
# measure_ttft (mocked client)
# ---------------------------------------------------------------------------


class TestMeasureTTFT:
    def test_basic(self) -> None:
        client = _make_mock_client(num_cached_tokens=64)

        async def _run() -> tuple[float, float | None, int]:
            return await measure_ttft(client, [1, 2, 3])

        wall_ttft, engine_ttft, num_cached = asyncio.run(_run())
        assert wall_ttft >= 0.0
        assert engine_ttft == pytest.approx(0.005)
        assert num_cached == 64
        client.generate_tokens.assert_called_once()

    def test_no_metrics(self) -> None:
        client = AsyncMock()
        client.generate_tokens = AsyncMock(
            return_value={"text": ["out"], "num_cached_tokens": None}
        )

        async def _run() -> tuple[float, float | None, int]:
            return await measure_ttft(client, [1])

        wall_ttft, engine_ttft, num_cached = asyncio.run(_run())
        assert wall_ttft >= 0.0
        assert engine_ttft is None
        assert num_cached == 0

    def test_custom_params(self) -> None:
        client = _make_mock_client()

        async def _run() -> tuple[float, float | None, int]:
            return await measure_ttft(client, [1], max_tokens=5, temperature=0.5)

        asyncio.run(_run())
        call_kwargs = client.generate_tokens.call_args
        assert call_kwargs[1]["max_tokens"] == 5
        assert call_kwargs[1]["temperature"] == 0.5


# ---------------------------------------------------------------------------
# run_profiling (mocked client, small sweep)
# ---------------------------------------------------------------------------


class TestRunProfiling:
    def test_full_sweep(self) -> None:
        client = _make_mock_client(num_cached_tokens=32)
        config = ProfilingConfig(
            prompt_lengths=[64, 128],
            prefix_ratios=[0.0, 0.5],
            num_trials=2,
        )

        results = asyncio.run(run_profiling(client, config))

        expected_count = 2 * 2 * 2  # lengths * ratios * trials
        assert len(results) == expected_count
        for r in results:
            assert r.prompt_tokens in [64, 128]
            assert r.prefix_ratio in [0.0, 0.5]
            assert 0 <= r.trial < 2
            assert r.ttft >= 0.0

    def test_reset_called_each_trial(self) -> None:
        client = _make_mock_client()
        config = ProfilingConfig(
            prompt_lengths=[64],
            prefix_ratios=[0.0, 0.5],
            num_trials=2,
        )
        asyncio.run(run_profiling(client, config))
        # reset() should be called once per measurement (4 total).
        assert client.reset.call_count == 4

    def test_priming_called_for_nonzero_ratio(self) -> None:
        client = _make_mock_client()
        config = ProfilingConfig(
            prompt_lengths=[64],
            prefix_ratios=[0.0, 0.5],
            num_trials=1,
        )
        asyncio.run(run_profiling(client, config))
        # ratio=0.0: 1 generate_tokens (measure only)
        # ratio=0.5: 2 generate_tokens (prime + measure)
        # Total = 3
        assert client.generate_tokens.call_count == 3


# ---------------------------------------------------------------------------
# Server process management
# ---------------------------------------------------------------------------


class TestStartServerProcess:
    @patch("kvswitch.vllm.profiling.mp")
    def test_spawns_process(self, mock_mp: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx
        mock_proc = MagicMock()
        mock_ctx.Process.return_value = mock_proc

        proc = start_server_process("127.0.0.1", 9000, ["--model", "test-model"])

        mock_mp.get_context.assert_called_once_with("spawn")
        mock_ctx.Process.assert_called_once()
        call_kwargs = mock_ctx.Process.call_args
        assert call_kwargs[1]["args"] == ("127.0.0.1", 9000, ["--model", "test-model"])
        mock_proc.start.assert_called_once()
        assert proc is mock_proc


class TestStopServerProcess:
    def test_already_exited(self) -> None:
        proc = MagicMock()
        proc.is_alive.return_value = False
        stop_server_process(proc)
        proc.terminate.assert_not_called()

    def test_terminate_then_join(self) -> None:
        proc = MagicMock()
        proc.is_alive.side_effect = [True, False]
        proc.exitcode = 0
        stop_server_process(proc)
        proc.terminate.assert_called_once()
        proc.join.assert_called_once_with(timeout=10)


class TestWaitForServer:
    def test_ready_immediately(self) -> None:
        client = _make_mock_client()
        proc = MagicMock()
        proc.is_alive.return_value = True

        asyncio.run(wait_for_server(client, proc, timeout=5.0, poll_interval=0.01))
        client.health.assert_called()

    def test_process_crashed(self) -> None:
        client = _make_mock_client()
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 1

        with pytest.raises(RuntimeError, match="exited early"):
            asyncio.run(wait_for_server(client, proc, timeout=1.0, poll_interval=0.01))

    def test_timeout(self) -> None:
        client = AsyncMock()
        client.health = AsyncMock(side_effect=ConnectionRefusedError)
        proc = MagicMock()
        proc.is_alive.return_value = True

        with pytest.raises(TimeoutError):
            asyncio.run(wait_for_server(client, proc, timeout=0.05, poll_interval=0.01))


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


class TestCSV:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "results.csv"
        results = [
            ProfilingResult(
                prompt_tokens=256,
                prefix_ratio=0.5,
                trial=0,
                ttft=0.0123,
                engine_ttft=0.0100,
                num_cached_tokens=128,
            ),
            ProfilingResult(
                prompt_tokens=512,
                prefix_ratio=0.0,
                trial=1,
                ttft=0.0456,
                engine_ttft=None,
                num_cached_tokens=0,
            ),
        ]
        save_results_csv(results, path)

        loaded = load_results_csv(path)
        assert len(loaded) == 2

        assert loaded[0].prompt_tokens == 256
        assert loaded[0].prefix_ratio == pytest.approx(0.5)
        assert loaded[0].ttft == pytest.approx(0.0123)
        assert loaded[0].engine_ttft == pytest.approx(0.0100)
        assert loaded[0].num_cached_tokens == 128

        assert loaded[1].engine_ttft is None
        assert loaded[1].num_cached_tokens == 0

    def test_csv_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "cols.csv"
        save_results_csv([], path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "results.csv"
        save_results_csv([], path)
        assert path.exists()
