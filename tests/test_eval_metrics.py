"""Tests for kvswitch.eval.metrics."""

from pathlib import Path

import pytest

from kvswitch.eval.metrics import (
    RequestMetric,
    _percentile,
    compute_summary,
    load_experiment_results,
    save_experiment_results,
)


def _make_metric(**overrides) -> RequestMetric:
    defaults = dict(
        request_id=0,
        baseline="l4_rr",
        e2e_latency_ms=20.0,
        simulated_ttft_ms=10.0,
        simulated_tpot_ms=3.0,
        simulated_e2e_ms=19.0,
        routing_overhead_ms=0.0,
        matched_blocks=2,
        worker_id="w0",
        prompt_tokens=256,
        output_tokens=16,
        prefix_group="group_0",
        scheduled_time_s=0.0,
        actual_send_time_s=0.0,
    )
    defaults.update(overrides)
    return RequestMetric(**defaults)


class TestPercentile:
    def test_median_odd(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_median_even(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)

    def test_p99(self) -> None:
        values = list(range(100))
        assert _percentile(values, 99) == pytest.approx(98.01)

    def test_single_value(self) -> None:
        assert _percentile([42.0], 50) == pytest.approx(42.0)

    def test_empty(self) -> None:
        assert _percentile([], 50) == 0.0


class TestComputeSummary:
    def test_basic(self) -> None:
        metrics = [
            _make_metric(e2e_latency_ms=10.0, actual_send_time_s=0.0),
            _make_metric(e2e_latency_ms=20.0, actual_send_time_s=0.5),
            _make_metric(e2e_latency_ms=30.0, actual_send_time_s=1.0),
        ]
        summary = compute_summary(metrics)
        assert summary["n_requests"] == 3
        assert summary["e2e_mean_ms"] == pytest.approx(20.0)
        assert summary["e2e_p50_ms"] == pytest.approx(20.0)
        assert summary["ttft_mean_ms"] == pytest.approx(10.0)
        assert summary["throughput_rps"] == pytest.approx(3.0)

    def test_empty(self) -> None:
        assert compute_summary([]) == {}

    def test_cache_hit_rate(self) -> None:
        metrics = [
            _make_metric(matched_blocks=4, prompt_tokens=64),  # 4 / (64/16) = 1.0
            _make_metric(matched_blocks=0, prompt_tokens=64),  # 0 / 4 = 0.0
        ]
        summary = compute_summary(metrics)
        assert summary["cache_hit_rate_mean"] == pytest.approx(0.5)


class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        metrics = [_make_metric(request_id=0), _make_metric(request_id=1)]
        path = tmp_path / "results.json"

        save_experiment_results(
            experiment="test",
            config={"seed": 42},
            results={"l4_rr": metrics},
            path=path,
        )

        loaded = load_experiment_results(path)
        assert loaded["experiment"] == "test"
        assert loaded["config"]["seed"] == 42
        assert len(loaded["results"]["l4_rr"]["per_request"]) == 2
        assert "summary" in loaded["results"]["l4_rr"]
        assert loaded["results"]["l4_rr"]["summary"]["n_requests"] == 2
