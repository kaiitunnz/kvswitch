"""Tests for kvswitch.eval.workload."""

from pathlib import Path

from kvswitch.eval.workload import (
    WorkloadConfig,
    WorkloadRequest,
    generate_system_prompts,
    load_workload,
    save_workload,
)


class TestGenerateSystemPrompts:
    def test_correct_count_and_length(self) -> None:
        prompts = generate_system_prompts(num_groups=3, tokens_per_group=128, seed=42)
        assert len(prompts) == 3
        for key, tokens in prompts.items():
            assert key.startswith("group_")
            assert len(tokens) == 128

    def test_deterministic(self) -> None:
        a = generate_system_prompts(3, 64, seed=1)
        b = generate_system_prompts(3, 64, seed=1)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = generate_system_prompts(1, 64, seed=1)
        b = generate_system_prompts(1, 64, seed=2)
        assert a["group_0"] != b["group_0"]

    def test_token_range(self) -> None:
        prompts = generate_system_prompts(
            1, 100, seed=0, token_id_min=50, token_id_max=100
        )
        assert all(50 <= t <= 100 for t in prompts["group_0"])


class TestWorkloadSerialization:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        requests = [
            WorkloadRequest(
                request_id=0,
                prompt_token_ids=[1, 2, 3],
                scheduled_time=0.0,
                prefix_group="group_0",
                max_tokens=16,
            ),
            WorkloadRequest(
                request_id=1,
                prompt_token_ids=[4, 5, 6, 7],
                scheduled_time=0.1,
                prefix_group="none",
                max_tokens=8,
            ),
        ]
        path = tmp_path / "workload.json"
        save_workload(requests, path)

        loaded = load_workload(path)
        assert len(loaded) == 2
        assert loaded[0].request_id == 0
        assert loaded[0].prompt_token_ids == [1, 2, 3]
        assert loaded[0].prefix_group == "group_0"
        assert loaded[1].scheduled_time == 0.1
        assert loaded[1].max_tokens == 8

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "wl.json"
        save_workload([], path)
        assert path.exists()


class TestWorkloadConfig:
    def test_defaults(self) -> None:
        cfg = WorkloadConfig()
        assert cfg.num_requests == 200
        assert cfg.request_rate == 10.0
        assert cfg.prefix_sharing_ratio == 0.5
        assert cfg.num_prefix_groups == 3
        assert cfg.max_output_tokens == 256

    def test_custom(self) -> None:
        cfg = WorkloadConfig(num_requests=50, request_rate=5.0)
        assert cfg.num_requests == 50
        assert cfg.request_rate == 5.0
