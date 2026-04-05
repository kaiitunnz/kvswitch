"""Tests for kvswitch.eval.workload."""

from pathlib import Path

from kvswitch.eval.workload import (
    WorkloadConfig,
    WorkloadGenerator,
    WorkloadRequest,
    load_workload,
    save_workload,
)
from kvswitch.sdk.hashing import compute_truncated_hashes


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        if text == "prompt":
            return list(range(255))
        if text == "reply":
            return list(range(10))
        return list(range(255))


class TestWorkloadSerialization:
    def test_save_and_load_preserve_prefix_hashes(self, tmp_path: Path) -> None:
        requests = [
            WorkloadRequest(
                request_id=1,
                prompt_token_ids=[1, 2, 3],
                scheduled_time=0.5,
                prefix_group="group_0",
                max_tokens=4,
                prefix_hashes=[10, 20],
            ),
            WorkloadRequest(
                request_id=2,
                prompt_token_ids=[4, 5],
                scheduled_time=1.5,
                prefix_group="none",
                max_tokens=2,
                prefix_hashes=None,
            ),
        ]
        path = tmp_path / "workload.json"

        save_workload(requests, path)

        assert load_workload(path) == requests


class TestWorkloadGeneration:
    def test_generate_uses_sdk_hashing_contract(self, monkeypatch) -> None:
        config = WorkloadConfig(
            dataset_path=Path("unused.json"),
            num_requests=1,
            request_rate=0.0,
            prefix_sharing_ratio=1.0,
            num_prefix_groups=1,
            system_prompt_tokens=256,
            max_prompt_tokens=1024,
            hash_key="kvswitch-eval",
        )
        generator = WorkloadGenerator(config)

        monkeypatch.setattr(
            "kvswitch.eval.workload.load_sharegpt_conversations",
            lambda path, max_items=None: [("prompt", "reply")],
        )
        monkeypatch.setattr(generator, "_get_tokenizer", lambda: _DummyTokenizer())

        request = generator.generate()[0]

        assert len(request.prompt_token_ids) == 511
        assert request.prefix_hashes == compute_truncated_hashes(
            request.prompt_token_ids,
            config.hash_key.encode("utf-8"),
        )
        assert request.prefix_hashes is not None
        assert len(request.prefix_hashes) == 1
        assert request.max_tokens == 10  # inferred from GPT reply tokenization
