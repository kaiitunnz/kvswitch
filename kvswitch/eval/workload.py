"""Workload generator for KVSwitch evaluation.

Loads ShareGPT conversations, injects shared system prompts at a
configurable ratio, tokenizes with the model's tokenizer, and
generates Poisson-distributed arrival times.

The generated workload is serialized to JSON so it can be passed to
the Mininet ``workload_client`` CLI tool running inside a host
namespace.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kvswitch.sdk.hashing import compute_truncated_hashes

logger = logging.getLogger(__name__)
DEFAULT_HASH_KEY = "kvswitch-eval"


@dataclass
class WorkloadRequest:
    """A single request in the evaluation workload."""

    request_id: int
    prompt_token_ids: list[int]
    scheduled_time: float  # seconds from experiment start
    prefix_group: str  # "group_0" … "group_N" or "none"
    max_tokens: int
    prefix_hashes: list[int] | None = None  # truncated 32-bit hashes for shim header


@dataclass
class WorkloadConfig:
    """Parameters for workload generation."""

    dataset_path: Path = Path("data/ShareGPT_V3_unfiltered_cleaned_split.json")
    num_requests: int = 200
    request_rate: float = 10.0  # lambda for Poisson (requests/sec)
    prefix_sharing_ratio: float = 0.5  # fraction assigned to a prefix group
    num_prefix_groups: int = 3
    system_prompt_tokens: int = 256  # tokens per injected system prompt
    max_prompt_tokens: int = 2048
    max_output_tokens: int = 256
    seed: int = 42
    model: str = "meta-llama/Llama-3.2-3B-Instruct"
    hash_key: str = DEFAULT_HASH_KEY


# ---------------------------------------------------------------------------
# ShareGPT loading
# ---------------------------------------------------------------------------


def load_sharegpt_conversations(
    path: Path, max_items: int | None = None
) -> list[tuple[str, str | None]]:
    """Extract the first human turn and the following GPT reply from each conversation.

    Returns a list of ``(human_text, gpt_reply_or_none)`` pairs.
    """
    with open(path) as f:
        data = json.load(f)

    pairs: list[tuple[str, str | None]] = []
    for item in data:
        conversations = item.get("conversations", [])
        human_text: str | None = None
        gpt_reply: str | None = None
        for turn in conversations:
            if human_text is None and turn.get("from") == "human":
                text = turn.get("value", "").strip()
                if text:
                    human_text = text
            elif human_text is not None and turn.get("from") == "gpt":
                gpt_reply = turn.get("value", "").strip() or None
                break
        if human_text:
            pairs.append((human_text, gpt_reply))
        if max_items is not None and len(pairs) >= max_items:
            break

    return pairs


# ---------------------------------------------------------------------------
# System prompt generation
# ---------------------------------------------------------------------------


def generate_system_prompts(
    num_groups: int,
    tokens_per_group: int,
    seed: int,
    token_id_min: int = 100,
    token_id_max: int = 10000,
) -> dict[str, list[int]]:
    """Generate deterministic system prompt token sequences per group."""
    rng = random.Random(seed)
    prompts: dict[str, list[int]] = {}
    for i in range(num_groups):
        key = f"group_{i}"
        prompts[key] = [
            rng.randint(token_id_min, token_id_max) for _ in range(tokens_per_group)
        ]
    return prompts


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------


class WorkloadGenerator:
    """Generates a reproducible evaluation workload from ShareGPT data."""

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config
        self._tokenizer: Any = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from vllm.tokenizers import get_tokenizer

            self._tokenizer = get_tokenizer(self.config.model)
        return self._tokenizer

    def generate(self) -> list[WorkloadRequest]:
        """Generate the full workload: tokenize, inject prefixes, schedule."""
        cfg = self.config
        rng = random.Random(cfg.seed)

        # Load ShareGPT conversations (human turn + GPT reply).
        raw_pairs = load_sharegpt_conversations(
            cfg.dataset_path, max_items=cfg.num_requests * 2
        )
        if not raw_pairs:
            raise ValueError(f"No conversations loaded from {cfg.dataset_path}")
        logger.info("Loaded %d raw prompts from ShareGPT", len(raw_pairs))

        tokenizer = self._get_tokenizer()
        tokenized: list[tuple[list[int], int]] = []
        for human_text, gpt_reply in raw_pairs:
            prompt_ids = tokenizer.encode(human_text)
            if not prompt_ids:
                continue
            prompt_ids = prompt_ids[: cfg.max_prompt_tokens]
            # Infer output tokens from the GPT reply; fall back to default.
            if gpt_reply:
                output_tokens = min(
                    len(tokenizer.encode(gpt_reply)), cfg.max_output_tokens
                )
            else:
                output_tokens = cfg.max_output_tokens
            output_tokens = max(output_tokens, 1)
            tokenized.append((prompt_ids, output_tokens))
            if len(tokenized) >= cfg.num_requests:
                break

        if len(tokenized) < cfg.num_requests:
            logger.warning(
                "Only %d usable prompts (requested %d); recycling",
                len(tokenized),
                cfg.num_requests,
            )
            while len(tokenized) < cfg.num_requests:
                tokenized.append(tokenized[rng.randint(0, len(tokenized) - 1)])

        # Generate system prompts for prefix groups.
        system_prompts = generate_system_prompts(
            cfg.num_prefix_groups, cfg.system_prompt_tokens, cfg.seed
        )

        # Assign prefix groups and build final token sequences.
        group_keys = list(system_prompts.keys())
        requests: list[WorkloadRequest] = []
        t = 0.0

        for i in range(cfg.num_requests):
            user_tokens, output_tokens = tokenized[i]

            # Assign to a prefix group with probability prefix_sharing_ratio.
            if rng.random() < cfg.prefix_sharing_ratio:
                group = rng.choice(group_keys)
                prompt_ids = system_prompts[group] + user_tokens
            else:
                group = "none"
                prompt_ids = user_tokens

            # Truncate to max_prompt_tokens.
            prompt_ids = prompt_ids[: cfg.max_prompt_tokens]

            # Poisson inter-arrival time.
            if cfg.request_rate > 0:
                t += rng.expovariate(cfg.request_rate)
            else:
                t += 0.0  # All at once.

            # Compute prefix hashes for the KVSwitch shim header.
            prefix_hashes = compute_truncated_hashes(
                prompt_ids, cfg.hash_key.encode("utf-8")
            )

            requests.append(
                WorkloadRequest(
                    request_id=i,
                    prompt_token_ids=prompt_ids,
                    scheduled_time=t,
                    prefix_group=group,
                    max_tokens=output_tokens,
                    prefix_hashes=prefix_hashes,
                )
            )

        logger.info(
            "Generated %d requests (rate=%.1f/s, prefix_sharing=%.0f%%, groups=%d)",
            len(requests),
            cfg.request_rate,
            cfg.prefix_sharing_ratio * 100,
            cfg.num_prefix_groups,
        )
        return requests


# ---------------------------------------------------------------------------
# Serialization (for passing to Mininet host CLI)
# ---------------------------------------------------------------------------


def save_workload(requests: list[WorkloadRequest], path: Path) -> None:
    """Write workload to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in requests]
    path.write_text(json.dumps(data))
    logger.info("Saved %d requests to %s", len(requests), path)


def load_workload(path: Path) -> list[WorkloadRequest]:
    """Read workload from a JSON file."""
    data = json.loads(path.read_text())
    return [
        WorkloadRequest(
            request_id=r["request_id"],
            prompt_token_ids=r["prompt_token_ids"],
            scheduled_time=r["scheduled_time"],
            prefix_group=r["prefix_group"],
            max_tokens=r["max_tokens"],
            prefix_hashes=(
                [int(value) for value in r["prefix_hashes"]]
                if r.get("prefix_hashes") is not None
                else None
            ),
        )
        for r in data
    ]
