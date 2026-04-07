"""Shared token-block and prefix-hash helpers."""

import hashlib
import struct
from collections.abc import Iterable, Sequence
from typing import Any, Final

ROOT_PARENT_HASH: Final[bytes] = b"\x00" * 32


def chunk_token_ids(
    token_ids: Sequence[int],
    chunk_size: int,
    cacheable_only: bool = True,
) -> list[list[int]]:
    """Split token IDs into fixed-size chunks.

    By default, only full cacheable chunks are returned so the resulting block
    sequence matches prefix-cache semantics.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    limit = len(token_ids)
    if cacheable_only:
        limit -= limit % chunk_size

    return [list(token_ids[i : i + chunk_size]) for i in range(0, limit, chunk_size)]


def pack_token_ids(token_ids: Sequence[int]) -> bytes:
    """Pack token IDs into a deterministic big-endian byte string."""
    return struct.pack(f"!{len(token_ids)}I", *token_ids)


def cumulative_sha256_block(parent_hash: bytes, token_ids: Sequence[int]) -> bytes:
    """Hash one block in a cumulative SHA-256 prefix chain."""
    digest = hashlib.sha256()
    digest.update(parent_hash)
    digest.update(pack_token_ids(token_ids))
    return digest.digest()


def cumulative_sha256_chain(
    token_ids: Sequence[int],
    block_size: int,
) -> list[bytes]:
    """Compute cumulative SHA-256 hashes for all full cacheable blocks."""
    hashes: list[bytes] = []
    parent = ROOT_PARENT_HASH
    for block_tokens in chunk_token_ids(token_ids, chunk_size=block_size):
        parent = cumulative_sha256_block(parent, block_tokens)
        hashes.append(parent)
    return hashes


def normalize_prefix_hashes(
    prefix_hashes: Iterable[Any],
    max_hashes: int | None = None,
) -> list[int]:
    """Normalize prefix hashes as 32-bit unsigned integers."""
    if max_hashes is not None and max_hashes < 0:
        raise ValueError("max_hashes must be non-negative")

    normalized = [int(value) & 0xFFFFFFFF for value in prefix_hashes]
    if max_hashes is not None:
        return normalized[:max_hashes]
    return normalized


def prefix_chain(
    prefix_hashes: Iterable[Any],
    max_hashes: int | None = None,
) -> list[tuple[int, ...]]:
    """Expand a flat prefix-hash list into all incremental prefix tuples."""
    normalized = normalize_prefix_hashes(prefix_hashes, max_hashes=max_hashes)
    return [tuple(normalized[:i]) for i in range(1, len(normalized) + 1)]


def spine_prefix_key(prefix_hashes: Iterable[Any]) -> tuple[int, ...]:
    """Return the coarse spine prefix key from a prefix-hash list."""
    normalized = normalize_prefix_hashes(prefix_hashes, max_hashes=1)
    if not normalized:
        raise ValueError("expected at least one prefix hash")
    return (normalized[0],)


def leaf_prefix_key(
    prefix_hashes: Iterable[Any],
    depth: int = 4,
) -> tuple[int, ...]:
    """Return the leaf prefix key truncated to the requested depth."""
    normalized = normalize_prefix_hashes(prefix_hashes, max_hashes=depth)
    if not normalized:
        raise ValueError("expected at least one prefix hash")
    return tuple(normalized)


def format_prefix_key(prefix_hashes: Iterable[Any]) -> str:
    """Format a prefix tuple as dotted 8-hex-digit fields."""
    return ".".join(f"{value:08x}" for value in normalize_prefix_hashes(prefix_hashes))
