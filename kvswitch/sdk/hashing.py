"""Client-side hashing helpers for KVSwitch shim headers.

These helpers implement cumulative HMAC-SHA256 prefix hashing aligned with
prefix-cache semantics: only full cacheable token blocks contribute hashes.
"""

import hashlib
import hmac
import struct
from typing import Final

from kvswitch.utils.prefix import (
    ROOT_PARENT_HASH,
)
from kvswitch.utils.prefix import chunk_token_ids as _chunk_token_ids
from kvswitch.utils.prefix import (
    pack_token_ids,
)

CHUNK_SIZE: Final[int] = 256
MAX_HEADER_HASHES: Final[int] = 4


def chunk_token_ids(
    token_ids: list[int],
    chunk_size: int = CHUNK_SIZE,
    cacheable_only: bool = True,
) -> list[list[int]]:
    """Split token IDs into chunks.

    By default, only full cacheable chunks are returned so the resulting hash
    chain matches prefix-cache semantics.
    """
    return _chunk_token_ids(
        token_ids,
        chunk_size=chunk_size,
        cacheable_only=cacheable_only,
    )


def compute_hash_chain(
    token_ids: list[int],
    key: bytes,
    chunk_size: int = CHUNK_SIZE,
    max_hashes: int = MAX_HEADER_HASHES,
) -> list[bytes]:
    """Compute a cumulative HMAC-SHA256 chain for full cacheable chunks."""
    if max_hashes <= 0:
        return []

    parent = ROOT_PARENT_HASH
    hashes: list[bytes] = []
    for chunk in chunk_token_ids(
        token_ids,
        chunk_size=chunk_size,
        cacheable_only=True,
    ):
        payload = parent + pack_token_ids(chunk)
        parent = hmac.new(key, payload, hashlib.sha256).digest()
        hashes.append(parent)
        if len(hashes) >= max_hashes:
            break
    return hashes


def truncate_hash_chain(chain: list[bytes]) -> list[int]:
    """Truncate SHA-256 digests to 32-bit unsigned integers."""
    return [struct.unpack("!I", digest[:4])[0] for digest in chain]


def compute_truncated_hashes(
    token_ids: list[int],
    key: bytes,
    chunk_size: int = CHUNK_SIZE,
    max_hashes: int = MAX_HEADER_HASHES,
) -> list[int]:
    """Convenience wrapper returning the 32-bit KVSwitch shim hashes."""
    return truncate_hash_chain(
        compute_hash_chain(
            token_ids,
            key,
            chunk_size=chunk_size,
            max_hashes=max_hashes,
        )
    )
