"""Tests for kvswitch.sdk helpers."""

import hashlib
import hmac
import struct

import pytest

from kvswitch.sdk.hashing import (
    ROOT_PARENT_HASH,
    chunk_token_ids,
    compute_hash_chain,
    compute_truncated_hashes,
)
from kvswitch.sdk.header import HEADER_SIZE, KVSwitchShimHeader


class TestHashingHelpers:
    def test_chunk_token_ids_drops_partial_tail_by_default(self) -> None:
        token_ids = list(range(300))
        chunks = chunk_token_ids(token_ids, chunk_size=128)
        assert len(chunks) == 2
        assert all(len(chunk) == 128 for chunk in chunks)

    def test_compute_hash_chain_matches_manual_hmac_chain(self) -> None:
        token_ids = list(range(1, 513))
        key = b"secret"

        chain = compute_hash_chain(token_ids, key, chunk_size=256, max_hashes=2)

        first_payload = ROOT_PARENT_HASH + struct.pack("!256I", *token_ids[:256])
        first = hmac.new(key, first_payload, hashlib.sha256).digest()
        second_payload = first + struct.pack("!256I", *token_ids[256:512])
        second = hmac.new(key, second_payload, hashlib.sha256).digest()

        assert chain == [first, second]

    def test_compute_truncated_hashes_ignores_non_cacheable_tail(self) -> None:
        key = b"secret"
        full_block = list(range(256))
        with_tail = list(range(300))

        assert compute_truncated_hashes(with_tail, key) == compute_truncated_hashes(
            full_block, key
        )


class TestShimHeader:
    def test_binary_round_trip(self) -> None:
        header = KVSwitchShimHeader.from_hashes(
            [0x11111111, 0x22222222, 0x33333333],
            version=1,
            flags=7,
            req_id=42,
        )

        encoded = header.encode()
        decoded = KVSwitchShimHeader.decode(encoded)

        assert len(encoded) == HEADER_SIZE
        assert decoded == header
        assert decoded.active_hashes() == (0x11111111, 0x22222222, 0x33333333)

    def test_dict_round_trip(self) -> None:
        header = KVSwitchShimHeader.from_hashes([1, 2, 3, 4], flags=5, req_id=9)
        restored = KVSwitchShimHeader.from_dict(header.to_dict())
        assert restored == header

    def test_rejects_too_many_hashes(self) -> None:
        with pytest.raises(ValueError):
            KVSwitchShimHeader.from_hashes([1, 2, 3, 4, 5])
