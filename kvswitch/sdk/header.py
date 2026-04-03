"""KVSwitch shim header encode/decode helpers."""

import struct
from dataclasses import dataclass
from typing import Any, Final

MAX_HASHES: Final[int] = 4
HEADER_FORMAT: Final[str] = "!BBHIIII"
HEADER_SIZE: Final[int] = struct.calcsize(HEADER_FORMAT)


@dataclass(frozen=True)
class KVSwitchShimHeader:
    """Binary shim header carrying up to four cumulative prefix hashes."""

    version: int
    n_chunks: int
    flags: int
    req_id: int
    hashes: tuple[int, int, int, int]

    @classmethod
    def from_hashes(
        cls,
        hashes: list[int] | tuple[int, ...],
        version: int = 1,
        flags: int = 0,
        req_id: int = 0,
    ) -> "KVSwitchShimHeader":
        if not 0 <= version <= 0x0F:
            raise ValueError("version must fit in 4 bits")
        if not 0 <= len(hashes) <= MAX_HASHES:
            raise ValueError(f"expected at most {MAX_HASHES} hashes")
        if not 0 <= flags <= 0xFF:
            raise ValueError("flags must fit in 8 bits")
        if not 0 <= req_id <= 0xFFFF:
            raise ValueError("req_id must fit in 16 bits")

        padded = [0, 0, 0, 0]
        for idx, value in enumerate(hashes):
            padded[idx] = int(value) & 0xFFFFFFFF

        return cls(
            version=version,
            n_chunks=len(hashes),
            flags=flags,
            req_id=req_id,
            hashes=tuple(padded),  # type: ignore
        )

    def encode(self) -> bytes:
        packed_vn = ((self.version & 0x0F) << 4) | (self.n_chunks & 0x0F)
        return struct.pack(
            HEADER_FORMAT,
            packed_vn,
            self.flags,
            self.req_id,
            *self.hashes,
        )

    @classmethod
    def decode(cls, payload: bytes) -> "KVSwitchShimHeader":
        if len(payload) != HEADER_SIZE:
            raise ValueError(f"expected {HEADER_SIZE} bytes, got {len(payload)}")
        packed_vn, flags, req_id, h0, h1, h2, h3 = struct.unpack(
            HEADER_FORMAT,
            payload,
        )
        version = (packed_vn >> 4) & 0x0F
        n_chunks = packed_vn & 0x0F
        if n_chunks > MAX_HASHES:
            raise ValueError("n_chunks exceeds supported hash slots")
        return cls(
            version=version,
            n_chunks=n_chunks,
            flags=flags,
            req_id=req_id,
            hashes=(h0, h1, h2, h3),
        )

    def active_hashes(self) -> tuple[int, ...]:
        return self.hashes[: self.n_chunks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_chunks": self.n_chunks,
            "flags": self.flags,
            "req_id": self.req_id,
            "hashes": list(self.active_hashes()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KVSwitchShimHeader":
        hashes = [int(value) for value in payload.get("hashes", [])]
        return cls.from_hashes(
            hashes,
            version=int(payload.get("version", 1)),
            flags=int(payload.get("flags", 0)),
            req_id=int(payload.get("req_id", 0)),
        )
