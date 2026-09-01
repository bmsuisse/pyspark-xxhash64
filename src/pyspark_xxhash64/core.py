"""Pure-Python port of the XXH64 algorithm (Cyan4973/xxHash), bit-for-bit
compatible with the C reference implementation and with
``org.apache.spark.sql.catalyst.expressions.XXH64`` (Spark's own Java port
of the same reference algorithm).

Spark does not modify xxHash in any way -- ``XXH64.hashUnsafeBytes`` etc. are
a straight re-implementation of the upstream C code. That means any correct
XXH64 implementation (this one, or the compiled ``xxhash`` PyPI package)
already matches Spark for raw byte sequences; the Spark-specific behavior
that no generic library provides lives one layer up, in how each SQL type is
turned into bytes/ints before hashing (see ``hasher.py``).
"""
from __future__ import annotations

MASK64 = (1 << 64) - 1

PRIME64_1 = 0x9E3779B185EBCA87
PRIME64_2 = 0xC2B2AE3D27D4EB4F
PRIME64_3 = 0x165667B19E3779F9
PRIME64_4 = 0x85EBCA77C2B2AE63
PRIME64_5 = 0x27D4EB2F165667C5


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & MASK64


def _round(acc: int, lane: int) -> int:
    acc = (acc + lane * PRIME64_2) & MASK64
    acc = _rotl64(acc, 31)
    return (acc * PRIME64_1) & MASK64


def _merge_round(acc: int, val: int) -> int:
    val = _round(0, val)
    acc = (acc ^ val) & MASK64
    return (acc * PRIME64_1 + PRIME64_4) & MASK64


def _avalanche(h64: int) -> int:
    h64 ^= h64 >> 33
    h64 = (h64 * PRIME64_2) & MASK64
    h64 ^= h64 >> 29
    h64 = (h64 * PRIME64_3) & MASK64
    h64 ^= h64 >> 32
    return h64


def xxh64_unsigned(data: bytes, seed: int = 0) -> int:
    """XXH64(data, seed) as an unsigned 64-bit integer."""
    seed &= MASK64
    length = len(data)
    offset = 0

    if length >= 32:
        v1 = (seed + PRIME64_1 + PRIME64_2) & MASK64
        v2 = (seed + PRIME64_2) & MASK64
        v3 = seed
        v4 = (seed - PRIME64_1) & MASK64

        limit = length - 32
        while offset <= limit:
            v1 = _round(v1, int.from_bytes(data[offset:offset + 8], "little")); offset += 8
            v2 = _round(v2, int.from_bytes(data[offset:offset + 8], "little")); offset += 8
            v3 = _round(v3, int.from_bytes(data[offset:offset + 8], "little")); offset += 8
            v4 = _round(v4, int.from_bytes(data[offset:offset + 8], "little")); offset += 8

        h64 = (_rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)) & MASK64
        h64 = _merge_round(h64, v1)
        h64 = _merge_round(h64, v2)
        h64 = _merge_round(h64, v3)
        h64 = _merge_round(h64, v4)
    else:
        h64 = (seed + PRIME64_5) & MASK64

    h64 = (h64 + length) & MASK64

    while offset + 8 <= length:
        k1 = _round(0, int.from_bytes(data[offset:offset + 8], "little"))
        h64 = (h64 ^ k1) & MASK64
        h64 = (_rotl64(h64, 27) * PRIME64_1 + PRIME64_4) & MASK64
        offset += 8

    if offset + 4 <= length:
        h64 ^= int.from_bytes(data[offset:offset + 4], "little") * PRIME64_1 & MASK64
        h64 &= MASK64
        h64 = (_rotl64(h64, 23) * PRIME64_2 + PRIME64_3) & MASK64
        offset += 4

    while offset < length:
        h64 ^= (data[offset] * PRIME64_5) & MASK64
        h64 = (_rotl64(h64, 11) * PRIME64_1) & MASK64
        offset += 1

    return _avalanche(h64)


def _to_signed64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value >= (1 << 63) else value


def xxh64(data: bytes, seed: int = 0) -> int:
    """XXH64(data, seed) as a *signed* 64-bit integer (Spark's ``LongType``)."""
    return _to_signed64(xxh64_unsigned(data, seed))
