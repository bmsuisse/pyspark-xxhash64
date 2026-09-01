"""Spark-compatible ``xxhash64(col1, col2, ...)`` semantics on top of the
pure XXH64 core in :mod:`pyspark_xxhash64.core`.

This reimplements ``org.apache.spark.sql.catalyst.expressions.XxHash64`` /
``HashExpression`` from Spark's Catalyst engine
(sql/catalyst/.../expressions/hash.scala): each column's hash becomes the
seed for the next column, nulls pass the running hash through unchanged, and
containers (array/map/struct) recurse the same way over their elements.

Default seed is 42, matching ``pyspark.sql.functions.xxhash64``.
"""
from __future__ import annotations

import math
import struct
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Tuple

from .core import xxh64_unsigned, MASK64, _to_signed64

DEFAULT_SEED = 42

_NAN_FLOAT32_BITS = 0x7FC00000
_NAN_FLOAT64_BITS = 0x7FF8000000000000


def _hash_int32(value: int, seed: int) -> int:
    return xxh64_unsigned(struct.pack("<i", value), seed)


def _hash_int64(value: int, seed: int) -> int:
    return xxh64_unsigned(struct.pack("<q", value), seed)


def _hash_bytes(data: bytes, seed: int) -> int:
    return xxh64_unsigned(data, seed)


def _float_bits(value: float) -> int:
    # Mirrors Java's Float.floatToIntBits: canonicalizes every NaN bit
    # pattern to a single representative value (floatToRawIntBits would not).
    if math.isnan(value):
        return _NAN_FLOAT32_BITS
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return 0
    packed = struct.pack("<f", value)
    return struct.unpack("<i", packed)[0]


def _double_bits(value: float) -> int:
    # Mirrors Java's Double.doubleToLongBits (see _float_bits).
    if math.isnan(value):
        return _NAN_FLOAT64_BITS
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return 0
    packed = struct.pack("<d", value)
    return struct.unpack("<q", packed)[0]


def java_biginteger_bytes(n: int) -> bytes:
    """Minimal two's-complement big-endian bytes, matching Java's
    ``BigInteger.toByteArray()`` (used by Spark for ``Decimal`` values whose
    precision does not fit in a ``long``, i.e. precision > 18)."""
    length = 1
    while True:
        try:
            return n.to_bytes(length, "big", signed=True)
        except OverflowError:
            length += 1


def decimal_to_unscaled(value: Decimal, scale: int) -> int:
    """Convert a :class:`decimal.Decimal` to the unscaled integer Spark
    would store for a ``DecimalType(precision, scale)`` column, i.e.
    ``round(value * 10**scale)`` using half-up rounding (Spark's default
    rounding mode when adjusting a Decimal's scale)."""
    quant = Decimal(1).scaleb(-scale)
    return int(value.quantize(quant, rounding=ROUND_HALF_UP).scaleb(scale))


def _type_name(dtype: Any) -> str:
    return type(dtype).__name__


def compute_hash(value: Any, dtype: Any, seed: int) -> int:
    """Chain ``value``'s Spark-typed hash onto ``seed``. Returns an
    *unsigned* 64-bit int (the running accumulator); the public API converts
    the final result to Spark's signed ``LongType`` representation."""
    if value is None:
        return seed

    name = _type_name(dtype)

    if name == "NullType":
        return seed

    if name == "BooleanType":
        return _hash_int32(1 if value else 0, seed)

    if name in ("ByteType", "ShortType", "IntegerType", "DateType", "YearMonthIntervalType"):
        return _hash_int32(int(value), seed)

    if name in ("LongType", "TimestampType", "TimestampNTZType", "DayTimeIntervalType"):
        return _hash_int64(int(value), seed)

    if name == "FloatType":
        return _hash_int32(_float_bits(float(value)), seed)

    if name == "DoubleType":
        return _hash_int64(_double_bits(float(value)), seed)

    if name == "StringType":
        return _hash_bytes(value.encode("utf-8"), seed)

    if name == "BinaryType":
        return _hash_bytes(bytes(value), seed)

    if name == "DecimalType":
        precision = getattr(dtype, "precision", 38)
        scale = getattr(dtype, "scale", 0)
        if isinstance(value, Decimal):
            unscaled = decimal_to_unscaled(value, scale)
        else:
            unscaled = int(value)
        if precision <= 18:
            return _hash_int64(unscaled, seed)
        return _hash_bytes(java_biginteger_bytes(unscaled), seed)

    if name == "ArrayType":
        element_type = dtype.elementType
        result = seed
        for element in value:
            result = compute_hash(element, element_type, result)
        return result

    if name == "MapType":
        key_type = dtype.keyType
        value_type = dtype.valueType
        result = seed
        items = value.items() if hasattr(value, "items") else value
        for k, v in items:
            result = compute_hash(k, key_type, result)
            result = compute_hash(v, value_type, result)
        return result

    if name == "StructType":
        result = seed
        if hasattr(value, "items") and not isinstance(value, (list, tuple)):
            values = [value[f.name] for f in dtype.fields]
        else:
            values = list(value)
        for field_value, struct_field in zip(values, dtype.fields):
            result = compute_hash(field_value, struct_field.dataType, result)
        return result

    raise TypeError(f"Unsupported Spark type for xxhash64: {dtype!r}")


def xxhash64(*columns: Tuple[Any, Any], seed: int = DEFAULT_SEED) -> int:
    """Spark-compatible ``xxhash64(col1, col2, ...)``.

    Each argument is a ``(value, dtype)`` pair, ``dtype`` being one of the
    types in :mod:`pyspark_xxhash64.types` (or a real ``pyspark.sql.types``
    instance -- dispatch is duck-typed on the class name). Returns a signed
    64-bit int, exactly like the column PySpark's
    ``pyspark.sql.functions.xxhash64(*cols)`` produces.

    >>> from pyspark_xxhash64 import types as T
    >>> xxhash64(("hello", T.StringType()))
    -4367754540140381902
    """
    running = seed & MASK64
    for value, dtype in columns:
        running = compute_hash(value, dtype, running)
    return _to_signed64(running)
