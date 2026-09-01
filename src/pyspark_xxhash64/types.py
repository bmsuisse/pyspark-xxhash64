"""Minimal mirror of the ``pyspark.sql.types`` classes that matter for
hashing. Deliberately not a full type system -- just enough shape
(``elementType``, ``keyType``/``valueType``, ``fields``, ``precision``/
``scale``) for :mod:`pyspark_xxhash64.hasher` to dispatch on, so schemas read
the same as they would in PySpark and a real ``pyspark.sql.types.StructType``
can be passed in directly (duck-typed, see ``hasher._type_name``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


class DataType:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class NullType(DataType):
    pass


class BooleanType(DataType):
    pass


class ByteType(DataType):
    pass


class ShortType(DataType):
    pass


class IntegerType(DataType):
    pass


class LongType(DataType):
    pass


class FloatType(DataType):
    pass


class DoubleType(DataType):
    pass


class StringType(DataType):
    pass


class BinaryType(DataType):
    pass


class DateType(DataType):
    """Value must be an int: days since the 1970-01-01 epoch."""


class TimestampType(DataType):
    """Value must be an int: microseconds since the 1970-01-01 epoch (UTC)."""


class TimestampNTZType(TimestampType):
    pass


class DayTimeIntervalType(LongType):
    """ANSI day-time interval; Spark stores it as total microseconds, so it
    hashes exactly like ``LongType``."""


class YearMonthIntervalType(IntegerType):
    """ANSI year-month interval; Spark stores it as total months, so it
    hashes exactly like ``IntegerType``."""


@dataclass(frozen=True)
class DecimalType(DataType):
    precision: int = 10
    scale: int = 0


@dataclass(frozen=True)
class ArrayType(DataType):
    elementType: DataType
    containsNull: bool = True


@dataclass(frozen=True)
class MapType(DataType):
    keyType: DataType
    valueType: DataType
    valueContainsNull: bool = True


@dataclass(frozen=True)
class StructField:
    name: str
    dataType: DataType
    nullable: bool = True


@dataclass(frozen=True)
class StructType(DataType):
    fields: list = field(default_factory=list)
