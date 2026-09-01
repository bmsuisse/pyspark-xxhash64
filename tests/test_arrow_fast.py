"""Tests for the native Rust Arrow fast path (pyspark_xxhash64.arrow).

Skipped entirely if either `pyarrow` or the compiled `spark_xxhash64_arrow`
extension isn't installed -- build the latter with:

    cd rust/crates/spark-xxhash64-arrow && maturin develop --release

pyarrow is used here only as a convenient way to build test input and read
the result back (via `pa.array(result)`) -- the extension itself does not
require pyarrow specifically, it works with anything implementing the Arrow
PyCapsule Interface (`__arrow_c_array__`). See test_arrow_fast_nanoarrow.py
for a pyarrow-free round trip proving that.

Correctness is checked against pyspark_xxhash64.hasher.xxhash64 (the pure
Python implementation, itself verified against a real Spark session -- see
tests/test_spark_crosscheck.py and the README), so a match here transitively
means a match against real Spark too.
"""
from decimal import Decimal

import pytest

pa = pytest.importorskip("pyarrow")
pytest.importorskip("spark_xxhash64_arrow")

from pyspark_xxhash64 import arrow as fast  # noqa: E402
from pyspark_xxhash64 import types as T  # noqa: E402
from pyspark_xxhash64.hasher import xxhash64  # noqa: E402


def ref(values, dtype, seed=42):
    return [seed if v is None else xxhash64((v, dtype), seed=seed) for v in values]


def hashed(array, seed=42):
    """Run the fast path and read the result back via pyarrow, regardless
    of whether it's our ``ArrowArrayExport`` (single array) or a real
    ``pyarrow.ChunkedArray`` (chunked input) -- both implement the Arrow
    PyCapsule Interface / already have ``to_pylist``."""
    result = fast.xxhash64_array(array, seed)
    if hasattr(result, "to_pylist"):
        return result.to_pylist()
    return pa.array(result).to_pylist()


def test_strings_match_reference_and_spark_suite_vectors():
    values = ["AAA", "AAA  ", "aaa", "aaa   ", None, "hello"]
    out = hashed(pa.array(values))
    assert out == ref(values, T.StringType())
    assert out[:4] == [3965631622972380050, 196039582279068044, 2465751751477118478, -2249763606958050730]


def test_large_string_and_binary():
    values = ["a", "bb", None, "ccc"]
    out = hashed(pa.array(values, type=pa.large_string()))
    assert out == ref(values, T.StringType())

    bvalues = [b"\x00\x01", None, b"\xff" * 40]
    out_bin = hashed(pa.array(bvalues, type=pa.binary()))
    assert out_bin == ref(bvalues, T.BinaryType())

    out_lbin = hashed(pa.array(bvalues, type=pa.large_binary()))
    assert out_lbin == ref(bvalues, T.BinaryType())


def test_all_integer_widths_and_bool():
    i8 = [1, -1, 0, 127, -128, None]
    assert hashed(pa.array(i8, type=pa.int8())) == ref(i8, T.ByteType())

    i16 = [1, -1, 0, 32767, -32768, None]
    assert hashed(pa.array(i16, type=pa.int16())) == ref(i16, T.ShortType())

    i32 = [1, -1, 0, 2147483647, -2147483648, None]
    assert hashed(pa.array(i32, type=pa.int32())) == ref(i32, T.IntegerType())

    i64 = [1, -1, 0, 2**62, -(2**62), None]
    assert hashed(pa.array(i64, type=pa.int64())) == ref(i64, T.LongType())

    b = [True, False, None]
    assert hashed(pa.array(b)) == ref(b, T.BooleanType())


def test_float_and_double_special_values():
    f32 = [1.5, -0.0, 0.0, float("nan"), -float("nan"), None]
    assert hashed(pa.array(f32, type=pa.float32())) == ref(f32, T.FloatType())

    f64 = [1.5, -0.0, 0.0, float("nan"), None]
    assert hashed(pa.array(f64, type=pa.float64())) == ref(f64, T.DoubleType())


def test_date32():
    import datetime

    dates = [datetime.date(1970, 1, 1), datetime.date(2023, 6, 15), None]
    days = [None if d is None else (d - datetime.date(1970, 1, 1)).days for d in dates]
    out = hashed(pa.array(dates, type=pa.date32()))
    assert out == ref(days, T.DateType())


def test_timestamp_microsecond_and_nanosecond_truncation():
    import datetime

    ts = [datetime.datetime(2023, 6, 15, 12, 30, 45, 123456), None]
    micros = [None if t is None else int((t - datetime.datetime(1970, 1, 1)).total_seconds() * 1_000_000) for t in ts]
    out_us = hashed(pa.array(ts, type=pa.timestamp("us")))
    assert out_us == ref(micros, T.TimestampType())

    out_ns = hashed(pa.array(ts, type=pa.timestamp("ns")))
    assert out_ns == out_us  # nanosecond input truncates to microseconds, same as Spark's TimestampType


def test_timestamp_millisecond_and_second_units():
    ms = hashed(pa.array([1234, None], type=pa.timestamp("ms")))
    s = hashed(pa.array([5, None], type=pa.timestamp("s")))
    us_from_ms = hashed(pa.array([1_234_000, None], type=pa.timestamp("us")))
    us_from_s = hashed(pa.array([5_000_000, None], type=pa.timestamp("us")))
    assert ms == us_from_ms
    assert s == us_from_s


def test_negative_nanosecond_timestamp_floors_towards_negative_infinity():
    # -1500ns floors to -2us (Spark's Math.floorDiv convention), not -1us
    # truncated-toward-zero -- these differ for pre-1970 values.
    ns = hashed(pa.array([-1500], type=pa.timestamp("ns")))
    us = hashed(pa.array([-2], type=pa.timestamp("us")))
    assert ns == us


def test_decimal_precision_boundary():
    small = [Decimal("12345.67"), Decimal("-999.99"), None]
    out = hashed(pa.array(small, type=pa.decimal128(10, 2)))
    assert out == ref(small, T.DecimalType(10, 2))

    big = [Decimal("123456789012345.67890"), None]
    out_big = hashed(pa.array(big, type=pa.decimal128(30, 5)))
    assert out_big == ref(big, T.DecimalType(30, 5))


def test_chunked_array_reassembles_correctly():
    values = ["a", "b", "c", "d", "e"]
    chunked = pa.chunked_array([pa.array(values[:2]), pa.array(values[2:])])
    out = fast.xxhash64_array(chunked)
    assert isinstance(out, pa.ChunkedArray)
    assert out.to_pylist() == ref(values, T.StringType())


def test_empty_array():
    assert hashed(pa.array([], type=pa.utf8())) == []


def test_all_null_array():
    values = [None, None, None]
    assert hashed(pa.array(values, type=pa.utf8())) == [42, 42, 42]


def test_output_implements_arrow_pycapsule_interface():
    result = fast.xxhash64_array(pa.array(["x"]))
    assert hasattr(result, "__arrow_c_array__")
    schema_capsule, array_capsule = result.__arrow_c_array__()
    assert schema_capsule.__class__.__name__ == "PyCapsule"
    assert array_capsule.__class__.__name__ == "PyCapsule"
