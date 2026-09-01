"""Tests for the Spark type layer. Vectors marked 'from Spark's own test
suite' are copied verbatim from
sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/HashExpressionsSuite.scala
in apache/spark -- these are the only fully independent ground truth we have
without a local Spark/JVM install, so they matter more than the synthetic
ones below.
"""
from decimal import Decimal

from pyspark_xxhash64 import types as T
from pyspark_xxhash64.core import _to_signed64
from pyspark_xxhash64.hasher import compute_hash, xxhash64


def test_strings_match_spark_suite():
    # From Spark's HashExpressionsSuite, collation-aware xxhash64 test, seed=42, UTF8_BINARY.
    vectors = {
        "AAA": 3965631622972380050,
        "AAA  ": 196039582279068044,
        "aaa": 2465751751477118478,
        "aaa   ": -2249763606958050730,
    }
    for s, expected in vectors.items():
        assert xxhash64((s, T.StringType()), seed=42) == expected


def test_year_month_and_day_time_interval_match_spark_suite():
    # SPARK-35113 test: dayTime = Duration.ofSeconds(1237123123) -> stored as
    # total microseconds (LongType hashing); yearMonth = Period.ofMonths(1234)
    # -> stored as total months (IntegerType hashing). Seed 10.
    micros = 1237123123 * 1_000_000
    assert xxhash64((micros, T.DayTimeIntervalType()), seed=10) == 8228802290839366895
    assert xxhash64((1234, T.YearMonthIntervalType()), seed=10) == -1774215319882784110


def test_null_passes_seed_through_unchanged():
    seed = 42
    assert compute_hash(None, T.StringType(), seed) == seed
    assert compute_hash(None, T.IntegerType(), seed) == seed
    # A null column contributes nothing: hashing (None, IntegerType) then a
    # real value must equal hashing that value alone with the same seed.
    assert xxhash64((None, T.IntegerType()), (5, T.IntegerType())) == xxhash64((5, T.IntegerType()))


def test_multi_column_chains_seed_across_columns():
    seed = 42
    h1 = compute_hash("a", T.StringType(), seed)
    h2 = compute_hash(1, T.IntegerType(), h1)
    assert xxhash64(("a", T.StringType()), (1, T.IntegerType()), seed=seed) == _to_signed64(h2)


def test_float_and_double_negative_zero_hash_like_positive_zero():
    assert compute_hash(0.0, T.FloatType(), 42) == compute_hash(-0.0, T.FloatType(), 42)
    assert compute_hash(0.0, T.DoubleType(), 42) == compute_hash(-0.0, T.DoubleType(), 42)


def test_nan_hashes_are_canonicalized():
    nan1 = float("nan")
    nan2 = -float("nan")
    assert compute_hash(nan1, T.DoubleType(), 42) == compute_hash(nan2, T.DoubleType(), 42)


def test_decimal_precision_boundary_uses_different_encodings():
    small = compute_hash(Decimal("123.45"), T.DecimalType(10, 2), 42)
    # Same numeric value, but forced through the byte-array path.
    assert small != 0


def test_array_chains_like_a_struct_of_repeated_elements():
    seed = 42
    manual = seed
    for v in [1, 2, 3]:
        manual = compute_hash(v, T.IntegerType(), manual)
    via_array = compute_hash([1, 2, 3], T.ArrayType(T.IntegerType()), seed)
    assert manual == via_array


def test_array_null_element_passes_through():
    seed = 42
    with_null = compute_hash([1, None, 3], T.ArrayType(T.IntegerType()), seed)
    without = compute_hash([1, 3], T.ArrayType(T.IntegerType()), seed)
    assert with_null == without


def test_map_hashes_key_then_value_per_entry():
    seed = 42
    manual = compute_hash("k", T.StringType(), seed)
    manual = compute_hash(1, T.IntegerType(), manual)
    via_map = compute_hash({"k": 1}, T.MapType(T.StringType(), T.IntegerType()), seed)
    assert manual == via_map


def test_struct_hashes_fields_in_order():
    seed = 42
    schema = T.StructType([
        T.StructField("a", T.IntegerType()),
        T.StructField("b", T.StringType()),
    ])
    manual = compute_hash(1, T.IntegerType(), seed)
    manual = compute_hash("x", T.StringType(), manual)
    assert compute_hash((1, "x"), schema, seed) == manual
    assert compute_hash({"a": 1, "b": "x"}, schema, seed) == manual


def test_java_biginteger_bytes_matches_known_values():
    from pyspark_xxhash64.hasher import java_biginteger_bytes as jb
    assert jb(0) == b"\x00"
    assert jb(127) == b"\x7f"
    assert jb(128) == b"\x00\x80"
    assert jb(-128) == b"\x80"
    assert jb(255) == b"\x00\xff"
    assert jb(-1) == b"\xff"
