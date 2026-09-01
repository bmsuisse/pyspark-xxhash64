"""Direct cross-check against a real, local PySpark session.

Skipped by default (needs a JVM + pyspark, which aren't always on hand) --
run with a JDK on PATH/JAVA_HOME and `pyspark` installed to exercise it:

    uv pip install -e '.[test]' pyspark
    JAVA_HOME=/path/to/jdk PATH="$JAVA_HOME/bin:$PATH" pytest tests/test_spark_crosscheck.py -v

This is what actually proved out the parts of hasher.py that have no
hardcoded vector in Spark's own test suite: float/double NaN and -0.0
handling, decimal (both the <=18 and >18 precision encodings), array, map
(needs `spark.sql.legacy.allowHashOnMapType=true` -- modern Spark refuses to
hash MapType columns at all otherwise), struct, nested array-of-struct,
multi-column chaining, and top-level null passthrough. All matched on
Spark 4.2.0 / Python 3.14 / OpenJDK 17.

One gotcha this test caught: `spark.createDataFrame([("Alice", 30)], [...])`
without an explicit schema infers plain Python ints as LongType, not
IntegerType -- get the Spark type wrong and you silently get IntegerType's
4-byte encoding hashed instead of LongType's 8-byte one, which produces a
completely different (but still "valid-looking") hash.
"""
import datetime
import shutil
from decimal import Decimal

import pytest

pyspark = pytest.importorskip("pyspark")

from pyspark_xxhash64 import types as T  # noqa: E402
from pyspark_xxhash64.hasher import xxhash64  # noqa: E402

if not shutil.which("java"):
    pytest.skip("no java on PATH", allow_module_level=True)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.legacy.allowHashOnMapType", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def spark_hash(spark, spark_schema, value):
    from pyspark.sql import functions as F
    from pyspark.sql import types as ST

    df = spark.createDataFrame([(value,)], schema=ST.StructType([ST.StructField("x", spark_schema)]))
    return df.select(F.xxhash64("x")).collect()[0][0]


CASES = []


def case(label, spark_type_factory, spark_value, our_dtype, our_value):
    CASES.append((label, spark_type_factory, spark_value, our_dtype, our_value))


def _build_cases():
    from pyspark.sql import types as ST

    case("float_pos", ST.FloatType, 3.14, T.FloatType(), 3.14)
    case("float_neg_zero", ST.FloatType, -0.0, T.FloatType(), -0.0)
    case("float_nan", ST.FloatType, float("nan"), T.FloatType(), float("nan"))
    case("float_neg_nan", ST.FloatType, -float("nan"), T.FloatType(), -float("nan"))
    case("float_inf", ST.FloatType, float("inf"), T.FloatType(), float("inf"))
    case("double_pos", ST.DoubleType, 3.14159265, T.DoubleType(), 3.14159265)
    case("double_neg_zero", ST.DoubleType, -0.0, T.DoubleType(), -0.0)
    case("double_nan", ST.DoubleType, float("nan"), T.DoubleType(), float("nan"))
    case("byte", ST.ByteType, 42, T.ByteType(), 42)
    case("short", ST.ShortType, 12345, T.ShortType(), 12345)
    case("int", ST.IntegerType, -998877, T.IntegerType(), -998877)
    case("long", ST.LongType, 1234567890123, T.LongType(), 1234567890123)
    case("bool_true", ST.BooleanType, True, T.BooleanType(), True)
    case("string_unicode", ST.StringType, "héllo wörld 日本語", T.StringType(), "héllo wörld 日本語")
    case("binary", ST.BinaryType, bytearray(b"\x00\x01\x02\xff"), T.BinaryType(), b"\x00\x01\x02\xff")

    d = datetime.date(2023, 6, 15)
    days = (d - datetime.date(1970, 1, 1)).days
    case("date", ST.DateType, d, T.DateType(), days)

    ts = datetime.datetime(2023, 6, 15, 12, 30, 45, 123456)
    micros = int((ts - datetime.datetime(1970, 1, 1)).total_seconds() * 1_000_000)
    case("timestamp_ntz", ST.TimestampNTZType, ts, T.TimestampNTZType(), micros)

    case("decimal_p10s2", lambda: ST.DecimalType(10, 2), Decimal("12345.67"), T.DecimalType(10, 2), Decimal("12345.67"))
    case("decimal_p30s5", lambda: ST.DecimalType(30, 5), Decimal("123456789012345.67890"), T.DecimalType(30, 5), Decimal("123456789012345.67890"))
    case("decimal_p18_boundary", lambda: ST.DecimalType(18, 0), Decimal("999999999999999999"), T.DecimalType(18, 0), Decimal("999999999999999999"))
    case("decimal_p19_boundary", lambda: ST.DecimalType(19, 0), Decimal("9999999999999999999"), T.DecimalType(19, 0), Decimal("9999999999999999999"))

    case("array_int", lambda: ST.ArrayType(ST.IntegerType()), [1, 2, 3], T.ArrayType(T.IntegerType()), [1, 2, 3])
    case("array_with_null", lambda: ST.ArrayType(ST.IntegerType()), [1, None, 3], T.ArrayType(T.IntegerType()), [1, None, 3])
    case("array_empty", lambda: ST.ArrayType(ST.IntegerType()), [], T.ArrayType(T.IntegerType()), [])
    case("map_str_int", lambda: ST.MapType(ST.StringType(), ST.IntegerType()), {"a": 1, "b": 2},
         T.MapType(T.StringType(), T.IntegerType()), {"a": 1, "b": 2})

    struct_schema = ST.StructType([ST.StructField("a", ST.IntegerType()), ST.StructField("b", ST.StringType())])
    our_struct = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    case("struct", lambda: struct_schema, (1, "x"), our_struct, (1, "x"))
    case("array_of_struct", lambda: ST.ArrayType(struct_schema), [(1, "x"), (2, "y")],
         T.ArrayType(our_struct), [(1, "x"), (2, "y")])


_build_cases()


@pytest.mark.parametrize("label,spark_type_factory,spark_value,our_dtype,our_value",
                          CASES, ids=[c[0] for c in CASES])
def test_matches_real_spark(spark, label, spark_type_factory, spark_value, our_dtype, our_value):
    expected = spark_hash(spark, spark_type_factory(), spark_value)
    assert xxhash64((our_value, our_dtype)) == expected


def test_multi_column_matches_real_spark(spark):
    from pyspark.sql import functions as F

    df = spark.createDataFrame([("Alice", 30)], ["name", "age"])
    expected = df.select(F.xxhash64("name", "age")).collect()[0][0]
    # Bare Python ints infer as LongType without an explicit schema.
    assert xxhash64(("Alice", T.StringType()), (30, T.LongType())) == expected


def test_top_level_null_matches_real_spark(spark):
    from pyspark.sql import types as ST

    expected = spark_hash(spark, ST.IntegerType(), None)
    assert xxhash64((None, T.IntegerType())) == expected == 42
