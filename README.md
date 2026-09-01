# pyspark-xxhash64

A pure-Python, dependency-free reimplementation of PySpark's
`pyspark.sql.functions.xxhash64(*cols)`, so you can compute the exact same
row hashes Spark would produce without a JVM, without `pyspark` installed,
and (eventually) from Postgres.

## Why this exists

`pyspark.sql.functions.xxhash64` is not just the standard xxHash64 algorithm
run over your bytes. It's two layers:

1. **The XXH64 primitive** (Cyan4973/xxHash) -- this part is generic, and
   the `xxhash` PyPI package (C bindings to the real xxHash library) already
   computes it correctly. Nothing Spark-specific here.
2. **Spark's Catalyst type serialization** -- how each SQL type gets turned
   into bytes/ints before hashing, and how multiple columns and nested
   values (arrays/maps/structs) get chained into one hash by feeding each
   value's hash back in as the seed for the next. *This* part is
   Spark-specific and isn't provided by any existing generic hash library
   (checked: `xxhash`, `pyhash`, ClickHouse's `sparkXxHash64` only covers
   strings). This package implements it, ported directly from
   `org.apache.spark.sql.catalyst.expressions.{HashExpression,XxHash64,XXH64}`
   in `apache/spark`.

## Install

```bash
pip install -e .
```

## Usage

```python
from pyspark_xxhash64 import xxhash64, types as T

# Equivalent to: df.select(F.xxhash64(F.col("name"), F.col("age")))
xxhash64(("Alice", T.StringType()), (30, T.IntegerType()))

# Custom seed, equivalent to F.xxhash64(..., F.lit(seed))  is NOT how Spark's
# seed works -- Spark's seed is fixed per-expression (default 42), not a
# literal column. Pass it as a keyword instead:
xxhash64(("Alice", T.StringType()), seed=42)
```

Supported types (mirroring `pyspark.sql.types`, see `types.py`): `NullType`,
`BooleanType`, `ByteType`, `ShortType`, `IntegerType`, `LongType`,
`FloatType`, `DoubleType`, `DecimalType(precision, scale)`, `StringType`,
`BinaryType`, `DateType` (int days since epoch), `TimestampType` /
`TimestampNTZType` (int microseconds since epoch), `DayTimeIntervalType`,
`YearMonthIntervalType`, `ArrayType`, `MapType`, `StructType`.

You can pass a real `pyspark.sql.types` instance too -- dispatch is by class
name, not `isinstance`, so no hard dependency on `pyspark` is needed.

### Not implemented

- Legacy `CalendarIntervalType` (the pre-3.2, no-ANSI interval type stored as
  a months/days/microseconds triple). Rare in modern Spark; PRs welcome.
- Collation-aware hashing for non-`UTF8_BINARY` collations (ICU sort keys).
  Plain `StringType` (`UTF8_BINARY`, Spark's default) is fully supported and
  verified against Spark's own test vectors.
- `TimestampNanosVal` (nanosecond-precision timestamps, Spark 4.x preview).

## Verification

There's no JVM/pyspark in the sandbox this was built in, so correctness was
established two ways instead of "run it next to a real Spark cluster":

1. **The XXH64 core** (`core.py`) was checked against the actual C reference
   implementation from [Cyan4973/xxHash](https://github.com/Cyan4973/xxHash)
   (compiled locally with gcc, not reimplemented from memory) across 41
   vectors spanning every branch of the algorithm: 0-length input, the
   32-byte block loop, 8/4/1-byte remainders, and negative/out-of-`int32`
   seeds. See `tests/test_core_vectors.py`.
2. **The Spark type layer** (`hasher.py`) was checked against hardcoded
   expected values copied verbatim out of Spark's own
   `HashExpressionsSuite.scala` test file (`apache/spark`, `sql/catalyst`):
   4 string vectors (seed 42, e.g. `xxhash64("AAA") == 3965631622972380050`)
   and the `SPARK-35113` day-time/year-month interval vectors (seed 10).
   See `tests/test_hasher.py`.

Together these pin down: the hashing primitive itself, integer/long
fast-path encoding (via the interval vectors, which resolve to plain
`LongType`/`IntegerType` hashing), UTF-8 string hashing, and the multi-value
seed-chaining behavior. Types without an independent Spark test vector
available here (float/double NaN and `-0.0` canonicalization, decimal,
array/map/struct chaining) are implemented by directly porting Catalyst's
Scala/Java source and covered by self-consistency tests, but have **not**
been cross-checked against a real Spark run -- do that before relying on
them for anything precision-critical, e.g.:

```python
# in a real PySpark session
from pyspark.sql import functions as F
df.select(F.xxhash64(F.col("x")).alias("h")).show()
```

## Postgres

A Postgres-native implementation (SQL/PLpgSQL function, or C extension for
speed) is planned as a follow-up in `postgres/`, once the Python
implementation above has been validated against a real Spark cluster.

## Running tests

```bash
uv pip install -e '.[test]'
pytest
```
