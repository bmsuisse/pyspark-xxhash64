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

Correctness was established three ways, each covering a different layer:

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
3. **A real local Spark session.** A portable JDK 17 + `pyspark` were
   installed and every type was compared directly against
   `df.select(F.xxhash64("col"))` on Spark 4.2.0: float/double including NaN
   and `-0.0`, byte/short/int/long/bool, string (incl. non-ASCII), binary,
   date, timestamp, decimal (both the precision<=18 long-encoding and the
   precision>18 `BigInteger`-byte-array encoding, including the exact
   boundary at precision 18 vs 19), array (incl. null elements and empty
   arrays), map, struct, nested array-of-struct, multi-column hashing, and
   top-level null. All matched exactly. See `tests/test_spark_crosscheck.py`
   (skipped by default -- needs a JVM + `pyspark` to run).

Two non-obvious things that verification run surfaced, both now documented
in the test file:

- **Maps**: modern Spark (`spark.sql.legacy.allowHashOnMapType`) refuses to
  hash `MapType` columns by default with an analysis error -- it's not that
  the hash is undefined, Spark just won't compute it unless you opt back
  into the legacy behavior. This library will hash a Python dict as a map
  regardless; make sure that's actually what you want.
- **Type inference matters, exactly.** `spark.createDataFrame([("Alice", 30)], [...])`
  without an explicit schema infers a bare Python int as `LongType`, not
  `IntegerType`. Since `IntegerType` and `LongType` hash via different byte
  lengths (4 vs 8 bytes), picking the wrong one silently produces a
  different, still-plausible-looking hash. Always match the *actual* Spark
  column type, not the Python value's type.

## Fast path: Arrow, native Rust

The pure-Python `xxhash64()` above processes one value at a time -- fine for
small data, a bottleneck at scale. For a whole Arrow column hashed natively
(no per-row Python overhead, ~175x faster on strings), see [`rust/`](rust/).
No `pyarrow` dependency -- both directions go through the standardized
[Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html),
so `nanoarrow` (much lighter than pyarrow) works just as well:

```python
import nanoarrow as na
from pyspark_xxhash64 import arrow as fast

result = fast.xxhash64_array(na.Array(["hello", "world"], schema=na.string()))
na.Array(result).to_pylist()
```

(`pyarrow.Array` works identically if that's what you're already using --
`fast.xxhash64_array` doesn't care which Arrow implementation produced its
input, and its return value works with any of them too.)

Requires building the `spark-xxhash64-arrow` extension with maturin (see
`rust/README.md`); it's an optional extra, not a dependency of the base
package.

## Fast path: DuckDB extension

`rust/crates/spark-xxhash64-duckdb` is a DuckDB loadable extension exposing
`spark_xxhash64(VARCHAR) -> BIGINT` in SQL, built against DuckDB's new
stable C Extension API (not the older C++ extension template):

```sql
LOAD 'spark_xxhash64.duckdb_extension';
SELECT spark_xxhash64('hello'); -- -4367754540140381902
```

See `rust/crates/spark-xxhash64-duckdb/README.md` for build steps and
scope (currently VARCHAR only).

## Fast path: Postgres extension

`rust/crates/spark-xxhash64-postgres` is a Postgres extension (built with
[pgrx](https://github.com/pgcentralfoundation/pgrx)) exposing
`spark_xxhash64(text) -> bigint`:

```sql
CREATE EXTENSION spark_xxhash64_postgres;
SELECT spark_xxhash64('hello'); -- -4367754540140381902
```

See `rust/crates/spark-xxhash64-postgres/README.md` for build/install
steps -- including how this was done against a real Postgres cluster
without OS root access (extracting build dependencies from `.deb`s and
loading the extension via Postgres 18's `extension_control_path`/
`dynamic_library_path` GUCs).

## Running tests

```bash
uv pip install -e '.[test]'
pytest
```
