# rust/

Native, vectorized implementations of the Spark-compatible `xxhash64`, for
when the pure-Python package (`../src/pyspark_xxhash64`) is too slow at
scale. This is a Cargo workspace with four crates:

- **`spark-xxhash64-core`** -- the actual logic: Spark's per-type byte
  encoding (mirrors `pyspark_xxhash64.hasher`) applied directly to Arrow
  arrays via [`arrow-rs`](https://github.com/apache/arrow-rs), using the
  [`xxhash-rust`](https://crates.io/crates/xxhash-rust) crate for the XXH64
  primitive itself (same reasoning as the Python package: don't
  reimplement a well-tested generic hash function, only the Spark-specific
  layer on top). No Python dependency -- reusable from a future DuckDB
  extension or anywhere else that hands you an Arrow array.
- **`spark-xxhash64-pyarrow`** -- a thin [PyO3](https://pyo3.rs) +
  [maturin](https://www.maturin.rs) wrapper exposing that as a Python
  extension module, converting to/from pyarrow zero-copy via `arrow-rs`'s
  `pyarrow` feature (the Arrow C Data Interface -- no serialization, no data
  copy for the input array).
- **`spark-xxhash64-duckdb`** -- a DuckDB loadable extension exposing
  `spark_xxhash64(VARCHAR) -> BIGINT` in SQL, built against DuckDB's new
  stable C Extension API via [`quack-rs`](https://crates.io/crates/quack-rs)
  (no C++, not the older extension template). Reuses the same
  `spark-xxhash64-core::scalar` functions. See its own README for scope,
  build steps, and a real NULL-handling gotcha it surfaced.
- **`spark-xxhash64-postgres`** -- a Postgres extension exposing
  `spark_xxhash64(text) -> bigint`, built with
  [pgrx](https://github.com/pgcentralfoundation/pgrx). Also reuses
  `spark-xxhash64-core::scalar`, and hits the same NULL-handling gotcha as
  the DuckDB extension. See its own README -- it also documents how this
  was built and installed against a real Postgres cluster *without* root
  (extracting build deps from `.deb`s with `dpkg -x`, and loading the
  extension via Postgres 18's `extension_control_path`/
  `dynamic_library_path` GUCs instead of the root-owned system
  directories).

## Why this exists

The pure-Python `xxhash64()` in the parent package processes one value at a
time; at scale that's the bottleneck, not the hash algorithm. Hashing an
entire pyarrow column natively removes essentially all of the per-row
Python overhead. Measured on this machine (release build, 2M random
ASCII strings, 5-30 chars each):

| Path | Throughput |
|---|---|
| Pure Python (`pyspark_xxhash64.hasher.xxhash64`) | ~235K rows/sec |
| Native Rust (`pyspark_xxhash64.arrow.xxhash64_array`) | ~41M rows/sec |

(~175x. Your mileage will vary with string length and CPU, but the pure
Python path is always going to be call-per-value-bound.)

## Supported types

boolean, int8/16/32/64, float32/64 (with Spark's NaN-canonicalization and
`-0.0` handling), utf8, large_utf8, binary, large_binary, date32, timestamp
(any unit -- non-microsecond units are converted to microseconds first,
matching Spark's `TimestampType`), decimal128 (both the precision<=18
long-encoding and the precision>18 big-integer-byte-array encoding). Nulls
hash to the seed unchanged, exactly like the Python package and like Spark.

**Not yet implemented**: list/struct/map (nested types) and decimal256.
These fall back to iterating in Python with `pyspark_xxhash64.hasher` for
now. Contributions welcome -- the recursive logic already exists in
`hasher.py` as a reference; porting it to Arrow's `ListArray`/`StructArray`/
`MapArray` is the main remaining gap.

## Building

```bash
# from rust/crates/spark-xxhash64-pyarrow, into whatever venv is active
maturin develop --release
```

```python
import pyarrow as pa
from pyspark_xxhash64 import arrow as fast

fast.xxhash64_array(pa.array(["hello", "world"]))
# <pyarrow.lib.Int64Array [-4367754540140381902, ...]>
```

## Verification

Every type is compared in `../tests/test_arrow_fast.py` against
`pyspark_xxhash64.hasher.xxhash64` -- the pure-Python implementation, which
was itself checked directly against a real local Spark 4.2.0 session (see
the parent README's "Verification" section). A match here is therefore a
match against real Spark, transitively. `spark-xxhash64-core` additionally
has its own unit tests re-checking the hardcoded Spark test-suite vectors
(strings, day-time/year-month intervals) independently, in Rust, in
`crates/spark-xxhash64-core/src/lib.rs`.
