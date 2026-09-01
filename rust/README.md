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
  extension module, converting to/from Arrow zero-copy via the [Arrow
  PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
  (the standardized `__arrow_c_array__` protocol -- not pyarrow-specific,
  see "No pyarrow dependency" below).
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
entire Arrow column natively removes essentially all of the per-row Python
overhead. Measured on this machine (release build, 2M random ASCII strings,
5-30 chars each):

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
import nanoarrow as na
from pyspark_xxhash64 import arrow as fast

result = fast.xxhash64_array(na.Array(["hello", "world"], schema=na.string()))
na.Array(result).to_pylist()
# [-4367754540140381902, ...]
```

## No pyarrow dependency

Despite the crate/module names (kept for historical reasons -- this started
as a pyarrow-specific wrapper), neither `spark-xxhash64-pyarrow` nor
`pyspark_xxhash64.arrow` requires `pyarrow` to be installed. Both directions
go through the standardized Arrow PyCapsule Interface
(`__arrow_c_array__`):

- **Input**: `arrow-rs`'s `FromPyArrow` already checks for
  `__arrow_c_array__` before falling back to a pyarrow-specific import, so
  a `nanoarrow.Array`, `polars.Series.to_arrow()` result, or anything else
  implementing the protocol works as-is.
- **Output** needed an actual fix: `arrow-rs`'s built-in `ToPyArrow` for
  `ArrayData` unconditionally does `py.import("pyarrow")` to construct the
  result as a literal `pyarrow.Array`, which would force a `pyarrow`
  import even for an all-`nanoarrow` caller. `spark-xxhash64-pyarrow`
  doesn't use that -- it implements the export side of the PyCapsule
  protocol itself (`ArrowArrayExport` in `src/lib.rs`, including handling
  `requested_schema` so e.g. `pa.chunked_array(chunks, type=pa.int64())`
  still works), so the crate never touches pyarrow.

`../tests/test_arrow_fast_nanoarrow.py` proves this with a `nanoarrow`-only
round trip and a check (only meaningful run in isolation, since other test
files in the same `pytest` process legitimately do import pyarrow) that
`pyarrow` never ends up in `sys.modules`. It was also verified by building
the wheel and running it in a venv with only `nanoarrow` installed, no
`pyarrow` anywhere -- see that test file's docstring for the command.

`pyspark_xxhash64.arrow.xxhash64_array` still imports `pyarrow` -- but only
lazily, and only to special-case `pyarrow.ChunkedArray` input (chunking is
an inherently pyarrow-specific concept, not part of the PyCapsule
protocol); it's skipped entirely if `pyarrow` isn't installed or the input
isn't a `ChunkedArray`.

## Verification

Every type is compared in `../tests/test_arrow_fast.py` against
`pyspark_xxhash64.hasher.xxhash64` -- the pure-Python implementation, which
was itself checked directly against a real local Spark 4.2.0 session (see
the parent README's "Verification" section). A match here is therefore a
match against real Spark, transitively. `spark-xxhash64-core` additionally
has its own unit tests re-checking the hardcoded Spark test-suite vectors
(strings, day-time/year-month intervals) independently, in Rust, in
`crates/spark-xxhash64-core/src/lib.rs`.
