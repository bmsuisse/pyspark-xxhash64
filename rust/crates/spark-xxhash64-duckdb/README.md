# spark-xxhash64-duckdb

A DuckDB loadable extension exposing `spark_xxhash64(VARCHAR) -> BIGINT`,
bit-for-bit compatible with `pyspark.sql.functions.xxhash64(string_col)`.

Built entirely against DuckDB's new, stable **C Extension API** (stable
since DuckDB v1.2), via the [`quack-rs`](https://crates.io/crates/quack-rs)
crate -- no C, no C++, and specifically **not** the older C++ extension
template. That was a hard requirement for this crate: DuckDB's own docs
note that writing a Rust extension traditionally forces you through the
CMake/C++ template with hand-written glue code; the new C API plus
`quack-rs` avoids that entirely.

Reuses `spark-xxhash64-core::scalar` for the actual hashing -- the same
per-type encoding used by the pyarrow extension -- so the Spark
compatibility logic exists in exactly one place.

## Scope

Only `spark_xxhash64(VARCHAR) -> BIGINT` is wired up. The published
`quack-rs` version (0.13.0) doesn't yet have the typed `map1_str`-style
closures used in the pyarrow crate (those are `main`-branch-only as of this
writing), so this uses the raw `ScalarFunctionBuilder` + an
`unsafe extern "C"` callback directly -- see `src/lib.rs`, it's about 40
lines.

Extending to more types (BIGINT, INTEGER, DOUBLE, BLOB, ...) means adding
more overloads of `spark_xxhash64` via DuckDB function-set registration,
each calling the matching `spark_xxhash64_core::scalar::hash_*` function --
the hashing logic already exists, only the DuckDB-side registration per
type is missing.

## A real gotcha this surfaced

DuckDB's default scalar-function NULL handling is SQL's NULL-in-NULL-out:
a NULL argument short-circuits to a NULL result without even calling your
callback. That is **not** what Spark's `xxhash64(NULL)` does -- Spark
returns the seed unchanged (42, not SQL NULL) for a null single-column
value. This extension registers `NullHandling::SpecialNullHandling` so
DuckDB hands NULL rows to the callback instead of intercepting them, and
the callback writes the seed itself. Skipping this gets you a
plausible-looking but wrong answer (NULL instead of 42) -- there's no error
to catch it.

## Building and loading

```bash
# from rust/
cargo build -p spark-xxhash64-duckdb --release

# once, to get the metadata-footer tool:
cargo install quack-rs --bin append_metadata

append_metadata target/release/libspark_xxhash64_duckdb.so \
    target/release/spark_xxhash64.duckdb_extension \
    --duckdb-version v1.2.0 \
    --platform linux_amd64
```

```python
import duckdb

con = duckdb.connect(config={"allow_unsigned_extensions": True})
con.execute("LOAD 'rust/target/release/spark_xxhash64.duckdb_extension';")
con.sql("SELECT spark_xxhash64('hello')").fetchone()
# (-4367754540140381902,)
```

(`allow_unsigned_extensions` is needed because this is a local unsigned
build, not one distributed through DuckDB's extension repository.)

## Verification

`../../tests/test_duckdb_extension.py` (skipped unless `duckdb` is
installed and the extension above has been built) checks: the same
hardcoded Spark test-suite string vectors used everywhere else in this
repo, a scalar literal call, NULL handling (the gotcha above), Unicode and
empty strings, and a full table column. All checked against
`pyspark_xxhash64.hasher.xxhash64`, which was itself verified directly
against a real Spark 4.2.0 session -- see the main README.

Tested against DuckDB 1.5.5 (Python `duckdb` package), within `quack-rs`
0.13's supported range of DuckDB 1.4.x/1.5.x (both expose C API version
`v1.2.0`).
