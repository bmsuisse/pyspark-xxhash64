"""Tests for the DuckDB loadable extension (rust/crates/spark-xxhash64-duckdb).

Skipped if the `duckdb` Python package isn't installed, or the extension
hasn't been built + packaged yet. Build it with:

    cd rust
    cargo build -p spark-xxhash64-duckdb --release
    cargo install quack-rs --bin append_metadata   # once
    append_metadata target/release/libspark_xxhash64_duckdb.so \\
        target/release/spark_xxhash64.duckdb_extension \\
        --duckdb-version v1.2.0 --platform linux_amd64

Correctness is checked against pyspark_xxhash64.hasher.xxhash64, itself
verified against a real Spark session (see the main README).

Non-obvious thing this uncovered: DuckDB's default NULL handling is SQL's
NULL-in-NULL-out, which is *not* what Spark's xxhash64(NULL) does (Spark
returns the seed unchanged, not SQL NULL, for a single-column call). The
extension registers `SpecialNullHandling` so DuckDB hands NULL rows to the
callback instead of short-circuiting them -- see lib.rs.
"""
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

EXTENSION_PATH = (
    Path(__file__).parent.parent / "rust" / "target" / "release" / "spark_xxhash64.duckdb_extension"
)
if not EXTENSION_PATH.exists():
    pytest.skip(f"extension not built at {EXTENSION_PATH}", allow_module_level=True)

from pyspark_xxhash64 import types as T  # noqa: E402
from pyspark_xxhash64.hasher import xxhash64  # noqa: E402


@pytest.fixture(scope="module")
def con():
    connection = duckdb.connect(config={"allow_unsigned_extensions": True})
    connection.execute(f"LOAD '{EXTENSION_PATH}';")
    yield connection
    connection.close()


def ref(values):
    return [42 if v is None else xxhash64((v, T.StringType())) for v in values]


def test_matches_spark_test_suite_vectors(con):
    vectors = {
        "AAA": 3965631622972380050,
        "AAA  ": 196039582279068044,
        "aaa": 2465751751477118478,
        "aaa   ": -2249763606958050730,
    }
    for s, expected in vectors.items():
        got = con.execute("SELECT spark_xxhash64(?)", [s]).fetchone()[0]
        assert got == expected


def test_scalar_literal(con):
    assert con.execute("SELECT spark_xxhash64('hello')").fetchone()[0] == -4367754540140381902


def test_null_returns_seed_not_sql_null(con):
    assert con.execute("SELECT spark_xxhash64(NULL::VARCHAR)").fetchone()[0] == 42


def test_unicode_and_empty_string(con):
    values = ["héllo wörld 日本語", ""]
    for v in values:
        got = con.execute("SELECT spark_xxhash64(?)", [v]).fetchone()[0]
        assert got == xxhash64((v, T.StringType()))


def test_column_of_mixed_values(con):
    values = ["AAA", "aaa", None, "hello", ""]
    con.execute("CREATE TABLE t(s VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?)", [(v,) for v in values])
    got = [row[0] for row in con.execute("SELECT spark_xxhash64(s) FROM t").fetchall()]
    assert got == ref(values)
