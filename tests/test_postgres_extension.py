r"""Tests for the Postgres extension (rust/crates/spark-xxhash64-postgres).

Skipped if `psycopg` isn't installed, or the extension isn't reachable on
the local Postgres cluster. Build + install it with (no root needed --
headers/openssl/pkg-config/libclang were extracted from .debs with
`apt-get download` + `dpkg -x` into ~/pg-dev-headers and ~/local-deps, and
the extension itself is loaded via Postgres 18's extension_control_path /
dynamic_library_path GUCs instead of the root-owned system directories):

    source ~/pg-dev-headers/env.sh
    cd rust
    cargo pgrx package --pg-config "$(which pg_config)" -p spark_xxhash64_postgres

    cp target/release/spark_xxhash64_postgres-pg18/usr/share/postgresql/18/extension/spark_xxhash64_postgres* \\
        ~/pg-extensions/share/extension/
    cp target/release/spark_xxhash64_postgres-pg18/usr/lib/postgresql/18/lib/spark_xxhash64_postgres.so \\
        ~/pg-extensions/lib/

    psql -d postgres -c "ALTER SYSTEM SET extension_control_path = '\$system:$HOME/pg-extensions/share';"
    psql -d postgres -c "ALTER SYSTEM SET dynamic_library_path = '\$libdir:$HOME/pg-extensions/lib';"
    psql -d postgres -c "SELECT pg_reload_conf();"
    psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS spark_xxhash64_postgres;"

Correctness is checked against pyspark_xxhash64.hasher.xxhash64, itself
verified against a real Spark session (see the main README).

Same NULL-handling gotcha as the DuckDB extension: the Rust function takes
Option<&str> rather than &str, so pgrx does NOT generate a STRICT SQL
function -- a STRICT function would short-circuit a NULL argument straight
to SQL NULL, but Spark's xxhash64(NULL) returns the seed (42) unchanged,
not SQL NULL.
"""
import os

import pytest

psycopg = pytest.importorskip("psycopg")

PG_HOST = os.environ.get("PGHOST", "/var/run/postgresql")
PG_PORT = int(os.environ.get("PGPORT", "54322"))


def _connect():
    return psycopg.connect(host=PG_HOST, port=PG_PORT, dbname="postgres")


try:
    _conn = _connect()
    with _conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'spark_xxhash64_postgres'"
        )
        if cur.fetchone() is None:
            pytest.skip("spark_xxhash64_postgres extension not installed", allow_module_level=True)
except psycopg.OperationalError:
    pytest.skip(f"cannot reach postgres at {PG_HOST}:{PG_PORT}", allow_module_level=True)

from pyspark_xxhash64 import types as T  # noqa: E402
from pyspark_xxhash64.hasher import xxhash64  # noqa: E402


@pytest.fixture(scope="module")
def con():
    conn = _connect()
    yield conn
    conn.close()


def scalar(con, sql, params=None):
    with con.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def test_matches_spark_test_suite_vectors(con):
    vectors = {
        "AAA": 3965631622972380050,
        "AAA  ": 196039582279068044,
        "aaa": 2465751751477118478,
        "aaa   ": -2249763606958050730,
    }
    for s, expected in vectors.items():
        assert scalar(con, "SELECT spark_xxhash64(%s)", (s,)) == expected


def test_scalar_literal(con):
    assert scalar(con, "SELECT spark_xxhash64('hello')") == -4367754540140381902


def test_null_returns_seed_not_sql_null(con):
    assert scalar(con, "SELECT spark_xxhash64(NULL::text)") == 42


def test_unicode_and_empty_string(con):
    for v in ["héllo wörld 日本語", ""]:
        assert scalar(con, "SELECT spark_xxhash64(%s)", (v,)) == xxhash64((v, T.StringType()))


def test_table_column(con):
    values = ["AAA", "aaa", None, "hello", ""]
    with con.cursor() as cur:
        cur.execute("CREATE TEMP TABLE t(s text)")
        cur.executemany("INSERT INTO t VALUES (%s)", [(v,) for v in values])
        cur.execute("SELECT spark_xxhash64(s) FROM t ORDER BY s NULLS LAST")
        got = sorted(row[0] for row in cur.fetchall())
    expected = sorted(42 if v is None else xxhash64((v, T.StringType())) for v in values)
    assert got == expected
