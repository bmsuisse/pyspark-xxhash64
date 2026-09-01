# spark-xxhash64-postgres

A Postgres extension exposing `spark_xxhash64(text) -> bigint`, bit-for-bit
compatible with `pyspark.sql.functions.xxhash64(string_col)`.

Built with [pgrx](https://github.com/pgcentralfoundation/pgrx), which
generates the SQL/Datum/FFI boilerplate from a `#[pg_extern]` annotation.
Reuses `spark-xxhash64-core::scalar::hash_str` -- the same function the
DuckDB extension calls -- so there is exactly one Rust implementation of
Spark's per-type encoding across every host.

## Scope

Only `spark_xxhash64(text) -> bigint` is wired up, same rationale as the
DuckDB extension: it's the primary use case (row/column hashing over text),
and extending to more types means adding more `#[pg_extern]` functions
calling the matching `spark_xxhash64_core::scalar::hash_*` -- the hashing
logic already exists.

## A real gotcha this surfaced (same shape as the DuckDB one)

The function takes `Option<&str>`, not `&str`. pgrx makes a plain `&str`
argument generate a `STRICT` SQL function, meaning Postgres itself
short-circuits any NULL argument straight to a NULL result without calling
the function at all -- SQL's usual NULL-in-NULL-out. That's *not* what
Spark's `xxhash64(NULL)` does: Spark returns the seed (42) unchanged for a
null single-column value. `Option<&str>` makes pgrx generate a non-STRICT
function, so the Rust code sees the NULL and returns the seed itself.

## Building and installing without root

This was built and tested in a sandbox where the extension author has
Postgres superuser but not OS root -- no `apt install postgresql-server-dev-18`,
no writing into `/usr/share/postgresql/18/extension` or
`/usr/lib/postgresql/18/lib`. Two things made that possible:

1. **Getting the build dependencies without root.** `apt-get download`
   doesn't need root (only installing via `apt-get install`/`dpkg -i`
   does), and `dpkg -x <deb> <dir>` extracts a package's files anywhere
   without installing it system-wide. This is how the Postgres server
   headers (`postgresql-server-dev-18`), OpenSSL dev headers
   (`libssl-dev`), `pkgconf`, and `libclang-19` (needed by pgrx's bindgen
   step) were obtained -- see `~/pg-dev-headers` and `~/local-deps` in this
   sandbox, wired together by `~/pg-dev-headers/env.sh` (sets `PATH`,
   `PKG_CONFIG_PATH` + `PKG_CONFIG_SYSROOT_DIR`, `LIBCLANG_PATH`,
   `BINDGEN_EXTRA_CLANG_ARGS=-resource-dir=...`) and a `pg_config` wrapper
   script that delegates to the real `/usr/bin/pg_config` for everything
   except `--includedir`/`--includedir-server`, which it redirects to the
   extracted headers. None of this is committed to the repo (it's
   sandbox-local setup) -- reproduce it with the same `apt-get download` +
   `dpkg -x` steps wherever you're building.

2. **Loading the extension without writing to root-owned directories.**
   Postgres 18 added `extension_control_path` and `dynamic_library_path`
   GUCs (the latter already existed) that let a superuser point Postgres at
   *additional* directories to search for `.control`/`.sql` files and
   `.so` libraries, on top of the compiled-in `$system`/`$libdir`
   defaults -- no filesystem write access to the Postgres install tree
   needed, only SQL-level superuser:

   ```sql
   ALTER SYSTEM SET extension_control_path = '$system:/home/YOU/pg-extensions/share';
   ALTER SYSTEM SET dynamic_library_path = '$libdir:/home/YOU/pg-extensions/lib';
   SELECT pg_reload_conf();
   ```

   (Also needed: the Postgres server process, running as OS user
   `postgres`, must be able to *traverse* into your home directory to
   reach those paths -- `chmod o+x ~` if your home directory doesn't
   already allow it. That grants directory traversal only, not file
   listing/reading, since none of the intermediate directories get `+r`.)

### Full sequence

```bash
source ~/pg-dev-headers/env.sh   # PATH, PKG_CONFIG_*, LIBCLANG_PATH, etc.

# once: point pgrx at the existing system Postgres instead of having it
# download/compile its own
cargo install --locked cargo-pgrx
cargo pgrx init --pg18 "$(which pg_config)"   # the wrapper script, not /usr/bin/pg_config

cd rust
cargo pgrx package -p spark_xxhash64_postgres --pg-config "$(which pg_config)"

mkdir -p ~/pg-extensions/share/extension ~/pg-extensions/lib
D=target/release/spark_xxhash64_postgres-pg18
cp "$D/usr/share/postgresql/18/extension/spark_xxhash64_postgres.control" ~/pg-extensions/share/extension/
cp "$D/usr/share/postgresql/18/extension/spark_xxhash64_postgres--"*.sql ~/pg-extensions/share/extension/
cp "$D/usr/lib/postgresql/18/lib/spark_xxhash64_postgres.so" ~/pg-extensions/lib/

psql -d postgres -c "ALTER SYSTEM SET extension_control_path = '\$system:$HOME/pg-extensions/share';"
psql -d postgres -c "ALTER SYSTEM SET dynamic_library_path = '\$libdir:$HOME/pg-extensions/lib';"
psql -d postgres -c "SELECT pg_reload_conf();"
psql -d postgres -c "CREATE EXTENSION spark_xxhash64_postgres;"
psql -d postgres -c "SELECT spark_xxhash64('hello');"
# -4367754540140381902
```

### Known gap: `cargo pgrx test`

`cargo pgrx test` manages its own separate throwaway Postgres data
directory (under `~/.pgrx/data-18`) for running the `#[pg_test]` functions
in `src/lib.rs`, but its extension-install step copies straight to
`pg_config`'s reported `PKGLIBDIR`/`SHAREDIR` (the real, root-owned system
paths) rather than going through the `extension_control_path` /
`dynamic_library_path` workaround above -- so it fails with `Permission
denied` in this no-root setup. The `#[pg_test]` functions in `src/lib.rs`
are the same assertions as `../../tests/test_postgres_extension.py`, which
*does* pass, running against the real system cluster via the GUC-based
install above. Fixing `cargo pgrx test` itself would mean also pointing
`pg_config`'s reported `PKGLIBDIR`/`SHAREDIR` at writable directories and
setting the same GUCs on pgrx's own dedicated test cluster -- doable, just
not done here since the Python-side test already gives full, real
coverage.

## Verification

`../../tests/test_postgres_extension.py` (skipped unless `psycopg` is
installed and the extension is reachable) checks: the same hardcoded Spark
test-suite string vectors used everywhere else in this repo, a scalar
literal call, NULL handling (the gotcha above), Unicode and empty strings,
and a full table column. All checked against
`pyspark_xxhash64.hasher.xxhash64`, itself verified directly against a real
Spark 4.2.0 session -- see the main README.

Tested against a real local PostgreSQL 18.4 (Ubuntu/PGDG) cluster, pgrx
0.19.2.
