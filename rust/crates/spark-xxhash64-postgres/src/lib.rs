//! Postgres extension exposing `spark_xxhash64(text) -> bigint`, bit-for-bit
//! compatible with `pyspark.sql.functions.xxhash64(string_col)` (seed fixed
//! at 42, matching Spark's own public API -- Spark does not let callers pass
//! a custom seed either).
//!
//! Built with [pgrx](https://github.com/pgcentralfoundation/pgrx), which
//! generates the SQL/Datum/FFI boilerplate from the `#[pg_extern]`
//! annotation below.
//!
//! Reuses `spark-xxhash64-core::scalar::hash_str`, the same function the
//! DuckDB extension calls, so there is exactly one Rust implementation of
//! Spark's per-type encoding across every host.

use pgrx::prelude::*;

::pgrx::pg_module_magic!(name, version);

const SPARK_DEFAULT_SEED: i64 = 42;

/// `Option<&str>` (not `&str`) so a NULL argument reaches this function
/// instead of pgrx generating a STRICT SQL function that short-circuits to
/// SQL NULL -- Spark's `xxhash64(NULL)` returns the seed (42) unchanged,
/// not SQL NULL, and a STRICT function can't express that. See the
/// equivalent gotcha (and fix) in the sibling DuckDB extension's README.
#[pg_extern(immutable, parallel_safe)]
fn spark_xxhash64(s: Option<&str>) -> i64 {
    match s {
        Some(s) => spark_xxhash64_core::scalar::hash_str(s, SPARK_DEFAULT_SEED),
        None => SPARK_DEFAULT_SEED,
    }
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    // Copied from Spark's HashExpressionsSuite.scala, UTF8_BINARY, seed=42.
    #[pg_test]
    fn strings_match_spark_test_suite_vectors() {
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('AAA')").unwrap(),
            Some(3965631622972380050i64)
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('AAA  ')").unwrap(),
            Some(196039582279068044i64)
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('aaa')").unwrap(),
            Some(2465751751477118478i64)
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('aaa   ')").unwrap(),
            Some(-2249763606958050730i64)
        );
    }

    #[pg_test]
    fn hello_matches_c_reference_vector() {
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('hello')").unwrap(),
            Some(-4367754540140381902i64)
        );
    }

    #[pg_test]
    fn null_returns_seed_not_sql_null() {
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64(NULL::text)").unwrap(),
            Some(42i64)
        );
    }

    #[pg_test]
    fn unicode_and_empty_string() {
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('')").unwrap(),
            crate::spark_xxhash64(Some("")).into()
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT spark_xxhash64('héllo wörld 日本語')").unwrap(),
            crate::spark_xxhash64(Some("héllo wörld 日本語")).into()
        );
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
