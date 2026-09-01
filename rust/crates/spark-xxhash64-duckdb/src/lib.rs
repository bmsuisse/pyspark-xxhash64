//! DuckDB loadable extension exposing `spark_xxhash64(VARCHAR) -> BIGINT`,
//! bit-for-bit compatible with `pyspark.sql.functions.xxhash64(string_col)`
//! (seed fixed at 42, matching Spark's own public API -- Spark does not let
//! callers pass a custom seed either).
//!
//! Built entirely against DuckDB's new, stable C Extension API (stable
//! since DuckDB v1.2) via the `quack-rs` crate -- no C++, and not the older
//! C++ extension template, per the project's requirement.
//!
//! The actual per-type encoding logic lives in `spark-xxhash64-core`
//! (`scalar::hash_str`), shared with the pyarrow extension crate, so there
//! is exactly one Rust implementation of "how does Spark turn a value into
//! bytes before hashing", not one per host.
//!
//! Only VARCHAR is wired up so far (quack-rs 0.13's published API does not
//! yet have the typed `map1_str`-style closures used for the pyarrow crate
//! -- those are `main`-branch-only -- so this uses the raw scalar-function
//! builder directly). Extending to other types (BIGINT, INTEGER, DOUBLE,
//! BLOB, ...) means adding more raw callbacks under DuckDB function-set
//! overloads, each calling the matching `spark_xxhash64_core::scalar::hash_*`
//! function -- the hashing logic already exists, only the DuckDB-side
//! registration is missing.

use libduckdb_sys::{duckdb_connection, duckdb_data_chunk, duckdb_function_info, duckdb_vector};
use quack_rs::data_chunk::DataChunk;
use quack_rs::error::ExtensionError;
use quack_rs::scalar::ScalarFunctionBuilder;
use quack_rs::types::{NullHandling, TypeId};
use quack_rs::vector::VectorWriter;

const SPARK_DEFAULT_SEED: i64 = 42;

/// Called once per data chunk by DuckDB's expression executor.
///
/// # Safety
/// Invoked by DuckDB with its own valid handles: `input` has exactly one
/// VARCHAR column (declared at registration) and `output` is a writable
/// BIGINT vector with capacity for `chunk.size()` rows.
unsafe extern "C" fn xxhash64_varchar(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    // SAFETY: `input` is the chunk DuckDB handed this callback.
    let chunk = unsafe { DataChunk::from_raw(input) };
    // SAFETY: one VARCHAR parameter was declared, so column 0 exists.
    let reader = unsafe { chunk.reader(0) };
    // SAFETY: `output` is the writable BIGINT vector DuckDB handed this callback.
    let mut writer = unsafe { VectorWriter::from_vector(output) };

    for row in 0..chunk.size() {
        // SAFETY: `row` is within the chunk.
        let hash = if unsafe { reader.is_valid(row) } {
            // SAFETY: the column was declared VARCHAR.
            let s = unsafe { reader.read_str(row) };
            spark_xxhash64_core::scalar::hash_str(s, SPARK_DEFAULT_SEED)
        } else {
            // Spark's xxhash64(NULL) returns the seed unchanged, not SQL
            // NULL -- this function registers SpecialNullHandling so DuckDB
            // hands us the row instead of short-circuiting to NULL itself.
            SPARK_DEFAULT_SEED
        };
        // SAFETY: `row` is within the output vector's capacity.
        unsafe { writer.write_i64(row, hash) };
    }
}

fn register(con: duckdb_connection) -> Result<(), ExtensionError> {
    // SAFETY: `con` is the connection DuckDB handed the entry point.
    unsafe {
        ScalarFunctionBuilder::try_new("spark_xxhash64")?
            .param(TypeId::Varchar)
            .returns(TypeId::BigInt)
            .function(xxhash64_varchar)
            .null_handling(NullHandling::SpecialNullHandling)
            .register(con)?;
    }
    Ok(())
}

quack_rs::entry_point!(spark_xxhash64_init_c_api, |con| register(con));
