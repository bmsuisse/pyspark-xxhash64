//! No `#[test]`s in this crate: `pyo3`'s `extension-module` feature (needed
//! to build the loadable `.so` maturin ships) doesn't link against
//! `libpython`, which is exactly what a plain `cargo test` binary needs --
//! so tests here wouldn't link. Coverage is instead entirely transitive
//! through `../../tests/test_arrow_fast.py`, which exercises this function
//! (compiled via `maturin develop`) from Python for every supported type;
//! the actual hashing logic being tested lives in `spark-xxhash64-core`,
//! which does have its own `#[test]`s, and this crate is just a thin
//! Arrow-C-Data-Interface conversion wrapper around it.

use arrow::array::{make_array, Array, ArrayData};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Spark-compatible `xxhash64(col)` for a single pyarrow Array, computed
/// natively in Rust (no per-row Python overhead). `seed` defaults to 42,
/// matching `pyspark.sql.functions.xxhash64`.
#[pyfunction]
#[pyo3(signature = (array, seed=42))]
fn xxhash64(py: Python<'_>, array: &Bound<'_, PyAny>, seed: i64) -> PyResult<PyObject> {
    let data = ArrayData::from_pyarrow_bound(array)
        .map_err(|e| PyTypeError::new_err(format!("expected a pyarrow Array: {e}")))?;
    let arr = make_array(data);
    let hashed = spark_xxhash64_core::hash_array(arr.as_ref(), seed)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    hashed.into_data().to_pyarrow(py)
}

#[pymodule]
fn spark_xxhash64_pyarrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(xxhash64, m)?)?;
    Ok(())
}
