//! No `#[test]`s in this crate: `pyo3`'s `extension-module` feature (needed
//! to build the loadable `.so` maturin ships) doesn't link against
//! `libpython`, which is exactly what a plain `cargo test` binary needs --
//! so tests here wouldn't link. Coverage is instead entirely transitive
//! through `../../tests/test_arrow_fast.py`, which exercises this function
//! (compiled via `maturin develop`) from Python for every supported type;
//! the actual hashing logic being tested lives in `spark-xxhash64-core`,
//! which does have its own `#[test]`s, and this crate is just a thin
//! Arrow-C-Data-Interface conversion wrapper around it.
//!
//! Arrow-implementation-agnostic, not pyarrow-specific, despite the
//! `arrow-rs` crate feature this depends on being named "pyarrow" upstream
//! (see `Cargo.toml`). Neither direction requires `pyarrow` to be
//! installed: the input accepts anything implementing the standardized
//! [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
//! (`__arrow_c_array__`) -- `arrow-rs`'s `FromPyArrow` already checks for
//! that before falling back to a pyarrow-specific import, so a `nanoarrow`,
//! `polars`, or `pandas` (arrow-backed) array all work here already. The
//! *output*, though, needed a fix: `arrow-rs`'s built-in `ToPyArrow` for
//! `ArrayData` unconditionally does `py.import("pyarrow")` to construct the
//! result as a literal `pyarrow.Array` -- so calling this function forced a
//! `pyarrow` import even for an all-nanoarrow caller. `ArrowArrayExport`
//! below implements the same PyCapsule protocol *as the return value*
//! instead, so the caller picks how to consume it (`pa.array(result)`,
//! `nanoarrow.Array(result)`, `polars.from_arrow(result)`, ...) and this
//! crate never imports pyarrow itself.

use std::ffi::CString;

use arrow::array::{make_array, Array, ArrayData};
use arrow::datatypes::DataType;
use arrow::ffi::{to_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::pyarrow::FromPyArrow;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// A one-shot Arrow-PyCapsule-Interface-compliant export of a single array.
/// `__arrow_c_array__` consumes it (matching the FFI structs' own
/// single-owner release-callback semantics) -- calling it twice errors.
// `unsendable`: FFI_ArrowArray/FFI_ArrowSchema hold raw pointers and are not
// Send/Sync, but this object is only ever created and consumed within a
// single GIL-holding call on one thread, so that's fine.
#[pyclass(unsendable)]
struct ArrowArrayExport {
    array: Option<FFI_ArrowArray>,
    schema: Option<FFI_ArrowSchema>,
}

#[pymethods]
impl ArrowArrayExport {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &mut self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        // requested_schema is a raw "arrow_schema" PyCapsule per the spec
        // (not an object with __arrow_c_schema__ -- that's for values, this
        // is for the schema itself). We always produce int64, so accept a
        // request only if it's asking for exactly that; a caller like
        // `pyarrow.chunked_array(chunks, type=pa.int64())` passes this,
        // and it should just work rather than erroring on principle.
        if let Some(requested) = &requested_schema {
            let capsule = requested.downcast::<PyCapsule>().map_err(|_| {
                PyValueError::new_err(
                    "spark_xxhash64_arrow: requested_schema must be an 'arrow_schema' PyCapsule",
                )
            })?;
            // SAFETY: caller-provided per the Arrow PyCapsule Interface
            // contract; a bad capsule name/contents is caught below.
            let schema_ptr = unsafe { capsule.reference::<FFI_ArrowSchema>() };
            let requested_type = DataType::try_from(schema_ptr).map_err(|e| {
                PyValueError::new_err(format!(
                    "spark_xxhash64_arrow: invalid requested_schema: {e}"
                ))
            })?;
            if requested_type != DataType::Int64 {
                return Err(PyValueError::new_err(format!(
                    "spark_xxhash64_arrow: cannot satisfy requested_schema {requested_type:?} \
                     (this function always returns int64)"
                )));
            }
        }
        let array = self.array.take().ok_or_else(|| {
            PyRuntimeError::new_err("__arrow_c_array__ already consumed this export")
        })?;
        let schema = self.schema.take().ok_or_else(|| {
            PyRuntimeError::new_err("__arrow_c_array__ already consumed this export")
        })?;
        let schema_capsule = PyCapsule::new(py, schema, Some(c_name("arrow_schema")))?;
        let array_capsule = PyCapsule::new(py, array, Some(c_name("arrow_array")))?;
        Ok((schema_capsule, array_capsule))
    }
}

fn c_name(name: &str) -> CString {
    CString::new(name).expect("capsule name has no interior null byte")
}

/// Spark-compatible `xxhash64(col)` for a single Arrow array, computed
/// natively in Rust (no per-row Python overhead). `seed` defaults to 42,
/// matching `pyspark.sql.functions.xxhash64`.
///
/// Accepts any input implementing the Arrow PyCapsule Interface (a
/// `pyarrow.Array`, `nanoarrow.Array`, `polars.Series.to_arrow()` result,
/// etc.) and returns one too -- convert the result with whichever Arrow
/// library you're already using, e.g. `pa.array(result)` or
/// `na.Array(result)`.
#[pyfunction]
#[pyo3(signature = (array, seed=42))]
fn xxhash64(array: &Bound<'_, PyAny>, seed: i64) -> PyResult<ArrowArrayExport> {
    let data = ArrayData::from_pyarrow_bound(array)
        .map_err(|e| PyTypeError::new_err(format!("expected an Arrow array: {e}")))?;
    let arr = make_array(data);
    let hashed = spark_xxhash64_core::hash_array(arr.as_ref(), seed)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (array, schema) = to_ffi(&hashed.into_data()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(ArrowArrayExport {
        array: Some(array),
        schema: Some(schema),
    })
}

#[pymodule]
fn spark_xxhash64_arrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowArrayExport>()?;
    m.add_function(wrap_pyfunction!(xxhash64, m)?)?;
    Ok(())
}
