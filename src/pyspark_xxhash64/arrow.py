"""Fast path: Spark-compatible ``xxhash64()`` over a whole Arrow column at
once, computed natively in Rust (see ``rust/crates/spark-xxhash64-pyarrow``)
instead of one Python value at a time via :mod:`pyspark_xxhash64.hasher`.

Despite the crate/module names (kept for historical reasons -- this started
as a pyarrow-specific wrapper), **pyarrow is not required**. Both directions
go through the standardized `Arrow PyCapsule Interface
<https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
(``__arrow_c_array__``): the input can be a ``pyarrow.Array``,
``nanoarrow.Array``, a ``polars.Series.to_arrow()`` result, or anything else
implementing that protocol, and the return value implements it too --
convert it with whichever Arrow library you're already using
(``pa.array(result)``, ``na.Array(result)``, ...). pyarrow is only imported
here at all for the ``pyarrow.ChunkedArray`` convenience case below, and
only if it's already installed.

Requires the optional ``spark-xxhash64-pyarrow`` wheel (built with maturin
from this repo's ``rust/`` workspace) to be installed alongside this
package -- it is not a hard dependency of ``pyspark_xxhash64`` itself.
"""
from __future__ import annotations

from typing import Any

try:
    import spark_xxhash64_pyarrow as _native
except ImportError as _exc:  # pragma: no cover - exercised via skip in tests
    _native = None
    _import_error = _exc

DEFAULT_SEED = 42


def _require_native() -> Any:
    if _native is None:
        raise ImportError(
            "pyspark_xxhash64.arrow requires the compiled 'spark-xxhash64-pyarrow' "
            "extension. Build it with `maturin develop` (or `pip install`) from "
            "rust/crates/spark-xxhash64-pyarrow, see that folder's README."
        ) from _import_error
    return _native


def xxhash64_array(array: Any, seed: int = DEFAULT_SEED) -> Any:
    """Spark-compatible ``xxhash64(col)`` for one Arrow array (anything
    implementing ``__arrow_c_array__``, e.g. ``pyarrow.Array`` or
    ``nanoarrow.Array``) or ``pyarrow.ChunkedArray``. Returns a value
    implementing the same protocol (an ``ArrowArrayExport``, or a
    ``pyarrow.ChunkedArray`` for chunked input) -- convert it with whichever
    Arrow library you're using, e.g. ``pa.array(result)``.

    Supported Arrow types: boolean, int8/16/32/64, float32/64, utf8,
    large_utf8, binary, large_binary, date32, timestamp (any unit --
    microsecond matches Spark's ``TimestampType`` directly; other units are
    converted to microseconds first), decimal128. Nested types
    (list/struct/map) are not yet supported by the native extension --
    fall back to :func:`pyspark_xxhash64.hasher.xxhash64` for those.
    """
    native = _require_native()

    try:
        import pyarrow as pa
    except ImportError:
        pa = None

    if pa is not None and isinstance(array, pa.ChunkedArray):
        hashed_chunks = [native.xxhash64(chunk, seed) for chunk in array.chunks]
        return pa.chunked_array(hashed_chunks, type=pa.int64())

    return native.xxhash64(array, seed)
