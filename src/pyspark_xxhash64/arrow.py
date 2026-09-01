"""Fast path: Spark-compatible ``xxhash64()`` over a whole pyarrow column at
once, computed natively in Rust (see ``rust/crates/spark-xxhash64-pyarrow``)
instead of one Python value at a time via :mod:`pyspark_xxhash64.hasher`.

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
    """Spark-compatible ``xxhash64(col)`` for one pyarrow ``Array`` or
    ``ChunkedArray``, returned as the same kind of pyarrow value (``int64``).

    Supported Arrow types: boolean, int8/16/32/64, float32/64, utf8,
    large_utf8, binary, large_binary, date32, timestamp (any unit --
    microsecond matches Spark's ``TimestampType`` directly; other units are
    converted to microseconds first), decimal128. Nested types
    (list/struct/map) are not yet supported by the native extension --
    fall back to :func:`pyspark_xxhash64.hasher.xxhash64` for those.
    """
    native = _require_native()

    import pyarrow as pa

    if isinstance(array, pa.ChunkedArray):
        hashed_chunks = [native.xxhash64(chunk, seed) for chunk in array.chunks]
        return pa.chunked_array(hashed_chunks, type=pa.int64())

    return native.xxhash64(array, seed)
