"""Proves the Rust fast path (pyspark_xxhash64.arrow) works without pyarrow
installed at all -- both directions go through the standardized Arrow
PyCapsule Interface (__arrow_c_array__), not a pyarrow-specific API. See
rust/crates/spark-xxhash64-pyarrow/src/lib.rs for how the output achieves
this (arrow-rs's own ToPyArrow hard-imports pyarrow; this crate implements
the export protocol itself instead, to avoid that).

Skipped unless `nanoarrow` and the compiled `spark_xxhash64_pyarrow`
extension are installed. Deliberately does NOT skip if pyarrow is missing --
that's the point of this file.
"""
import pytest

na = pytest.importorskip("nanoarrow")
pytest.importorskip("spark_xxhash64_pyarrow")

from pyspark_xxhash64 import arrow as fast  # noqa: E402


def test_1_native_extension_never_imports_pyarrow():
    # Named to run first in this file (pytest runs tests in declaration
    # order within a module): calls the compiled extension directly,
    # bypassing pyspark_xxhash64.arrow's own `try: import pyarrow` (an
    # *optional* convenience probe for the ChunkedArray case, unrelated to
    # whether the extension itself needs it -- see that module's
    # docstring). This is the part that actually matters: arrow-rs's
    # built-in ToPyArrow would have forced a pyarrow import here if
    # lib.rs's ArrowArrayExport didn't implement the export side of the
    # PyCapsule protocol itself.
    import sys

    if "pyarrow" in sys.modules:
        # Something else in this pytest run already imported pyarrow
        # (e.g. test_arrow_fast.py, if collected first) -- this check can
        # only prove anything when run in isolation, e.g.:
        #   pytest tests/test_arrow_fast_nanoarrow.py
        pytest.skip("pyarrow already imported by another test in this run")

    import spark_xxhash64_pyarrow as native

    arr = na.Array(["hello"], schema=na.string())
    result = native.xxhash64(arr, 42)
    na.Array(result).to_pylist()

    assert "pyarrow" not in sys.modules


def test_nanoarrow_round_trip_matches_spark_suite_vectors():
    values = ["AAA", "AAA  ", "aaa", "aaa   ", None, "hello"]
    arr = na.Array(values, schema=na.string())

    result = fast.xxhash64_array(arr)
    assert hasattr(result, "__arrow_c_array__")

    out = na.Array(result).to_pylist()
    assert out == [
        3965631622972380050,
        196039582279068044,
        2465751751477118478,
        -2249763606958050730,
        42,
        -4367754540140381902,
    ]
