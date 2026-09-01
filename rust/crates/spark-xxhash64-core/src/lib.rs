//! Rust port of the Spark-type-serialization layer from the sibling Python
//! package (`pyspark_xxhash64.hasher`), operating directly on Arrow arrays
//! instead of one Python value at a time. The XXH64 primitive itself comes
//! from the `xxhash-rust` crate (a widely used, independently correct
//! implementation) -- this crate's job is only the Spark-specific part:
//! per-type byte encoding and null passthrough. See that package's README
//! for why that split exists and how it was verified against a real Spark
//! session; this crate reuses the exact same encoding rules.

use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit};
use arrow::error::ArrowError;

const NAN32_BITS: u32 = 0x7fc0_0000;
const NAN64_BITS: u64 = 0x7ff8_0000_0000_0000;

#[inline]
fn hash_i32(v: i32, seed: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(&v.to_le_bytes(), seed)
}

#[inline]
fn hash_i64(v: i64, seed: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(&v.to_le_bytes(), seed)
}

#[inline]
fn hash_bytes(v: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(v, seed)
}

/// Mirrors Java's `Float.floatToIntBits`: canonicalizes every NaN bit
/// pattern to one representative value, and -0.0 to 0 (Spark special-cases
/// this so `-0.0f` hashes identically to `0.0f`).
#[inline]
fn float_bits(v: f32) -> u32 {
    if v.is_nan() {
        NAN32_BITS
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Mirrors Java's `Double.doubleToLongBits` (see `float_bits`).
#[inline]
fn double_bits(v: f64) -> u64 {
    if v.is_nan() {
        NAN64_BITS
    } else if v == 0.0 {
        0
    } else {
        v.to_bits()
    }
}

/// Minimal two's-complement big-endian bytes, matching Java's
/// `BigInteger.toByteArray()` (used by Spark for `Decimal` values whose
/// precision does not fit in a `long`, i.e. precision > 18).
fn java_biginteger_bytes(v: i128) -> Vec<u8> {
    let full = v.to_be_bytes(); // 16 bytes, big-endian two's complement
    for len in 1..=16usize {
        let bits = len * 8;
        if bits >= 128 {
            return full.to_vec();
        }
        let min = -(1i128 << (bits - 1));
        let max = (1i128 << (bits - 1)) - 1;
        if v >= min && v <= max {
            return full[16 - len..].to_vec();
        }
    }
    unreachable!()
}

macro_rules! primitive_hash_i32 {
    ($array:expr, $len:expr, $seed:expr, $seed_u:expr, $arr_ty:ty) => {{
        let arr = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        (0..$len)
            .map(|i| {
                if arr.is_null(i) {
                    $seed
                } else {
                    hash_i32(arr.value(i) as i32, $seed_u) as i64
                }
            })
            .collect()
    }};
}

macro_rules! primitive_hash_i64 {
    ($array:expr, $len:expr, $seed:expr, $seed_u:expr, $arr_ty:ty) => {{
        let arr = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        (0..$len)
            .map(|i| {
                if arr.is_null(i) {
                    $seed
                } else {
                    hash_i64(arr.value(i) as i64, $seed_u) as i64
                }
            })
            .collect()
    }};
}

macro_rules! bytes_hash {
    ($array:expr, $len:expr, $seed:expr, $seed_u:expr, $arr_ty:ty) => {{
        let arr = $array.as_any().downcast_ref::<$arr_ty>().unwrap();
        (0..$len)
            .map(|i| {
                if arr.is_null(i) {
                    $seed
                } else {
                    hash_bytes(arr.value(i).as_ref(), $seed_u) as i64
                }
            })
            .collect()
    }};
}

/// Spark-compatible `xxhash64(col)` for a single Arrow array, vectorized:
/// every row is hashed natively, with no per-row Python overhead. `seed`
/// defaults to 42 in Spark (`pyspark.sql.functions.xxhash64`'s seed).
///
/// A null element hashes to `seed` unchanged (Spark's null-passthrough
/// behavior for a single-column `xxhash64`) -- the output array therefore
/// never has nulls, matching Spark.
pub fn hash_array(array: &dyn Array, seed: i64) -> Result<Int64Array, ArrowError> {
    let len = array.len();
    let seed_u = seed as u64;

    let values: Vec<i64> = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        seed
                    } else {
                        hash_i32(if arr.value(i) { 1 } else { 0 }, seed_u) as i64
                    }
                })
                .collect()
        }
        DataType::Int8 => primitive_hash_i32!(array, len, seed, seed_u, Int8Array),
        DataType::Int16 => primitive_hash_i32!(array, len, seed, seed_u, Int16Array),
        DataType::Int32 => primitive_hash_i32!(array, len, seed, seed_u, Int32Array),
        DataType::Date32 => primitive_hash_i32!(array, len, seed, seed_u, Date32Array),
        DataType::Int64 => primitive_hash_i64!(array, len, seed, seed_u, Int64Array),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            primitive_hash_i64!(array, len, seed, seed_u, TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        seed
                    } else {
                        // Spark's TimestampType is microsecond-precision;
                        // truncate (not round) to match integer division
                        // semantics used when Spark itself downcasts.
                        hash_i64(arr.value(i).div_euclid(1_000), seed_u) as i64
                    }
                })
                .collect()
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        seed
                    } else {
                        hash_i32(float_bits(arr.value(i)) as i32, seed_u) as i64
                    }
                })
                .collect()
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        seed
                    } else {
                        hash_i64(double_bits(arr.value(i)) as i64, seed_u) as i64
                    }
                })
                .collect()
        }
        DataType::Utf8 => bytes_hash!(array, len, seed, seed_u, StringArray),
        DataType::LargeUtf8 => bytes_hash!(array, len, seed, seed_u, LargeStringArray),
        DataType::Binary => bytes_hash!(array, len, seed, seed_u, BinaryArray),
        DataType::LargeBinary => bytes_hash!(array, len, seed, seed_u, LargeBinaryArray),
        DataType::Decimal128(precision, _scale) => {
            let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let precision = *precision;
            (0..len)
                .map(|i| {
                    if arr.is_null(i) {
                        seed
                    } else {
                        let unscaled = arr.value(i);
                        if precision <= 18 {
                            hash_i64(unscaled as i64, seed_u) as i64
                        } else {
                            hash_bytes(&java_biginteger_bytes(unscaled), seed_u) as i64
                        }
                    }
                })
                .collect()
        }
        other => {
            return Err(ArrowError::NotYetImplemented(format!(
                "spark-xxhash64: unsupported Arrow type for xxhash64: {other:?}"
            )))
        }
    };

    Ok(Int64Array::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_match_spark_test_suite_vectors() {
        // Copied from Spark's HashExpressionsSuite.scala, UTF8_BINARY, seed=42.
        let arr = StringArray::from(vec!["AAA", "AAA  ", "aaa", "aaa   "]);
        let out = hash_array(&arr, 42).unwrap();
        assert_eq!(
            out.values(),
            &[
                3965631622972380050i64,
                196039582279068044,
                2465751751477118478,
                -2249763606958050730,
            ]
        );
    }

    #[test]
    fn hello_matches_c_reference_vector() {
        let arr = StringArray::from(vec!["hello"]);
        let out = hash_array(&arr, 42).unwrap();
        assert_eq!(out.value(0), -4367754540140381902i64);
    }

    #[test]
    fn null_passes_seed_through() {
        let arr = StringArray::from(vec![None, Some("x")]);
        let out = hash_array(&arr, 42).unwrap();
        assert_eq!(out.value(0), 42);
        assert_ne!(out.value(1), 42);
    }

    #[test]
    fn day_time_and_year_month_interval_vectors() {
        // SPARK-35113: dayTime -> total microseconds (LongType hashing),
        // yearMonth -> total months (IntegerType hashing). Seed 10.
        let micros = 1237123123i64 * 1_000_000;
        let long_arr = Int64Array::from(vec![micros]);
        assert_eq!(hash_array(&long_arr, 10).unwrap().value(0), 8228802290839366895u64 as i64);

        let int_arr = Int32Array::from(vec![1234]);
        assert_eq!(hash_array(&int_arr, 10).unwrap().value(0), -1774215319882784110i64);
    }

    #[test]
    fn float_double_negative_zero_and_nan() {
        let f = Float32Array::from(vec![0.0f32, -0.0f32, f32::NAN, -f32::NAN]);
        let out = hash_array(&f, 42).unwrap();
        assert_eq!(out.value(0), out.value(1));
        assert_eq!(out.value(2), out.value(3));

        let d = Float64Array::from(vec![0.0f64, -0.0f64]);
        let out = hash_array(&d, 42).unwrap();
        assert_eq!(out.value(0), out.value(1));
    }

    #[test]
    fn decimal_precision_boundary() {
        // precision <= 18 -> hash the i64 unscaled value directly.
        let small = Decimal128Array::from(vec![123457i128])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let via_small = hash_array(&small, 42).unwrap().value(0);
        let via_long = hash_array(&Int64Array::from(vec![123457i64]), 42).unwrap().value(0);
        assert_eq!(via_small, via_long);

        // precision > 18 -> BigInteger-byte-array path, must differ from
        // the raw i64 hash of the same numeric unscaled value.
        let big = Decimal128Array::from(vec![123457i128])
            .with_precision_and_scale(30, 5)
            .unwrap();
        let via_big = hash_array(&big, 42).unwrap().value(0);
        assert_ne!(via_big, via_long);
    }
}
