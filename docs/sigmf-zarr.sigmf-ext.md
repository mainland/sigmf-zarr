# The `sigmf-zarr` SigMF extension namespace v0.1.0

## Abstract

This document defines the `sigmf-zarr` extension namespace for the Signal
Metadata Format (SigMF). The extension describes the logical sample tensor and
its Zarr storage properties when a recording is represented by the SigMF-Zarr
container format.

This is an initial format draft. Compatibility is not guaranteed between alpha
releases, and the schema will not be declared stable until the initial-format
work is complete.

## 0 Datatypes

The extension does not add a standard SigMF dataset datatype. Standard
`core:datatype` values continue to describe the byte representation used when a
recording is exported as conventional SigMF.

The Zarr `samples` array uses a NumPy-compatible Boolean, integer,
floating-point, or complex element dtype. Complex SigMF samples are stored as
real components with an explicit length-2 `iq` axis, or as elements of a
complex dtype without an `iq` axis. Each element of a complex dtype holds one
I/Q pair, so a complex dtype must not be combined with an `iq` axis.

## 1 Global

The `sigmf-zarr` extension adds the following fields to the `global` SigMF
object:

| name | required | type | description |
| --- | --- | --- | --- |
| `dtype` | true | string | Canonical little-endian NumPy dtype string for the `samples` array element type. |
| `sample-shape` | true | uint[] | Shape of an unbatched recording, or shape of one item in a batched recording. Every dimension is positive. |
| `sample-axes` | true | string[] | Semantic axis names corresponding to `sample-shape`. |
| `num-channels` | true | uint | Length of the explicit `channel` axis, or `1` when no channel axis is present. The value is positive. |
| `checksum` | true | string or null | Chunk checksum name. The initial draft permits `crc32c` or `null`. A recording without sample checksums declares `null`. |

The field names in metadata are prefixed with `sigmf-zarr:`, for example
`sigmf-zarr:sample-axes`.

A recording must declare this namespace exactly once in `core:extensions`,
with `version` `0.1.0` and `optional` `true`.

`dtype` must have the NumPy kind `b`, `i`, `u`, `f`, or `c`. Text, byte-string,
date-time, time-delta, object, void, and structured dtypes are not permitted.
`dtype` must use little-endian byte order (`<`), or `|` for a dtype with no
byte order, such as `|i1`. `dtype` must be the canonical NumPy spelling, which
is the `str` attribute of the NumPy dtype, such as `<f4` or `|i1`. Other
spellings, such as `float32`, `=f4`, or `>i1`, are not permitted. A writer
converts big-endian sample values to little-endian byte order, so the recorded
dtype does not depend on the source byte order or the Zarr format. The
`core:datatype` byte order continues to govern export to standard SigMF.

`sample-axes` must have the same length as `sample-shape`, must contain
exactly one `time` axis, must not contain duplicate names, and must not contain
the reserved runtime axis name `item`. An `iq` axis may appear at most once and
has length 2. `sample-axes` must not contain `iq` when `dtype` is complex. A
`channel` axis may appear at most once.

A recording is batched when the `samples` array has one more axis than
`sample-shape`. The leading runtime axis is then named `item`.

The namespace also names two attributes of Zarr index arrays, which are not
SigMF metadata fields. `sigmf-zarr:valid` is a Boolean. The value `false` marks
an index whose mutation did not complete or whose aligned axis changed, and
readers must reject such an index. Absence or `true` means valid. Readers must
also reject an index whose `sigmf-zarr:valid` value is not a Boolean. The
optional `sigmf-zarr:invalid-reason` string explains `sigmf-zarr:valid=false`.
The complete [format model](format.md) defines the index model.

## 2 Captures

The extension does not add fields to SigMF capture objects.

## 3 Annotations

The extension does not add fields to SigMF annotation objects.

## 4 Collections

The extension does not add fields to the standard SigMF collection object.
SigMF-Zarr containers may represent collections as Zarr groups, as specified by
the complete [format model](format.md).

## 5 Example

```json
{
  "global": {
    "core:datatype": "cf32_le",
    "core:sample_rate": 1000000.0,
    "core:version": "1.2.0",
    "core:extensions": [
      {
        "name": "sigmf-zarr",
        "version": "0.1.0",
        "optional": true
      }
    ],
    "sigmf-zarr:dtype": "<f4",
    "sigmf-zarr:sample-shape": [2, 1024],
    "sigmf-zarr:sample-axes": ["iq", "time"],
    "sigmf-zarr:num-channels": 1,
    "sigmf-zarr:checksum": "crc32c"
  },
  "captures": [{"core:sample_start": 0}],
  "annotations": []
}
```
