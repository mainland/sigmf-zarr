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

The Zarr `samples` array uses a NumPy-compatible element dtype. Complex SigMF
samples are stored as real components with an explicit length-2 `iq` axis.

## 1 Global

The `sigmf-zarr` extension adds the following fields to the `global` SigMF
object:

| name | required | type | description |
| --- | --- | --- | --- |
| `dtype` | true | string | NumPy dtype string for the `samples` array element type. |
| `sample-shape` | true | uint[] | Shape of an unbatched recording, or shape of one item in a batched recording. |
| `sample-axes` | true | string[] | Semantic axis names corresponding to `sample-shape`. |
| `num-channels` | true | uint | Length of the explicit `channel` axis, or `1` when no channel axis is present. |
| `checksum` | true | string or null | Chunk checksum name. The initial draft permits `crc32c` or `null`. |

The field names in metadata are prefixed with `sigmf-zarr:`, for example
`sigmf-zarr:sample-axes`.

`sample-axes` must contain exactly one `time` axis, must not contain duplicate
names, and must not contain the reserved runtime axis name `item`. An `iq` axis,
when present, has length 2. A `channel` axis may appear at most once.

A recording is batched when the `samples` array has one more axis than
`sample-shape`. The leading runtime axis is then named `item`.

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
