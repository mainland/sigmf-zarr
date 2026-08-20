# Usage

Use SigMF-Zarr as a Python API for building Zarr-backed signal stores.

## Installation

Install a published release with pip:

```bash
python -m pip install sigmf-zarr
```

Install the package from the repository in editable mode during development:

```bash
uv sync
```

Run commands from the repository environment:

```bash
.venv/bin/python -m pytest
```

## Create a recording

A store location string is a URL only when it starts with an explicit scheme
followed by `://`, such as `s3://`, `http://`, `file://`, or `memory://`, or a
chained scheme followed by `::`, such as `simplecache::s3://bucket/store.zarr`.
Any other string, and every `pathlib.Path` or other path-like object, names a
local directory. SigMF-Zarr does not parse a local path as a URL, so characters
such as `#` and `?` are part of the directory name. The command line applies
the same rule to store arguments.

Create a store with one unbatched real-valued recording:

```python
import numpy as np

from sigmf_zarr import SigMFZarrStore

store = SigMFZarrStore.create("example.zarr", overwrite=True)
recording = store.recordings.create(
    "tone",
    batched=False,
    sample_dtype=np.float32,
    sample_shape=(1024,),
    sample_axes=("time",),
    global_metadata={
        "core:datatype": "rf32_le",
        "core:sample_rate": 1_000_000.0,
        "core:version": "1.2.0",
    },
    captures=[
        {
            "core:sample_start": 0,
            "core:frequency": 915_000_000.0,
        }
    ],
)
```

`SigMFZarrStore.create` defaults to Zarr format 3. To create the same logical
schema using the Zarr format 2 physical layout, pass `zarr_format=2`:

```python
store = SigMFZarrStore.create(
    "example-v2.zarr",
    overwrite=True,
    zarr_format=2,
)
```

`SigMFZarrStore.open` auto-detects both formats. This compatibility uses
zarr-python 3.2 or newer. It is not compatibility with the old zarr-python 2.x
package.

`SigMFZarrStore.create` refuses a target that already contains a Zarr
hierarchy unless `overwrite=True`. `SigMFZarrStore.open` never creates a store.
It opens read-only by default and accepts `mode="r+"` for writes. The
`SigMFZarrStore` constructor follows Zarr access modes. Its default mode, `"a"`,
opens an existing store or initializes an empty target. Requesting a Zarr
format that differs from an existing store raises `ValueError`.

Recordings and collections follow the same rule. `store.recordings.open()`
raises `KeyError` for a missing recording, and `store.recordings.create()`
raises `ValueError` for an existing one unless `overwrite=True`. Use a store as
a context manager, or call `close()`, to release its backend.

When `sample_chunks` is omitted, new recordings use logical chunks of about
1 MiB along the `item` axis of a batched recording or the `time` axis of an
unbatched recording. Pass `sample_chunks` to choose a different layout.

`sample_dtype` must be a Boolean, integer, floating-point, or complex dtype.
Recording creation raises `ValueError` for text, byte-string, date-time,
time-delta, object, void, and structured dtypes. Each element of a complex
dtype holds one I/Q pair, so recording creation also raises `ValueError` for a
complex dtype with an `iq` axis. Pass `sample_axes` explicitly for a complex
recording whose shape would infer an `iq` axis, such as `(2, 1024)`.

Extension arrays, indexes, and store-wide arrays must not use a structured or
void dtype. Array hashes do not cover structured field names, so
`add_extension_array()`, `add_index()`, and `SigMFZarrStore.create_array()`
raise `ValueError` for such values. Store each field as a separate array.

`sample_compressor`, and the `compressor` and `compressors` arguments of
`add_extension_array()` and `SigMFZarrStore.create_array()`, accept these
forms in both Zarr formats:

- `"auto"` for the default Zstandard compressor
- `None` for no compression
- A numcodecs `Blosc`, `Zstd`, or `GZip` instance
- A `zarr.codecs` `BloscCodec`, `ZstdCodec`, or `GzipCodec` instance

SigMF-Zarr translates each codec to the equivalent codec of the store's
format. Zarr format 2 also accepts any other numcodecs codec. Zarr format 3
also accepts any other bytes-to-bytes codec and its JSON mapping. Any other
compressor raises `ValueError` before any write. `compressors` also accepts a
sequence of codecs, applied in order. Zarr format 2 accepts at most one.

The recording stores SigMF-like `global`, `captures`, and `annotations`
metadata as Zarr group attributes. SigMF-Zarr adds storage metadata such as
`sigmf-zarr:sample-shape` and `sigmf-zarr:sample-axes` to the recording
`global` object.

## Sample axes

Use `sample_axes` to make the tensor layout explicit. Standard SigMF serializes
a `cf*` signal as interleaved complex samples in its flat `.sigmf-data` byte
stream. SigMF-Zarr instead stores the components in one real-valued Zarr array
with an explicit `iq` axis of length `2`: index `0` is I and index `1` is Q.
They are not separate arrays.

```python
recording = store.recordings.create(
    "array",
    batched=False,
    sample_dtype=np.float32,
    sample_shape=(4, 2, 2048),
    sample_axes=("channel", "iq", "time"),
    global_metadata={
        "core:datatype": "cf32_le",
        "core:num_channels": 4,
        "core:sample_rate": 2_000_000.0,
        "core:version": "1.2.0",
    },
    overwrite=True,
)

assert recording.sample_axes == ("channel", "iq", "time")
assert recording.runtime_axes == ("channel", "iq", "time")
assert recording.axis_index("time") == 2
assert recording.sample_count == 2048
```

For a batched recording, `sample_shape` describes one logical item and the
leading runtime axis is inferred as `item`:

```python
recording = store.recordings.create(
    "snippets",
    batched=True,
    sample_dtype=np.float32,
    sample_shape=(2, 128),
    sample_axes=("iq", "time"),
    overwrite=True,
)

assert recording.batched is True
assert recording.runtime_axes == ("item", "iq", "time")
```

Batching is derived from the stored array rank and
`sigmf-zarr:sample-shape`. There is no separate stored batching flag.

## Integrity

New sample arrays use per-chunk CRC32C checksums by default. Zarr validates the
checksum automatically whenever a chunk is read. Zarr format 3 represents it in
the codec pipeline, while Zarr format 2 represents it as a numcodecs filter.
Disable checksums only when a caller has a specific reason to accept
unprotected sample chunks:

```python
recording = store.recordings.create(
    "unchecked",
    batched=False,
    sample_shape=(1024,),
    sample_checksum=None,
)
```

Recording and collection names must identify immediate child groups. Names
must be nonempty and must not contain path separators, `.` or `..` path
components, or a NUL character.

Recording, collection, index, split, and extension-array names must not contain
control characters or lone surrogates, which are the Unicode categories `Cc`
and `Cs`. Such a name could forge terminal output or fail to encode as UTF-8.
The API raises `ValueError` for such a name, and `validate_store()` reports one
that another tool stored.

Index, split, and extension-array names must identify one array below their
group. A name may contain nested groups separated by `/`. It must not start or
end with `/`, contain an empty, `.`, or `..` segment, or contain a NUL
character. A name that identifies a group is not an index. Lookups and
`mutate_index()` raise `KeyError` for it, and `overwrite=True` does not replace
the group. A name below an existing array is rejected with `ValueError` before
any write. For example, with index `ids` present, `ids/c` is rejected, because
Zarr format 3 stores the chunks of `ids` under that prefix.

The Zarr metadata document names `.zarray`, `.zattrs`, `.zgroup`, `.zmetadata`,
and `zarr.json` are reserved in both Zarr formats. A recording or collection
name must not be a reserved name, and no segment of an index, split, or
extension-array name may be one. The API raises `ValueError` for a reserved
name before any write, and `validate_store()` reports a recording or collection
that another tool stored under a reserved name.

Collection creation and `set_recording_ids()` reject identifiers that do not
name an existing recording or that appear more than once.
`store.recordings.remove()` rejects a recording that a collection references.
Update the collection first.
