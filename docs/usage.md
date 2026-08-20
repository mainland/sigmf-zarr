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
checksum automatically whenever a chunk is read. Format 3 represents it in the
codec pipeline, while Zarr format 2 represents it as a numcodecs filter. Disable
checksums only when a caller has a specific reason to accept unprotected sample
chunks:

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

Collection creation and `set_recording_ids()` reject identifiers that do not
name an existing recording. `store.recordings.remove()` rejects a recording
that a collection references. Update the collection first.
