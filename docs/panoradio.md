# Panoradio HF dataset

The `sigmf-zarr import panoradio` command imports the complex NumPy samples
and CSV tags in the
[Panoradio HF radio signal classification dataset](https://panoradio-sdr.de/radio-signal-classification-dataset/).
The published dataset contains 172800 items with 2048 complex samples each,
covering 18 high-frequency (HF) transmission modes at a sample rate of 6000
samples per second. The transmission-mode labels describe signal classes that
include protocols and modulation schemes.

## Command-line import

Import the sample file with its corresponding tags file:

```bash
sigmf-zarr import panoradio dataset_panoradio_hf.npy panoradio.zarr \
  --tags-file dataset_panoradio_hf_tags.csv
```

The source must be a nonempty `complex64` or `complex128` `.npy` array with shape
`(item, time)`. The importer also accepts compatible subsets with fewer items
or a different positive time length. The published item count, time length,
and mode vocabulary are not validation requirements.

The importer maps the source file into memory and converts bounded batches
to the SigMF-Zarr real-valued layout `(item, iq, time)`. The `iq` axis contains
I at index `0` and Q at index `1`. A `complex64` source produces `float32`
components, and a `complex128` source produces `float64` components. The
published file uses `complex128`. The importer preserves source row order and
component precision.
Set `--batch-size` to limit the items converted and copied per write. The
default is `4096`. The complete sample array is not loaded into memory, but
the aligned tag arrays remain in memory until they are written.

The recording name defaults to `panoradio`. The command and Python API record
`global["panoradio:source_dataset"]` as the source file name without its final
suffix. Use `--source-dataset` or the Python `source_dataset` argument to set
a different name. The importer sets `core:sample_rate` to `6000` and
`core:datatype` to `cf32_le` or `cf64_le` according to the component precision.

## Tag indexes

The CSV must have exactly the columns `idx`, `mode`, and `snr`, in any order.
The published header is `idx, mode, snr`. Spaces around the header fields and
values are ignored. Each row contains:

- A zero-based sample row index in `idx`
- A transmission-mode label in `mode`
- An integer signal-to-noise ratio (SNR) in dB in `snr`

The `idx` values must identify every source sample row exactly once. The
importer uses `idx` to align tags, so the CSV rows may appear in any order.
Duplicate, missing, and out-of-range row indexes are rejected. For a subset
copied into a new `.npy` file, renumber its tag indexes from zero.

The importer creates typed indexes aligned with the `item` axis:

| Index | Type | Meaning |
| --- | --- | --- |
| `mode_id` | `int32` | ID in the index's sorted `labels` attribute |
| `snr_db` | `int16` | SNR in dB |

The `mode_id` index uses the field `panoradio:mode`. Its labels preserve the
source transmission-mode names. The `snr_db` index uses the field
`panoradio:snr` and the unit `dB`. The importer derives the vocabulary from
the supplied tags and does not require all published modes or SNR levels.

## Storage options

The command accepts the shared `--recording-name`, `--overwrite-store`,
`--overwrite-recording`, `--zarr-format`, `--sample-compression`, and
`--sample-compression-level` options. New stores use Zarr format 3. Existing
stores have their Zarr format detected automatically. Use `--zarr-format 2`
to create a Zarr format 2 store.

Automatic Zarr format 3 storage targets approximately 256 KiB logical chunks
and 4 MiB physical shards. Use `--sample-shard-batch` to set the number of
items per shard, or `--no-sample-sharding` to disable sharding. Zarr format 2
does not support shards and cannot be combined with `--sample-shard-batch`.

Sample compression defaults to Zstandard. Select `zstd`, `lz4`, or `lz4hc` to
use the corresponding Blosc algorithm. Use `--sample-compression-level` to
set the Blosc level from `0` through `9`, or select `none` to disable
compression.

## Python API

```python
from sigmf_zarr import import_panoradio_dataset

store = import_panoradio_dataset(
    "panoradio.zarr",
    "dataset_panoradio_hf.npy",
    "dataset_panoradio_hf_tags.csv",
    batch_size=256,
)

recording = store.recordings["panoradio"]
print(recording.samples.shape)
print(recording.index("mode_id").attrs["labels"])
```

The API accepts `sample_chunks`, `sample_shards`, `automatic_sharding`, and
`sample_compressor` to configure sample storage. Use `global_metadata` to add
recording metadata. The importer sets the source dataset name, datatype, and
sample rate independently of `global_metadata`.

## Import provenance

Imports declare the project-defined `panoradio` metadata namespace at version
`0.1.0`. The [provenance record](provenance.md) identifies both the original
sample file and the CSV truth file by SHA-512. It records class ordering, the
half-open source row range, sample rate, and the precision-preserving conversion
from complex values to I/Q components. Inputs must remain unchanged while
hashing and importing. Hashing adds a complete read of both source files.
