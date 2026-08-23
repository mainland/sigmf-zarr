# Chad Spooner CSPB datasets

The `sigmf-zarr import cspb` command imports the RF machine-learning datasets
published by Chad Spooner on the Cyclostationary Signal Processing Blog.
Their published names include CSPB.ML.2018, CSPB.ML.2022, CSPB.ML.2023, and later
corrected or generalized releases. The importer uses the dataset name supplied
by the user rather than inferring a particular release from a filename.

For new work, use the corrected
[CSPB.ML.2018R2](https://cyclostationary.blog/2023/09/25/cspb-ml-2018r2-correcting-an-rng-flaw-in-cspb-ml-2018/)
and
[CSPB.ML.2022R2](https://cyclostationary.blog/2023/10/02/cspb-ml-2022r2-correcting-an-rng-flaw-in-cspb-ml-2022/)
releases where applicable. The complete set of releases and Spooner's MATLAB
reader are linked from the
[CSP Blog dataset page](https://cyclostationary.blog/data-sets/).

## The `.tim` format

Spooner's
[`read_binary.m`](https://cyclostationary.blog/wp-content/uploads/2015/09/read_binary.doc)
defines a compact binary layout:

1. One signed 32-bit integer containing `1` for real data or `2` for complex
   data.
2. One signed 32-bit integer containing the logical sample count $N$.
3. Either $N$ real float32 values or $2N$ float32 values alternating I and Q.

The format has no explicit byte-order marker. `read_tim()` detects byte order
from the real/complex header field and validates the declared sample count
against the payload length. It accepts little- and big-endian files.

```python
from sigmf_zarr import read_tim

samples = read_tim("signal_1.tim")
```

Like Spooner's MATLAB reader, `read_tim()` returns a one-dimensional float32
array for a real file or complex64 array for a complex file. During dataset
import, complex samples are converted to SigMF-Zarr's real-valued `(iq, time)`
representation.

## Command-line import

The source can be:

- One `.tim` file
- One ZIP batch containing `.tim` members
- A directory containing extracted `.tim` files, ZIP batches, or both

ZIP batches are read directly and are not extracted to temporary storage.
Files are ordered by the numeric suffix in names such as `signal_123.tim` or
`psk_mixtures_123.tim`.

The following command is recommended for an extracted CSPB.ML.2018R2 dataset.
It uses Blosc with Zstandard at a balanced compression level, retains automatic
sharding, and imports the published truth file:

```bash
sigmf-zarr import cspb CSPB.ML.2018R2 cspb.zarr \
  --truth-file signal_record_C_2023.txt \
  --source-dataset CSPB.ML.2018R2 \
  --sample-compression zstd \
  --sample-compression-level 3
```

The source directory may instead contain the downloaded batch ZIP files:

```bash
sigmf-zarr import cspb downloaded-batches cspb.zarr \
  --truth-file signal_record_C_2023.txt \
  --source-dataset CSPB.ML.2018R2
```

Repeat `--truth-file` when a source directory combines datasets whose metadata
is split across files. This is useful for the CSPB.ML.2023 single-signal and
two-signal truth files:

```bash
sigmf-zarr import cspb psk-mixtures cspb.zarr \
  --truth-file PM_single_truth_10000.txt \
  --truth-file PM_two_truth_10000.txt \
  --source-dataset CSPB.ML.2023
```

Truth data is optional. Releases such as CSPB.ML.2023G1 that intentionally
withhold labels can still be imported. The resulting recording retains signal
IDs and source filenames but has no label-derived indexes.

The command shares the standard import options:

```text
--recording-name
--overwrite-store
--overwrite-recording
--zarr-format {2,3}
```

It also accepts `--batch-size`, `--sample-compression`,
`--sample-compression-level`, `--sample-shard-batch`, and
`--no-sample-sharding`. Existing Zarr format 2 stores are detected
automatically. New format-3 stores group CSPB items into logical chunks
targeting approximately 256 KiB and physical shards targeting approximately 4
MiB. Use `--sample-shard-batch` to override the derived item count. Format-2
stores and format-3 imports with `--no-sample-sharding` use sample-major chunks
targeting approximately 4 MiB instead.

Sample compression defaults to Zstandard. Select `zstd`, `lz4`, or `lz4hc` to
use the corresponding Blosc algorithm. Use `--sample-compression-level` to set
the Blosc level from 0 through 9, or select `none` to disable compression.

## Truth metadata

The importer detects two published truth layouts:

`cspb-ml`
: The nine-field layout documented for
  [CSPB.ML.2018](https://cyclostationary.blog/2019/02/15/data-set-for-the-machine-learning-challenge/)
  and reused by CSPB.ML.2022 and their corrected releases. It contains the
  signal index, modulation, base symbol period, carrier offset, excess
  bandwidth, resampling factors, in-band SNR, and noise spectral density.

`psk-mixtures`
: The `Index_N` single- and two-signal layouts documented for
  [CSPB.ML.2023](https://cyclostationary.blog/2023/02/02/psk-qam-cochannel-data-set-for-modulation-recognition-researchers-cspb-ml-2023/).
  They contain symbol rate, carrier offset, modulation type and variant, and
  signal power for each component signal.

Dense single-signal fields become typed indexes aligned with the `item` axis:

| Index | Meaning |
| --- | --- |
| `signal_id` | Numeric suffix of the `.tim` filename |
| `source_file` | Direct filename or `archive.zip:member` location |
| `signal_count` | Number of component signals described by truth data |
| `mod_class_id` | Modulation label ID for single-signal items |
| `symbol_rate` | Normalized symbols per sample |
| `carrier_offset` | Normalized cycles per sample |
| `base_symbol_period` | Base period in samples, when supplied |
| `excess_bandwidth` | SRRC excess-bandwidth value, when supplied |
| `inband_snr_db` | In-band SNR, when supplied |
| `signal_power_db` | Component power for PSK Mixtures, when supplied |

Additional published scalar fields are also retained as indexes. A cochannel
item cannot be represented by one modulation ID, so its component-signal list
is stored in per-item metadata under `global["cspb:signals"]`. The
`signal_count` index remains available for inexpensive selection.

When selected truth records contain different fields, the importer also stores
their complete component descriptions in per-item metadata. Dense indexes
contain only fields available for every item. This preserves metadata when
combining the supported truth-file formats.

The published CSPB.ML.2023 truth data contains a small number of modulation
type/variant pairs not defined by its accompanying table. The importer does
not guess their meaning. It preserves both numeric codes and assigns a label
such as `UNKNOWN-1-0`.

Every imported file must have a matching row when any truth files are
supplied. Extra truth rows are allowed, which permits importing one batch with
a truth file describing the complete release.

## Python API

```python
from sigmf_zarr import import_cspb_dataset

store = import_cspb_dataset(
    "cspb.zarr",
    "downloaded-batches",
    truth_paths=("signal_record_C_2023.txt",),
    source_dataset="CSPB.ML.2018R2",
    batch_size=64,
)
```

All files placed in one recording must agree on real versus complex encoding
and sample shape. The importer detects and normalizes each file's byte order.
Use separate recording names when combining releases with different signal
lengths or encodings.
