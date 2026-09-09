# RadioML

```{warning}
RadioML 2016 and RML22 datasets use Python pickle files. Pickle loading can
execute code. Run `sigmf-zarr import radioml2016` and `sigmf-zarr import rml22`
only on files obtained from a trusted source.
```

SigMF-Zarr provides subcommands under `sigmf-zarr import` for the RadioML 2016
and RML22 pickle datasets and the RadioML 2018.01A HDF5 dataset. All three
importers create a batched recording with sample axes `(iq, time)` and
item-aligned metadata indexes:

- The `indexes/mod_class_id` array contains integer modulation class IDs. Its
  `labels` attribute maps each ID to a modulation name.
- The `indexes/snr_db` array contains SNR values in dB.

The importers also record the source dataset name in
`global["radioml:source_dataset"]` when one is provided. The command-line
interfaces default this name to the source file name without its final suffix.

## RadioML 2016

Imports declare the project-defined `radioml` metadata namespace at version
`0.1.0`. The [import provenance](provenance.md) records source identities,
class mappings, item ordering, and sample conversions. Inputs must remain
unchanged while the importer hashes and reads them.

The following command is recommended for a new RadioML 2016 store. It uses
Blosc with Zstandard at a balanced compression level and retains automatic
sharding:

```bash
sigmf-zarr import radioml2016 RML2016.10a.pkl store.zarr \
  --source-dataset RML2016.10a \
  --recording-name RML2016.10a \
  --sample-compression zstd \
  --sample-compression-level 3
```

The importer supports legacy Python 2 pickles and uses `latin1` decoding by
default. Change it with `--encoding` if a particular pickle requires another
encoding. The modulation labels are read from the mapping keys and stored in a
stable sorted order.

The decoded pickle mapping remains in memory. The importer converts and writes
its samples in bounded batches without concatenating a second complete sample
tensor. Use `--batch-size` or the Python `batch_size` argument to set the maximum
items per write. The default is `4096`. The aligned label arrays remain in
memory until they are written.

Samples are explicitly converted to float32 and declare `core:datatype` as
`cf32_le`. Provenance hashes each original decoded bucket before conversion and
records its labels, item count, and output start. These hashes identify the
decoded mapping, not the original pickle bytes.

The equivalent Python API accepts the decoded mapping:

```python
import pickle

from sigmf_zarr import import_radioml2016_dataset

with open("RML2016.10a.pkl", "rb") as handle:
    dataset = pickle.load(handle, encoding="latin1")

store = import_radioml2016_dataset(
    "store.zarr",
    dataset,
    source_dataset="RML2016.10a",
    recording_name="RML2016.10a",
    overwrite_store=True,
)
```

## RadioML 2018.01A

The following command is recommended for a new RadioML 2018.01A store. It uses
the same compression and automatic-sharding settings:

```bash
sigmf-zarr import radioml2018 GOLD_XYZ_OSC.0001_1024.hdf5 store.zarr \
  --source-dataset RML2018.01A \
  --recording-name RML2018.01A \
  --sample-compression zstd \
  --sample-compression-level 3
```

RadioML 2018.01A stores samples, one-hot modulation labels, and SNR values in
the HDF5 datasets `X`, `Y`, and `Z`. The importer streams bounded batches from
`X`, transposes samples from `(item, time, iq)` to `(item, iq, time)`, and
writes the aligned indexes without loading the full sample dataset into
memory. Use `--batch-size` to change the HDF5 read size. Use
`--samples-dataset`, `--labels-dataset`, and `--snr-dataset` when a compatible
file uses different HDF5 paths.

The HDF5 importer accepts float32 and float64 components, preserves their
precision, and declares `cf32_le` or `cf64_le` accordingly. Its provenance
includes the original file SHA-512, selected HDF5 paths, class order, axis
conversion, and half-open source row range. File hashing adds one complete
source-file read before sample copying.

The Python API streams directly from the HDF5 file:

```python
from sigmf_zarr import import_radioml2018_dataset

store = import_radioml2018_dataset(
    "store.zarr",
    "GOLD_XYZ_OSC.0001_1024.hdf5",
    source_dataset="RML2018.01A",
    recording_name="RML2018.01A",
    overwrite_store=True,
)
```

### Modulation class order

The HDF5 `Y` dataset contains one-hot labels but does not contain their class
names. The original `classes.txt` distributed by DeepSig has an incorrect
order. By default, SigMF-Zarr uses the corrected 24-class mapping from
`classes-fixed.json` in the
[Kaggle mirror of the dataset](https://www.kaggle.com/datasets/pinxau1000/radioml2018):

```text
OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK,
16APSK, 32APSK, 64APSK, 128APSK,
16QAM, 32QAM, 64QAM, 128QAM, 256QAM,
AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC,
FM, GMSK, OQPSK
```

Use `--classes-file` to override this mapping for a dataset with a different
one-hot column order:

```bash
sigmf-zarr import radioml2018 GOLD_XYZ_OSC.0001_1024.hdf5 store.zarr \
  --classes-file classes-fixed.json
```

The file may contain a JSON string array, a Python-style `classes = [...]`
assignment, or one class per non-empty line. Python-style files are parsed
without executing their contents.

## RML22

Import an RML22 pickle mapping with the `sigmf-zarr import rml22` command:

```bash
sigmf-zarr import rml22 RML22.01A.pkl store.zarr \
  --source-dataset RML22.01A \
  --recording-name RML22.01A \
  --sample-compression zstd \
  --sample-compression-level 3
```

The source is a dictionary keyed by `(modulation, snr_db)` pairs. Modulation
names are strings, and SNR values are integers. Each value is a NumPy array
with shape `(item, 2, time)`, where the second axis contains I/Q components.
All arrays must have the same item shape. Class subsets and unequal numbers
of items per key are supported. The common time length does not have to be
128. The importer derives the modulation vocabulary from the keys and
preserves the label strings in sorted order.

The inspected `RML22.01A.pkl` dataset contains 462000 float32 items with shape
`(2, 128)`, grouped under 231 keys with 2000 items each. Its 11 modulation
labels include `AM-SSB`, and its SNR values range from -20 through 20 dB in
2 dB steps. These counts and labels describe that file and are not validation
requirements.

The `sigmf-zarr import rml22` command records `global["radioml:dataset_version"]`
as `"2022"` and defaults the recording name to `rml22`. The command defaults
`radioml:source_dataset` to the source file name without its final suffix,
such as `RML22.01A`. Use `--source-dataset` to override it. The Python API
records `null` for this metadata field when `source_dataset` is not supplied.

RML22 reuses the pickle decoding, input validation, and bounded sample writer
in `sigmf_zarr.radioml2016`. The complete decoded mapping remains in memory.
The importer does not concatenate a second complete sample tensor. Set
`--batch-size` or the Python `batch_size` argument to limit each sample write.
The default is `4096` items. The aligned label arrays also remain in memory
until they are written. String decoding defaults to `latin1` and can be
changed with `--encoding`.

The Python API uses `import_radioml2016_dataset()` with
`dataset_version="2022"` and an explicit recording name:

```python
import pickle

from sigmf_zarr import import_radioml2016_dataset

with open("RML22.01A.pkl", "rb") as handle:
    dataset = pickle.load(handle, encoding="latin1")

store = import_radioml2016_dataset(
    "store.zarr",
    dataset,
    dataset_version="2022",
    source_dataset="RML22.01A",
    recording_name="RML22.01A",
    overwrite_store=True,
)
```

## Storage options

All three commands accept `--sample-compression`, `--sample-compression-level`,
`--sample-shard-batch`, and `--no-sample-sharding`. Like the standard SigMF
importer, they support `--recording-name`, `--overwrite-store`,
`--overwrite-recording`, and `--zarr-format`.

Sample compression defaults to Zstandard. Select `zstd`, `lz4`, or `lz4hc` to
use the corresponding Blosc algorithm. Use `--sample-compression-level` to set
the Blosc level from 0 through 9, or select `none` to disable compression.

New stores use Zarr format 3 by default. Automatic layout selection groups
RadioML items into logical chunks targeting approximately 256 KiB and physical
shards targeting approximately 4 MiB. A float32 RadioML 2016 or RML22 array
with item shape `(2, 128)` uses 256 items per chunk and 4096 items per shard.
A float32 RadioML 2018 array with item shape `(2, 1024)` uses 32 items per
chunk and 512 items per shard. Use `--sample-shard-batch` to override the
derived shard item count or `--no-sample-sharding` to use larger unsharded
chunks.

Existing stores have their format detected automatically. Pass
`--zarr-format 2` to create a Zarr format 2 store or to require format 2 for an
existing store. Format 2 uses chunks targeting approximately 4 MiB and cannot
be combined with `--sample-shard-batch`. Run any command with `--help` for
the complete option list.
