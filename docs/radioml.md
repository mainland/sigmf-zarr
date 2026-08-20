# RadioML

```{warning}
RadioML 2016 datasets use Python pickle files. Pickle loading can execute code.
Run `sigmf-zarr import radioml2016` only on files obtained from a trusted source.
```

SigMF-Zarr provides subcommands under `sigmf-zarr import` for RadioML 2016
pickle datasets and the RadioML 2018.01A HDF5 dataset. Both importers create a batched recording
with sample axes `(iq, time)` and item-aligned metadata indexes:

- The `indexes/mod_class_id` array contains integer modulation class IDs. Its
  `labels` attribute maps each ID to a modulation name.
- The `indexes/snr_db` array contains SNR values in dB.

The importers also record the source dataset name in
`global["radioml:source_dataset"]` when one is provided.

## RadioML 2016

Imports declare the project-defined `radioml` metadata namespace at version
`0.1.0`. The [import provenance](provenance.md) records source identities,
class mappings, item ordering, and sample conversions. Inputs must remain
unchanged while the importer hashes and reads them.

Use `sigmf-zarr import radioml2016` for RadioML 2016 pickle mappings:

```bash
sigmf-zarr import radioml2016 RML2016.10a.pkl store.zarr \
  --source-dataset RML2016.10a \
  --recording-name RML2016.10a
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

Use `sigmf-zarr import radioml2018` for the HDF5 dataset:

```bash
sigmf-zarr import radioml2018 GOLD_XYZ_OSC.0001_1024.hdf5 store.zarr \
  --source-dataset RML2018.01A \
  --recording-name RML2018.01A
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

## Storage options

Both commands accept `--sample-compression`, `--sample-compression-level`, and
`--sample-shard-batch`. Like the standard SigMF importer, they support
`--recording-name`, `--overwrite-store`, `--overwrite-recording`, and
`--zarr-format`. Existing stores have their format detected automatically, and
new stores use Zarr format 3 by default. Pass `--zarr-format 2` to create a
Zarr format 2 store or to require Zarr format 2 for an existing store. Because
sharding is a Zarr format 3 feature, `--zarr-format 2` cannot be combined with
`--sample-shard-batch`. Run either command with `--help` for the full option
list.
