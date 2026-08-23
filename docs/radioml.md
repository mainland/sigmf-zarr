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

Both commands accept `--sample-compression`, `--sample-compression-level`,
`--sample-shard-batch`, and `--no-sample-sharding`. Like the standard SigMF
importer, they support `--recording-name`, `--overwrite-store`,
`--overwrite-recording`, and `--zarr-format`.

Sample compression defaults to Zstandard. Select `zstd`, `lz4`, or `lz4hc` to
use the corresponding Blosc algorithm. Use `--sample-compression-level` to set
the Blosc level from 0 through 9, or select `none` to disable compression.

New stores use Zarr format 3 by default. Automatic layout selection groups
RadioML items into logical chunks targeting approximately 256 KiB and physical
shards targeting approximately 4 MiB. A float32 RadioML 2016 array with item
shape `(2, 128)` uses 256 items per chunk and 4096 items per shard. A float32
RadioML 2018 array with item shape `(2, 1024)` uses 32 items per chunk and 512
items per shard. Use `--sample-shard-batch` to override the derived shard item
count or `--no-sample-sharding` to use larger unsharded chunks.

Existing stores have their format detected automatically. Pass
`--zarr-format 2` to create a Zarr format 2 store or to require format 2 for an
existing store. Format 2 uses chunks targeting approximately 4 MiB and cannot
be combined with `--sample-shard-batch`. Run either command with `--help` for
the complete option list.
