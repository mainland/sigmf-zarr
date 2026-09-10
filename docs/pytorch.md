# PyTorch datasets

Install the optional adapter with `pip install 'sigmf-zarr[pytorch]'`.
Import `RecordingDataset` from `sigmf_zarr.pytorch`. The base package does not
import or require PyTorch. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
for CPU, CUDA, and other accelerator installation options.

## Select samples and targets

`RecordingDataset` reads fixed-shape items from a batched recording. Both
`targets` and `split` are required keyword arguments. Target mapping keys name
outputs. Values name item-aligned recording indexes. An empty mapping selects
no targets. `split=None` selects every item. Otherwise, provide a split index
name and one partition label or a sequence of labels.

```python
from torch.utils.data import DataLoader

from sigmf_zarr.pytorch import RecordingDataset


def main() -> None:
    """Read batches from an imported RadioML recording."""
    dataset = RecordingDataset(
        "radioml.zarr",
        recording="radioml2016",
        targets={"modulation": "mod_class_id", "snr_db": "snr_db"},
        split=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    for batch in loader:
        print(batch["samples"].shape, batch["targets"]["modulation"].shape)
    dataset.close()


if __name__ == "__main__":
    main()
```

For an existing named split, use a selection such as
`split=("splits/session", "train")` or
`split=("splits/session", ["train", "validation"])`. Selected items retain
source recording order. `dataset.source_indices` returns a detached array of
original recording positions. Each item has these fields:

- `samples`: A CPU tensor in stored sample axis order
- `targets`: A mapping of selected output names to scalar CPU tensors
- `item_index`: The original recording position

PyTorch's default collation stacks these tensors and item positions.
`dataset.sample_axes` describes the stored per-item axes. The adapter preserves
numerical dtype, signedness, precision, and sample shape. It converts byte order
to the host's native order. It does not convert I/Q to complex values,
transpose axes, normalize samples, or cast class IDs to `int64` automatically.

Targets retain raw index values. `dataset.labels("item", "modulation")`
returns the selected target's source string lookup table, or `None` if it has
no table. The adapter requires categorical tables to contain unique nonempty
strings. It does not compare targets with JSON metadata.
`dataset.metadata(position)` reads resolved shared and local JSON for a
dataset position through the recording's normal metadata accessor.

## Transform items

The constructor accepts these optional callables:

- `sample_transform`: Transforms the sample tensor
- `item_target_transforms`: Maps selected output names to target transforms
- `item_transform`: Transforms the complete item after the other transforms

Conversion precedes transforms. Sample transforms run before target transforms,
and the joint item transform runs last. Transforms must return CPU tensors.
Use `item_transform` when one operation changes both samples and target meaning.
It must retain the selected target keys and original `item_index`. Target
transforms may change scalar shapes, provided the results can be collated.
Applications must choose sample and target shapes compatible with their
collator. Label tables and axis descriptors describe storage before transforms.

Repeated positions receive separate transform calls in request order and
independent mutable tensor storage. `__getitems__()` performs a batched storage
selection and returns a list of ordinary items for the collator. Standard
PyTorch samplers, including `DistributedSampler`, operate on dataset positions.

## Worker lifetime and validation

The source recording must remain unchanged while the dataset or its loaders
are in use. Construction validates layout, target descriptors, and selected
split assignments. It does not scan all target payloads or per-item JSON.
Selected target values are checked when read. Nonfinite targets, validity
masks, unsupported tensor dtypes, and out-of-range category IDs are rejected.
Requested JSON entries are validated when accessed.

Split selection does not revalidate source-group isolation. Use the recording's
`validate_split()` explicitly when that claim needs checking. Normal loading
does not compute whole-array SHA-512 hashes. Zarr codec checks remain active.

Pass a path or URL and optional `storage_options` containing JSON configuration
values. Live stores, clients, and sessions are not accepted. Handles open
lazily and belong to the process that opened them. Serialization omits handles,
and local fork workers reopen after detecting a process change. `close()`
releases handles in the calling process. Later access opens them again.

Remote URLs require `spawn` or `forkserver` workers. Remote access in fork
workers raises an error before opening backend handles. Backend packages such
as S3Fs must be installed separately. See the
[S3Fs multiprocessing requirements](https://s3fs.readthedocs.io/en/latest/index.html#multiprocessing).
Worker transforms must be serializable by the selected start method.
Applications control random seeds and worker initialization through the normal
PyTorch loader APIs.

This adapter supports dense item targets. Signal tables, nullable targets,
and signal-specific collation are deferred.

## Dataset manifests

`sigmf_zarr.experiments.dataset_manifest()` records explicit target paths,
lookup tables, split assignments, selected partition labels, source item
ordering, and software versions. Callers must provide JSON descriptions of
transforms and experiment parameters. The function does not infer the behavior
of Python callables.

Manifest creation scans logical samples and metadata, validates declared split
isolation, and checks an existing split input binding when present. Keep the
recording unchanged during the scan and experiment. The resulting hashes detect
input changes. They do not certify source truth or guarantee identical training
results across devices. Store the returned JSON with experiment results.
