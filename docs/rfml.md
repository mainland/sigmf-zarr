# RFML dataset profile

This document defines the initial `rfml-dataset` profile, version `0.1.0`, for
RF machine learning metadata in SigMF-Zarr. It is a project-defined draft,
not a canonical SigMF extension. Compatibility follows the project's alpha
format policy.

The profile assigns meanings to existing item-aligned indexes. It adds no
storage structure, metadata synchronization, or default training target.
Existing importers and the PyTorch adapter work without adopting it. Generic
store validation checks index structure and integrity. It does not enforce
these field-specific semantic requirements or certify source correctness.

## Namespace and declaration

The namespace is `rfml-dataset`. It does not reuse the historical `rfml`
namespace, which SigMF removed from its canonical extensions. The
[removal discussion](https://github.com/sigmf/SigMF/pull/233) describes that
extension's separate labeling use case and documentation limitations.

A recording that uses this profile must include this entry in its shared
`global["core:extensions"]` list, preserving other extension declarations:

```json
{"name": "rfml-dataset", "version": "0.1.0", "optional": true}
```

The extension is optional for interpreting stored samples. It is not optional
for interpreting a field's profile-specific meaning. Declaring the profile
neither validates nor converts existing fields. Per-item JSON cannot override
the shared extension list.

Use standard SigMF fields for recording provenance. The
[SigMF signal extension](https://github.com/sigmf/SigMF/blob/main/extensions/signal.sigmf-ext.md)
provides structured annotation details such as modulation family, order,
standard, and channel bandwidth. Dataset-native class labels do not imply an
automatic mapping to those fields.

## Representation and selection

Each dense profile index must have `axis="item"` and `kind="metadata"`. Its
`field` attribute uses one of the names below. Index names remain local,
caller-selected identifiers. Index values must cover every item. Nullable
columns and constituent-signal storage are deferred.

JSON and indexes remain independent. No duplicate JSON representation is
required. `recording.index(name)` selects raw values, while
`recording.decode_index(name, selection=...)` explicitly selects category
lookup. `recording.find_indexes(field, axis="item")` discovers all matching
names without choosing among them. Same-field indexes may disagree.
`resolved_item_metadata()` continues to return JSON without index projection.

Changing samples or JSON does not refresh an index. Callers must check the
source before relying on previously derived values. Matching hashes and array
lengths do not establish that a label or measurement describes those samples.

## Dense fields

All fields are optional. Boolean values are not numeric measurements, IDs, or
counts. Real measurements must be finite. Writers must retain unknown or incompatible
values in their source representation instead of substituting zero.

| Field | Stored values | Unit | Meaning |
| --- | --- | --- | --- |
| `rfml-dataset:class` | Integer category IDs | None | Dataset-native task class |
| `rfml-dataset:modulation` | Integer category IDs | None | Dataset-native modulation label for one identified signal |
| `rfml-dataset:snr` | Integer or floating-point values | `dB` | Scene SNR with the measurement descriptor below |
| `rfml-dataset:signal_count` | Nonnegative integers | None | Number of labeled constituent signals in the item |
| `rfml-dataset:carrier_offset` | Real values in $[-0.5, 0.5)$ | `cycles/sample` | Nominal carrier relative to stored baseband DC |
| `rfml-dataset:symbol_rate` | Positive real values | `symbols/sample` | Symbol rate relative to the stored output sample clock |
| `rfml-dataset:source_id` | Integers or nonempty strings | None | Source-file or generated-source identity |
| `rfml-dataset:session_id` | Integers or nonempty strings | None | Collection-session identity |
| `rfml-dataset:emitter_id` | Integers or nonempty strings | None | Physical or simulated emitter identity |

### Classes and identities

Class and modulation indexes must have a nonempty `labels` list of unique,
nonempty strings. Values must be integer IDs in the range
$0 \le i < \operatorname{len}(\mathrm{labels})$. Strings are case-sensitive and
retain the supplied vocabulary. IDs refer only to that index's lookup table.
Equal IDs in different recordings need not describe the same class. Equal
strings do not establish equivalent class definitions across datasets.

`class` permits protocol names, transmission modes, and composite task labels.
`modulation` identifies the modulation of one known signal. A multi-signal
item must not receive a single constituent's modulation as its item target.
A composite scene label may use `class` with its dataset's definition.

Identities are scoped to the source collection described by the recording.
Without an explicit cross-recording correspondence, consumers must treat them
as recording-local. A source filename or item number must not be presented as
a physical emitter or collection-session ID. Different source files do not
prove that their contents derive from different transmissions. Group-isolation
claims remain explicit split-validation operations.

`signal_count` counts labeled constituents. It does not assert that no other
signals or interference are present. A value of zero requires a known empty
label set. Unknown counts must not be encoded as zero.

### Normalized rates and frequency

Carrier offset and symbol rate apply only to an item with one identified
signal and one receiver channel. Carrier frequency uses the convention
$e^{j2\pi f n}$ and refers to stored samples after resampling and translation.
It is not receiver oscillator error or the center of a labeled bandwidth.
Symbol rate is measured in symbols per stored output sample. It must not be
inferred from a base symbol period without accounting for resampling.

This profile defines no common bandwidth field. Occupied bandwidth, channel
bandwidth, and annotation frequency bounds describe different quantities.
Retain their source definitions. Writers must not invent an absolute sample
rate to convert normalized frequencies into hertz.

## SNR measurement descriptor

An `rfml-dataset:snr` index must have `unit="dB"` and a JSON object in its
`measurement` attribute. All values in the index must use the same convention.

| Entry | Required value for dense item SNR |
| --- | --- |
| `definition` | Nonempty versioned identifier or source reference specifying the power estimates and measurement procedure |
| `signal_reference` | `scene` |
| `time_support` | `item` or `signal_interval`, as specified by the definition |
| `noise_bandwidth` | `full_sample_band` or `signal_band` |
| `interference` | `excluded` |

The ratio is $10\log_{10}(P_s/P_n)$. The definition must identify which desired
signals contribute to scene power, the receiver channel, the processing stage,
and how signal and noise powers are obtained. A `signal_interval` definition
must identify its interval-selection rule. A `signal_band` definition must
identify the band-selection rule and the metadata needed to reconstruct it.
Per-channel SNR columns are outside this version's scope.

An unknown convention does not satisfy this contract. A ratio with undesired
signals in its denominator is not this field's SNR. A nominal generator label
must not be described as a measurement on stored samples unless its definition
supports that interpretation. Such values remain usable under source-specific
fields. Consumers must examine measurement descriptors before combining SNR
targets across datasets. Identical units or descriptors alone do not prove
comparable source conditions.

A source reference is a definition, not a freshness mechanism. Updating the
samples does not recompute the measurement or its descriptor. Automatic SNR
projection into JSON is not part of this profile implementation.

## Importer adoption decisions

The following decisions concern semantic eligibility for future explicit
adoption. Importers continue to write their existing source fields. No index
renaming, sample conversion, or automatic profile declaration occurs.

| Source representation | Profile candidate | Decision |
| --- | --- | --- |
| RadioML 2016 and RML22 `mod_class_id` | `modulation` | Preserve imported IDs and the exact selected lookup table. Do not infer RML22 measurement semantics from pickle layout compatibility. |
| RadioML 2018 `mod_class_id` | `modulation` | Preserve the configured class order corresponding to one-hot columns. Column position alone does not determine a label name. |
| RadioML `snr_db` | `snr` | Retain `radioml:snr`. A complete measurement convention has not been established for each supported release in this review. |
| Panoradio `mode_id` | `class` | Preserve mode names. Do not replace them with inferred modulation-family labels. |
| Panoradio `snr_db` | `snr` | Retain `panoradio:snr` until the measurement stage and power/band conventions are established. |
| CSPB single-signal `mod_class_id` | `modulation` | Preserve the imported lookup and raw truth codes. Unknown-code labels must not be treated as identified modulation schemes. |
| CSPB `signal_count` | `signal_count` | Count truth components, not inferred emitters or all physically present signals. |
| CSPB `source_file` | `source_id` | A file locator may identify a file within the recording. It does not establish source-transmission independence. |
| CSPB normalized rates and offsets | `symbol_rate`, `carrier_offset` | Verify release-specific reference clock and carrier sign before adoption. Preserve source fields during this review. |
| CSPB in-band SNR | `snr` | Retain `cspb:inband_snr_db` until its band and power conventions are encoded. |
| CSPB cochannel components | Deferred signal columns | Preserve `cspb:signals` JSON and all component truth. Do not collapse component values into one dense item target. |

The [RadioML 2016 generator](https://github.com/radioML/dataset/blob/master/generate_RML2016.10a.py)
uses the SNR key to configure a channel model before taking and scaling output
windows. It does not separately estimate signal and noise powers for each
stored window. This supports retaining the supplied generator labels rather
than asserting a new per-window measurement. The
[DeepSig dataset descriptions](https://www.deepsig.ai/datasets/)
distinguish the releases but do not establish every descriptor entry above.
These conclusions do not assert that further source material cannot establish
a release-specific convention.

The [Panoradio publication page](https://panoradio-sdr.de/radio-signal-classification-dataset/)
distinguishes transmission modes from their modulation schemes and describes
noise and fading during generation. The reviewed description does not resolve
every required SNR convention. For CSPB, the author's
[single-signal dataset description](https://cyclostationary.blog/2019/02/15/data-set-for-the-machine-learning-challenge/)
and [cochannel truth description](https://cyclostationary.blog/2023/02/02/psk-qam-cochannel-data-set-for-modulation-recognition-researchers-cspb-ml-2023/)
identify source truth that must remain available during any later conversion.

## Writing a native profile index

This example creates a temporary recording with a declared native modulation
vocabulary. The samples illustrate storage, not a waveform generator.

```python
from tempfile import TemporaryDirectory

import numpy as np

from sigmf_zarr import SigMFZarrStore

with TemporaryDirectory() as directory:
    store = SigMFZarrStore.create(directory)
    recording = store.recordings.open(
        "example",
        batched=True,
        sample_shape=(2, 128),
        sample_axes=("iq", "time"),
        global_metadata={
            "core:extensions": [
                {
                    "name": "rfml-dataset",
                    "version": "0.1.0",
                    "optional": True,
                }
            ]
        },
    )
    recording.append_samples(np.zeros((2, 2, 128), dtype=np.float32))
    recording.add_index(
        "mod_class_id",
        np.array([0, 1], dtype=np.int16),
        axis="item",
        field="rfml-dataset:modulation",
        labels=["BPSK", "QPSK"],
    )
    print(recording.find_indexes("rfml-dataset:modulation", axis="item"))
    print(recording.decode_index("mod_class_id", selection=[1, 0]).tolist())
```

The output is `('mod_class_id',)` followed by `['QPSK', 'BPSK']`. The profile
declaration coexists with the storage extension. Ordinary JSON access does not
include the modulation index. The PyTorch adapter can select `mod_class_id`
explicitly without inspecting the profile declaration.

## Deferred work

Signal tables, nullable targets, explicit JSON projection, and source-to-profile
conversion APIs require separate implementation. They must preserve explicit
source selection and independent JSON/index state. This dense profile does not
reserve a signal-table layout or change the SigMF-Zarr schema version.
