# Index queries

`sigmf_zarr.query` is part of the core library. It uses Lark to parse query
expressions and NumPy to evaluate bounded batches of dense item indexes.
Installation does not require the viewer or any optional extra.

## Select items

```python
import numpy as np
from tempfile import TemporaryDirectory

from sigmf_zarr.query import compile_query
from sigmf_zarr.store import SigMFZarrStore

with TemporaryDirectory() as path:
    store = SigMFZarrStore.create(path)
    recording = store.recordings.open(
        "example",
        batched=True,
        sample_shape=(2, 8),
        sample_axes=("iq", "time"),
    )
    recording.append_samples(np.zeros((3, 2, 8), dtype=np.float32))
    recording.add_index(
        "mod_class_id", [0, 1, 0], axis="item",
        field="rfml-dataset:modulation", labels=["BPSK", "QPSK"],
    )
    recording.add_index(
        "snr_db", [-10, 0, 10], axis="item", field="example:snr", unit="dB",
    )
    query = compile_query('label(mod_class_id) == "BPSK" and snr_db >= 0')
    print(query.select(recording).tolist())
```

The result is `[2]`, the original item position. Compiled queries can be reused
across recordings. Each selection validates the target recording's descriptors
and binds its own categorical lookup tables. `query.index_names` lists the
referenced index names. `IndexQuery()` selects all items. Explicit empty query
text is an error in `compile_query()`.

## Syntax

| Expression | Meaning |
| --- | --- |
| `snr_db >= 10` | Compare raw numeric values |
| `mod_class_id in [0, 2]` | Test membership in a list of raw values |
| `label(mod_class_id) == "QPSK"` | Compare a categorical label |
| `index("splits/example") == 1` | Reference an exact index name containing punctuation |
| `label("targets/modulation") in ["BPSK", "QPSK"]` | Compare labels for a quoted index name |
| `enabled == true` | Compare Boolean values |
| `not (snr_db < 0 or snr_db > 20)` | Negate a compound condition |

Comparisons support `==`, `!=`, `<`, `<=`, `>`, `>=`, and `in`. Strings and
Booleans support only equality, inequality, and membership. Operands use JSON
syntax: double-quoted strings, finite numbers, lowercase Booleans, and lists
of scalars for `in`. Ordering is numeric. Arithmetic, index-to-index comparisons,
arbitrary function calls, and JSON metadata access are not supported.

Comparison binds more tightly than `not`, followed by `and`, then `or`.
Parentheses override precedence. Keywords are lowercase. Use `index("name")`
when a name conflicts with query syntax.

Bare names identify stored indexes, not their `field` attributes. `label()`
uses the named index's own `labels` attribute. Unknown labels do not equal any
stored category. Invalid category IDs fail evaluation. Index values never fall
back to recording or item JSON.

## Missing values and errors

Only dense indexes with one value per item are supported. Indexes with a
`validity` mask are rejected. Nonfinite numeric values are unknown rather than
true or false. Negation preserves unknown values, so `not (snr_db >= 10)` does
not select missing SNR values. A true branch can still make an `or` expression
true. A false branch makes an `and` expression false. Only true results select
items.

Syntax errors include line and column information. Missing indexes, incompatible
operand types, invalid index descriptors, and invalid category IDs fail the
operation. Validation also applies to branches that would not affect a Boolean
result. The evaluator does not short-circuit index reads.

Numeric comparisons preserve integer precision. Compatible operands use native
NumPy operations. Mixed numeric types or integer operands outside an index's
range use exact Python numeric comparisons when native conversion could round
values.

## Execution and cancellation

`query.select(recording, batch_size=4096)` reads each referenced index once per
batch, even if the query refers to it more than once. Label operands are mapped
to integer IDs before scanning. The evaluator does not read samples or item
JSON, decode per-item label strings, or create a persistent index structure.
Uncached queries still scan the referenced index arrays.

Memory scales with the returned matches and batch size, referenced indexes,
and expression complexity. Results are ascending original positions. They are
an independent selection snapshot and do not update when recording contents
change. Open recordings with structural validation when descriptor-only opening
is required.

The optional `progress(examined, total)` and `cancelled()` callbacks have no GUI
dependencies. Cancellation raises `concurrent.futures.CancelledError` without
returning partial results. Checks occur between index reads and batches.
In-flight backend reads finish under their backend timeout configuration.
