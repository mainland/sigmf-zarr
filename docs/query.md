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
    recording = store.recordings.create(
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

Compiled queries and `IndexFilter` conditions are immutable and hashable.
`IndexFilter` stores the operand list of an `in` comparison as a tuple, so
`IndexFilter("snr_db", "in", [0, 10]).value` is `(0, 10)`.

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
Parentheses override precedence. Keywords are lowercase. A keyword must not be
followed directly by a letter, digit, or underscore, so `snr >= 10 android == 1`
is a syntax error rather than `snr >= 10 and roid == 1`.

Bare names must not be `true`, `false`, `and`, `or`, or `in`. Write
`index("and")` or `label("and")` for an index with a reserved name. `not`,
`index`, and `label` cannot begin a bare comparison either. Use `index("name")`
for these names and for names that contain punctuation.

Numeric literals must not exceed 100 characters. `compile_query()` raises
`ValueError` with the line and column of the literal for a longer number, an
invalid string escape, or an operand that the comparison does not accept.

An expression must not enclose a comparison in more than 100 levels of `not`,
`and`, and `or`. Redundant parentheses do not count toward this limit.
`compile_query()` raises `ValueError` for a deeper expression.

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

Syntax and operand errors include line and column information. Missing indexes,
incompatible operand types, invalid index descriptors, and invalid category IDs
fail the operation. Validation also applies to branches that would not affect a
Boolean result. The evaluator does not short-circuit index reads.

Numeric comparisons preserve integer precision. Compatible operands use native
NumPy operations. Mixed numeric types or integer operands outside an index's
range use exact Python numeric comparisons when native conversion could round
values.

## Execution and cancellation

`query.select(recording, batch_size=4096)` opens each referenced index once and
evaluates it in batches of at most `batch_size` values. Reads are aligned to
stored chunks, so each chunk is decoded once, even if the query refers to its
index more than once or a chunk is longer than a batch. Label
tables are validated once, and label operands are mapped to integer IDs before
scanning. Each batch then checks only that category IDs are within the table.
The evaluator does not read samples or item JSON, decode per-item label
strings, or create a persistent index structure. Uncached queries still scan
the referenced index arrays.

Memory scales with the returned matches, the batch size, one stored chunk per
referenced index, and expression complexity. Results are ascending original
positions. They are an independent selection snapshot and do not update when
recording contents change. Open recordings with structural validation when
descriptor-only opening is required.

The optional `progress(examined, total)` and `cancelled()` callbacks have no GUI
dependencies. Cancellation raises `concurrent.futures.CancelledError` without
returning partial results. Checks occur between index reads and batches.
In-flight backend reads finish under their backend timeout configuration.

## Combine with JSON filtering

`sigmf_zarr.filtering.filter_items()` accepts an `index_query` string or
compiled `IndexQuery`, existing `IndexFilter` conditions, and an optional
JMESPath `metadata_query`. Index conditions are combined with the index query
using AND. The JSON predicate evaluates resolved item metadata only for
surviving index candidates. Items are resolved one at a time and share the
recording's shared metadata instead of copying it. Candidates wait until their
`item_metadata` chunk is complete, so each chunk is decoded once. Memory
therefore scales with the matches, one batch and one stored chunk of index
values, the encoded item JSON of one stored chunk, the shared metadata, and one
decoded item. JMESPath requires the `query` extra. This optional JSON facility
is separate from the core index query language.
