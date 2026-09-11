"""Compiled, batched queries over explicitly named dense item indexes.

This core module does not read samples or item JSON and has no GUI
dependencies.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from lark import Lark, Token, Tree, UnexpectedInput

from sigmf_zarr._chunks import SpanReader
from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.json import JSONValue, json_value
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFRecording

FilterOperator = Literal["==", "!=", "<", "<=", ">", ">=", "in"]
"""Supported comparisons.

Multiple filters are combined with AND.
"""

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


type QueryScalar = str | int | float | bool
"""Scalar operand of an index comparison."""


@dataclass(frozen=True)
class IndexFilter:
    """One comparison against an explicitly named item index."""

    name: str
    """Exact array name relative to the recording's indexes group."""

    operator: FilterOperator
    """Comparison to apply to each value."""

    value: QueryScalar | Sequence[QueryScalar]
    """Comparison scalar, or a tuple of scalars for membership.

    A list operand is stored as a tuple, so filters are hashable and
    validated operands cannot change.
    """

    decode: bool = False
    """Whether to compare decoded string labels instead of raw values."""

    def __post_init__(self) -> None:
        """Detach the operand and reject unsupported comparisons.

        Raises:
            ValueError: If the operator or operand is invalid.
        """
        if self.operator not in {*_OPERATORS, "in"}:
            raise ValueError(f"Unsupported filter operator: {self.operator}")
        value = json_value(
            list(self.value) if isinstance(self.value, tuple) else self.value,
            name="filter operand",
        )
        operands = value if self.operator == "in" else [value]
        if not isinstance(operands, list) or any(
            not isinstance(v, str | int | float | bool) for v in operands
        ):
            raise ValueError("Use scalar operands, or a list for 'in'")
        if any(isinstance(v, float) and not np.isfinite(v) for v in operands):
            raise ValueError("Filter operands must be finite")
        object.__setattr__(
            self,
            "value",
            tuple(cast(list[QueryScalar], operands))
            if self.operator == "in"
            else value,
        )

    def _operands(self) -> tuple[QueryScalar, ...]:
        """Return the validated operands as a tuple.

        Returns:
            Membership operands, or the single comparison operand.
        """
        if self.operator == "in":
            return cast(tuple[QueryScalar, ...], self.value)
        return (cast(QueryScalar, self.value),)

    def _prepare(
        self, array: ReadOnlyArray, labels: list[JSONValue] | None
    ) -> None:
        """Validate operand types against a bound index.

        Args:
            array: Dense item index selected by this comparison.
            labels: Validated label table for a decoded comparison.

        Raises:
            ValueError: If the labels or operands are invalid.
        """
        kind = array.dtype.kind
        if self.decode:
            assert labels is not None
            if not all(isinstance(label, str) for label in labels):
                raise ValueError("Decoded filters require string labels")
            kind = "U"
        valid = {
            "b": lambda v: isinstance(v, bool),
            "i": lambda v: type(v) in {int, float},
            "u": lambda v: type(v) in {int, float},
            "f": lambda v: type(v) in {int, float},
            "U": lambda v: isinstance(v, str),
            "T": lambda v: isinstance(v, str),
            "O": lambda v: isinstance(v, str),
        }
        if kind not in valid or not all(
            valid[kind](v) for v in self._operands()
        ):
            raise ValueError("Filter operand type does not match index dtype")
        if kind in {"b", "U", "T", "O"} and self.operator not in {
            "==",
            "!=",
            "in",
        }:
            raise ValueError("Strings and Booleans support ==, !=, and in")


_GRAMMAR = r"""
?start: disjunction
?disjunction: conjunction (_OR conjunction)* -> any_of
?conjunction: negation (_AND negation)* -> all_of
?negation: "not" negation -> negate
         | "(" disjunction ")"
         | predicate
predicate: reference OP operand
?reference: NAME -> raw
          | "index" "(" ESCAPED_STRING ")" -> quoted
          | "label" "(" NAME ")" -> decoded
          | "label" "(" ESCAPED_STRING ")" -> decoded_quoted
?operand: scalar | "[" [scalar ("," scalar)*] "]" -> values
?scalar: NUMBER | ESCAPED_STRING | BOOLEAN
OP: "==" | "!=" | "<=" | ">=" | "<" | ">" | /in\b/
BOOLEAN: /(true|false)\b/
_AND: /and\b/
_OR: /or\b/
NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/
NAME: /[a-zA-Z_][a-zA-Z_0-9]*/
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


@dataclass(frozen=True)
class _Node:
    """Boolean expression with comparisons at its leaves."""

    kind: str
    """Predicate, all_of, any_of, or negate."""

    children: tuple[_Node, ...] = ()
    """Child expressions in source order."""

    predicate: IndexFilter | None = None
    """Comparison for a predicate leaf."""

    def predicates(self) -> Iterator[IndexFilter]:
        """Visit comparisons in source order.

        Yields:
            Leaf comparisons.
        """
        if self.predicate is not None:
            yield self.predicate
        for child in self.children:
            yield from child.predicates()


@lru_cache(maxsize=1)
def _parser() -> Lark:
    """Construct the shared grammar parser on first use.

    Returns:
        LALR parser without filesystem caches.
    """
    return Lark(
        _GRAMMAR,
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )


# Maximum number of not, and, and or operators enclosing a comparison.
# Redundant parentheses do not count. The limit bounds the recursion depth
# of evaluation and of the generated dataclass methods.
_MAX_QUERY_DEPTH = 100

# Words that parse as literals or operators where an operand or operator is
# expected. Rejecting them as bare names keeps each spelling unambiguous.
_RESERVED_NAMES = frozenset({"true", "false", "and", "or", "in"})

# Longest accepted numeric literal. Index values are at most 64 bits wide, so
# longer literals have no use, and the bound keeps integer parsing cheap.
_MAX_NUMBER_LENGTH = 100


def _error(item: Token | Tree[Token], fallback: Token, message: str) -> str:
    """Prefix a diagnostic with the query position of a parsed item.

    Args:
        item: Token or subtree that caused the error.
        fallback: Token whose position is used for an empty subtree.
        message: Description of the problem.

    Returns:
        Message that names the line and column of the item.
    """
    if isinstance(item, Tree):
        line, column = (
            (fallback.line, fallback.column)
            if item.meta.empty
            else (item.meta.line, item.meta.column)
        )
    else:
        line, column = item.line, item.column
    return f"Invalid index query at line {line}, column {column}: {message}"


def _literal(token: Token) -> QueryScalar:
    """Decode one JSON literal token.

    Args:
        token: Number, string, or Boolean token.

    Returns:
        Decoded scalar. The grammar admits no other literal.

    Raises:
        ValueError: If the literal is too long or is not valid JSON. The
            message names the token's line and column.
    """
    text = str(token)
    if token.type == "NUMBER" and len(text) > _MAX_NUMBER_LENGTH:
        raise ValueError(
            _error(
                token,
                token,
                f"numeric literal exceeds {_MAX_NUMBER_LENGTH} characters",
            )
        )
    try:
        return cast(QueryScalar, json.loads(text))
    except ValueError as exc:
        # JSONDecodeError positions are relative to the literal itself.
        message = exc.msg if isinstance(exc, json.JSONDecodeError) else exc
        raise ValueError(
            _error(token, token, f"invalid literal: {message}")
        ) from exc


def _predicate(tree: Tree[Token]) -> IndexFilter:
    """Convert one parsed comparison to a validated filter.

    Args:
        tree: Parsed predicate tree.

    Returns:
        Validated comparison.

    Raises:
        ValueError: If a name is reserved or a comparison operand is
            invalid. The message names the query line and column.
    """
    reference, operation, operand = tree.children
    reference = cast(Tree[Token], reference)
    operation = cast(Token, operation)
    token = cast(Token, reference.children[0])
    if reference.data in {"quoted", "decoded_quoted"}:
        # The grammar admits only string literals here.
        name = str(_literal(token))
    else:
        name = str(token)
        if name in _RESERVED_NAMES:
            raise ValueError(
                _error(
                    token,
                    token,
                    f"{name!r} is reserved. Write index({json.dumps(name)}) "
                    "to name an index",
                )
            )
    value = (
        [_literal(cast(Token, child)) for child in operand.children]
        if isinstance(operand, Tree)
        else _literal(operand)
    )
    try:
        return IndexFilter(
            name,
            cast(FilterOperator, str(operation)),
            value,
            decode=reference.data in {"decoded", "decoded_quoted"},
        )
    except ValueError as exc:
        raise ValueError(_error(operand, operation, str(exc))) from exc


def _node(tree: Tree[Token]) -> _Node:
    """Convert a parsed expression to validated comparisons.

    Args:
        tree: Parsed expression tree.

    Returns:
        Boolean expression node.

    Raises:
        ValueError: If a comparison operand is invalid or the expression
            nests more than 100 levels of not, and, and or.
    """
    # Convert iteratively: each redundant parenthesis pair adds two parse
    # tree levels, so recursion would fail on valid input. Children are
    # pushed in reverse to validate predicates in source order.
    converted: dict[int, tuple[_Node, int]] = {}
    stack = [(tree, False)]
    while stack:
        current, expanded = stack.pop()
        if current.data == "predicate":
            leaf = _Node("predicate", predicate=_predicate(current))
            converted[id(current)] = (leaf, 0)
        elif not expanded:
            stack.append((current, True))
            stack.extend(
                (cast(Tree[Token], child), False)
                for child in reversed(current.children)
            )
        else:
            children = [converted.pop(id(c)) for c in current.children]
            if len(children) == 1 and current.data != "negate":
                converted[id(current)] = children[0]
                continue
            depth = 1 + max(d for _, d in children)
            if depth > _MAX_QUERY_DEPTH:
                raise ValueError(
                    "Index query nests more than "
                    f"{_MAX_QUERY_DEPTH} levels of not, and, and or"
                )
            node = _Node(str(current.data), tuple(n for n, _ in children))
            converted[id(current)] = (node, depth)
    return converted[id(tree)][0]


def compile_query(expression: str) -> IndexQuery:
    """Parse an index-only expression for repeated use across recordings.

    Args:
        expression: Comparisons joined by and, or, not, and parentheses.
            Use index("name/path") for quoted names and label(name) for
            categorical labels. Literals use JSON syntax.

    Returns:
        Compiled query. Index descriptors are validated when selecting items.

    Raises:
        ValueError: If the expression is empty, malformed, has bad operands,
            or nests more than 100 levels of not, and, and or. Redundant
            parentheses do not count toward this limit.
    """
    if not expression.strip():
        raise ValueError("Index query must not be empty")
    try:
        root = _node(_parser().parse(expression))
    except UnexpectedInput as exc:
        raise ValueError(
            f"Invalid index query at line {exc.line}, column {exc.column}:\n"
            + exc.get_context(expression)
        ) from exc
    return IndexQuery(expression, root)


def _operands(
    values: npt.NDArray[Any], operands: list[Any]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Choose native comparisons when conversion preserves numeric values.

    Args:
        values: One batch of index values.
        operands: Validated scalar operands.

    Returns:
        Arrays for exact comparisons. Mixed or out-of-range integer operands
        use Python object comparisons to avoid NumPy promotion rounding.
    """
    kind = values.dtype.kind
    if kind in {"i", "u"}:
        bounds = np.iinfo(values.dtype)
        if all(
            type(v) is int and bounds.min <= v <= bounds.max for v in operands
        ):
            return values, np.asarray(operands, dtype=values.dtype)
    elif kind == "f" and values.dtype.itemsize <= 8:
        # Binary64 represents every integer through 2**53 exactly. Larger
        # integer operands need the object fallback to avoid rounded equality.
        if all(type(v) is float or abs(v) <= 2**53 for v in operands):
            return values.astype(np.float64, copy=False), np.asarray(
                operands, dtype=np.float64
            )
    elif kind == "b":
        return values, np.asarray(operands, dtype=bool)
    elif kind in {"U", "T", "O"}:
        return values, np.asarray(operands, dtype=object)
    return values.astype(object), np.asarray(operands, dtype=object)


@dataclass(frozen=True)
class _Prepared:
    """Recording-specific comparison with an optional integer label lookup."""

    condition: IndexFilter
    """Source comparison and operand types."""

    array: ReadOnlyArray
    """Validated item index."""

    label_ids: npt.NDArray[np.int64] | None
    """Matching category IDs, resolved once from the index's label table."""

    label_count: int | None
    """Length of the validated label table for a decoded comparison."""

    def evaluate(
        self, values: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """Evaluate true and false masks, leaving nonfinite values unknown.

        Args:
            values: Cached batch for this index.

        Returns:
            True and false masks. Unknown positions are false in both masks.
        """
        condition = self.condition
        valid = (
            np.isfinite(values)
            if values.dtype.kind in {"i", "u", "f"}
            else np.ones(len(values), dtype=bool)
        )
        if self.label_ids is not None:
            # Labels were resolved once. No per-item string decoding is needed.
            left, right = _operands(values, self.label_ids.tolist())
            result = np.isin(left, right)
            if condition.operator == "!=":
                result = ~result
        else:
            left, right = _operands(values, list(condition._operands()))
            # Ordered object comparisons with NaN set the invalid flag. NaN
            # positions are masked as unknown below, so the flag is expected.
            with np.errstate(invalid="ignore"):
                if condition.operator == "in":
                    result = np.isin(left, right)
                else:
                    # Python scalar extraction preserves exact mixed
                    # comparisons.
                    result = np.asarray(
                        _OPERATORS[condition.operator](left, right[0]),
                        dtype=bool,
                    )
        # Keep unknown separate from false: negating a comparison against NaN
        # must not turn that item into a match.
        return result & valid, ~result & valid


def _prepare(
    recording: SigMFRecording, conditions: Sequence[IndexFilter]
) -> dict[int, _Prepared]:
    """Validate descriptors and resolve label operands before reading values.

    Each distinct index is opened, and each label table validated, once.

    Args:
        recording: Source recording.
        conditions: Leaf comparisons.

    Returns:
        Prepared comparisons keyed by source comparison identity.

    Raises:
        ValueError: If a descriptor or operand is incompatible.
        KeyError: If an index does not exist.
    """
    length = len(recording)
    arrays: dict[str, ReadOnlyArray] = {}
    tables: dict[str, list[JSONValue]] = {}
    lookups: dict[str, dict[str, list[int]]] = {}
    result = {}
    for condition in conditions:
        name = condition.name
        array = arrays.get(name)
        if array is None:
            array = recording.index(name)
            if array.attrs.get("axis") != "item" or "validity" in array.attrs:
                raise ValueError("Filters require dense item indexes")
            if array.shape != (length,):
                raise ValueError("Filters require one value per item")
            arrays[name] = array
        table = None
        if condition.decode:
            table = tables.get(name)
            if table is None:
                table = validate_categorical_values(
                    np.empty(0, dtype=array.dtype),
                    labels=array.attrs.get("labels"),
                )
                tables[name] = table
        condition._prepare(array, table)
        label_ids = None
        if table is not None:
            lookup = lookups.get(name)
            if lookup is None:
                # Several category IDs may have the same label. Collect every
                # matching ID rather than assuming the table is injective.
                lookup = {}
                for i, label in enumerate(table):
                    lookup.setdefault(cast(str, label), []).append(i)
                lookups[name] = lookup
            label_ids = np.asarray(
                sorted(
                    {
                        i
                        for label in condition._operands()
                        for i in lookup.get(cast(str, label), [])
                    }
                ),
                dtype=np.int64,
            )
        result[id(condition)] = _Prepared(
            condition,
            array,
            label_ids,
            None if table is None else len(table),
        )
    return result


def _evaluate(
    node: _Node,
    prepared: dict[int, _Prepared],
    batches: dict[str, npt.NDArray[Any]],
    size: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Combine comparison masks using three-valued Boolean logic.

    Args:
        node: Expression to evaluate.
        prepared: Bound leaf comparisons.
        batches: One cached batch per referenced index.
        size: Batch item count.

    Returns:
        True and false masks. Negation preserves unknown values.
    """
    if node.predicate is not None:
        return prepared[id(node.predicate)].evaluate(
            batches[node.predicate.name]
        )
    if node.kind == "negate":
        yes, no = _evaluate(node.children[0], prepared, batches, size)
        return no, yes
    # AND is false if any child is false, and OR is true if any child is true.
    # The other outcome requires all children, leaving unknown in neither
    # mask. These identities also make an empty conjunction select all items.
    conjunction = node.kind == "all_of"
    yes = np.full(size, conjunction, dtype=bool)
    no = ~yes
    for child in node.children:
        child_yes, child_no = _evaluate(child, prepared, batches, size)
        if conjunction:
            yes &= child_yes
            no |= child_no
        else:
            yes |= child_yes
            no &= child_no
    return yes, no


@dataclass(frozen=True)
class IndexQuery:
    """Parsed expression independent of recording or viewer state.

    Use compile_query() to parse text. IndexQuery() selects every item.
    """

    expression: str = ""
    """Original query text."""

    _root: _Node = _Node("all_of")
    """Validated syntax tree, without bound storage handles."""

    @property
    def index_names(self) -> tuple[str, ...]:
        """Exact index names referenced by this expression.

        Returns:
            Unique names in sorted order.
        """
        return tuple(sorted({c.name for c in self._root.predicates()}))

    def select(
        self,
        recording: SigMFRecording,
        *,
        batch_size: int = 4096,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> npt.NDArray[np.int64]:
        """Return matching original item positions without reading item JSON.

        Args:
            recording: Batched recording, preferably opened structurally.
            batch_size: Maximum index values evaluated at once.
            cancelled: Cooperative cancellation callback.
            progress: Callback receiving examined and total item counts.

        Returns:
            Ascending original positions. Memory scales with matches and,
            for each referenced index, one batch plus one stored chunk. Each
            stored chunk is decoded once. Selection does not modify the
            store.

        Raises:
            ValueError: If an index, operand, or batch size is unsupported.
            KeyError: If an index is absent.
            concurrent.futures.CancelledError: If cancelled. No partial result
                is returned.
        """
        matches = [
            batch
            for batch in self._iter_batches(
                recording,
                batch_size=batch_size,
                cancelled=cancelled,
                progress=progress,
            )
            if batch.size
        ]
        return np.concatenate(matches) if matches else np.empty(0, np.int64)

    def _iter_batches(
        self,
        recording: SigMFRecording,
        *,
        filters: Sequence[IndexFilter] = (),
        batch_size: int = 4096,
        cancelled: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[npt.NDArray[np.int64]]:
        """Yield index candidates for an optional downstream JSON filter.

        Args:
            recording: Batched source recording.
            filters: Additional comparisons combined with AND.
            batch_size: Maximum values evaluated per index per batch.
            cancelled: Cooperative cancellation callback.
            progress: Callback receiving examined and total item counts.

        Yields:
            Matching original positions for each contiguous batch.

        Raises:
            ValueError: If the recording or index descriptors are unsupported.
            KeyError: If an index is absent.
            concurrent.futures.CancelledError: If cancelled.
        """
        if not recording.batched:
            raise ValueError("Item filtering requires a batched recording")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        _check_cancelled(cancelled)
        root = _Node(
            "all_of",
            (
                self._root,
                *(_Node("predicate", predicate=c) for c in filters),
            ),
        )
        prepared = _prepare(recording, list(root.predicates()))
        # Predicates bind independently, but comparisons of the same index
        # share one array read per batch, including raw and decoded predicates.
        # Each reader decodes a stored chunk once, however long the chunk.
        arrays = {
            p.condition.name: SpanReader(p.array) for p in prepared.values()
        }
        decoded = {
            p.condition.name: p.label_count
            for p in prepared.values()
            if p.label_count is not None
        }
        total = len(recording)
        if progress is not None:
            progress(0, total)
        for start in range(0, total, batch_size):
            batches = {}
            stop = min(start + batch_size, total)
            _check_cancelled(cancelled)
            for name, array in arrays.items():
                _check_cancelled(cancelled)
                values = batches[name] = array.read(start, stop)
                # _prepare validated the label table and the integer dtype,
                # so each batch needs only a bounds check.
                if name in decoded and (
                    np.any(values < 0) or np.any(values >= decoded[name])
                ):
                    raise ValueError(
                        "Categorical ID is outside the labels lookup table"
                    )
            keep, _ = _evaluate(root, prepared, batches, stop - start)
            yield np.flatnonzero(keep).astype(np.int64) + start
            _check_cancelled(cancelled)
            if progress is not None:
                progress(stop, total)
        _check_cancelled(cancelled)


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    """Raise when the caller requests cancellation.

    Args:
        cancelled: Optional cooperative cancellation callback.

    Raises:
        concurrent.futures.CancelledError: If cancellation was requested.
    """
    if cancelled is not None and cancelled():
        raise CancelledError("Filtering cancelled")


__all__ = [
    "FilterOperator",
    "IndexFilter",
    "IndexQuery",
    "QueryScalar",
    "compile_query",
]
