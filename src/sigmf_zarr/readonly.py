"""Read-only views over Zarr arrays and groups."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from zarr.core.array import Array, ArrayConfigLike
from zarr.core.group import Group


class ReadOnlyAttributes(Mapping[str, Any]):
    """Read-only live view of Zarr attributes."""

    _attrs: Mapping[str, Any]
    """Backing Zarr attribute mapping."""

    def __init__(self, attrs: Mapping[str, Any]) -> None:
        """Initialize an attribute view.

        Args:
            attrs: Backing Zarr attribute mapping.
        """
        self._attrs = attrs

    def __getitem__(self, key: str) -> Any:
        """Return one attribute value.

        Args:
            key: Attribute key.

        Returns:
            Stored attribute value.
        """
        # Blocking assignment to the mapping does not protect nested lists or
        # dictionaries. Detach each value while keeping the view itself live.
        return deepcopy(self._attrs[key])

    def __iter__(self) -> Iterator[str]:
        """Iterate over attribute keys.

        Returns:
            Iterator over keys.
        """
        return iter(self._attrs)

    def __len__(self) -> int:
        """Return the number of attributes.

        Returns:
            Attribute count.
        """
        return len(self._attrs)


class _ReadOnlyIndexer:
    """Read-only wrapper for a Zarr selection indexer."""

    _indexer: Any
    """Backing Zarr indexer."""

    def __init__(self, indexer: Any) -> None:
        """Initialize an indexer view.

        Args:
            indexer: Backing Zarr selection indexer.
        """
        self._indexer = indexer

    def __getitem__(self, selection: object) -> object:
        """Read a selection.

        Args:
            selection: Zarr selection expression.

        Returns:
            Selected array values.
        """
        return self._indexer[selection]

    def __setitem__(self, selection: object, value: object) -> None:
        """Reject a write through the indexer.

        Args:
            selection: Zarr selection expression.
            value: Values that would be written.

        Raises:
            TypeError: Always, because the view is read-only.
        """
        del selection, value
        raise TypeError("Zarr array view is read-only. Use a mutation context")


class ReadOnlyArray:
    """Read-only facade for a Zarr array."""

    # New Zarr methods must be reviewed before they become public here.
    _READ_MEMBERS: ClassVar[frozenset[str]] = frozenset(
        {
            "basename",
            "cdata_shape",
            "compressor",
            "compressors",
            "config",
            "fill_value",
            "filters",
            "get_basic_selection",
            "get_block_selection",
            "get_coordinate_selection",
            "get_mask_selection",
            "get_orthogonal_selection",
            "info",
            "info_complete",
            "name",
            "nbytes",
            "nbytes_stored",
            "nchunks",
            "nchunks_initialized",
            "order",
            "path",
            "read_chunk_sizes",
            "serializer",
            "shards",
            "size",
            "write_chunk_sizes",
        }
    )
    """Public Zarr members reviewed for read-only delegation."""

    _array: Array
    """Backing writable Zarr array, kept private."""

    def __init__(self, array: Array) -> None:
        """Initialize an array view.

        Args:
            array: Backing Zarr array.
        """
        object.__setattr__(self, "_array", array)

    def __getitem__(self, selection: object) -> Any:
        """Read a selection.

        Args:
            selection: Zarr selection expression.

        Returns:
            Selected array values.
        """
        return self._array[selection]  # type: ignore[index]

    def __setitem__(self, selection: object, value: object) -> None:
        """Reject direct array writes.

        Args:
            selection: Zarr selection expression.
            value: Values that would be written.

        Raises:
            TypeError: Always, because the view is read-only.
        """
        del selection, value
        raise TypeError("Zarr array view is read-only. Use a mutation context")

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[Any]:
        """Materialize the logical array as a NumPy array.

        Args:
            dtype: Optional target NumPy dtype.
            copy: Optional NumPy copy policy.

        Returns:
            Materialized NumPy array.
        """
        # Always detach the NumPy result from storage so subsequent writes to
        # the materialized array cannot bypass integrity invalidation.
        values = np.asarray(self._array, dtype=dtype)
        return np.array(values, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Return the first-axis length.

        Returns:
            First-axis length.
        """
        return int(self._array.shape[0])

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            Read-only array representation.
        """
        return f"ReadOnlyArray({self._array!r})"

    def __getattr__(self, name: str) -> object:
        """Delegate explicitly supported reads to the backing array.

        Args:
            name: Member name.

        Returns:
            Backing array member.

        Raises:
            AttributeError: If the member is outside the supported read API.
        """
        if name not in self._READ_MEMBERS:
            raise AttributeError(
                f"{type(self).__name__!s} has no writable member {name!r}. "
                "Use a mutation context"
            )
        member = getattr(self._array, name)
        return member if callable(member) else deepcopy(member)

    @property
    def read_only(self) -> bool:
        """Whether this view permits reads only.

        Returns:
            True for every read-only view.
        """
        return True

    def with_config(self, config: ArrayConfigLike) -> ReadOnlyArray:
        """Return a read-only view with a different runtime configuration.

        Args:
            config: Zarr runtime configuration for the new view.

        Returns:
            Configured read-only array view.
        """
        return ReadOnlyArray(self._array.with_config(config))

    @property
    def attrs(self) -> ReadOnlyAttributes:
        """Read-only array attributes.

        Returns:
            Read-only attribute mapping.
        """
        return ReadOnlyAttributes(self._array.attrs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical array shape.

        Returns:
            Logical dimension lengths.
        """
        return tuple(int(size) for size in self._array.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        """Logical chunk shape.

        Returns:
            Chunk dimension lengths.
        """
        return tuple(int(size) for size in self._array.chunks)

    @property
    def ndim(self) -> int:
        """Logical array rank.

        Returns:
            Number of dimensions.
        """
        return int(self._array.ndim)

    @property
    def dtype(self) -> Any:
        """Logical array dtype.

        Returns:
            Zarr or NumPy dtype object.
        """
        return self._array.dtype

    @property
    def metadata(self) -> Any:
        """Physical array metadata.

        Returns:
            Zarr array metadata object.
        """
        return deepcopy(self._array.metadata)

    @property
    def blocks(self) -> _ReadOnlyIndexer:
        """Read-only block indexer.

        Returns:
            Read-only block selection wrapper.
        """
        return _ReadOnlyIndexer(self._array.blocks)

    @property
    def oindex(self) -> _ReadOnlyIndexer:
        """Read-only orthogonal indexer.

        Returns:
            Read-only orthogonal selection wrapper.
        """
        return _ReadOnlyIndexer(self._array.oindex)

    @property
    def vindex(self) -> _ReadOnlyIndexer:
        """Read-only vectorized indexer.

        Returns:
            Read-only vectorized selection wrapper.
        """
        return _ReadOnlyIndexer(self._array.vindex)


class ReadOnlyGroup:
    """Read-only facade for a Zarr group and its descendants."""

    _READ_MEMBERS: ClassVar[frozenset[str]] = frozenset(
        {
            "basename",
            "info",
            "info_complete",
            "keys",
            "metadata",
            "name",
            "nmembers",
            "path",
            "synchronizer",
            "tree",
        }
    )
    """Public Zarr members reviewed for read-only delegation."""

    _group: Group
    """Backing writable Zarr group, kept private."""

    def __init__(self, group: Group) -> None:
        """Initialize a group view.

        Args:
            group: Backing Zarr group.
        """
        object.__setattr__(self, "_group", group)

    @staticmethod
    def _wrap(member: Array | Group) -> ReadOnlyArray | ReadOnlyGroup:
        """Wrap one Zarr group member.

        Args:
            member: Backing array or group.

        Returns:
            Corresponding read-only view.
        """
        # Wrapping every descendant prevents a writable child from escaping
        # through group traversal.
        if isinstance(member, Array):
            return ReadOnlyArray(member)
        return ReadOnlyGroup(member)

    def __getitem__(self, name: str) -> ReadOnlyArray | ReadOnlyGroup:
        """Return a read-only child member.

        Args:
            name: Child name or relative path.

        Returns:
            Read-only array or group view.
        """
        return type(self)._wrap(self._group[name])

    def __setitem__(self, name: str, value: object) -> None:
        """Reject assignment through the group.

        Args:
            name: Child name.
            value: Value that would be assigned.

        Raises:
            TypeError: Always, because the view is read-only.
        """
        del name, value
        raise TypeError("Zarr group view is read-only")

    def __delitem__(self, name: str) -> None:
        """Reject deletion through the group.

        Args:
            name: Child name.

        Raises:
            TypeError: Always, because the view is read-only.
        """
        del name
        raise TypeError("Zarr group view is read-only")

    def __contains__(self, name: object) -> bool:
        """Return whether a child exists.

        Args:
            name: Candidate child name.

        Returns:
            True when the child exists.
        """
        return isinstance(name, str) and name in self._group

    def get(
        self,
        name: str,
        default: object = None,
    ) -> ReadOnlyArray | ReadOnlyGroup | object:
        """Return a read-only child or a default value.

        Args:
            name: Child name or relative path.
            default: Value returned when the child is absent.

        Returns:
            Read-only child view or ``default``.
        """
        member = self._group.get(name, default)
        if isinstance(member, Array | Group):
            return type(self)._wrap(member)
        return member

    def __len__(self) -> int:
        """Return the number of immediate children.

        Returns:
            Immediate child count.
        """
        return len(self._group)

    def __getattr__(self, name: str) -> object:
        """Delegate explicitly supported reads to the backing group.

        Args:
            name: Member name.

        Returns:
            Backing group member.

        Raises:
            AttributeError: If the member is outside the supported read API.
        """
        if name not in self._READ_MEMBERS:
            raise AttributeError(
                f"{type(self).__name__!s} has no writable member {name!r}"
            )
        member = getattr(self._group, name)
        return member if callable(member) else deepcopy(member)

    @property
    def read_only(self) -> bool:
        """Whether this view permits reads only.

        Returns:
            True for every read-only view.
        """
        return True

    @property
    def attrs(self) -> ReadOnlyAttributes:
        """Read-only group attributes.

        Returns:
            Read-only attribute mapping.
        """
        return ReadOnlyAttributes(self._group.attrs)

    def arrays(self) -> Iterator[tuple[str, ReadOnlyArray]]:
        """Iterate over immediate child arrays.

        Returns:
            Iterator of child names and read-only array views.
        """
        for name, array in self._group.arrays():
            yield name, ReadOnlyArray(array)

    def array_keys(self) -> tuple[str, ...]:
        """Return immediate child array names.

        Returns:
            Child array names.
        """
        return tuple(self._group.array_keys())

    def array_values(self) -> Iterator[ReadOnlyArray]:
        """Iterate over read-only immediate child arrays.

        Returns:
            Iterator of read-only array views.
        """
        for _, array in self.arrays():
            yield array

    def groups(self) -> Iterator[tuple[str, ReadOnlyGroup]]:
        """Iterate over immediate child groups.

        Returns:
            Iterator of child names and read-only group views.
        """
        for name, group in self._group.groups():
            yield name, ReadOnlyGroup(group)

    def group_keys(self) -> tuple[str, ...]:
        """Return immediate child group names.

        Returns:
            Child group names.
        """
        return tuple(self._group.group_keys())

    def group_values(self) -> Iterator[ReadOnlyGroup]:
        """Iterate over read-only immediate child groups.

        Returns:
            Iterator of read-only group views.
        """
        for _, group in self.groups():
            yield group

    def members(
        self,
        max_depth: int | None = 0,
        *,
        use_consolidated_for_children: bool = True,
    ) -> tuple[tuple[str, ReadOnlyArray | ReadOnlyGroup], ...]:
        """Return read-only descendants.

        Args:
            max_depth: Maximum traversal depth.
            use_consolidated_for_children: Whether to use consolidated child
                metadata when available.

        Returns:
            Child paths paired with read-only views.
        """
        return tuple(
            (name, type(self)._wrap(member))
            for name, member in self._group.members(
                max_depth,
                use_consolidated_for_children=(
                    use_consolidated_for_children
                ),
            )
        )


__all__ = [
    "ReadOnlyArray",
    "ReadOnlyAttributes",
    "ReadOnlyGroup",
]
