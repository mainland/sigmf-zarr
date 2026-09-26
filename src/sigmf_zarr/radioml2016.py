"""Import RadioML 2016 pickle datasets into SigMF-Zarr."""

from __future__ import annotations

import _codecs
import codecs
import io
import math
import os
import pickle
import pickletools
import warnings
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr._import_target import (
    remove_new_store_on_failure,
)
from sigmf_zarr.integrity import calculate_array_sha512
from sigmf_zarr.json import JSONObject
from sigmf_zarr.provenance import import_metadata
from sigmf_zarr.store import SigMFZarrStore, ZarrFormat
from sigmf_zarr.store._transaction import recording_import_transaction

RadioML2016Key = tuple[str, int]
"""RadioML 2016 mapping key of `(modulation_class, snr_db)`."""

RadioML2016Value = npt.NDArray[np.floating[Any]]
"""RadioML 2016 IQ batch with shape `(N, 2, T)`."""

RadioML2016Dict = Mapping[RadioML2016Key, RadioML2016Value]
"""Canonical in-memory RadioML 2016 pickle mapping."""

_PLAIN_DTYPE_CODES = frozenset(
    {"f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"}
)
"""NumPy type codes accepted in pickled dtypes."""

_MAX_OPCODES = 1 << 20
"""Maximum number of opcodes in a pickle."""

_MAX_MARK_DEPTH = 16
"""Maximum number of open `MARK` opcodes."""

_MAX_TUPLE_DEPTH = 2
"""Maximum tuple nesting, as in `(1, (N, 2, T), dtype, False, data)`."""

_MAX_TUPLE_LENGTH = 32
"""Maximum tuple length, which also bounds the number of array dimensions."""

_MAX_NUMBER_LINE = 32
"""Maximum length of a text integer or memo index argument."""

_MAX_GLOBAL_LINE = 128
"""Maximum length of a text module or global name argument."""

_MAX_LONG_BYTES = 8
"""Maximum length of a binary integer argument."""

_INT_RANGE = range(-(1 << 63), 1 << 63)
"""Integers that a binary integer argument of `_MAX_LONG_BYTES` can hold.

Text integers are limited to the same range. Python does not randomize
integer hashes, and a wider range would let a pickle choose many distinct
integers with one hash value.
"""

_MEMO_RANGE = range(1 << 32)
"""Memo indexes that the 4-byte `LONG_BINPUT` argument can hold."""

_MAX_ENTRIES = 1 << 16
"""Maximum number of entries in a pickled dictionary."""

_MAX_BATCHES = 1 << 12
"""Maximum number of batches that an import accepts.

Each batch adds a provenance source of about 400 bytes, and every append
rereads the recording metadata. The released RadioML 2016 and RML22 files
have at most 231 batches.
"""

_MAX_LABEL_LENGTH = 256
"""Maximum length of a modulation label in characters."""

_SNR_RANGE = range(np.iinfo(np.int16).min, np.iinfo(np.int16).max + 1)
"""SNR values that the `int16` SNR index can hold."""

_BUFFER_BYTES = 1 << 16
"""Byte strings of at least this length are kept as mutable buffers.

An array then uses the buffer as its storage instead of a copy, as NumPy does
when it unpickles an array from a byte string.
"""

_OPCODE_NAMES = {
    ord(opcode.code): opcode.name for opcode in pickletools.opcodes
}
"""Pickle opcode names for error messages."""


class _Global:
    """Inert placeholder for a global that NumPy array pickles name."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        """Create a placeholder.

        Args:
            name: Qualified global name for error messages.
        """
        self.name = name


_NDARRAY = _Global("numpy.ndarray")
_DTYPE = _Global("numpy.dtype")
_RECONSTRUCT = _Global("numpy.core.multiarray._reconstruct")
_FROMBUFFER = _Global("numpy.core.numeric._frombuffer")
_ENCODE = _Global("_codecs.encode")

_PICKLE_GLOBALS = {
    ("_codecs", "encode"): _ENCODE,
    ("numpy", "dtype"): _DTYPE,
    ("numpy", "ndarray"): _NDARRAY,
    ("numpy.core.multiarray", "_reconstruct"): _RECONSTRUCT,
    ("numpy._core.multiarray", "_reconstruct"): _RECONSTRUCT,
    ("numpy.core.numeric", "_frombuffer"): _FROMBUFFER,
    ("numpy._core.numeric", "_frombuffer"): _FROMBUFFER,
}
"""Globals that NumPy array pickles name, including Python 2 module names."""


class _Tuple:
    """Tuple built by a pickle, with its nesting depth."""

    __slots__ = ("depth", "items")
    depth: int
    items: tuple[object, ...]

    def __init__(self, items: tuple[object, ...]) -> None:
        """Validate the items and record the nesting depth.

        Args:
            items: Tuple items.

        Raises:
            ValueError: If the tuple is too long or too deep, or holds a
                value that NumPy array pickles do not put in tuples.
        """
        if len(items) > _MAX_TUPLE_LENGTH:
            raise ValueError(
                f"a tuple has more than {_MAX_TUPLE_LENGTH} items"
            )
        depth = 1
        for item in items:
            if isinstance(item, _Tuple):
                depth = max(depth, item.depth + 1)
            elif not isinstance(item, _TUPLE_ITEM_TYPES):
                raise ValueError(f"a tuple cannot hold {_describe(item)}")
        if depth > _MAX_TUPLE_DEPTH:
            raise ValueError(
                f"tuples are nested more than {_MAX_TUPLE_DEPTH} levels deep"
            )
        self.items = items
        self.depth = depth


class _DTypeSpec:
    """Validated description of a plain real numeric dtype."""

    __slots__ = ("byteorder", "code")

    def __init__(self, code: str) -> None:
        """Create a dtype description that still needs its byte order.

        Args:
            code: Type code from `_PLAIN_DTYPE_CODES`.
        """
        self.code = code
        self.byteorder: str | None = None

    def dtype(self) -> np.dtype[Any]:
        """Return the described dtype.

        Returns:
            New NumPy dtype.

        Raises:
            ValueError: If the pickle never set the byte order.
        """
        if self.byteorder is None:
            raise ValueError(f"dtype {self.code!r} has no recorded state")
        return np.dtype(self.byteorder + self.code)


class _ArraySpec:
    """Array placeholder that receives its data from a `BUILD` opcode."""

    __slots__ = ("array",)

    def __init__(self) -> None:
        """Create an array placeholder without data."""
        self.array: npt.NDArray[Any] | None = None


class _Dict:
    """Dictionary built by a pickle."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        """Create an empty dictionary."""
        self.entries: dict[RadioML2016Key, _ArraySpec] = {}


_SCALAR_TYPES = (type(None), bool, int, str, bytes)
_TUPLE_ITEM_TYPES = (*_SCALAR_TYPES, bytearray, _Global, _DTypeSpec)


def _describe(value: object) -> str:
    """Name the kind of a pickle value for error messages.

    Args:
        value: Value built by the pickle.

    Returns:
        Short description.
    """
    if isinstance(value, _Global):
        return f"global {value.name}"
    names = {
        _Tuple: "a tuple",
        _DTypeSpec: "a dtype",
        _ArraySpec: "an array",
        _Dict: "a dictionary",
    }
    return names.get(type(value), f"a {type(value).__name__}")


def _text(value: object) -> str | None:
    """Return a short pickled string as text.

    Python 2 pickles store strings that the unpickler may leave as bytes.

    Args:
        value: Value built by the pickle.

    Returns:
        String value, or `None` if the value is not a string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("latin1")
    return None


def _is_int(value: object, *allowed: int) -> bool:
    """Return whether a value is an integer or Boolean from a set.

    Args:
        value: Value built by the pickle.
        *allowed: Accepted values.

    Returns:
        Whether the value is accepted.
    """
    return isinstance(value, int) and value in allowed


class _RestrictedUnpickler:
    """Interpreter for the pickle opcodes that NumPy array mappings use.

    The interpreter never calls a function that the file names. It builds
    arrays from validated type codes, shapes, and data, and checks each
    declared size against the file size before it allocates memory. Its other
    allocations are values read from the file and at most one small object
    per opcode, so memory use is at most a few times the file size plus about
    120 bytes per opcode. Dictionary keys and memo indexes are integers and
    strings from bounded ranges, so their hash values cannot be chosen to
    collide in bulk.
    """

    def __init__(
        self, handle: io.BufferedIOBase, size: int, encoding: str
    ) -> None:
        """Prepare to read a pickle.

        Args:
            handle: Binary file positioned at the start of the pickle.
            size: File size in bytes.
            encoding: Encoding of Python 2 strings.
        """
        self._handle = handle
        self._size = size
        self._pos = 0
        self._encoding = encoding
        self._stack: list[object] = []
        self._marks: list[list[object]] = []
        self._memo: dict[int, object] = {}
        self._array_bytes = 0
        self._encoded_bytes = 0
        self._label_chars = 0
        self._dispatch: dict[int, Callable[[], None]] = {
            pickle.PROTO[0]: self._load_proto,
            pickle.FRAME[0]: self._load_frame,
            pickle.MARK[0]: self._load_mark,
            pickle.NONE[0]: lambda: self._stack.append(None),
            pickle.NEWTRUE[0]: lambda: self._stack.append(True),
            pickle.NEWFALSE[0]: lambda: self._stack.append(False),
            pickle.INT[0]: self._load_int,
            pickle.BININT[0]: lambda: self._push_int(4, signed=True),
            pickle.BININT1[0]: lambda: self._push_int(1),
            pickle.BININT2[0]: lambda: self._push_int(2),
            pickle.LONG[0]: self._load_long,
            pickle.LONG1[0]: lambda: self._load_long_binary(1),
            pickle.LONG4[0]: lambda: self._load_long_binary(4),
            pickle.STRING[0]: self._load_string,
            pickle.BINSTRING[0]: lambda: self._load_binstring(4),
            pickle.SHORT_BINSTRING[0]: lambda: self._load_binstring(1),
            pickle.UNICODE[0]: self._load_unicode,
            pickle.BINUNICODE[0]: lambda: self._load_binunicode(4),
            pickle.SHORT_BINUNICODE[0]: lambda: self._load_binunicode(1),
            pickle.BINUNICODE8[0]: lambda: self._load_binunicode(8),
            pickle.BINBYTES[0]: lambda: self._load_binbytes(4),
            pickle.SHORT_BINBYTES[0]: lambda: self._load_binbytes(1),
            pickle.BINBYTES8[0]: lambda: self._load_binbytes(8),
            pickle.BYTEARRAY8[0]: self._load_bytearray8,
            pickle.EMPTY_TUPLE[0]: lambda: self._stack.append(_Tuple(())),
            pickle.TUPLE[0]: self._load_tuple,
            pickle.TUPLE1[0]: lambda: self._load_short_tuple(1),
            pickle.TUPLE2[0]: lambda: self._load_short_tuple(2),
            pickle.TUPLE3[0]: lambda: self._load_short_tuple(3),
            pickle.EMPTY_DICT[0]: lambda: self._stack.append(_Dict()),
            pickle.DICT[0]: self._load_dict,
            pickle.SETITEM[0]: self._load_setitem,
            pickle.SETITEMS[0]: self._load_setitems,
            pickle.PUT[0]: self._load_put,
            pickle.BINPUT[0]: lambda: self._put(self._uint(1)),
            pickle.LONG_BINPUT[0]: lambda: self._put(self._uint(4)),
            pickle.MEMOIZE[0]: lambda: self._put(len(self._memo)),
            pickle.GET[0]: self._load_get,
            pickle.BINGET[0]: lambda: self._get(self._uint(1)),
            pickle.LONG_BINGET[0]: lambda: self._get(self._uint(4)),
            pickle.GLOBAL[0]: self._load_global,
            pickle.STACK_GLOBAL[0]: self._load_stack_global,
            pickle.REDUCE[0]: self._load_reduce,
            pickle.BUILD[0]: self._load_build,
        }

    def load(self) -> dict[RadioML2016Key, npt.NDArray[Any]]:
        """Interpret the pickle.

        Returns:
            Dictionary of arrays.

        Raises:
            ValueError: If the pickle is malformed, uses an unsupported
                opcode, global, or value, or exceeds a limit.
        """
        for _ in range(_MAX_OPCODES):
            code = self._read(1)[0]
            if code == pickle.STOP[0]:
                return self._finish(self._pop())
            handler = self._dispatch.get(code)
            if handler is None:
                name = _OPCODE_NAMES.get(code, f"0x{code:02x}")
                raise ValueError(f"unsupported pickle opcode {name}")
            handler()
        raise ValueError(f"the pickle has more than {_MAX_OPCODES} opcodes")

    def _read(self, count: int) -> bytes:
        """Read bytes that must be present in the file.

        Args:
            count: Number of bytes.

        Returns:
            Bytes read.

        Raises:
            ValueError: If the file ends first.
        """
        if count > self._size - self._pos:
            raise ValueError("the pickle is truncated")
        data = self._handle.read(count)
        if len(data) != count:
            raise ValueError("the pickle is truncated")
        self._pos += count
        return data

    def _read_buffer(self, count: int) -> bytearray:
        """Read bytes that must be present in the file into a new buffer.

        Args:
            count: Number of bytes.

        Returns:
            Buffer read.

        Raises:
            ValueError: If the file ends first.
        """
        if count > self._size - self._pos:
            raise ValueError("the pickle is truncated")
        buffer = bytearray(count)
        if self._handle.readinto(buffer) != count:
            raise ValueError("the pickle is truncated")
        self._pos += count
        return buffer

    def _readline(self, limit: int | None = None) -> bytes:
        """Read a newline-terminated argument.

        Args:
            limit: Maximum length including the newline, or `None` for the
                rest of the file.

        Returns:
            Line without its newline.

        Raises:
            ValueError: If the line is unterminated or too long.
        """
        remaining = self._size - self._pos
        line = self._handle.readline(
            remaining if limit is None else min(limit, remaining)
        )
        self._pos += len(line)
        if not line.endswith(b"\n"):
            raise ValueError("a text argument is unterminated or too long")
        return line[:-1]

    def _uint(self, size: int) -> int:
        """Read a little-endian unsigned integer argument.

        Args:
            size: Width in bytes.

        Returns:
            Integer value.
        """
        return int.from_bytes(self._read(size), "little")

    def _pop(self) -> object:
        """Pop a value from the stack.

        Returns:
            Top value.

        Raises:
            ValueError: If the stack is empty.
        """
        if not self._stack:
            raise ValueError("an opcode needs a value on an empty stack")
        return self._stack.pop()

    def _top(self) -> object:
        """Return the top of the stack.

        Returns:
            Top value.

        Raises:
            ValueError: If the stack is empty.
        """
        if not self._stack:
            raise ValueError("an opcode needs a value on an empty stack")
        return self._stack[-1]

    def _pop_mark(self) -> list[object]:
        """Pop the values above the innermost mark.

        Returns:
            Values pushed since the mark.

        Raises:
            ValueError: If there is no open mark.
        """
        if not self._marks:
            raise ValueError("an opcode needs a MARK")
        items = self._stack
        self._stack = self._marks.pop()
        return items

    def _decode_string(self, data: bytes) -> str:
        """Decode a Python 2 string.

        Args:
            data: Encoded string.

        Returns:
            Decoded string.

        Raises:
            ValueError: If the string cannot be decoded.
        """
        try:
            return data.decode(self._encoding)
        except LookupError as exc:
            raise ValueError(f"cannot decode strings: {exc}") from exc

    def _load_proto(self) -> None:
        """Check a protocol version."""
        version = self._read(1)[0]
        if version > pickle.HIGHEST_PROTOCOL:
            raise ValueError(f"unsupported pickle protocol {version}")

    def _load_frame(self) -> None:
        """Skip a frame header, which only groups the following opcodes."""
        if self._uint(8) > self._size - self._pos:
            raise ValueError("the pickle is truncated")

    def _load_mark(self) -> None:
        """Open a mark."""
        if len(self._marks) >= _MAX_MARK_DEPTH:
            raise ValueError(f"more than {_MAX_MARK_DEPTH} marks are open")
        self._marks.append(self._stack)
        self._stack = []

    @staticmethod
    def _text_int(text: bytes) -> int:
        """Parse a text integer argument in the range of the binary opcodes.

        Args:
            text: Integer digits.

        Returns:
            Integer value.

        Raises:
            ValueError: If the text is not an integer, or the integer does
                not fit in `_MAX_LONG_BYTES` bytes.
        """
        value = int(text)
        if value not in _INT_RANGE:
            raise ValueError("an integer is too large")
        return value

    def _load_int(self) -> None:
        """Push a text integer, which Python 2 also uses for Booleans."""
        line = self._readline(_MAX_NUMBER_LINE)
        if line in {b"00", b"01"}:
            self._stack.append(line == b"01")
        else:
            self._stack.append(self._text_int(line))

    def _push_int(self, size: int, *, signed: bool = False) -> None:
        """Push a little-endian binary integer.

        Args:
            size: Width in bytes.
            signed: Whether the integer is signed.
        """
        self._stack.append(
            int.from_bytes(self._read(size), "little", signed=signed)
        )

    def _load_long(self) -> None:
        """Push a text long integer."""
        line = self._readline(_MAX_NUMBER_LINE).rstrip(b"L")
        self._stack.append(self._text_int(line))

    def _load_long_binary(self, size: int) -> None:
        """Push a binary long integer.

        Args:
            size: Width of the length field in bytes.
        """
        count = self._uint(size)
        if count > _MAX_LONG_BYTES:
            raise ValueError("an integer is too large")
        self._push_int(count, signed=True)

    def _load_string(self) -> None:
        """Push a quoted and escaped Python 2 string."""
        line = self._readline()
        if len(line) < 2 or line[0] != line[-1] or line[:1] not in b"'\"":
            raise ValueError("a STRING argument is not quoted")
        # Python 2 never writes the escapes that Python deprecates.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                data = _codecs.escape_decode(line[1:-1])[0]
            except DeprecationWarning as exc:
                raise ValueError(f"a STRING argument has an {exc}") from exc
        self._stack.append(self._decode_string(data))

    def _load_binstring(self, size: int) -> None:
        """Push a length-prefixed Python 2 string.

        Args:
            size: Width of the length field in bytes.
        """
        count = int.from_bytes(self._read(size), "little", signed=size > 1)
        if count < 0:
            raise ValueError("a string has a negative length")
        self._stack.append(self._decode_string(self._read(count)))

    def _load_unicode(self) -> None:
        """Push a text Unicode string."""
        self._stack.append(str(self._readline(), "raw-unicode-escape"))

    def _load_binunicode(self, size: int) -> None:
        """Push a length-prefixed UTF-8 string.

        Args:
            size: Width of the length field in bytes.
        """
        data = self._read(self._uint(size))
        self._stack.append(str(data, "utf-8", "surrogatepass"))

    def _load_binbytes(self, size: int) -> None:
        """Push a length-prefixed byte string.

        Args:
            size: Width of the length field in bytes.
        """
        count = self._uint(size)
        if count < _BUFFER_BYTES:
            self._stack.append(self._read(count))
        else:
            self._stack.append(self._read_buffer(count))

    def _load_bytearray8(self) -> None:
        """Push a length-prefixed byte array."""
        self._stack.append(self._read_buffer(self._uint(8)))

    def _load_tuple(self) -> None:
        """Push a tuple of the values above the innermost mark."""
        # Pop before binding the append, which must target the outer stack.
        result = _Tuple(tuple(self._pop_mark()))
        self._stack.append(result)

    def _load_short_tuple(self, count: int) -> None:
        """Push a tuple of the top values.

        Args:
            count: Number of values.
        """
        if len(self._stack) < count:
            raise ValueError("an opcode needs a value on an empty stack")
        items = tuple(self._stack[-count:])
        del self._stack[-count:]
        self._stack.append(_Tuple(items))

    def _load_dict(self) -> None:
        """Push a dictionary of the pairs above the innermost mark."""
        result = _Dict()
        self._set_items(result, self._pop_mark())
        self._stack.append(result)

    def _load_setitem(self) -> None:
        """Add the top key and value to the dictionary below them."""
        value = self._pop()
        key = self._pop()
        self._set_items(self._top(), [key, value])

    def _load_setitems(self) -> None:
        """Add the pairs above the innermost mark to a dictionary."""
        items = self._pop_mark()
        self._set_items(self._top(), items)

    def _set_items(self, target: object, items: list[object]) -> None:
        """Add alternating keys and values to a dictionary.

        Args:
            target: Dictionary built by the pickle.
            items: Alternating keys and values.

        Raises:
            ValueError: If the target is not a dictionary, a key or value
                is not supported, the dictionary would hold more than
                `_MAX_ENTRIES` entries, or the labels of all keys added so far
                would hold more characters than the file has bytes.
        """
        if not isinstance(target, _Dict):
            raise ValueError(f"cannot add items to {_describe(target)}")
        if len(items) % 2:
            raise ValueError("a dictionary has a key without a value")
        entries = target.entries
        for key, value in zip(items[::2], items[1::2], strict=True):
            if not isinstance(value, _ArraySpec):
                raise ValueError(
                    f"a dictionary value is {_describe(value)}, not an array"
                )
            entry_key = self._key(key)
            # The importer copies the label of every entry into its
            # provenance, and a memoized label costs the file only once.
            self._label_chars += len(entry_key[0])
            if self._label_chars > self._size:
                raise ValueError(
                    "dictionary keys hold more label text than the file"
                )
            # As with pickle.load, a repeated key replaces the earlier value.
            if len(entries) >= _MAX_ENTRIES and entry_key not in entries:
                raise ValueError(
                    f"a dictionary has more than {_MAX_ENTRIES} entries"
                )
            entries[entry_key] = value

    @staticmethod
    def _key(key: object) -> RadioML2016Key:
        """Convert a pickled dictionary key.

        Args:
            key: Key built by the pickle.

        Returns:
            Modulation label and SNR.

        Raises:
            ValueError: If the key is not a pair of a string of at most
                `_MAX_LABEL_LENGTH` characters and a non-Boolean integer in
                the `int16` range.
        """
        if isinstance(key, _Tuple) and len(key.items) == 2:
            label, snr = key.items
            if (
                type(label) is str
                and len(label) <= _MAX_LABEL_LENGTH
                and type(snr) is int
                and snr in _SNR_RANGE
            ):
                return label, snr
        raise ValueError(
            "a dictionary key must be a (modulation, snr) pair of a string "
            f"of at most {_MAX_LABEL_LENGTH} characters and an int16 integer"
        )

    def _memo_index(self) -> int:
        """Read a text memo index in the range of the binary opcodes.

        Returns:
            Memo index.

        Raises:
            ValueError: If the index is not an integer, or does not fit in
                the 4-byte argument of `LONG_BINPUT`.
        """
        index = int(self._readline(_MAX_NUMBER_LINE))
        if index not in _MEMO_RANGE:
            raise ValueError(f"memo index {index} is out of range")
        return index

    def _load_put(self) -> None:
        """Store the top value under a text memo index."""
        self._put(self._memo_index())

    def _put(self, index: int) -> None:
        """Store the top value in the memo.

        Each entry costs an opcode, so the opcode limit bounds the memo.

        Args:
            index: Memo index in `_MEMO_RANGE`.
        """
        self._memo[index] = self._top()

    def _load_get(self) -> None:
        """Push a value stored under a text memo index."""
        self._get(self._memo_index())

    def _get(self, index: int) -> None:
        """Push a value from the memo.

        Args:
            index: Memo index.

        Raises:
            ValueError: If the index was never stored.
        """
        try:
            self._stack.append(self._memo[index])
        except KeyError:
            raise ValueError(f"memo index {index} is undefined") from None

    def _load_global(self) -> None:
        """Push the placeholder for a global named by text arguments."""
        module = self._readline(_MAX_GLOBAL_LINE).decode("utf-8")
        name = self._readline(_MAX_GLOBAL_LINE).decode("utf-8")
        self._stack.append(self._find_global(module, name))

    def _load_stack_global(self) -> None:
        """Push the placeholder for a global named by stack strings."""
        name = self._pop()
        module = self._pop()
        if not isinstance(module, str) or not isinstance(name, str):
            raise ValueError("STACK_GLOBAL needs two strings")
        self._stack.append(self._find_global(module, name))

    @staticmethod
    def _find_global(module: str, name: str) -> _Global:
        """Return the placeholder for an allowed global.

        Args:
            module: Module named by the pickle.
            name: Global name within the module.

        Returns:
            Placeholder.

        Raises:
            ValueError: If the global is not allowed.
        """
        try:
            return _PICKLE_GLOBALS[module, name]
        except KeyError:
            raise ValueError(
                f"unsupported global {module[:100]}.{name[:100]}"
            ) from None

    def _load_reduce(self) -> None:
        """Apply an allowed global to its arguments."""
        args = self._pop()
        function = self._pop()
        if not isinstance(args, _Tuple):
            raise ValueError(f"REDUCE arguments are {_describe(args)}")
        items = args.items
        result: object
        if function is _RECONSTRUCT:
            if (
                len(items) != 3
                or items[0] is not _NDARRAY
                or not isinstance(items[1], _Tuple)
                or items[1].items != (0,)
                or _text(items[2]) != "b"
            ):
                raise ValueError("unsupported _reconstruct arguments")
            result = _ArraySpec()
        elif function is _DTYPE:
            if (
                len(items) != 3
                or _text(items[0]) not in _PLAIN_DTYPE_CODES
                or not _is_int(items[1], 0)
                or not _is_int(items[2], 1)
            ):
                raise ValueError("unsupported dtype arguments")
            result = _DTypeSpec(cast(str, _text(items[0])))
        elif function is _ENCODE:
            result = self._encode(items)
        elif function is _FROMBUFFER:
            if len(items) != 4 or _text(items[3]) not in {"C", "F"}:
                raise ValueError("unsupported _frombuffer arguments")
            result = _ArraySpec()
            result.array = self._array(
                items[2], items[1], _text(items[3]) == "F", items[0]
            )
        else:
            raise ValueError(f"cannot call {_describe(function)}")
        self._stack.append(result)

    def _encode(self, items: tuple[object, ...]) -> bytes | bytearray:
        """Return the Latin-1 bytes that Python 3 pickles encode as text.

        Args:
            items: `_codecs.encode` arguments.

        Returns:
            Encoded bytes, as a buffer if they are long.

        Raises:
            ValueError: If the arguments are unsupported, or the encoded
                strings would exceed the file size.
        """
        # Check the type first, because an unhashable encoding cannot be
        # looked up in a set.
        if (
            len(items) != 2
            or not isinstance(items[0], str)
            or not isinstance(items[1], str)
            or items[1] not in {"latin1", "latin-1"}
        ):
            raise ValueError("unsupported _codecs.encode arguments")
        # A memoized string can be encoded many times.
        if len(items[0]) > self._size - self._encoded_bytes:
            raise ValueError("encoded strings hold more data than the file")
        self._encoded_bytes += len(items[0])
        if len(items[0]) < _BUFFER_BYTES:
            return items[0].encode("latin1")
        return bytearray(items[0], "latin1")

    def _load_build(self) -> None:
        """Set the state of a dtype or array placeholder."""
        state = self._pop()
        target = self._top()
        if not isinstance(state, _Tuple):
            raise ValueError(f"BUILD state is {_describe(state)}")
        items = state.items
        if isinstance(target, _DTypeSpec) and target.byteorder is None:
            byteorders = {"|"} if target.code[1] == "1" else {"<", ">"}
            if (
                len(items) != 8
                or not _is_int(items[0], 3)
                or _text(items[1]) not in byteorders
                or items[2:] != (None, None, None, -1, -1, 0)
            ):
                raise ValueError(f"unsupported state for dtype {target.code}")
            target.byteorder = cast(str, _text(items[1]))
        elif isinstance(target, _ArraySpec) and target.array is None:
            if (
                len(items) != 5
                or not _is_int(items[0], 1)
                or not _is_int(items[3], 0, 1)
            ):
                raise ValueError("unsupported array state")
            array = self._array(items[1], items[2], bool(items[3]), items[4])
            # NumPy converts unpickled array state to native byte order.
            if not array.dtype.isnative:
                array = array.astype(array.dtype.newbyteorder("="))
            target.array = array
        else:
            raise ValueError(f"cannot set the state of {_describe(target)}")

    def _array(
        self, shape: object, dtype: object, fortran: bool, data: object
    ) -> npt.NDArray[Any]:
        """Build an array from validated data.

        Args:
            shape: Pickled shape.
            dtype: Pickled dtype.
            fortran: Whether the data is in Fortran order.
            data: Pickled array bytes, or a string of their Latin-1 code
                points.

        Returns:
            Writable array that uses a copy of the data, or the data itself
            if it is a buffer.

        Raises:
            ValueError: If the shape, dtype, or data is invalid, or the
                arrays would hold more data than the file.
        """
        if not (
            isinstance(shape, _Tuple)
            and all(type(size) is int and size >= 0 for size in shape.items)
        ):
            raise ValueError("an array shape is not a tuple of sizes")
        if not isinstance(dtype, _DTypeSpec):
            raise ValueError(f"an array dtype is {_describe(dtype)}")
        if not isinstance(data, str | bytes | bytearray):
            raise ValueError(f"array data is {_describe(data)}")
        array_dtype = dtype.dtype()
        sizes = cast(tuple[int, ...], shape.items)
        nbytes = math.prod(sizes) * array_dtype.itemsize
        if len(data) != nbytes:
            raise ValueError(
                f"array data has {len(data)} bytes, but shape {sizes} "
                f"and dtype {array_dtype} need {nbytes}"
            )
        # Data can be shared through the memo, so check before allocating.
        if nbytes > self._size - self._array_bytes:
            raise ValueError("arrays hold more data than the file")
        self._array_bytes += nbytes
        if isinstance(data, str):
            data = bytearray(data, "latin1")
        elif isinstance(data, bytes):
            data = bytearray(data)
        array: npt.NDArray[Any] = np.frombuffer(data, dtype=array_dtype)
        return array.reshape(sizes, order="F" if fortran else "C")

    def _finish(self, value: object) -> dict[RadioML2016Key, npt.NDArray[Any]]:
        """Convert the final pickle value.

        Args:
            value: Value before `STOP`.

        Returns:
            Dictionary of arrays.

        Raises:
            ValueError: If the value is not a dictionary of arrays, or the
                dictionary refers to more array data than the file holds.
        """
        if not isinstance(value, _Dict):
            raise ValueError(f"the pickle holds {_describe(value)}")
        result: dict[RadioML2016Key, npt.NDArray[Any]] = {}
        for key, spec in value.entries.items():
            if spec.array is None:
                raise ValueError("an array has no recorded state")
            result[key] = spec.array
        # A dictionary can refer to one array many times.
        array_bytes = sum(array.nbytes for array in result.values())
        if array_bytes > self._size:
            raise ValueError(
                f"arrays hold {array_bytes} bytes, more than the "
                f"{self._size}-byte file"
            )
        return result


def load_radioml2016_pickle(
    path: str | PathLike[str], *, encoding: str
) -> object:
    """Load a legacy RadioML 2016 pickle file.

    The loader does not use `pickle.load`. It interprets only the opcodes
    that pickles of NumPy array mappings use, in protocols 0 through 5,
    including Python 2 pickles. It never calls a function that the file
    names, and rejects any global other than those NumPy array pickles name.
    The result must be a dictionary of at most 65,536 entries whose keys are
    `(modulation, snr)` pairs and whose values are arrays. Each modulation
    label must be a string of at most 256 characters, and each SNR an integer
    in the `int16` range other than a Boolean. Each array must have a
    little- or big-endian integer or floating-point dtype of 1, 2, 4, or 8
    bytes. The loader builds each dtype itself from a validated type code
    and byte order, and places validated data in a new writable array. As
    with `pickle.load`, arrays have native byte order unless a protocol 5
    pickle records another.

    A pickle stores array bytes verbatim, so the arrays cannot hold more
    data than the file. The loader checks each array against this bound
    before allocating it, and rejects a dictionary that refers to the same
    array more than once if the total exceeds the file size. The labels of
    all keys have the same bound in characters, because a pickle can refer
    to one stored label many times. The loader also limits the number of
    opcodes, open marks, and tuple length and nesting. Text integers and memo
    indexes must lie in the ranges that the binary opcodes can encode.

    Args:
        path: Pickle file path.
        encoding: String encoding for Python 2 pickle compatibility.

    Returns:
        Dictionary of NumPy arrays.

    Raises:
        ValueError: If the encoding is unknown, or if the pickle is malformed,
            references an unsupported global, opcode, or value, exceeds a
            limit, or holds arrays with more data than the file.
    """
    path = Path(path)
    try:
        codecs.lookup(encoding)
    except LookupError as exc:
        raise ValueError(f"Unknown pickle encoding {encoding!r}") from exc
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        try:
            return _RestrictedUnpickler(handle, size, encoding).load()
        except ValueError as exc:
            raise ValueError(f"Invalid RadioML pickle {path}: {exc}") from exc


def as_radioml2016_dict(obj: object) -> RadioML2016Dict:
    """Validate a candidate RadioML 2016 pickle mapping.

    Args:
        obj: Candidate mapping.

    Returns:
        Validated RadioML 2016 mapping.

    Raises:
        TypeError: If the object does not have the expected mapping shape, a
            key is not a pair of a string and a non-Boolean integer, or an
            array is not real numeric data.
    """
    if not isinstance(obj, Mapping):
        raise TypeError(
            "Expected a mapping from (mod_class, snr_db) to IQ arrays."
        )
    if not obj:
        raise TypeError("Expected the RadioML mapping to be non-empty.")

    for key, value in obj.items():
        # A Boolean is an int, but not an SNR.
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], str)
            and isinstance(key[1], int)
            and not isinstance(key[1], bool)
        ):
            raise TypeError(
                "Expected keys of the form (mod_class: str, snr_db: int)."
            )
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 3
            or value.shape[1] != 2
        ):
            shape = getattr(value, "shape", None)
            raise TypeError(
                "Expected values to be 3D NumPy arrays with shape "
                f"(N, 2, T), got shape {shape}"
            )
        # Object arrays would hold arbitrary Python values, and complex or
        # structured values have no lossless float32 sample representation.
        # A dtype with a subarray or flags can hide object references
        # behind a numeric kind.
        dtype = value.dtype
        if (
            dtype.kind not in "iuf"
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.hasobject
            or dtype.flags
        ):
            raise TypeError(
                f"Expected real numeric IQ arrays, got dtype {value.dtype}"
            )
    return cast(RadioML2016Dict, obj)


def flatten_radioml2016_dataset(
    dataset: RadioML2016Dict,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int16],
    npt.NDArray[np.int16],
    list[str],
]:
    """Flatten a RadioML 2016 mapping into aligned arrays.

    Args:
        dataset: Mapping keyed by `(modulation_class, snr_db)`.

    Returns:
        IQ samples, modulation IDs, SNR labels, and modulation names.

    Raises:
        ValueError: If any batch has an incompatible shape.
    """
    # Pickle mappings preserve no portable class-ID convention. Sorting makes
    # both item order and the generated label table deterministic.
    mod_classes = sorted({mod_class for mod_class, _ in dataset})
    mod_class_to_id = {
        mod_class: index for index, mod_class in enumerate(mod_classes)
    }
    iq_batches: list[npt.NDArray[np.float32]] = []
    mod_labels: list[npt.NDArray[np.int16]] = []
    snr_labels: list[npt.NDArray[np.int16]] = []

    for mod_class, snr_value in sorted(dataset):
        batch = np.asarray(dataset[(mod_class, snr_value)], dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1] != 2:
            raise ValueError(
                "Expected shape (N, 2, T) for "
                f"({mod_class}, {snr_value}), got {batch.shape}"
            )
        count = int(batch.shape[0])
        iq_batches.append(batch)
        mod_labels.append(
            np.full(
                (count,),
                mod_class_to_id[mod_class],
                dtype=np.int16,
            )
        )
        snr_labels.append(np.full((count,), snr_value, dtype=np.int16))

    return (
        np.concatenate(iq_batches, axis=0),
        np.concatenate(mod_labels, axis=0),
        np.concatenate(snr_labels, axis=0),
        mod_classes,
    )


def _radioml2016_layout(
    dataset: RadioML2016Dict,
) -> tuple[tuple[int, ...], int, list[str]]:
    """Validate shapes and labels before creating destination storage.

    Args:
        dataset: Validated RadioML mapping.

    Returns:
        Per-item shape, total item count, and sorted modulation names.

    Raises:
        ValueError: If the mapping has more than 4,096 batches, a batch has
            no items, or sample shapes or labels cannot be represented.
    """
    # Each batch adds a provenance source that copies its modulation label,
    # and a small file can declare many one-item batches.
    if len(dataset) > _MAX_BATCHES:
        raise ValueError(
            f"RadioML mappings may hold at most {_MAX_BATCHES} batches, "
            f"got {len(dataset)}"
        )
    sample_shape = tuple(next(iter(dataset.values())).shape[1:])
    if any(size <= 0 for size in sample_shape):
        raise ValueError("RadioML sample dimensions must be positive")
    if any(batch.shape[1:] != sample_shape for batch in dataset.values()):
        raise ValueError("RadioML batches must have the same sample shape")
    # An empty batch costs no file data but adds a hashed provenance source,
    # so many empty batches would let a small file expand the import.
    for key, batch in dataset.items():
        if not batch.shape[0]:
            raise ValueError(
                f"RadioML batch {key!r} must contain at least one item"
            )
    item_count = sum(int(batch.shape[0]) for batch in dataset.values())
    mod_classes = sorted({mod_class for mod_class, _ in dataset})
    if len(mod_classes) > np.iinfo(np.int16).max + 1:
        raise ValueError("RadioML modulation IDs must fit in int16")
    if any(len(label) > _MAX_LABEL_LENGTH for label in mod_classes):
        raise ValueError(
            "RadioML modulation labels must have at most "
            f"{_MAX_LABEL_LENGTH} characters"
        )
    for _, snr_value in dataset:
        if snr_value not in _SNR_RANGE:
            raise ValueError("RadioML SNR labels must fit in int16")
    return sample_shape, item_count, mod_classes


@remove_new_store_on_failure
def import_radioml2016_dataset(
    store_path: str | PathLike[str],
    dataset: RadioML2016Dict | object,
    *,
    source_dataset: str | None = None,
    dataset_version: str = "2016",
    recording_name: str = "radioml2016",
    overwrite_store: bool = False,
    overwrite_recording: bool = False,
    batch_size: int = 4096,
    global_metadata: JSONObject | None = None,
    iq_chunks: tuple[int, int, int] | None = None,
    sample_shards: ShardsLike | None = None,
    sample_compressor: CompressorLike = "auto",
    zarr_format: ZarrFormat | None = None,
) -> SigMFZarrStore:
    """Import a pickle mapping in the RadioML 2016 format.

    Args:
        store_path: Target SigMF-Zarr store.
        dataset: RadioML 2016 mapping or candidate object.
        source_dataset: Optional source dataset name for metadata.
        dataset_version: Dataset generation recorded in source metadata.
            Defaults to "2016".
        recording_name: Recording name to create.
        overwrite_store: Whether to replace an existing SigMF-Zarr store at
            `store_path`. The target must be absent, an empty directory, or
            a SigMF-Zarr store.
        overwrite_recording: Whether to replace an existing recording.
        batch_size: Maximum number of sample items copied per write.
            Consecutive source batches share a write when they fit.
        global_metadata: Optional SigMF global metadata to merge.
        iq_chunks: Optional IQ sample chunks.
        sample_shards: Optional IQ sample shards.
        sample_compressor: Optional IQ sample compressor.
        zarr_format: Optional physical Zarr format requirement. Existing
            stores are auto-detected when omitted. New stores default to Zarr
            format 3.

    Returns:
        Updated SigMF-Zarr store.

    Raises:
        TypeError: If the object is not a supported pickle mapping.
        ValueError: If shapes, labels, or the batch size are invalid, or sample
            sharding is requested for Zarr format 2.
    """
    if zarr_format == 2 and sample_shards is not None:
        raise ValueError(
            "Sample sharding requires Zarr format 3. Omit sample_shards or "
            "create a Zarr format 3 store"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    dataset = as_radioml2016_dict(dataset)
    sample_shape, item_count, mod_classes = _radioml2016_layout(dataset)
    mod_class_to_id = {
        mod_class: index for index, mod_class in enumerate(mod_classes)
    }
    mod_class_id = np.empty(item_count, dtype=np.int16)
    snr_db = np.empty(item_count, dtype=np.int16)
    metadata = dict(global_metadata or {})
    if metadata.get("core:datatype", "cf32_le") != "cf32_le":
        raise ValueError("RadioML 2016 output requires core:datatype cf32_le")
    metadata["core:datatype"] = "cf32_le"
    metadata["radioml:source_dataset"] = source_dataset
    metadata["radioml:dataset_version"] = dataset_version

    sources: list[JSONObject] = []
    offset = 0
    for (modulation, snr), values in sorted(dataset.items()):
        sources.append(
            {
                "kind": "logical-array",
                "role": "samples-and-labels",
                "sha512": calculate_array_sha512(values),
                "modulation": modulation,
                "snr_db": snr,
                "items": int(values.shape[0]),
                "target_start": offset,
            }
        )
        offset += int(values.shape[0])
    metadata = import_metadata(
        metadata,
        sources=sources,
        operation="import_radioml2016_dataset",
        source_namespace="radioml",
        parameters={
            "order": "sorted-modulation-snr-then-source-row",
            "modulation_classes": list(mod_classes),
            "sample_conversion": "float32",
            "sample_axes": ["iq", "time"],
        },
    )

    store = SigMFZarrStore(
        store_path,
        overwrite=overwrite_store,
        zarr_format=zarr_format,
    )
    if recording_name in store.recordings and not overwrite_recording:
        raise ValueError(
            f"Recording {recording_name!r} already exists. Pass "
            "overwrite_recording=True to replace it"
        )
    with recording_import_transaction(
        store, recording_name, overwrite=overwrite_recording
    ):
        recording = store.recordings.create(
            recording_name,
            batched=True,
            sample_dtype=np.float32,
            sample_shape=sample_shape,
            sample_axes=("iq", "time"),
            global_metadata=metadata,
            sample_chunks=iq_chunks,
            sample_shards=sample_shards,
            sample_compressor=sample_compressor,
            overwrite=overwrite_recording,
        )
        # Walk the mapping in the same sorted order used to assign class IDs.
        # offset tracks items across source batches so both indexes stay
        # aligned without concatenating another full sample tensor. Each
        # append rewrites recording metadata, so small source batches are
        # coalesced into appends of batch_size items.
        offset = 0
        pending: list[npt.NDArray[np.float32]] = []
        pending_count = 0
        for mod_class, snr_value in sorted(dataset):
            batch = dataset[(mod_class, snr_value)]
            count = int(batch.shape[0])
            mod_class_id[offset : offset + count] = mod_class_to_id[mod_class]
            snr_db[offset : offset + count] = snr_value
            start = 0
            while start < count:
                stop = min(count, start + batch_size - pending_count)
                pending.append(np.asarray(batch[start:stop], dtype=np.float32))
                pending_count += stop - start
                start = stop
                if pending_count == batch_size:
                    recording.append_samples(np.concatenate(pending))
                    pending = []
                    pending_count = 0
            offset += count
        if pending:
            recording.append_samples(np.concatenate(pending))
        recording.add_index(
            "mod_class_id",
            mod_class_id,
            axis="item",
            field="radioml:mod_class",
            labels=list(mod_classes),
            overwrite=overwrite_recording,
        )
        recording.add_index(
            "snr_db",
            snr_db,
            axis="item",
            field="radioml:snr",
            unit="dB",
            overwrite=overwrite_recording,
        )
        recording.update_integrity()
        store._update_metadata_integrity_when_complete()
    return store


__all__ = [
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "as_radioml2016_dict",
    "flatten_radioml2016_dataset",
    "import_radioml2016_dataset",
    "load_radioml2016_pickle",
]
