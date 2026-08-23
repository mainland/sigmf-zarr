"""Shared command-line behavior for SigMF-Zarr importers."""

from __future__ import annotations

import argparse
import codecs
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, TypeGuard, get_args

from zarr.codecs import BloscCodec
from zarr.core.array import CompressorLike, ShardsLike

from sigmf_zarr.cli.command import Command, positive_int
from sigmf_zarr.store import ZarrFormat

BloscSampleCompression = Literal["zstd", "lz4", "lz4hc"]
"""Blosc algorithms exposed by the import commands."""

DEFAULT_COMPRESSION_LEVEL = 3
"""Blosc compression level used when none is given."""


def is_blosc_sample_compression(
    value: str,
) -> TypeGuard[BloscSampleCompression]:
    """Return whether a value names an exposed Blosc algorithm.

    Args:
        value: Compression name to test.

    Returns:
        Whether `value` is a supported Blosc sample compression name.
    """
    # Derive the runtime set from the Literal so command choices and static
    # narrowing cannot acquire separate, inconsistent codec lists.
    return value in get_args(BloscSampleCompression)


def compression_level(value: str) -> int:
    """Parse a Blosc compression level for ``argparse``.

    Args:
        value: Command-line value.

    Returns:
        Parsed compression level.

    Raises:
        argparse.ArgumentTypeError: If the value is not an integer from zero
            through nine.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= parsed <= 9:
        raise argparse.ArgumentTypeError("must be between 0 and 9")
    return parsed


def text_encoding(value: str) -> str:
    """Parse a text encoding name for ``argparse``.

    Args:
        value: Command-line value.

    Returns:
        Encoding name as given.

    Raises:
        argparse.ArgumentTypeError: If Python has no codec with this name.
    """
    try:
        codecs.lookup(value)
    except LookupError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown encoding: {value!r}"
        ) from exc
    return value


def add_sample_compression_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add sample compression options shared by all import commands.

    Args:
        parser: Argument parser to extend.
    """
    parser.add_argument(
        "--sample-compression",
        choices=("auto", "none", *get_args(BloscSampleCompression)),
        default="auto",
        help=(
            "Sample compression codec. Auto uses native Zstandard. Other "
            "named codecs use Blosc."
        ),
    )
    parser.add_argument(
        "--sample-compression-level",
        type=compression_level,
        default=None,
        help=(
            "Blosc compression level from 0 through 9. Defaults to 3. "
            "Requires a Blosc --sample-compression codec."
        ),
    )


def validate_sample_compression_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Reject a compression level for a codec that does not use one.

    Args:
        parser: Configured argument parser.
        args: Parsed command-line arguments.
    """
    if args.sample_compression_level is not None and (
        not is_blosc_sample_compression(args.sample_compression)
    ):
        parser.error(
            "--sample-compression-level requires --sample-compression "
            f"{'|'.join(get_args(BloscSampleCompression))}"
        )


def add_sample_sharding_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add sharding options shared by batched dataset import commands.

    Args:
        parser: Argument parser to extend.
    """
    sharding = parser.add_mutually_exclusive_group()
    sharding.add_argument(
        "--sample-shard-batch",
        type=positive_int,
        help=(
            "Items per physical sample shard, overriding automatic sizing "
            "(Zarr format 3 only)."
        ),
    )
    sharding.add_argument(
        "--no-sample-sharding",
        action="store_true",
        help=(
            "Disable automatic sample sharding. Larger logical chunks are "
            "used instead."
        ),
    )


def resolve_sample_compressor(
    compression: str,
    *,
    level: int | None,
) -> CompressorLike | None:
    """Resolve a command-line sample compression selection.

    Args:
        compression: Selected sample compression name.
        level: Blosc compression level, or `None` for the default level 3.

    Returns:
        Zarr compressor configuration.

    Raises:
        ValueError: If `compression` is unsupported.
    """
    if compression == "auto":
        # Leave automatic selection to the store, which chooses native
        # Zstandard codecs appropriate to the physical Zarr format.
        return "auto"
    if compression == "none":
        return None
    if is_blosc_sample_compression(compression):
        # Every explicit named algorithm uses Blosc so one level scale and
        # one cross-format translation path cover all command-line choices.
        return BloscCodec(
            cname=compression,
            clevel=DEFAULT_COMPRESSION_LEVEL if level is None else level,
        )
    raise ValueError(f"Unsupported sample compression codec: {compression!r}")


def resolve_sample_shards(
    shard_batch: int | None,
    sample_shape: tuple[int, ...],
    *,
    zarr_format: ZarrFormat | None,
) -> ShardsLike | None:
    """Resolve an explicit sample shard batch size.

    Args:
        shard_batch: Number of items per sample shard, if specified.
        sample_shape: Shape of one imported sample.
        zarr_format: Requested target Zarr format, if specified.

    Returns:
        Full shard shape, or `None` when sharding is not requested.

    Raises:
        ValueError: If sharding is requested for Zarr format 2.
    """
    if shard_batch is None:
        return None
    if zarr_format == 2:
        raise ValueError("--sample-shard-batch requires Zarr format 3")
    return (shard_batch, *sample_shape)


class ImportCommand(Command):
    """Base class for commands that import into a SigMF-Zarr store."""

    parameter_options: ClassVar[Mapping[str, str]] = {
        "overwrite_store": "--overwrite-store",
        "overwrite_recording": "--overwrite-recording",
        "overwrite_collection": "--overwrite-collection",
        "recording_name": "--recording-name",
        "zarr_format": "--zarr-format",
        "batch_size": "--batch-size",
    }
    """Importer parameters mapped to their command-line options."""

    source_help: str
    """Help text for the positional source argument."""

    default_recording_name: str | None
    """Default target recording name, or `None` for source-derived names."""

    def __init__(
        self,
        description: str,
        *,
        source_help: str,
        default_recording_name: str | None = None,
    ) -> None:
        """Initialize a shared import command.

        Args:
            description: Command-line description.
            source_help: Help text for the positional source argument.
            default_recording_name: Optional default recording name.
        """
        super().__init__(description)
        self.source_help = source_help
        self.default_recording_name = default_recording_name

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments common to every importer.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument("source", type=Path, help=self.source_help)
        parser.add_argument(
            "store",
            help="Target SigMF-Zarr store path or URL.",
        )
        parser.add_argument(
            "--zarr-format",
            type=int,
            choices=(2, 3),
            default=None,
            help=(
                "Physical format for a new target store. Existing stores "
                "are auto-detected. New stores default to Zarr format 3."
            ),
        )
        parser.add_argument(
            "--recording-name",
            default=self.default_recording_name,
            help="Target recording name.",
        )
        parser.add_argument(
            "--overwrite-store",
            action="store_true",
            help=(
                "Replace an existing SigMF-Zarr store at the target. The "
                "target must be absent, an empty directory, or a SigMF-Zarr "
                "store, and must not contain an input file."
            ),
        )
        parser.add_argument(
            "--overwrite-recording",
            action="store_true",
            help="Replace an existing recording with the same name.",
        )
        add_sample_compression_arguments(parser)
        self.add_import_arguments(parser)

    def validate_arguments(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
    ) -> None:
        """Reject option combinations that the import would ignore.

        Args:
            parser: Configured argument parser.
            args: Parsed command-line arguments.
        """
        validate_sample_compression_arguments(parser, args)

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add importer-specific arguments.

        Args:
            parser: Argument parser to extend.
        """
        return None


__all__ = [
    "DEFAULT_COMPRESSION_LEVEL",
    "BloscSampleCompression",
    "ImportCommand",
    "add_sample_compression_arguments",
    "add_sample_sharding_arguments",
    "compression_level",
    "is_blosc_sample_compression",
    "resolve_sample_compressor",
    "resolve_sample_shards",
    "text_encoding",
    "validate_sample_compression_arguments",
]
