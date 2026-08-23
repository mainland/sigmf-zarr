"""Import and export SigMF files and archives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast, get_args

from sigmf_zarr import SigMFZarrStore
from sigmf_zarr.cli import Command, CommandGroup, ImportCommand, StoreContext
from sigmf_zarr.cli.import_command import resolve_sample_compressor
from sigmf_zarr.cli.import_cspb import ImportCSPBCommand
from sigmf_zarr.cli.import_radioml2016 import ImportRadioML2016Command
from sigmf_zarr.cli.import_radioml2018 import ImportRadioML2018Command
from sigmf_zarr.cli.store import StoreCommand
from sigmf_zarr.sigmf import (
    export_sigmf,
    export_sigmf_archive,
    import_sigmf,
    import_sigmf_archive,
)

ImportFormat = Literal["auto", "sigmf", "sigmf-archive"]
"""Supported import format selectors."""

IMPORT_FORMAT_CHOICES = cast(tuple[ImportFormat, ...], get_args(ImportFormat))
"""Command-line choices for import format selection."""


def detect_import_format(
    source: Path,
    requested_format: ImportFormat,
    *,
    archive: bool,
) -> ImportFormat:
    """Resolve the import format from flags and the input path.

    Args:
        source: Source input path.
        requested_format: User-requested import format.
        archive: Whether the archive flag was provided.

    Returns:
        Resolved import format.

    Raises:
        ValueError: If the format cannot be inferred from the path.
    """
    if requested_format != "auto":
        return "sigmf-archive" if archive else requested_format
    if archive or source.suffix == ".sigmf":
        return "sigmf-archive"
    if source.suffix == ".sigmf-meta":
        return "sigmf"
    raise ValueError(
        "Could not infer import format from source path. "
        "Pass --format explicitly for SigMF input. See "
        "`sigmf-zarr import --help` for other importers."
    )


class ImportSigMFCommand(ImportCommand):
    """Import standard SigMF data into a SigMF-Zarr store."""

    name = "sigmf"
    help = "Import standard SigMF data into a SigMF-Zarr store"

    def __init__(self) -> None:
        """Initialize the import command."""
        super().__init__(
            "Import a standard SigMF recording or archive into a "
            "SigMF-Zarr store.",
            source_help="Input SigMF metadata file or archive.",
        )

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add import command arguments.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument(
            "--archive",
            action="store_true",
            help="Treat the input as a standard `.sigmf` archive.",
        )
        parser.add_argument(
            "--format",
            choices=IMPORT_FORMAT_CHOICES,
            default="auto",
            help=(
                "Input format. Defaults to auto-detection from the path "
                "suffix."
            ),
        )
        parser.add_argument(
            "--no-sample-sharding",
            action="store_true",
            help=(
                "Disable automatic sample sharding. Larger logical chunks "
                "are used instead."
            ),
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Run the import command.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        import_format = detect_import_format(
            args.source,
            args.format,
            archive=args.archive,
        )

        if import_format == "sigmf-archive":
            store = import_sigmf_archive(
                args.store,
                args.source,
                overwrite_store=args.overwrite_store,
                overwrite_recordings=args.overwrite_recording,
                automatic_sharding=not args.no_sample_sharding,
                sample_compressor=resolve_sample_compressor(
                    args.sample_compression,
                    level=args.sample_compression_level,
                ),
                zarr_format=args.zarr_format,
            )
            print(store.info())
            return 0

        if import_format == "sigmf":
            recording = import_sigmf(
                args.store,
                args.source,
                recording_name=args.recording_name,
                overwrite_store=args.overwrite_store,
                overwrite_recording=args.overwrite_recording,
                automatic_sharding=not args.no_sample_sharding,
                sample_compressor=resolve_sample_compressor(
                    args.sample_compression,
                    level=args.sample_compression_level,
                ),
                zarr_format=args.zarr_format,
            )
            print(recording.info())
            return 0

        raise ValueError(f"Unsupported SigMF import format: {import_format}")


class ExportCommand(Command):
    """Export SigMF-Zarr recordings to standard SigMF files."""

    name = "export"
    help = "Export SigMF-Zarr recordings to standard SigMF files"

    def __init__(self) -> None:
        """Initialize the export command."""
        super().__init__(
            "Export one recording or a selected set of recordings from a "
            "SigMF-Zarr store."
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add export command arguments.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument(
            "store",
            help="Input SigMF-Zarr store path or URL.",
        )
        parser.add_argument(
            "output",
            type=Path,
            help="Output SigMF path or archive.",
        )
        parser.add_argument(
            "--archive",
            action="store_true",
            help=(
                "Export a standard `.sigmf` archive instead of one "
                "recording."
            ),
        )
        parser.add_argument(
            "--recording-name",
            help="Recording name to export for single-recording exports.",
        )
        parser.add_argument(
            "--recording",
            dest="recording_names",
            action="append",
            default=None,
            help="Recording name to include in an archive export. Repeatable.",
        )
        parser.add_argument(
            "--collection-name",
            help="Optional collection name to include in an archive export.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace existing target files.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Export even when sample-indexed metadata appears stale.",
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Write compact JSON instead of pretty-printed metadata.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Run the export command.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.

        Raises:
            SystemExit: If a single-recording export omits
                `--recording-name`.
        """
        store = SigMFZarrStore.open(args.store)
        pretty = not args.compact

        if args.archive:
            archive_path = export_sigmf_archive(
                store,
                args.output,
                recording_names=args.recording_names,
                collection_name=args.collection_name,
                overwrite=args.overwrite,
                pretty=pretty,
                force=args.force,
            )
            print(f"exported archive: {archive_path}")
            return 0

        if args.recording_name is None:
            raise SystemExit(
                "--recording-name is required unless --archive is used"
            )

        meta_path = export_sigmf(
            store,
            args.recording_name,
            args.output,
            overwrite=args.overwrite,
            pretty=pretty,
            force=args.force,
        )
        print(f"exported recording: {meta_path}")
        return 0


class ImportGroup(CommandGroup):
    """Group the supported dataset import commands."""

    name = "import"
    help = "Import a supported dataset into a SigMF-Zarr store"

    def __init__(self) -> None:
        """Initialize the import command group."""
        super().__init__(
            "Import a supported dataset into a SigMF-Zarr store.",
            (
                ImportSigMFCommand(),
                ImportRadioML2016Command(),
                ImportRadioML2018Command(),
                ImportCSPBCommand(),
            ),
            destination="import_name",
        )


class SigMFCommand(CommandGroup):
    """Top-level SigMF-Zarr CLI command."""

    def __init__(self) -> None:
        """Initialize the top-level SigMF command."""
        context = StoreContext()
        super().__init__(
            "Inspect SigMF-Zarr stores and import or export supported data.",
            (
                StoreCommand(context),
                ImportGroup(),
                ExportCommand(),
            ),
            destination="command_name",
        )


def main() -> int:
    """Run the SigMF CLI.

    Returns:
        Process exit status.
    """
    return int(SigMFCommand().run() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
