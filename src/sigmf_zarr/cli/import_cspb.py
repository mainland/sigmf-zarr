"""Import Chad Spooner's CSPB ``.tim`` datasets into SigMF-Zarr."""

from __future__ import annotations

import argparse
from pathlib import Path

from sigmf_zarr.cli.command import (
    ImportCommand,
    add_sample_storage_arguments,
    positive_int,
    resolve_sample_compressor,
    resolve_sample_shards,
)
from sigmf_zarr.cspb import cspb_sample_shape, import_cspb_dataset


class ImportCSPBCommand(ImportCommand):
    """Import Spooner CSPB machine-learning datasets."""

    name = "cspb"
    help = "Import a CSPB dataset"

    def __init__(self) -> None:
        """Initialize the CSPB import command."""
        super().__init__(
            "Import Chad Spooner's CSPB .tim datasets into SigMF-Zarr.",
            source_help=(
                "Source .tim file, ZIP batch, or directory containing "
                "files and batches."
            ),
            default_recording_name="cspb",
        )

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add CSPB-specific import arguments.

        Args:
            parser: Argument parser to extend.
        """
        add_sample_storage_arguments(parser)
        parser.add_argument(
            "--truth-file",
            type=Path,
            action="append",
            default=[],
            help=(
                "Optional CSPB truth text file. Repeat for multiple truth "
                "files."
            ),
        )
        parser.add_argument(
            "--source-dataset",
            help="Published dataset name recorded in metadata.",
        )
        parser.add_argument(
            "--batch-size",
            type=positive_int,
            default=64,
            help="Number of .tim files copied per Zarr write.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Import a CSPB dataset.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        sample_shards = None
        if args.sample_shard_batch is not None:
            sample_shards = resolve_sample_shards(
                args,
                cspb_sample_shape(
                    args.source,
                ),
            )
        store = import_cspb_dataset(
            args.store,
            args.source,
            truth_paths=args.truth_file,
            source_dataset=args.source_dataset or args.source.stem,
            recording_name=args.recording_name,
            overwrite_store=args.overwrite_store,
            overwrite_recording=args.overwrite_recording,
            batch_size=args.batch_size,
            sample_shards=sample_shards,
            sample_compressor=resolve_sample_compressor(args),
            zarr_format=args.zarr_format,
        )
        print(store.info())
        return 0


__all__ = ["ImportCSPBCommand"]
