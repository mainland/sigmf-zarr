"""Command-line interface for Panoradio HF dataset imports."""

from __future__ import annotations

import argparse
from pathlib import Path

from sigmf_zarr.cli.command import positive_int
from sigmf_zarr.cli.import_command import (
    ImportCommand,
    add_sample_sharding_arguments,
    resolve_sample_compressor,
    resolve_sample_shards,
)
from sigmf_zarr.panoradio import (
    _load_panoradio_samples,
    import_panoradio_dataset,
)


class ImportPanoradioCommand(ImportCommand):
    """Import a Panoradio HF NumPy dataset and its CSV tags."""

    name = "panoradio"
    help = "Import Panoradio HF NumPy samples and CSV tags"

    def __init__(self) -> None:
        """Initialize the Panoradio import command."""
        super().__init__(
            "Stream a Panoradio HF dataset into SigMF-Zarr.",
            source_help="Source Panoradio complex NumPy .npy file.",
            default_recording_name="panoradio",
        )

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add Panoradio import arguments.

        Args:
            parser: Argument parser to extend.
        """
        add_sample_sharding_arguments(parser)
        parser.add_argument(
            "--tags-file",
            type=Path,
            required=True,
            help="CSV tags file aligned with the source sample rows.",
        )
        parser.add_argument(
            "--source-dataset",
            help="Source dataset name recorded in metadata.",
        )
        parser.add_argument(
            "--batch-size",
            type=positive_int,
            default=4096,
            help="Maximum sample items converted and copied per write.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Import a Panoradio HF dataset.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        samples = _load_panoradio_samples(args.source)
        sample_shape = (2, int(samples.shape[1]))
        del samples
        sample_shards = resolve_sample_shards(
            args.sample_shard_batch,
            sample_shape,
            zarr_format=args.zarr_format,
        )
        store = import_panoradio_dataset(
            args.store,
            args.source,
            args.tags_file,
            source_dataset=args.source_dataset,
            recording_name=args.recording_name,
            overwrite_store=args.overwrite_store,
            overwrite_recording=args.overwrite_recording,
            batch_size=args.batch_size,
            sample_shards=sample_shards,
            automatic_sharding=not args.no_sample_sharding,
            sample_compressor=resolve_sample_compressor(
                args.sample_compression,
                level=args.sample_compression_level,
            ),
            zarr_format=args.zarr_format,
        )
        print(store.info())
        return 0


__all__ = [
    "ImportPanoradioCommand",
    "import_panoradio_dataset",
]
