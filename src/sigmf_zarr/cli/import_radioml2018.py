"""Command-line interface for RadioML 2018 imports."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from sigmf_zarr.cli.command import (
    ImportCommand,
    add_sample_storage_arguments,
    positive_int,
    resolve_sample_compressor,
    resolve_sample_shards,
)
from sigmf_zarr.radioml2018 import (
    RADIOML2018_MODULATION_CLASSES,
    _hdf5_dataset,
    _sample_shape,
    import_radioml2018_dataset,
    load_modulation_classes,
)


class ImportRadioML2018Command(ImportCommand):
    """Import a RadioML 2018 HDF5 dataset."""

    name = "radioml2018"
    help = "Import a RadioML 2018 HDF5 dataset"

    def __init__(self) -> None:
        """Initialize the RadioML 2018 import command."""
        super().__init__(
            "Stream a RadioML 2018 HDF5 dataset into SigMF-Zarr.",
            source_help="Source RadioML 2018 HDF5 file.",
            default_recording_name="radioml2018",
        )

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add RadioML 2018 import arguments.

        Args:
            parser: Argument parser to extend.
        """
        add_sample_storage_arguments(parser)
        parser.add_argument(
            "--source-dataset",
            help="Source dataset name recorded in metadata.",
        )
        parser.add_argument(
            "--batch-size",
            type=positive_int,
            default=4096,
            help="Number of sample items copied per HDF5 read.",
        )
        parser.add_argument(
            "--samples-dataset",
            default="X",
            help="HDF5 path containing sample tensors.",
        )
        parser.add_argument(
            "--labels-dataset",
            default="Y",
            help="HDF5 path containing one-hot modulation labels.",
        )
        parser.add_argument(
            "--snr-dataset",
            default="Z",
            help="HDF5 path containing SNR labels.",
        )
        parser.add_argument(
            "--classes-file",
            type=Path,
            help=(
                "JSON, `classes = [...]`, or line-delimited modulation-class "
                "order."
            ),
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Import a RadioML 2018 dataset.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        with h5py.File(args.source, "r") as source:
            samples = _hdf5_dataset(
                source,
                args.samples_dataset,
                label="sample",
            )
            sample_shape, _ = _sample_shape(samples)
        sample_shards = resolve_sample_shards(args, sample_shape)
        store = import_radioml2018_dataset(
            args.store,
            args.source,
            source_dataset=args.source_dataset or args.source.stem,
            recording_name=args.recording_name,
            overwrite_store=args.overwrite_store,
            overwrite_recording=args.overwrite_recording,
            modulation_classes=load_modulation_classes(args.classes_file),
            samples_dataset=args.samples_dataset,
            labels_dataset=args.labels_dataset,
            snr_dataset=args.snr_dataset,
            batch_size=args.batch_size,
            sample_shards=sample_shards,
            automatic_sharding=not args.no_sample_sharding,
            sample_compressor=resolve_sample_compressor(args),
            zarr_format=args.zarr_format,
        )
        print(store.info())
        return 0


__all__ = [
    "ImportRadioML2018Command",
    "RADIOML2018_MODULATION_CLASSES",
    "import_radioml2018_dataset",
    "load_modulation_classes",
]
