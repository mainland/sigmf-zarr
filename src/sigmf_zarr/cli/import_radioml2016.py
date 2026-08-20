"""Command-line interface for RadioML 2016 imports."""

from __future__ import annotations

import argparse

from sigmf_zarr.cli.command import (
    ImportCommand,
    add_sample_storage_arguments,
    positive_int,
    resolve_sample_compressor,
    resolve_sample_shards,
)
from sigmf_zarr.radioml2016 import (
    RadioML2016Dict,
    RadioML2016Key,
    RadioML2016Value,
    as_radioml2016_dict,
    flatten_radioml2016_dataset,
    import_radioml2016_dataset,
    load_radioml2016_pickle,
)


class ImportRadioML2016Command(ImportCommand):
    """Import a RadioML 2016 pickle dataset."""

    name = "radioml2016"
    help = "Import a RadioML 2016 pickle dataset"

    def __init__(self) -> None:
        """Initialize the RadioML 2016 import command."""
        super().__init__(
            "Import a RadioML 2016 pickle mapping into SigMF-Zarr.",
            source_help="Trusted RadioML 2016 pickle file (may execute code).",
            default_recording_name="radioml2016",
        )

    def add_import_arguments(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """Add RadioML 2016 import arguments.

        Args:
            parser: Argument parser to extend.
        """
        add_sample_storage_arguments(parser)
        parser.add_argument(
            "--batch-size",
            type=positive_int,
            default=4096,
            help="Maximum sample items copied per write.",
        )
        parser.add_argument(
            "--encoding",
            default="latin1",
            help="String encoding for legacy Python 2 pickles.",
        )
        parser.add_argument(
            "--source-dataset",
            help="Source dataset name recorded in metadata.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Import a RadioML 2016 dataset.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        dataset = as_radioml2016_dict(
            load_radioml2016_pickle(args.source, encoding=args.encoding)
        )
        sample_shape = tuple(
            int(dim) for dim in next(iter(dataset.values())).shape[1:]
        )
        sample_shards = resolve_sample_shards(
            args,
            sample_shape,
        )
        store = import_radioml2016_dataset(
            args.store,
            dataset,
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


__all__ = [
    "ImportRadioML2016Command",
    "RadioML2016Dict",
    "RadioML2016Key",
    "RadioML2016Value",
    "as_radioml2016_dict",
    "flatten_radioml2016_dataset",
    "import_radioml2016_dataset",
    "load_radioml2016_pickle",
]
