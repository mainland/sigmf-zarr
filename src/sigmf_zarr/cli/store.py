"""Resource-oriented commands for inspecting SigMF-Zarr stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sigmf_zarr.cli.context import StoreContext
from sigmf_zarr.cli.group import CommandGroup
from sigmf_zarr.cli.output import OutputCommand
from sigmf_zarr.export_plan import plan_sigmf_export
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.rfml import validate_rfml
from sigmf_zarr.sigmf import export_sigmf, export_sigmf_archive
from sigmf_zarr.store import SigMFCollection, SigMFRecording, SigMFZarrStore
from sigmf_zarr.validation import validate_store


def store_result(store: SigMFZarrStore) -> dict[str, Any]:
    """Return a JSON-compatible store summary.

    Args:
        store: Store to summarize.

    Returns:
        Stable store summary fields.
    """
    return {
        "schema_name": store.SCHEMA_NAME,
        "schema_version": store.SCHEMA_VERSION,
        "zarr_format": store.zarr_format,
        "metadata_sha512": store.metadata_sha512,
        "recordings": list(store.list_recordings()),
        "collections": list(store.list_collections()),
        "indexes": sorted(
            name
            for name, member in store.indexes.members(max_depth=None)
            if isinstance(member, ReadOnlyArray)
        ),
    }


def recording_result(recording: SigMFRecording) -> dict[str, Any]:
    """Return a JSON-compatible recording summary.

    Args:
        recording: Recording to summarize.

    Returns:
        Stable recording summary fields.
    """
    return {
        "name": recording.name,
        "zarr_format": recording.zarr_format,
        "num_samples": len(recording),
        "batched": recording.batched,
        "samples_shape": [int(dim) for dim in recording.samples.shape],
        "samples_chunks": [int(dim) for dim in recording.samples.chunks],
        "sample_axes": list(recording.sample_axes),
        "sample_dtype": str(recording.sample_dtype),
        "sample_checksum": recording.sample_checksum,
        "sha512": recording.sha512,
        "sample_sha512": recording.sample_sha512,
        "metadata_sha512": recording.metadata_sha512,
        "num_channels": recording.num_channels,
        "has_item_metadata": recording.has_item_metadata,
        "global_keys": sorted(recording.global_metadata),
        "num_captures": len(recording.captures),
        "num_annotations": len(recording.annotations),
        "extensions": sorted(recording.extensions.group_keys()),
        "indexes": sorted(
            name
            for name, member in recording.indexes.members(max_depth=None)
            if isinstance(member, ReadOnlyArray)
        ),
    }


def collection_result(collection: SigMFCollection) -> dict[str, Any]:
    """Return a JSON-compatible collection summary.

    Args:
        collection: Collection to summarize.

    Returns:
        Stable collection summary fields.
    """
    return {
        "name": collection.name,
        "recording_ids": list(collection.recording_ids),
        "metadata_sha512": collection.metadata_sha512,
        "metadata_keys": sorted(collection.metadata),
    }


class StoreOutputCommand(OutputCommand):
    """Output command sharing a lazily opened store context."""

    context: StoreContext
    """Context used to open the selected store."""

    def __init__(self, context: StoreContext, description: str) -> None:
        """Initialize a store output command.

        Args:
            context: Shared store context.
            description: Command-line description.
        """
        super().__init__(description)
        self.context = context


class StoreInfoCommand(StoreOutputCommand):
    """Display a SigMF-Zarr store summary."""

    name = "info"
    help = "Show store information"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the store information command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Show information about a SigMF-Zarr store.")

    def handle(self, args: argparse.Namespace) -> int:
        """Display store information.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        store = self.context.open(args.store)
        self.write_output(args, text=store.info(), value=store_result(store))
        return 0


class StoreValidateCommand(StoreOutputCommand):
    """Validate a complete SigMF-Zarr store."""

    name = "validate"
    help = "Validate structure, metadata, indexes, and integrity"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the validation command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Validate a complete SigMF-Zarr store.")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add validation policy arguments.

        Args:
            parser: Argument parser to extend.
        """
        super().add_arguments(parser)
        parser.add_argument(
            "--skip-integrity",
            action="store_true",
            help="Validate structure without checking declared hashes.",
        )
        parser.add_argument(
            "--require-integrity",
            action="store_true",
            help="Require hashes for the store and every resource.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Validate the selected store.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Zero for a valid store and one when issues are found.
        """
        report = validate_store(
            self.context.open(args.store),
            verify_integrity=not args.skip_integrity,
            require_integrity=args.require_integrity,
        )
        self.write_output(args, text=report.format(), value=report.as_dict())
        return 0 if report.valid else 1


class StoreIntegrityVerifyCommand(StoreOutputCommand):
    """Verify all internal integrity hashes."""

    name = "verify"
    help = "Verify all internal integrity hashes"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the integrity verification command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Verify all internal integrity hashes.")

    def handle(self, args: argparse.Namespace) -> int:
        """Verify all hashes in the selected store.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Zero when every required hash matches and one otherwise.
        """
        report = validate_store(
            self.context.open(args.store),
            require_integrity=True,
        )
        self.write_output(args, text=report.format(), value=report.as_dict())
        return 0 if report.valid else 1


class StoreIntegrityUpdateCommand(StoreOutputCommand):
    """Recalculate all internal integrity hashes."""

    name = "update"
    help = "Recalculate all internal integrity hashes"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the integrity update command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Recalculate all internal integrity hashes.")

    def handle(self, args: argparse.Namespace) -> int:
        """Update hashes in the selected store.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        store = SigMFZarrStore.open(args.store, mode="r+")
        store.update_integrity()
        report = validate_store(store, require_integrity=True)
        self.write_output(args, text=report.format(), value=report.as_dict())
        return 0 if report.valid else 1


class StoreIntegrityCommand(CommandGroup):
    """Group store integrity operations."""

    name = "integrity"
    help = "Verify or update internal integrity hashes"

    def __init__(self, context: StoreContext) -> None:
        """Initialize integrity commands.

        Args:
            context: Shared store context.
        """
        super().__init__(
            "Verify or update internal integrity hashes.",
            (
                StoreIntegrityVerifyCommand(context),
                StoreIntegrityUpdateCommand(context),
            ),
            destination="integrity_action",
        )


class RecordingsCommand(StoreOutputCommand):
    """List recordings in a SigMF-Zarr store."""

    name = "recordings"
    help = "List recordings"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the recordings command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "List recordings in a SigMF-Zarr store.")

    def handle(self, args: argparse.Namespace) -> int:
        """List recordings.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        names = self.context.open(args.store).list_recordings()
        self.write_output(
            args,
            text="\n".join(names),
            value={"recordings": list(names)},
        )
        return 0


class RecordingInfoCommand(StoreOutputCommand):
    """Display one recording summary."""

    name = "info"
    help = "Show recording information"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the recording information command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Show information about one recording.")

    def handle(self, args: argparse.Namespace) -> int:
        """Display recording information.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        recording = self.context.open(args.store).recordings[args.name]
        self.write_output(
            args,
            text=recording.info(),
            value=recording_result(recording),
        )
        return 0


class RecordingExportCommand(StoreOutputCommand):
    """Export one recording as standard SigMF."""

    name = "export"
    help = "Export the recording"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the recording export command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Export one recording as standard SigMF.")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add export arguments.

        Args:
            parser: Argument parser to extend.
        """
        super().add_arguments(parser)
        parser.add_argument("output", type=Path, help="Output SigMF path.")
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
            "--allow-lossy",
            action="store_true",
            help=(
                "Warn and omit indexes, arrays, and channel metadata."
            ),
        )
        parser.add_argument(
            "--item-index",
            type=int,
            help="Export one nonnegative item position from a batch.",
        )
        parser.add_argument(
            "--project-index", action="append", default=[],
            help="Preserve this item index and its descriptors. Repeatable.",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Print a verified JSON export plan without writing files.",
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Write compact JSON metadata.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Export one recording.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        if args.dry_run:
            plan = plan_sigmf_export(
                self.context.open(args.store), args.name,
                item_index=args.item_index, project_indexes=args.project_index,
                force=args.force,
            )
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
            return int(bool(
                plan.rejected or plan.omitted and not args.allow_lossy
            ))
        options = (
            {"project_indexes": args.project_index}
            if args.project_index else {}
        )
        path = export_sigmf(
            self.context.open(args.store),
            args.name,
            args.output,
            overwrite=args.overwrite,
            pretty=not args.compact,
            force=args.force,
            allow_lossy=args.allow_lossy,
            item_index=args.item_index,
            **options,
        )
        self.write_output(
            args,
            text=f"exported recording: {path}",
            value={"recording": args.name, "output": str(path)},
        )
        return 0


class RecordingRFMLValidationCommand(StoreOutputCommand):
    """Validate the dense RFML profile independently of integrity and
    splits.
    """

    name = "validate-rfml"
    help = "Validate declared RFML metadata fields"

    def __init__(self, context: StoreContext) -> None:
        """Initialize profile validation.

        Args:
            context: Shared read-only store context.
        """
        super().__init__(context, "Validate the dense RFML metadata profile.")

    def handle(self, args: argparse.Namespace) -> int:
        """Check descriptors and values without reading samples.

        Args:
            args: Parsed recording selection and output format.

        Returns:
            Zero on conformance, or one when requirements are violated.
        """
        recording = self.context.open(args.store).recordings.open(
            args.name, create=False, validation="structural",
        )
        report = validate_rfml(recording)
        value = report.as_dict()
        self.write_output(args, text=json.dumps(value, indent=2), value=value)
        return int(not report.valid)


class RecordingCommand(CommandGroup):
    """Group commands that operate on one recording."""

    name = "recording"
    help = "Inspect or export one recording"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the recording command group.

        Args:
            context: Shared store context.
        """
        super().__init__(
            "Inspect or export one recording.",
            (
                RecordingInfoCommand(context), RecordingExportCommand(context),
                RecordingRFMLValidationCommand(context),
            ),
            destination="recording_action",
        )

    def add_group_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the recording name.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument("name", metavar="NAME", help="Recording name.")


class CollectionsCommand(StoreOutputCommand):
    """List collections in a SigMF-Zarr store."""

    name = "collections"
    help = "List collections"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the collections command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "List collections in a SigMF-Zarr store.")

    def handle(self, args: argparse.Namespace) -> int:
        """List collections.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        names = self.context.open(args.store).list_collections()
        self.write_output(
            args,
            text="\n".join(names),
            value={"collections": list(names)},
        )
        return 0


class CollectionInfoCommand(StoreOutputCommand):
    """Display one collection summary."""

    name = "info"
    help = "Show collection information"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the collection information command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Show information about one collection.")

    def handle(self, args: argparse.Namespace) -> int:
        """Display collection information.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        collection = self.context.open(args.store).collections[args.name]
        self.write_output(
            args,
            text=collection.info(),
            value=collection_result(collection),
        )
        return 0


class CollectionExportCommand(StoreOutputCommand):
    """Export one collection as a standard SigMF archive."""

    name = "export"
    help = "Export the collection"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the collection export command.

        Args:
            context: Shared store context.
        """
        super().__init__(context, "Export one collection as a SigMF archive.")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add archive export arguments.

        Args:
            parser: Argument parser to extend.
        """
        super().add_arguments(parser)
        parser.add_argument("output", type=Path, help="Output archive path.")
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace an existing archive.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Export even when sample-indexed metadata appears stale.",
        )
        parser.add_argument(
            "--allow-lossy",
            action="store_true",
            help=(
                "Warn and omit indexes, arrays, and channel metadata."
            ),
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Write compact JSON metadata.",
        )

    def handle(self, args: argparse.Namespace) -> int:
        """Export one collection.

        Args:
            args: Parsed command-line arguments.

        Returns:
            Process exit status.
        """
        path = export_sigmf_archive(
            self.context.open(args.store),
            args.output,
            collection_name=args.name,
            overwrite=args.overwrite,
            pretty=not args.compact,
            force=args.force,
            allow_lossy=args.allow_lossy,
        )
        self.write_output(
            args,
            text=f"exported collection: {path}",
            value={"collection": args.name, "output": str(path)},
        )
        return 0


class CollectionCommand(CommandGroup):
    """Group commands that operate on one collection."""

    name = "collection"
    help = "Inspect or export one collection"

    def __init__(self, context: StoreContext) -> None:
        """Initialize the collection command group.

        Args:
            context: Shared store context.
        """
        super().__init__(
            "Inspect or export one collection.",
            (CollectionInfoCommand(context), CollectionExportCommand(context)),
            destination="collection_action",
        )

    def add_group_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the collection name.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument("name", metavar="NAME", help="Collection name.")


class StoreCommand(CommandGroup):
    """Group resource-oriented operations on one SigMF-Zarr store."""

    name = "store"
    help = "Inspect a SigMF-Zarr store"

    def __init__(self, context: StoreContext | None = None) -> None:
        """Initialize the store command group.

        Args:
            context: Optional shared store context.
        """
        shared_context = context or StoreContext()
        super().__init__(
            "Inspect a SigMF-Zarr store and its resources.",
            (
                StoreInfoCommand(shared_context),
                StoreValidateCommand(shared_context),
                StoreIntegrityCommand(shared_context),
                RecordingsCommand(shared_context),
                RecordingCommand(shared_context),
                CollectionsCommand(shared_context),
                CollectionCommand(shared_context),
            ),
            destination="store_action",
        )

    def add_group_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the store location.

        Args:
            parser: Argument parser to extend.
        """
        parser.add_argument("store", help="SigMF-Zarr store path or URL.")


__all__ = [
    "CollectionCommand",
    "CollectionExportCommand",
    "CollectionInfoCommand",
    "CollectionsCommand",
    "RecordingCommand",
    "RecordingExportCommand",
    "RecordingInfoCommand",
    "RecordingsCommand",
    "StoreCommand",
    "StoreInfoCommand",
    "StoreIntegrityCommand",
    "StoreIntegrityUpdateCommand",
    "StoreIntegrityVerifyCommand",
    "StoreValidateCommand",
    "collection_result",
    "recording_result",
    "store_result",
]
