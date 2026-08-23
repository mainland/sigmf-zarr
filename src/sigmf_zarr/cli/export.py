"""Shared command-line export options and handlers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sigmf_zarr.sigmf import export_sigmf, export_sigmf_archive
from sigmf_zarr.store import SigMFZarrStore

type ExportResult = tuple[int, str, dict[str, Any]]
"""Exit status, human-readable text, and JSON value of one export."""


def add_export_arguments(
    parser: argparse.ArgumentParser, *, output_help: str
) -> None:
    """Add the output path and options shared by every export command.

    Args:
        parser: Argument parser to extend.
        output_help: Help text for the positional output path.
    """
    parser.add_argument("output", type=Path, help=output_help)
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
        help="Warn and omit indexes, arrays, and channel metadata.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON metadata instead of pretty-printed JSON.",
    )


def add_recording_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options that apply only to single-recording export.

    Args:
        parser: Argument parser to extend.
    """
    parser.add_argument(
        "--item-index",
        type=int,
        help="Export one nonnegative item position from a batch.",
    )


def export_recording(
    store: SigMFZarrStore, recording_name: str, args: argparse.Namespace
) -> ExportResult:
    """Export one recording.

    Args:
        store: Source store.
        recording_name: Recording to export.
        args: Parsed export arguments.

    Returns:
        Exit status, text, and JSON value.
    """
    path = export_sigmf(
        store,
        recording_name,
        args.output,
        overwrite=args.overwrite,
        pretty=not args.compact,
        force=args.force,
        allow_lossy=args.allow_lossy,
        item_index=args.item_index,
    )
    return (
        0,
        f"exported recording: {path}",
        {"recording": recording_name, "output": str(path)},
    )


def export_archive(
    store: SigMFZarrStore,
    args: argparse.Namespace,
    *,
    recording_names: list[str] | None,
    collection_name: str | None,
) -> Path:
    """Export recordings and an optional collection as a SigMF archive.

    Args:
        store: Source store.
        args: Parsed export arguments.
        recording_names: Explicit recordings, or `None` for the collection's
            recordings or every recording.
        collection_name: Optional collection to include.

    Returns:
        Written archive path.
    """
    return export_sigmf_archive(
        store,
        args.output,
        recording_names=recording_names,
        collection_name=collection_name,
        overwrite=args.overwrite,
        pretty=not args.compact,
        force=args.force,
        allow_lossy=args.allow_lossy,
    )


__all__ = [
    "ExportResult",
    "add_export_arguments",
    "add_recording_export_arguments",
    "export_archive",
    "export_recording",
]
