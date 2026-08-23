"""Shared command-line export options and handlers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sigmf_zarr._text import escape_text
from sigmf_zarr.export_plan import plan_sigmf_export
from sigmf_zarr.sigmf import export_sigmf, export_sigmf_archive
from sigmf_zarr.store import SigMFZarrStore

type ExportResult = tuple[int, str, dict[str, Any]]
"""Exit status, human-readable text, and JSON value of one export."""

EXPORT_OPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "overwrite": "--overwrite",
        "force": "--force",
        "allow_lossy": "--allow-lossy",
        "item_index": "--item-index",
        "project_indexes": "--project-index",
    }
)
"""Export parameters mapped to their command-line options."""


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
    parser.add_argument(
        "--project-index",
        action="append",
        default=[],
        help="Preserve this item index and its descriptors. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a verified JSON export plan without writing files.",
    )


def export_recording(
    store: SigMFZarrStore, recording_name: str, args: argparse.Namespace
) -> ExportResult:
    """Export one recording, or plan the export for a dry run.

    Args:
        store: Source store.
        recording_name: Recording to export.
        args: Parsed export arguments.

    Returns:
        Exit status, text, and JSON value. A dry run fails when the plan has
        rejections, or omissions without `--allow-lossy`.
    """
    if args.dry_run:
        plan = plan_sigmf_export(
            store,
            recording_name,
            item_index=args.item_index,
            project_indexes=args.project_index,
            force=args.force,
        )
        value = plan.as_dict()
        status = int(
            bool(plan.rejected or (plan.omitted and not args.allow_lossy))
        )
        return status, json.dumps(value, indent=2, sort_keys=True), value
    path = export_sigmf(
        store,
        recording_name,
        args.output,
        overwrite=args.overwrite,
        pretty=not args.compact,
        force=args.force,
        allow_lossy=args.allow_lossy,
        item_index=args.item_index,
        project_indexes=args.project_index,
    )
    return (
        0,
        f"exported recording: {escape_text(str(path))}",
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
    "EXPORT_OPTIONS",
    "ExportResult",
    "add_export_arguments",
    "add_recording_export_arguments",
    "export_archive",
    "export_recording",
]
