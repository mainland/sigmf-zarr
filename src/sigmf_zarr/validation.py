"""Conformance validation for complete SigMF-Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from sigmf_zarr.integrity import integrity_is_supported
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.store import SigMFZarrStore


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure at a logical store path."""

    path: str
    """Logical Zarr path associated with the failure."""

    message: str
    """Human-readable failure description."""

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation.

        Returns:
            Dictionary containing the issue path and message.
        """
        return {"path": self.path, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating one complete store."""

    issues: tuple[ValidationIssue, ...]
    """Validation failures in deterministic path order."""

    recordings_checked: int
    """Number of recording groups inspected."""

    collections_checked: int
    """Number of collection groups inspected."""

    indexes_checked: int
    """Number of recording and store indexes inspected."""

    @property
    def valid(self) -> bool:
        """Return whether no validation failures were found.

        Returns:
            True when the store conforms to all requested checks.
        """
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report.

        Returns:
            Stable validation summary and issue list.
        """
        return {
            "valid": self.valid,
            "recordings_checked": self.recordings_checked,
            "collections_checked": self.collections_checked,
            "indexes_checked": self.indexes_checked,
            "issues": [issue.as_dict() for issue in self.issues],
        }

    def format(self) -> str:
        """Return a human-readable validation report.

        Returns:
            Multiline validation summary.
        """
        summary = (
            f"valid={self.valid!r}, recordings={self.recordings_checked}, "
            f"collections={self.collections_checked}, "
            f"indexes={self.indexes_checked}"
        )
        if self.valid:
            return summary
        details = [f"{issue.path}: {issue.message}" for issue in self.issues]
        return "\n".join((summary, *details))


def _integrity_issue(
    *,
    path: str,
    integrity: object,
    verifies: bool,
    require_integrity: bool,
) -> ValidationIssue | None:
    """Return an integrity issue for one resource when applicable.

    Args:
        path: Logical resource path.
        integrity: Declared integrity object.
        verifies: Whether declared hashes match current content.
        require_integrity: Whether absent hashes are failures.

    Returns:
        Integrity issue, or None when the resource satisfies the policy.
    """
    if not integrity:
        if require_integrity:
            return ValidationIssue(path, "integrity metadata is required")
        return None
    if not integrity_is_supported(integrity):
        return ValidationIssue(
            path, "integrity metadata uses an unsupported algorithm or version"
        )
    if not verifies:
        return ValidationIssue(
            path, "integrity hashes do not match current content"
        )
    return None


def _recording_issues(
    store: SigMFZarrStore,
    recording_name: str,
    *,
    verify_integrity: bool,
    require_integrity: bool,
) -> tuple[list[ValidationIssue], int]:
    """Validate one recording and its indexes.

    Args:
        store: Store containing the recording.
        recording_name: Recording name to validate.
        verify_integrity: Whether to verify hashes that are present.
        require_integrity: Whether the recording must declare integrity.

    Returns:
        Validation issues and number of indexes checked.
    """
    path = f"recordings/{recording_name}"
    issues: list[ValidationIssue] = []
    indexes_checked = 0
    # Treat each resource as an independent validation boundary so one broken
    # recording does not prevent a complete store report.
    try:
        recording = store.recordings.open(recording_name, create=False)
        for index_name, member in sorted(
            recording.indexes.members(max_depth=None)
        ):
            if not isinstance(member, ReadOnlyArray):
                continue
            indexes_checked += 1
            recording.index(index_name)
        if verify_integrity or require_integrity:
            issue = _integrity_issue(
                path=path,
                integrity=recording.integrity,
                verifies=recording.verify_integrity(),
                require_integrity=require_integrity,
            )
            if issue is not None:
                issues.append(issue)
        if recording.sha512 is not None and not recording.verify_sha512():
            issues.append(
                ValidationIssue(path, "core:sha512 does not match sample data")
            )
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(ValidationIssue(path, str(exc)))
    return issues, indexes_checked


def _collection_issues(
    store: SigMFZarrStore,
    collection_name: str,
    *,
    recording_names: set[str],
    verify_integrity: bool,
    require_integrity: bool,
) -> list[ValidationIssue]:
    """Validate one collection and its recording references.

    Args:
        store: Store containing the collection.
        collection_name: Collection name to validate.
        recording_names: Recording names available in the store.
        verify_integrity: Whether to verify hashes that are present.
        require_integrity: Whether the collection must declare integrity.

    Returns:
        Validation issues for the collection.
    """
    path = f"collections/{collection_name}"
    issues: list[ValidationIssue] = []
    try:
        collection = store.collections.open(collection_name, create=False)
        missing = sorted(set(collection.recording_ids) - recording_names)
        if missing:
            issues.append(
                ValidationIssue(
                    path,
                    f"references missing recordings: {missing!r}",
                )
            )
        if verify_integrity or require_integrity:
            issue = _integrity_issue(
                path=path,
                integrity=collection.integrity,
                verifies=collection.verify_integrity(),
                require_integrity=require_integrity,
            )
            if issue is not None:
                issues.append(issue)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(ValidationIssue(path, str(exc)))
    return issues


def _store_index_issues(
    store: SigMFZarrStore,
) -> tuple[list[ValidationIssue], int]:
    """Validate every store-wide index.

    Args:
        store: Store whose indexes to validate.

    Returns:
        Validation issues and number of indexes checked.
    """
    issues: list[ValidationIssue] = []
    index_names = sorted(
        name
        for name, member in store.indexes.members(max_depth=None)
        if isinstance(member, ReadOnlyArray)
    )
    for index_name in index_names:
        try:
            store.index(index_name)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(
                ValidationIssue(f"indexes/{index_name}", str(exc))
            )
    return issues, len(index_names)


def _store_integrity_issues(
    store: SigMFZarrStore,
    *,
    require_integrity: bool,
) -> list[ValidationIssue]:
    """Validate root-store integrity metadata.

    Args:
        store: Store whose integrity metadata to validate.
        require_integrity: Whether the store must declare integrity.

    Returns:
        Root-store integrity issues.
    """
    try:
        issue = _integrity_issue(
            path="/",
            integrity=store.integrity,
            verifies=store.verify_integrity(),
            require_integrity=require_integrity,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [
            ValidationIssue("/", f"could not verify integrity: {exc}")
        ]
    return [] if issue is None else [issue]


def validate_store(
    store_or_path: SigMFZarrStore | str | PathLike[str],
    *,
    verify_integrity: bool = True,
    require_integrity: bool = False,
) -> ValidationReport:
    """Validate the complete logical structure and optional integrity hashes.

    Args:
        store_or_path: Open store or path understood by Zarr.
        verify_integrity: Whether to verify hashes that are present.
        require_integrity: Whether every resource must declare integrity.

    Returns:
        Complete validation report. Structural failures are reported rather
        than raised after the root store has opened successfully.

    Raises:
        ValueError: If the root is not a SigMF-Zarr store.
    """
    store = (
        store_or_path
        if isinstance(store_or_path, SigMFZarrStore)
        else SigMFZarrStore.open(store_or_path, mode="r")
    )
    issues: list[ValidationIssue] = []
    recording_names = store.list_recordings()
    collection_names = store.list_collections()
    indexes_checked = 0

    for recording_name in recording_names:
        recording_issues, recording_indexes = _recording_issues(
            store,
            recording_name,
            verify_integrity=verify_integrity,
            require_integrity=require_integrity,
        )
        issues.extend(recording_issues)
        indexes_checked += recording_indexes

    available_recordings = set(recording_names)
    for collection_name in collection_names:
        issues.extend(
            _collection_issues(
                store,
                collection_name,
                recording_names=available_recordings,
                verify_integrity=verify_integrity,
                require_integrity=require_integrity,
            )
        )

    index_issues, store_indexes = _store_index_issues(store)
    issues.extend(index_issues)
    indexes_checked += store_indexes

    if verify_integrity or require_integrity:
        issues.extend(
            _store_integrity_issues(
                store,
                require_integrity=require_integrity,
            )
        )

    return ValidationReport(
        # Stable ordering makes text output and automated comparisons
        # independent of the backing store's traversal order.
        issues=tuple(
            sorted(issues, key=lambda issue: (issue.path, issue.message))
        ),
        recordings_checked=len(recording_names),
        collections_checked=len(collection_names),
        indexes_checked=indexes_checked,
    )


__all__ = ["ValidationIssue", "ValidationReport", "validate_store"]
