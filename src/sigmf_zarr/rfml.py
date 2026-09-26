"""Explicit validation of the dense RFML dataset profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from sigmf_zarr.indexes import validate_categorical_values
from sigmf_zarr.json import JSONObject, json_object, json_object_list
from sigmf_zarr.readonly import ReadOnlyArray
from sigmf_zarr.validation import ValidationIssue

if TYPE_CHECKING:
    from sigmf_zarr.store import SigMFRecording

_UNITS = {
    "class": None, "modulation": None, "signal_count": None,
    "snr": "dB", "carrier_offset": "cycles/sample",
    "symbol_rate": "symbols/sample", "source_id": None,
    "session_id": None, "emitter_id": None,
}
_CATEGORIES = {"class", "modulation"}
_IDENTITIES = {"source_id", "session_id", "emitter_id"}


@dataclass(frozen=True)
class RFMLValidationReport:
    """Profile conformance results, independent of scientific ground truth."""

    issues: tuple[ValidationIssue, ...]
    """Descriptor or payload violations in the checked profile fields."""

    indexes_checked: int
    """Number of known profile indexes checked."""

    unvalidated_fields: tuple[str, ...]
    """Native or unknown fields whose meanings this validator does not
    check.
    """

    @property
    def valid(self) -> bool:
        """Return whether the checked profile requirements hold.

        Returns:
            True without violations. This does not certify source truth.
        """
        return not self.issues

    def as_dict(self) -> JSONObject:
        """Return explicit conformance and evidence boundaries as JSON.

        Returns:
            Report identifying checked and unvalidated fields.
        """
        return json_object({
            "valid": self.valid,
            "profile": "rfml-dataset", "version": "0.1.0",
            "indexes_checked": self.indexes_checked,
            "unvalidated_fields": list(self.unvalidated_fields),
            "source_truth_verified": False,
            "issues": [issue.as_dict() for issue in self.issues],
        }, name="RFML validation report")


def _measurement(attributes: JSONObject) -> None:
    """Check the SNR measurement descriptor without interpreting its source.

    Args:
        attributes: SNR index attributes.

    Raises:
        ValueError: If the descriptor omits a required convention.
    """
    measurement = json_object(
        attributes.get("measurement"), name="measurement"
    )
    definition = measurement.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("SNR measurement requires a nonempty definition")
    for name, allowed in (
        ("signal_reference", {"scene"}),
        ("time_support", {"item", "signal_interval"}),
        ("noise_bandwidth", {"full_sample_band", "signal_band"}),
        ("interference", {"excluded"}),
    ):
        value = measurement.get(name)
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"Invalid SNR measurement {name}")


def _descriptors(
    index: ReadOnlyArray, field: str, channels: int,
) -> JSONObject:
    """Validate a dense field's scope, unit, and required descriptors.

    Args:
        index: Selected profile index.
        field: Profile-local field name.
        channels: Number of receiver channels.

    Returns:
        Detached validated descriptors.

    Raises:
        ValueError: If descriptors conflict with the profile.
    """
    attributes = json_object(dict(index.attrs), name="index attributes")
    if (
        attributes.get("axis") != "item"
        or attributes.get("kind") != "metadata"
        or "validity" in attributes
    ):
        raise ValueError("Profile fields require dense item metadata indexes")
    if attributes.get("unit") != _UNITS[field]:
        raise ValueError(f"Field {field!r} requires unit {_UNITS[field]!r}")
    if field in {"carrier_offset", "symbol_rate"} and channels != 1:
        raise ValueError("Normalized rates require one receiver channel")
    if field in _CATEGORIES:
        labels = attributes.get("labels")
        if (
            not isinstance(labels, list) or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Category labels must be unique nonempty strings")
    if field == "snr":
        _measurement(attributes)
    return attributes


def _values(
    values: npt.NDArray[Any], field: str, attributes: JSONObject,
) -> None:
    """Check one bounded batch of profile values.

    Args:
        values: One-dimensional payload batch.
        field: Profile-local field name.
        attributes: Validated field descriptors.

    Raises:
        ValueError: If the values violate the field's domain.
    """
    if field in _CATEGORIES:
        labels = attributes["labels"]
        assert isinstance(labels, list)
        validate_categorical_values(values, labels=labels)
    elif field in _IDENTITIES:
        if values.dtype.kind not in "iuUOT" or any(
            not (
                isinstance(value, str) and bool(value)
                or isinstance(value, int | np.integer)
                and not isinstance(value, bool | np.bool_)
            )
            for value in values
        ):
            raise ValueError("Identities must be integers or nonempty strings")
    elif field == "signal_count":
        if values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError("Signal counts must be nonnegative integers")
    else:
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("Measurements must be finite real numbers")
        if field == "carrier_offset" and np.any(
            (values < -0.5) | (values >= 0.5)
        ):
            raise ValueError("Carrier offsets must lie in [-0.5, 0.5)")
        if field == "symbol_rate" and np.any(values <= 0):
            raise ValueError("Symbol rates must be positive")


def validate_rfml(
    recording: SigMFRecording, *, batch_size: int = 65536,
) -> RFMLValidationReport:
    """Check dense profile declarations and values without reading samples.

    Field meanings, source identities, and measurement definitions remain
    claims supplied by the writer. This validates their representation and
    declared numerical domains, not their physical truth or freshness. Each
    same-field index is checked independently, without selecting an authority.

    Args:
        recording: Recording whose RFML profile metadata is checked.
        batch_size: Maximum number of index values read at once.

    Returns:
        Profile violations and fields outside this validator's scope.

    Raises:
        ValueError: If batch_size is not a positive non-Boolean integer.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    issues = []
    unchecked = set()
    count = 0
    try:
        extensions = json_object_list(
            recording.global_metadata.get("core:extensions", []),
            name="core:extensions",
        )
        profile = [e for e in extensions if e.get("name") == "rfml-dataset"]
        if profile != [{
            "name": "rfml-dataset", "version": "0.1.0", "optional": True,
        }]:
            raise ValueError("Declare rfml-dataset version 0.1.0 as optional")
        if not recording.batched:
            raise ValueError("Dense RFML fields require a batched recording")
    except ValueError as exc:
        issues.append(ValidationIssue("global/core:extensions", str(exc)))
    for name, member in sorted(recording.indexes.members(max_depth=None)):
        if not isinstance(member, ReadOnlyArray):
            continue
        qualified = member.attrs.get("field")
        field = str(qualified).removeprefix("rfml-dataset:")
        if qualified != f"rfml-dataset:{field}" or field not in _UNITS:
            unchecked.add(str(qualified))
            continue
        count += 1
        try:
            index = recording.index(name)
            attributes = _descriptors(index, field, recording.num_channels)
            # Empty recordings still require the declared payload dtype.
            _values(np.empty(0, dtype=index.dtype), field, attributes)
            for start in range(0, len(recording), batch_size):
                _values(
                    np.asarray(index[start : start + batch_size]),
                    field, attributes,
                )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(ValidationIssue(f"indexes/{name}", str(exc)))
    return RFMLValidationReport(tuple(issues), count, tuple(sorted(unchecked)))


__all__ = ["RFMLValidationReport", "validate_rfml"]
