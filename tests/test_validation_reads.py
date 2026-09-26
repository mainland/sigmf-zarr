"""Tests for the sample reads that store validation performs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigmf_zarr.sigmf as sigmf_module
from sigmf_zarr import SigMFZarrStore, validate_store
from sigmf_zarr.store import SigMFRecording


@pytest.fixture
def hashed(tmp_path: Path) -> SigMFZarrStore:
    """Create a store whose recordings have every kind of hash.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Store with two hashed recordings.
    """
    store = SigMFZarrStore.create(tmp_path / "store.zarr")
    for name in ("a", "b"):
        recording = store.recordings.create(
            name,
            sample_shape=(2, 64),
            global_metadata={"core:datatype": "cf32_le"},
        )
        recording.set_samples(np.ones((2, 64), dtype=np.float32))
        recording.update_sha512()
    store.update_integrity()
    return store


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count full sample reads by hash kind.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Counter keyed by `native` and `sigmf`.
    """
    counts: Counter[str] = Counter()
    native = SigMFRecording.calculate_sample_sha512
    sigmf = sigmf_module.calculate_sha512

    def count_native(self: SigMFRecording) -> str:
        """Count a native sample hash.

        Returns:
            Original digest.
        """
        counts["native"] += 1
        return native(self)

    def count_sigmf(recording: SigMFRecording) -> str:
        """Count a standard SigMF dataset hash.

        Args:
            recording: Recording being hashed.

        Returns:
            Original digest.
        """
        counts["sigmf"] += 1
        return sigmf(recording)

    monkeypatch.setattr(
        SigMFRecording, "calculate_sample_sha512", count_native
    )
    monkeypatch.setattr(sigmf_module, "calculate_sha512", count_sigmf)
    return counts


def test_validation_reads_each_sample_hash_once(
    hashed: SigMFZarrStore, reads: Counter[str]
) -> None:
    """Store-wide verification must not recheck every recording.

    Args:
        hashed: Store with two hashed recordings.
        reads: Sample read counter.
    """
    report = validate_store(hashed)
    assert report.valid
    assert reads == Counter({"native": 2, "sigmf": 2})


def test_skip_integrity_reads_no_samples(
    hashed: SigMFZarrStore, reads: Counter[str]
) -> None:
    """Structural validation must not verify any declared hash.

    Args:
        hashed: Store with two hashed recordings.
        reads: Sample read counter.
    """
    hashed.recordings["a"].set_global_field("core:sha512", "0" * 128)
    report = validate_store(hashed, verify_integrity=False)
    assert report.valid
    assert reads == Counter()


def test_required_integrity_without_verification_checks_presence(
    hashed: SigMFZarrStore, reads: Counter[str]
) -> None:
    """Requiring hashes without verification checks only their presence.

    Args:
        hashed: Store with two hashed recordings.
        reads: Sample read counter.
    """
    assert validate_store(
        hashed, verify_integrity=False, require_integrity=True
    ).valid
    hashed.recordings["b"].set_global_field("test:edited", True)
    report = validate_store(
        hashed, verify_integrity=False, require_integrity=True
    )
    assert {issue.path for issue in report.issues} == {"/", "recordings/b"}
    assert reads == Counter()
