"""Annotation frequency references belong to the applicable capture."""

from sigmf_zarr.viewer.captures import sigmf_regions


def test_annotation_uses_its_own_capture_frequency_reference() -> None:
    """An unrelated RF capture must not hide a baseband annotation."""
    metadata = {
        "global": {"core:sample_rate": 100},
        "captures": [
            {"core:sample_start": 0, "core:frequency": 100e6},
            {"core:sample_start": 50},
        ],
        "annotations": [{
            "core:sample_start": 60, "core:sample_count": 10,
            "core:freq_lower_edge": -10, "core:freq_upper_edge": 10,
        }],
    }
    annotation = sigmf_regions(metadata, 100)[-1]
    assert annotation.bounds["frequency"].reference == "baseband"
    assert annotation.bounds["frequency"].start == -10
