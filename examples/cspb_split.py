"""Create a CSPB holdout that isolates reused constituent source signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigmf_zarr import SigMFZarrStore, capture_inputs
from sigmf_zarr.json import JSONObject
from sigmf_zarr.sources import source_groups

CSPB_SOURCES: JSONObject = {
    "field": "cspb:signals",
    "identity": "cspb:source_signal_index",
    "fallback_index": "signal_id",
}
"""Source reuse descriptor for truth-aligned native CSPB imports."""


def create_split(
    path: Path,
    *,
    recording: str = "cspb",
    name: str,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> dict[str, int]:
    """Create an evaluation split with transitive constituent isolation.

    This claim concerns source file reuse within one recording. Source IDs do
    not establish physical emitter or acquisition-session independence.
    A connected mixture graph cannot provide an independent holdout.

    Args:
        path: Writable imported CSPB store.
        recording: Recording with native CSPB truth metadata.
        name: New split index name, never overwritten.
        seed: Nonnegative generator seed.
        validation_fraction: Requested fraction of connected groups.

    Returns:
        Counts of connected groups and items assigned to each partition.

    Raises:
        ValueError: If truth is absent, identities are unknown, or fewer than
            two independent source groups exist.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    with SigMFZarrStore.open(path, mode="r+") as store:
        rec = store.recordings[recording]
        if not rec.global_metadata.get("cspb:truth_formats"):
            raise ValueError("CSPB source isolation requires imported truth")
        groups = source_groups(rec, CSPB_SOURCES)
        count = len(np.unique(groups))
        if count < 2:
            raise ValueError("Fewer than two independent source groups")
        held_out = min(count - 1, max(1, int(count * validation_fraction)))
        selected = rng.permutation(count)[:held_out]
        assignments = np.isin(groups, selected).astype(np.uint8)
        rec.add_split(
            name,
            assignments,
            labels=["train", "validation"],
            split_type="holdout",
            method="group_random",
            seed=seed,
            group_sources=CSPB_SOURCES,
            input_binding=capture_inputs(rec, indexes=["signal_id"]),
            generator="examples.cspb_split v1",
            description=(
                "Constituent source reuse isolation within this recording; "
                f"validation_group_fraction={validation_fraction}."
            ),
        )
        return {
            "groups": count,
            "train": int(np.sum(assignments == 0)),
            "validation": int(np.sum(assignments == 1)),
        }


def main() -> None:
    """Parse source selection and print evaluation split counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("--recording", default="cspb")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(
        json.dumps(
            create_split(
                args.store,
                recording=args.recording,
                name=args.name,
                seed=args.seed,
                validation_fraction=args.validation_fraction,
            )
        )
    )


if __name__ == "__main__":
    main()
