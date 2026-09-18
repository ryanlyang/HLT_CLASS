"""Plan an immutable, capacity-sufficient subset of an audited raw snapshot.

This module never reads particle arrays and never mutates the source snapshot.
It operates only on an already authenticated inventory.  The selected files
must subsequently be copied and re-inventoried at their destination; the plan
is not a substitute for destination byte verification.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from fractions import Fraction

import numpy as np

from .contracts import artifact, validate
from .inventory import validate_inventory
from .split_registry import EVALUATION_SIZE, TRAINING_SIZES
from .splits import ROLES, build_splits, validate_splits


ALGORITHM = "proportional_source_hash_prefix_first_capacity_with_headroom_v1"


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def verify_inventory_source_checksums(inventory: dict, lines: list[str]) -> None:
    """Compare an inventory with relative-path GNU sha256sum output."""
    validate_inventory(inventory)
    expected = {}
    for line in lines:
        if not line.strip():
            continue
        fields = line.rstrip("\r\n").split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("Invalid source checksum manifest line")
        digest, path = fields
        path = path.removeprefix("*")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or path in expected
        ):
            raise ValueError("Invalid or duplicate source checksum record")
        expected[path] = digest
    actual = {row["path"]: row["sha256"] for row in inventory["files"]}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(path for path in set(expected) & set(actual)
                         if expected[path] != actual[path])
        raise ValueError(
            "Source checksum manifest differs from local inventory: "
            f"missing={missing[:3]} extra={extra[:3]} changed={changed[:3]}"
        )


def _subset_inventory(parent: dict, records: list[dict]) -> dict:
    """Reconstruct the exact inventory payload for these existing records."""
    totals = np.sum(
        [row["selected_class_counts"] for row in records], axis=0,
        dtype=np.int64,
    ).tolist()
    return artifact(
        "INVENTORY",
        assumptions=parent["assumptions"],
        label_map=parent["label_map"],
        selection=parent["selection"],
        producer=parent["producer"],
        tree_policy=parent["tree_policy"],
        schema=parent["schema"],
        schema_sha256=parent["schema_sha256"],
        files=records,
        file_count=len(records),
        size_bytes=sum(row["size_bytes"] for row in records),
        entries=sum(row["entries"] for row in records),
        selected_class_counts=totals,
        particle_fields_audited=False,
        final_test_accessed=False,
    )


def _selection_order(inventory: dict) -> list[dict]:
    """Interleave deterministic per-source prefixes in source proportions."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in inventory["files"]:
        by_source[row["source"]].append(row)
    parent = inventory["content_hash"]
    for source, rows in by_source.items():
        rows.sort(key=lambda row: hashlib.sha256(
            f"JC2/partial-snapshot/v1/{parent}/{source}/{row['path']}".encode()
        ).digest())
    used = {source: 0 for source in by_source}
    result = []
    while len(result) < len(inventory["files"]):
        available = [source for source, rows in by_source.items() if used[source] < len(rows)]
        source = min(
            available,
            key=lambda name: (Fraction(used[name] + 1, len(by_source[name])), name),
        )
        result.append(by_source[source][used[source]])
        used[source] += 1
    return result


def build_partial_snapshot_plan(
    inventory: dict,
    *,
    training_sizes: tuple[int, ...] = TRAINING_SIZES,
    validation_size: int = EVALUATION_SIZE,
    test_size: int = EVALUATION_SIZE,
    headroom_fraction: float = 0.05,
    split_seed: int = 20260910,
    _validate_result: bool = True,
) -> dict:
    """Select the first deterministic file prefix satisfying every role.

    Headroom applies to total role capacity.  Exact per-class feasibility is
    independently proved by constructing and validating the real outer split.
    """
    parent_hash = validate_inventory(inventory)
    sizes = tuple(training_sizes)
    if (
        not sizes
        or list(sizes) != sorted(set(sizes))
        or any(not _positive_integer(value) for value in sizes)
        or not _positive_integer(validation_size)
        or not _positive_integer(test_size)
        or type(headroom_fraction) not in (int, float)
        or not math.isfinite(headroom_fraction)
        or headroom_fraction < 0
        or headroom_fraction >= 1
        or type(split_seed) is not int
        or split_seed < 0
    ):
        raise ValueError("Invalid partial snapshot capacity request")
    requested = {
        "train": max(sizes),
        "validation": validation_size,
        "final_test": test_size,
    }
    required = {
        role: int(math.ceil(value * (1.0 + headroom_fraction)))
        for role, value in requested.items()
    }
    chosen_inventory = chosen_splits = None
    last_error = None
    prefix = []
    for row in _selection_order(inventory):
        prefix.append(row)
        # Eleven classes in all three roles need at least 33 selected rows, but
        # avoiding tiny prefixes also keeps exception-driven probing cheap.
        if sum(sum(item["selected_class_counts"]) for item in prefix) < sum(required.values()):
            continue
        records = sorted(prefix, key=lambda item: item["path"])
        candidate = _subset_inventory(inventory, records)
        try:
            splits = build_splits(candidate, seed=split_seed)
        except ValueError as exc:
            last_error = str(exc)
            continue
        if all(splits["role_counts"][role] >= required[role] for role in ROLES):
            chosen_inventory, chosen_splits = candidate, splits
            break
    if chosen_inventory is None:
        details = f"; last split error: {last_error}" if last_error else ""
        raise ValueError(
            "Audited snapshot cannot satisfy requested role capacities with headroom"
            + details
        )
    selected_paths = [row["path"] for row in chosen_inventory["files"]]
    excluded_paths = sorted(
        set(row["path"] for row in inventory["files"]) - set(selected_paths)
    )
    result = artifact(
        "PARTIAL_SNAPSHOT_PLAN",
        parent_inventory_sha256=parent_hash,
        algorithm=ALGORITHM,
        split_seed=split_seed,
        training_sizes=list(sizes),
        evaluation_rows={"validation": validation_size, "final_test": test_size},
        headroom_fraction=float(headroom_fraction),
        requested_role_rows=requested,
        required_role_capacity=required,
        projected_inventory_sha256=chosen_inventory["content_hash"],
        projected_splits_sha256=chosen_splits["content_hash"],
        projected_role_counts=chosen_splits["role_counts"],
        projected_role_class_counts=chosen_splits["role_class_counts"],
        selected_file_count=len(selected_paths),
        selected_size_bytes=chosen_inventory["size_bytes"],
        selected_rows=sum(chosen_inventory["selected_class_counts"]),
        selected_class_counts=chosen_inventory["selected_class_counts"],
        selected_paths=selected_paths,
        excluded_paths=excluded_paths,
        destination_reinventory_required=True,
        destination_byte_verification_required=True,
        final_test_metadata_only=True,
        final_test_accessed=False,
    )
    if _validate_result:
        validate_partial_snapshot_plan(result, inventory)
    return result


def validate_partial_snapshot_plan(plan: dict, inventory: dict) -> str:
    digest = validate(plan, "PARTIAL_SNAPSHOT_PLAN")
    parent_hash = validate_inventory(inventory)
    if (
        plan["parent_inventory_sha256"] != parent_hash
        or plan["algorithm"] != ALGORITHM
        or plan["destination_reinventory_required"] is not True
        or plan["destination_byte_verification_required"] is not True
        or plan["final_test_metadata_only"] is not True
        or plan["final_test_accessed"] is not False
    ):
        raise ValueError("Partial snapshot plan parents or safety policy differ")
    expected = build_partial_snapshot_plan(
        inventory,
        training_sizes=tuple(plan["training_sizes"]),
        validation_size=plan["evaluation_rows"]["validation"],
        test_size=plan["evaluation_rows"]["final_test"],
        headroom_fraction=plan["headroom_fraction"],
        split_seed=plan["split_seed"],
        _validate_result=False,
    )
    if plan != expected:
        raise ValueError("Partial snapshot selection replay differs")
    return digest


def validate_partial_snapshot_destination(
    plan: dict,
    parent_inventory: dict,
    destination_inventory: dict,
    destination_splits: dict,
) -> None:
    """Bind re-inventoried destination bytes to the exact planned subset."""
    validate_partial_snapshot_plan(plan, parent_inventory)
    inventory_hash = validate_inventory(destination_inventory)
    split_hash = validate_splits(destination_splits, destination_inventory)
    if inventory_hash != plan["projected_inventory_sha256"]:
        raise ValueError("Destination inventory differs from planned subset")
    if split_hash != plan["projected_splits_sha256"]:
        raise ValueError("Destination outer split differs from planned subset")
    if [row["path"] for row in destination_inventory["files"]] != plan["selected_paths"]:
        raise ValueError("Destination path coverage differs from planned subset")
