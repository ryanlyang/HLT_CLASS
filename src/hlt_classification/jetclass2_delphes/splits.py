"""Deterministic file-grouped, class-balanced approximate 60/20/20 partition."""
from __future__ import annotations

import hashlib
import numpy as np

from .contracts import artifact, validate
from .inventory import validate_inventory
from .schema import CLASS_NAMES

ROLES = ("train", "validation", "final_test")


def build_splits(inventory: dict, *, seed: int = 20260910) -> dict:
    """Balance class deficits using files, never individual events.

    A file may mix labels. Greedy normalized deficit minimizes the squared
    class-count departure from target role fractions; hash order breaks ties.
    This proves file/content disjointness, NOT generator-event disjointness.
    """
    digest = validate_inventory(inventory)
    if type(seed) is not int or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    fractions = np.array([.6, .2, .2])
    totals = np.asarray(inventory["selected_class_counts"], np.int64)
    if np.any(totals == 0):
        raise ValueError("Every registered class must occur in the selected snapshot")
    def tie(row):
        return hashlib.sha256(f"JC2/split/v1/{seed}/{row['path']}".encode()).hexdigest()
    # Large files of rare classes first; deterministic but not filename bins.
    files = sorted(inventory["files"], key=lambda r: (
        -float(np.max(np.asarray(r["selected_class_counts"]) / totals)), tie(r),
    ))
    counts = np.zeros((3, len(CLASS_NAMES)), np.int64)
    assigned = {}
    for row in files:
        mass = np.asarray(row["selected_class_counts"], np.int64)
        objectives = []
        for role in range(3):
            trial = counts.copy()
            trial[role] += mass
            missing = int(np.count_nonzero((counts[role] == 0) & (mass > 0)))
            objectives.append((-missing, float(np.sum((trial / totals - fractions[:, None]) ** 2)), role))
        role = min(objectives)[2]
        counts[role] += mass
        assigned[row["path"]] = ROLES[role]
    result = artifact(
        "SPLITS", inventory_sha256=digest, selection_sha256=inventory["selection"]["content_hash"],
        seed=seed, target_fractions=dict(zip(ROLES, fractions.tolist())),
        algorithm="whole_file_class_coverage_then_normalized_deficit_greedy_v1",
        groups=[dict(path=row["path"], group_id=row["group_id"], role=assigned[row["path"]])
                for row in inventory["files"]],
        role_class_counts={role: counts[i].tolist() for i, role in enumerate(ROLES)},
        role_counts={role: int(counts[i].sum()) for i, role in enumerate(ROLES)},
        proven_disjointness="relative_file_paths_and_content_hashes",
        cross_file_generator_independence="provisional_unverified",
        final_test_accessed=False,
    )
    validate_splits(result, inventory)
    return result


def validate_splits(splits: dict, inventory: dict) -> str:
    if splits.get("contract") == "JETCLASS2_DELPHES_SPLIT_PROFILE/v1":
        from .split_registry import validate_split_profile
        return validate_split_profile(splits, inventory)
    digest = validate(splits, "SPLITS")
    if (splits["inventory_sha256"] != validate_inventory(inventory)
            or splits["selection_sha256"] != inventory["selection"]["content_hash"]):
        raise ValueError("Split parents differ")
    groups, files = splits["groups"], inventory["files"]
    if [g["path"] for g in groups] != [r["path"] for r in files]:
        raise ValueError("Split file coverage/order differs")
    counts = {role: np.zeros(len(CLASS_NAMES), np.int64) for role in ROLES}
    seen = set()
    for group, row in zip(groups, files):
        if (group["role"] not in ROLES or group["group_id"] != row["group_id"]
                or group["group_id"] in seen):
            raise ValueError("Split group overlap or invalid role")
        seen.add(group["group_id"])
        counts[group["role"]] += np.asarray(row["selected_class_counts"])
    for role, values in counts.items():
        if (np.any(values == 0) or values.tolist() != splits["role_class_counts"][role]
                or int(values.sum()) != splits["role_counts"][role]):
            raise ValueError(f"Split class coverage/counts invalid: {role}")
    return digest


def is_subset_profile(splits: dict) -> bool:
    return splits.get("contract") == "JETCLASS2_DELPHES_SPLIT_PROFILE/v1"


def file_population(splits: dict, file: dict, role: str) -> tuple[int, list[int]]:
    """Metadata for this exact profile, not the full reservoir's population."""
    if is_subset_profile(splits):
        row = next(r for r in splits["memberships"][role]["files"] if r["path"] == file["path"])
        return row["rows"], row["class_counts"]
    return sum(file["selected_class_counts"]), file["selected_class_counts"]
