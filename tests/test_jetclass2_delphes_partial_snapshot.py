from copy import deepcopy

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes.contracts import artifact, assumptions
from hlt_classification.jetclass2_delphes.partial_snapshot import (
    build_partial_snapshot_plan,
    validate_partial_snapshot_destination,
    validate_partial_snapshot_plan,
    verify_inventory_source_checksums,
)
from hlt_classification.jetclass2_delphes.splits import build_splits
from hlt_classification.jetclass2_delphes.provenance import source_record
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, label_map
from hlt_classification.jetclass2_delphes.selection import selection_policy


def synthetic_inventory(files_per_source=24, rows_per_class=100):
    rows = []
    schema = {"synthetic": "schema"}
    for source in ("train_higgs2p", "train_qcd"):
        for index in range(files_per_source):
            counts = [0] * len(CLASS_NAMES)
            if source == "train_qcd":
                counts[0] = rows_per_class * 10
            else:
                counts[1:] = [rows_per_class] * (len(CLASS_NAMES) - 1)
            digest = f"{len(rows) + 1:064x}"
            rows.append(dict(
                path=f"{source}/ntuple_{index:03d}.root", source=source,
                size_bytes=10_000, sha256=digest, tree_key="tree;1",
                entries=sum(counts), group_id=digest,
                raw_label_counts={}, matched_raw_label_counts={},
                selected_class_counts=counts,
                max_selected_particles={"hlt": 10, "offline": 10},
            ))
    rows.sort(key=lambda row: row["path"])
    totals = [sum(row["selected_class_counts"][i] for row in rows)
              for i in range(len(CLASS_NAMES))]
    from hlt_classification.data.cache_contracts import canonical_sha256
    return artifact(
        "INVENTORY", assumptions=assumptions(), label_map=label_map(),
        selection=selection_policy(),
        producer=source_record(
            "src/hlt_classification/jetclass2_delphes/inventory.py",
        ),
        tree_policy="latest_cycle_only_no_cycle_concatenation",
        schema=schema, schema_sha256=canonical_sha256(schema), files=rows,
        file_count=len(rows), size_bytes=sum(r["size_bytes"] for r in rows),
        entries=sum(r["entries"] for r in rows), selected_class_counts=totals,
        particle_fields_audited=False, final_test_accessed=False,
    )


def test_partial_snapshot_is_deterministic_and_capacity_safe():
    inventory = synthetic_inventory()
    plan = build_partial_snapshot_plan(
        inventory, training_sizes=(2_000,), validation_size=1_000,
        test_size=1_000, headroom_fraction=.05, split_seed=17,
    )
    assert validate_partial_snapshot_plan(plan, inventory) == plan["content_hash"]
    assert 0 < plan["selected_file_count"] < inventory["file_count"]
    assert plan["selected_paths"] == sorted(plan["selected_paths"])
    for role, required in plan["required_role_capacity"].items():
        assert plan["projected_role_counts"][role] >= required
    assert plan["destination_reinventory_required"] is True
    assert plan["final_test_accessed"] is False


def test_partial_snapshot_fails_when_capacity_is_insufficient():
    with pytest.raises(ValueError, match="cannot satisfy"):
        build_partial_snapshot_plan(
            synthetic_inventory(files_per_source=3, rows_per_class=2),
            training_sizes=(1_000,), validation_size=1_000, test_size=1_000,
        )


def test_partial_snapshot_tampering_fails():
    inventory = synthetic_inventory()
    plan = build_partial_snapshot_plan(
        inventory, training_sizes=(2_000,), validation_size=1_000,
        test_size=1_000, headroom_fraction=0,
    )
    bad = deepcopy(plan)
    bad["selected_paths"] = bad["selected_paths"][:-1]
    with pytest.raises(ValueError):
        validate_partial_snapshot_plan(with_content_hash(bad), inventory)


def test_partial_snapshot_destination_must_match_projected_artifacts():
    inventory = synthetic_inventory()
    plan = build_partial_snapshot_plan(
        inventory, training_sizes=(2_000,), validation_size=1_000,
        test_size=1_000, headroom_fraction=0,
    )
    selected = [row for row in inventory["files"] if row["path"] in plan["selected_paths"]]
    from hlt_classification.jetclass2_delphes.partial_snapshot import _subset_inventory
    destination = _subset_inventory(inventory, selected)
    splits = build_splits(destination, seed=plan["split_seed"])
    validate_partial_snapshot_destination(plan, inventory, destination, splits)
    bad = deepcopy(destination)
    bad["size_bytes"] += 1
    with pytest.raises(ValueError):
        validate_partial_snapshot_destination(
            plan, inventory, with_content_hash(bad), splits,
        )


def test_source_checksum_manifest_must_match_inventory():
    inventory = synthetic_inventory(files_per_source=2)
    lines = [f"{row['sha256']}  {row['path']}" for row in inventory["files"]]
    verify_inventory_source_checksums(inventory, lines)
    with pytest.raises(ValueError, match="differs"):
        verify_inventory_source_checksums(inventory, lines[:-1])
