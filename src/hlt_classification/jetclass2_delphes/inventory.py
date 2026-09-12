"""Read-only latest-cycle inventory and relocatable, hash-verified raw snapshot."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import uproot

from hlt_classification.data.cache_contracts import canonical_sha256, sha256_file
from .contracts import artifact, assumptions, relative_file, validate
from .schema import CLASS_NAMES, SCALARS, label_map, validate_label_map, validate_tree_schema
from .selection import selected_mask, selection_policy, validate_policy
from .provenance import source_record, validate_source_record


def latest_tree(root_file):
    keys = [key for key in root_file.keys(cycle=True, recursive=False)
            if key.rsplit(";", 1)[0] == "tree"]
    if not keys:
        raise ValueError("ROOT file has no tree")
    key = max(keys, key=lambda k: int(k.rsplit(";", 1)[1]))
    return key, root_file[key]


def build_inventory(data_root: Path, *, include_signal_file_qcd: bool = False,
                    step_size: int = 100_000) -> dict:
    root = Path(data_root).resolve(strict=True)
    if step_size < 1:
        raise ValueError("step_size must be positive")
    paths = sorted(root.rglob("*.root"), key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError("No ROOT files in dataset root")
    policy = selection_policy(include_signal_file_qcd=include_signal_file_qcd)
    records, names, hashes = [], set(), set()
    common_schema = None
    for number, path in enumerate(paths, 1):
        rel = path.relative_to(root).as_posix()
        path = relative_file(root, rel)
        if rel.casefold() in names:
            raise ValueError(f"Case-insensitive duplicate relative path: {rel}")
        names.add(rel.casefold())
        source = rel.split("/", 1)[0]
        if source not in policy["source_categories"]:
            raise ValueError(f"Unrecognized source file: {rel}")
        before = path.stat()
        digest = sha256_file(path)
        if digest in hashes:
            raise ValueError(f"Duplicate ROOT content (possible split leakage): {rel}")
        hashes.add(digest)
        raw_counts, matched_counts = Counter(), Counter()
        counts = np.zeros(len(CLASS_NAMES), np.int64)
        max_counts = {"hlt": 0, "offline": 0}
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            schema = validate_tree_schema(tree)
            if common_schema is None:
                common_schema = schema
            elif schema != common_schema:
                raise ValueError(f"ROOT schema differs: {rel}")
            for batch in tree.iterate(list(SCALARS), step_size=step_size, library="np"):
                keep, labels = selected_mask(batch["jet_label"], batch["hlt_matched"], source, policy)
                raw_counts.update(map(int, batch["jet_label"]))
                matched_counts.update(map(int, batch["jet_label"][batch["hlt_matched"]]))
                counts += np.bincount(labels[keep], minlength=len(CLASS_NAMES))
                for view, field in (("hlt", "hlt_jet_nparticles"), ("offline", "jet_nparticles")):
                    sizes = batch[field][keep]
                    if np.any(sizes <= 0):
                        raise ValueError(f"Selected empty jet in {rel}:{view}")
                    max_counts[view] = max(max_counts[view], int(sizes.max(initial=0)))
            entries = int(tree.num_entries)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed during inventory: {rel}")
        # A second content check prevents an in-place mutation from masquerading
        # as a stable snapshot merely by preserving size/mtime.
        if sha256_file(path) != digest:
            raise ValueError(f"File content changed during inventory: {rel}")
        records.append(dict(
            path=rel, source=source, size_bytes=after.st_size, sha256=digest,
            tree_key=key, entries=entries, group_id=digest,
            raw_label_counts={str(k): v for k, v in sorted(raw_counts.items())},
            matched_raw_label_counts={str(k): v for k, v in sorted(matched_counts.items())},
            selected_class_counts=counts.tolist(), max_selected_particles=max_counts,
        ))
        print(f"JC2 phase=inventory file={number}/{len(paths)} selected={int(counts.sum())} path={rel}", flush=True)
    return artifact(
        "INVENTORY", assumptions=assumptions(), label_map=label_map(), selection=policy,
        producer=source_record(*[f"src/hlt_classification/jetclass2_delphes/{name}.py"
                                for name in ("inventory", "schema", "selection", "contracts", "provenance")]),
        tree_policy="latest_cycle_only_no_cycle_concatenation",
        schema=common_schema, schema_sha256=canonical_sha256(common_schema), files=records,
        file_count=len(records), size_bytes=sum(r["size_bytes"] for r in records),
        entries=sum(r["entries"] for r in records),
        selected_class_counts=np.sum([r["selected_class_counts"] for r in records], axis=0).tolist(),
        particle_fields_audited=False, final_test_accessed=False,
    )


def validate_inventory(value: dict) -> str:
    digest = validate(value, "INVENTORY")
    if "producer" not in value:
        raise ValueError("Inventory lacks implementation provenance; regenerate, do not edit it")
    validate_source_record(value["producer"])
    validate(value["assumptions"], "ASSUMPTIONS")
    if value["assumptions"] != assumptions():
        raise ValueError("Provisional assumptions differ")
    validate_label_map(value["label_map"])
    validate_policy(value["selection"])
    if canonical_sha256(value["schema"]) != value["schema_sha256"]:
        raise ValueError("Schema digest differs")
    rows = value["files"]
    paths = [r["path"] for r in rows]
    hashes = [r["sha256"] for r in rows]
    if (not rows or paths != sorted(paths) or len(set(p.casefold() for p in paths)) != len(paths)
            or len(set(hashes)) != len(hashes) or len(rows) != value["file_count"]):
        raise ValueError("Inventory file coverage/uniqueness differs")
    for row in rows:
        relative_file(Path.cwd(), row["path"])
        if (row["group_id"] != row["sha256"] or row["source"] != row["path"].split("/")[0]
                or len(row["selected_class_counts"]) != len(CLASS_NAMES)
                or any(type(n) is not int or n < 0 for n in row["selected_class_counts"])):
            raise ValueError("Invalid file record")
    totals = np.sum([r["selected_class_counts"] for r in rows], axis=0).tolist()
    if (totals != value["selected_class_counts"]
            or sum(r["entries"] for r in rows) != value["entries"]
            or sum(r["size_bytes"] for r in rows) != value["size_bytes"]):
        raise ValueError("Inventory totals differ")
    return digest


def verify_snapshot(data_root: Path, inventory: dict) -> None:
    validate_inventory(inventory)
    root = Path(data_root).resolve(strict=True)
    actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.root"))
    if actual != [r["path"] for r in inventory["files"]]:
        raise ValueError("Snapshot contains missing or extra ROOT files")
    for i, row in enumerate(inventory["files"], 1):
        verify_file(root, row, inventory)
        print(f"JC2 phase=verify file={i}/{len(actual)}", flush=True)


def verify_file(root: Path, record: dict, inventory: dict) -> Path:
    path = relative_file(root, record["path"])
    if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"ROOT content differs: {record['path']}")
    with uproot.open(path) as handle:
        key, tree = latest_tree(handle)
        if (key != record["tree_key"] or tree.num_entries != record["entries"]
                or validate_tree_schema(tree) != inventory["schema"]):
            raise ValueError(f"ROOT cycle/schema differs: {record['path']}")
    return path
