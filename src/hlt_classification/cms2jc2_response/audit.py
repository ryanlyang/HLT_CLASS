"""Read-only source/role audits. This module never constructs fitted responses."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import uproot

from hlt_classification.scouting.labels import baseline_mask, multiclass_labels
from hlt_classification.scouting.schema import BASELINE_BRANCHES, LABEL_BRANCHES
from hlt_classification.scouting.splits import validate_split_manifest

from .contracts import artifact, sha256_file, safe_relative, validate
from .splits import pack_entries

CMS_SCALARS = tuple(sorted(set(BASELINE_BRANCHES + LABEL_BRANCHES)))


def latest_tree(handle):
    keys = [k for k in handle.keys(cycle=True, recursive=False) if k.rsplit(";", 1)[0] == "tree"]
    if not keys:
        raise ValueError("Missing ROOT tree")
    key = max(keys, key=lambda k: int(k.rsplit(";", 1)[1]))
    return key, handle[key]


def cms_particle_branches(side: str) -> tuple[str, ...]:
    if side not in {"offline", "hlt"}:
        raise ValueError("Unknown CMS side")
    branches = []
    prefixes = ("cpfcandlt", "npfcand") if side == "offline" else ("scoutpfcand",)
    for prefix in prefixes:
        fields = ["px", "py", "pz", "energy"]
        if prefix != "npfcand":
            fields += ["charge", "isChargedHad", "isEl", "isMu", "dxy", "dz", "dxysig", "dzsig"]
        if prefix in {"scoutpfcand", "npfcand"}:
            fields += ["isNeutralHad", "isGamma"]
        if prefix == "cpfcandlt":
            fields += ["isLostTrack"]
        branches.extend(f"{prefix}_{f}" for f in fields)
    return tuple(branches)


def inspect_headers(root: Path) -> dict:
    """Metadata only; safe before the historical split artifact is located."""
    root = root.resolve(strict=True)
    records = []
    for path in sorted(root.rglob("*.root")):
        relative = path.relative_to(root).as_posix()
        safe_relative(root, relative)
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            types = tree.typenames()
            required = set(CMS_SCALARS + cms_particle_branches("offline") + cms_particle_branches("hlt"))
            records.append(dict(path=relative, tree_key=key, raw_entries=int(tree.num_entries),
                                size_bytes=path.stat().st_size, branches=dict(sorted(types.items())),
                                missing_branches=sorted(required - set(types))))
    if not records:
        raise ValueError("No raw CMS files")
    return artifact("HEADER_AUDIT", files=records, file_count=len(records),
                    particle_arrays_read=False, fitted=False,
                    authenticated_for_fitting=False,
                    blockers=["Historical CMS split and content hashes required",
                              "Common physical semantics need documented review"])


def inventory_cms(root: Path, split: dict, *, chunk_rows: int = 100_000) -> dict:
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Invalid read chunk size")
    split_hash = validate_split_manifest(split, source_manifest_sha256=split["source_manifest_sha256"])
    all_files = [f for role in split["roles"].values() for f in role["files"]]
    if len({f["sha256"] for f in all_files}) != len(all_files):
        raise ValueError("Duplicate raw content across historical roles")
    root = root.resolve(strict=True)
    records = []
    for index, f in enumerate(split["roles"]["train"]["files"]):
        path = safe_relative(root, f["path"])
        if sha256_file(path) != f["sha256"]:
            raise ValueError(f"CMS checksum mismatch: {f['path']}")
        selected, counts = [], np.zeros(15, np.int64)
        with uproot.open(path) as handle:
            key, tree = latest_tree(handle)
            if tree.num_entries != f["raw_entries"]:
                raise ValueError("Latest-cycle entries differ from historical split")
            types = tree.typenames()
            needed = set(CMS_SCALARS + cms_particle_branches("offline") + cms_particle_branches("hlt"))
            if not needed <= set(types):
                raise ValueError(f"Missing required CMS fields: {sorted(needed-set(types))}")
            for start in range(0, tree.num_entries, chunk_rows):
                arrays = tree.arrays(list(CMS_SCALARS), entry_start=start,
                                     entry_stop=start+chunk_rows, library="np")
                labels = multiclass_labels(arrays)
                keep = baseline_mask(arrays) & (labels >= 0)
                selected.append(np.flatnonzero(keep) + start)
                counts += np.bincount(labels[keep], minlength=15)
        entries = np.concatenate(selected) if selected else np.empty(0, np.int64)
        if len(entries) != f["mapped_entries"] or counts.tolist() != list(f["class_counts"]):
            raise ValueError("Historical selected CMS population changed")
        if sha256_file(path) != f["sha256"]:
            raise ValueError("CMS source changed during audit")
        records.append(dict(path=f["path"], source=f["stratum"], sha256=f["sha256"],
                            original_role="train", tree_key=key, raw_entries=f["raw_entries"],
                            selected_entries=len(entries), class_counts=counts.tolist(),
                            entry_mask=pack_entries(entries, f["raw_entries"]), schema=types))
        print(f"CMS2JC2 phase=inventory file={index+1} selected={len(entries)}", flush=True)
    return artifact("CMS_INVENTORY", parents={"historical_split": split_hash}, files=records,
                    selected_entries=sum(f["selected_entries"] for f in records),
                    particle_arrays_read=False, selection="historical_baseline_mapped_v1")


def validate_inventory(value: dict):
    validate(value, "CMS_INVENTORY")
    from .splits import unpack_entries
    if len({f["sha256"] for f in value["files"]}) != len(value["files"]):
        raise ValueError("Duplicate CMS content")
    if sum(f["selected_entries"] for f in value["files"]) != value["selected_entries"]:
        raise ValueError("CMS inventory count differs")
    for f in value["files"]:
        if f["original_role"] != "train":
            raise PermissionError("Original held-out source in calibration")
        if len(unpack_entries(f["entry_mask"], f["raw_entries"])) != f["selected_entries"]:
            raise ValueError("CMS mask count differs")


def audit_jc2_sources(root: Path, inventory: dict, profile: dict) -> dict:
    from hlt_classification.jetclass2_delphes.inventory import validate_inventory as validate_jc2, verify_file
    from hlt_classification.jetclass2_delphes.splits import validate_splits
    from .bridge import JC2_FIELDS
    ih = validate_jc2(inventory); ph = validate_splits(profile,inventory)
    if profile.get("profile") != "TRAIN_500K":
        raise ValueError("Expected the registered 500k transfer profile")
    allowed = {r["path"] for r in profile["groups"] if r["role"] in {"train","validation"}}
    checked = []
    for row in inventory["files"]:
        if row["path"] not in allowed:
            continue
        path = verify_file(root,row,inventory)
        with uproot.open(path) as handle:
            key,tree = latest_tree(handle)
            if key != row["tree_key"] or tree.num_entries != row["entries"]:
                raise ValueError("JC2 latest-cycle identity differs")
            needed = {"jet_nparticles",*("part_"+f for f in JC2_FIELDS)}
            if not needed <= set(tree.keys()):
                raise ValueError("Missing JC2 offline bridge fields")
        checked.append(dict(path=row["path"],sha256=row["sha256"]))
        print(f"CMS2JC2 phase=jc2_source_authentication file={len(checked)}",flush=True)
    return artifact("JC2_SOURCE_AUDIT",parents={"inventory":ih,"profile":ph},files=checked,
                    particle_arrays_read=False,native_hlt_required=False,
                    historical_hlt_matched_selection_disclosed=True)
