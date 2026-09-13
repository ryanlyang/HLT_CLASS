"""Explicit row capabilities inside the frozen outer split (no test executor)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import awkward as ak
import numpy as np
import uproot

from ..contracts import row_identity
from ..inventory import validate_inventory, verify_file
from ..reader import Jet, _particles
from ..schema import PARTICLE_FIELDS
from ..selection import selected_mask
from ..split_registry import validate_split_profile, unpack_entries, nested_quotas
from .contracts import (artifact, validate, REGISTRY, OUTER_VAL, OUTER_TEST, ROWS,
                        publish_arrays, checked_payload, load_npz_arrays)
from hlt_classification.data.cache_contracts import array_sha256


def authenticate_profile(inventory, profile, *, production=True):
    validate_inventory(inventory)
    validate_split_profile(profile, inventory)
    if production and (profile["registry_sha256"] != REGISTRY or profile["profile"] != "TRAIN_500K"
            or profile["memberships"]["validation"]["content_hash"] != OUTER_VAL
            or profile["memberships"]["final_test"]["content_hash"] != OUTER_TEST
            or profile["role_counts"] != dict(train=500000, validation=1000000, final_test=1000000)):
        raise ValueError("Auxiliary study requires the exact frozen 500k/1M/1M split")


def _metadata(data_root, inventory, membership):
    blocks = []
    files = {row["path"]: row for row in membership["files"]}
    for file_index, file in enumerate(inventory["files"]):
        if file["path"] not in files:
            continue
        member = files[file["path"]]
        entries = unpack_entries(member["entry_mask"], file["entries"])
        if not len(entries):
            continue
        path = verify_file(data_root, file, inventory)
        with uproot.open(path) as handle:
            arrays = handle[file["tree_key"]].arrays(["jet_label", "hlt_matched"], library="np")
        keep, labels = selected_mask(arrays["jet_label"], arrays["hlt_matched"],
                                     file["source"], inventory["selection"])
        if not keep[entries].all() or np.bincount(labels[entries], minlength=11).tolist() != member["class_counts"]:
            raise ValueError("Frozen member eligibility/class counts differ")
        identities = np.array([np.frombuffer(bytes.fromhex(row_identity(
            inventory["content_hash"], file["path"], file["tree_key"], int(e))), np.uint8)
            for e in entries])
        blocks.append(dict(file_index=np.full(len(entries), file_index, np.int32),
                           entry=entries.astype(np.int64), labels=labels[entries].astype(np.int64),
                           identities=identities))
        verify_file(data_root, file, inventory)
    if not blocks:
        raise ValueError("Empty role")
    result = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    if len(result["labels"]) != membership["rows"]:
        raise ValueError("Metadata membership count differs")
    return result


def selection_indices(labels, identities, outer_hash, size):
    counts = np.bincount(labels, minlength=11)
    quotas = nested_quotas(counts.tolist(), [size])[0]
    if np.any(np.asarray(quotas) >= counts):
        raise ValueError("Both selection and report require every class")
    priorities = [hashlib.sha256(
        f"JC2/offline-aux/v1/val-select/20260912/{outer_hash}/{bytes(i).hex()}".encode()
    ).digest() for i in identities]
    chosen = []
    for label, quota in enumerate(quotas):
        # Stable original index is the canonical file/entry tie-break.
        group = np.flatnonzero(labels == label)
        chosen.extend(sorted(group, key=lambda i: (priorities[i], int(i)))[:quota])
    return np.sort(chosen)


def build_roles(data_root, inventory, profile, output_root, *, production=True, select_rows=200000):
    # Study specs serialize paths as strings; the native verifier takes Path.
    data_root = Path(data_root)
    authenticate_profile(inventory, profile, production=production)
    if production and select_rows != 200000:
        raise ValueError("Production VAL_SELECT must have 200000 rows")
    train = _metadata(data_root, inventory, profile["memberships"]["train"])
    val = _metadata(data_root, inventory, profile["memberships"]["validation"])
    chosen = selection_indices(val["labels"], val["identities"],
                               profile["memberships"]["validation"]["content_hash"], select_rows)
    mask = np.zeros(len(val["labels"]), bool)
    mask[chosen] = True
    roles = {}
    for role, arrays in (("TRAIN", train), ("VAL_SELECT", {k: v[mask] for k, v in val.items()}),
                         ("VAL_REPORT", {k: v[~mask] for k, v in val.items()})):
        roles[role] = dict(rows=len(arrays["labels"]),
                           class_counts=np.bincount(arrays["labels"], minlength=11).tolist(),
                           payload=publish_arrays(output_root, role + ".npz", arrays),
                           array_sha256={k: array_sha256(k, v) for k, v in arrays.items()},
                           identity_sha256=hashlib.sha256(arrays["identities"].tobytes()).hexdigest())
    return artifact("ROLE_SPLIT", inventory_sha256=inventory["content_hash"],
                    profile_sha256=profile["content_hash"], roles=roles,
                    outer_validation_sha256=profile["memberships"]["validation"]["content_hash"],
                    production=production, selection_rows=select_rows,
                    boundary="selected_row_interpretation_no_report_or_test_iteration")


def load_role(root, split, role):
    validate(split, "ROLE_SPLIT")
    if role not in ROWS:
        raise PermissionError("No final-test role")
    row = split["roles"][role]
    arrays = load_npz_arrays(checked_payload(root, row["payload"]))
    n = row["rows"]
    if (set(arrays) != {"file_index", "entry", "labels", "identities"}
            or arrays["identities"].dtype != np.uint8 or arrays["identities"].shape != (n, 32)
            or any(arrays[k].shape != (n,) for k in ("file_index", "entry", "labels"))
            or np.bincount(arrays["labels"], minlength=11).tolist() != row["class_counts"]
            or hashlib.sha256(arrays["identities"].tobytes()).hexdigest() != row["identity_sha256"]):
        raise ValueError("Role payload shape/identity/class coverage differs")
    if split["production"] and n != ROWS[role]:
        raise ValueError("Production role row count differs")
    pairs = np.rec.fromarrays([arrays["file_index"], arrays["entry"]])
    if len(np.unique(pairs)) != n or not np.array_equal(np.argsort(pairs, kind="stable"), np.arange(n)):
        raise ValueError("Role rows not unique canonical file/entry order")
    return arrays


def authorize_role(role, split, reporting_lock=None):
    if role not in ROWS:
        raise PermissionError("No auxiliary final-test access")
    if role == "VAL_REPORT":
        if reporting_lock is None:
            raise PermissionError("VAL_REPORT requires all twelve frozen confirmation models")
        validate(reporting_lock, "REPORTING_LOCK")
        if (reporting_lock["role_split_sha256"] != split["content_hash"]
                or len(reporting_lock["models"]) != 12
                or set(reporting_lock["models"]) != {
                    f"CONFIRM_{i:02d}_{arm}" for i in range(1, 4) for arm in ("CE", "COMP", "STRUCT", "BOTH")}):
            raise PermissionError("Reporting lock coverage differs")


def read_rows(data_root, inventory, split, metadata, *, role, include_offline,
              reporting_lock=None, file_index=None, step_size=2048, row_start=0, row_end=None):
    """Decode only authorized row objects. ROOT baskets may physically contain
    other rows, but these are never yielded, transformed, fitted or scored.
    The caller must load metadata through load_role, not a generic val reader.
    """
    authorize_role(role, split, reporting_lock)
    # Also covers independently spawned target and HLT-cache workers.
    data_root = Path(data_root)
    if {k: array_sha256(k, v) for k, v in metadata.items()} != split["roles"][role]["array_sha256"]:
        raise PermissionError("Reader metadata is not the authorized row capability")
    row_end = len(metadata["labels"]) if row_end is None else row_end
    if not 0 <= row_start < row_end <= len(metadata["labels"]):
        raise ValueError("Invalid bounded reader range")
    if split["inventory_sha256"] != inventory["content_hash"]:
        raise ValueError("Reader inventory parent differs")
    for fi in np.unique(metadata["file_index"]):
        if file_index is not None and fi != file_index:
            continue
        file = inventory["files"][int(fi)]
        index = np.flatnonzero(metadata["file_index"] == fi)
        index = index[(index >= row_start) & (index < row_end)]
        if not len(index):
            continue
        entries = metadata["entry"][index]
        path = verify_file(data_root, file, inventory)
        branches = ["jet_label", "hlt_matched", "hlt_jet_nparticles"] + ["hlt_part_" + f for f in PARTICLE_FIELDS]
        if include_offline:
            branches += ["jet_nparticles"] + ["part_" + f for f in PARTICLE_FIELDS]
        with uproot.open(path) as handle:
            tree = handle[file["tree_key"]]
            for start in range(0, file["entries"], step_size):
                lo, hi = np.searchsorted(entries, [start, start + step_size])
                if hi == lo:
                    continue
                arrays = tree.arrays(branches, entry_start=start, entry_stop=start + step_size, library="ak", how=dict)
                keep, labels = selected_mask(ak.to_numpy(arrays["jet_label"]), ak.to_numpy(arrays["hlt_matched"]),
                                             file["source"], inventory["selection"])
                for pos in range(lo, hi):
                    at, entry = int(index[pos]), int(entries[pos])
                    local = entry - start
                    identity = row_identity(inventory["content_hash"], file["path"], file["tree_key"], entry)
                    if (not keep[local] or labels[local] != metadata["labels"][at]
                            or bytes.fromhex(identity) != metadata["identities"][at].tobytes()):
                        raise ValueError("Reader row identity/eligibility differs")
                    hlt = _particles(arrays, local, "hlt_part_", int(arrays["hlt_jet_nparticles"][local]))
                    offline = (_particles(arrays, local, "part_", int(arrays["jet_nparticles"][local]))
                               if include_offline else None)
                    yield at, Jet(identity, int(labels[local]), hlt, offline)
        verify_file(data_root, file, inventory)
