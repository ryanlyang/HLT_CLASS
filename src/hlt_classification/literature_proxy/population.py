"""Label-blind train selection and a strictly offline-only ROOT capability."""
from __future__ import annotations

import hashlib
import heapq
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from hlt_classification.cms2jc2_response.bridge import JC2_FIELDS, Particles
from hlt_classification.jetclass2_delphes.contracts import row_identity
from hlt_classification.jetclass2_delphes.inventory import validate_inventory, verify_file
from hlt_classification.jetclass2_delphes.split_registry import validate_split_profile, unpack_entries
from .contracts import artifact, sha256_file
from .kernel import SEED

BRANCHES = ("jet_nparticles", *("part_" + f for f in JC2_FIELDS))


def select(inventory, profile, count):
    validate_inventory(inventory)
    validate_split_profile(profile, inventory)
    if type(count) is not int or not 1 <= count <= min(20_000, profile["role_counts"]["train"]):
        raise ValueError("Pilot requires 1..20000 registered training jets")
    files = {r["path"]: r for r in inventory["files"]}

    def candidates():
        for member in profile["memberships"]["train"]["files"]:
            record = files[member["path"]]
            for entry in unpack_entries(member["entry_mask"], record["entries"]):
                identity = row_identity(inventory["content_hash"], record["path"], record["tree_key"], int(entry))
                rank = hashlib.sha256(f"JC2_LIT_PILOT/v1/{SEED}/{identity}".encode()).digest()
                yield rank, record["path"], int(entry), identity

    chosen = sorted(heapq.nsmallest(count, candidates()), key=lambda r: (r[1], r[2]))
    selected = []
    for path in sorted({r[1] for r in chosen}):
        rows = [r for r in chosen if r[1] == path]
        selected.append(dict(path=path, entries=[r[2] for r in rows], identities=[r[3] for r in rows]))
    return artifact("POPULATION", parents=dict(inventory=inventory["content_hash"], profile=profile["content_hash"]),
                    role="train", jets=count, seed=SEED, files=selected,
                    selection="smallest_sha256_label_blind_within_existing_train_v1",
                    native_hlt_particles_accessed=False, validation_accessed=False, final_test_accessed=False)


def from_columns(columns):
    if set(columns) != set(JC2_FIELDS):
        raise ValueError("Offline physical branch set differs")
    a = np.column_stack([columns[f] for f in JC2_FIELDS])
    if a.ndim != 2 or a.shape[1] != 14:
        raise ValueError("Physical particle shape differs")
    flags = a[:, 5:10]
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("Nonbinary source PID")
    category = np.where(flags.sum(axis=1) == 1, flags.argmax(axis=1), 5)
    tracking = a[:, [10, 12, 11, 13]].copy()
    if np.any(np.isfinite(tracking[:, 2:]) & (tracking[:, 2:] < 0)):
        raise ValueError("Negative source uncertainty")
    valid = np.isfinite(tracking) & (a[:, 4, None] != 0)
    valid[:, 2:] &= (tracking[:, 2:] > 0) & valid[:, :2]
    tracking[~valid] = 0.
    return Particles(a[:, :4], a[:, 4], category, tracking, valid,
                     tuple(f"part:{i:08d}" for i in range(len(a))))


def read_file(data_root, inventory, profile, selected, *, step=512):
    """Authenticate membership before opening a ROOT file, even when used alone."""
    validate_inventory(inventory)
    validate_split_profile(profile, inventory)
    records = {r["path"]: r for r in inventory["files"]}
    members = {r["path"]: r for r in profile["memberships"]["train"]["files"]}
    name = selected["path"]
    if name not in members:
        raise PermissionError("Only training source files may be opened")
    record = records[name]
    entries = selected["entries"]
    allowed = set(unpack_entries(members[name]["entry_mask"], record["entries"]).tolist())
    ids = [row_identity(inventory["content_hash"], name, record["tree_key"], e) for e in entries]
    if (not entries or any(type(e) is not int for e in entries) or entries != sorted(set(entries))
            or not set(entries) <= allowed or ids != selected["identities"] or type(step) is not int or step < 1):
        raise ValueError("Selected training identities/entries differ")
    path = verify_file(Path(data_root), record, inventory)
    try:
        with uproot.open(path) as handle:
            tree = handle[record["tree_key"]]
            # Bucket only selected entries; never traverse non-selected baskets intentionally.
            buckets = {}
            for entry, identity in zip(entries, ids):
                buckets.setdefault(entry // step, []).append((entry, identity))
            for bucket, rows in buckets.items():
                start = bucket * step
                arrays = tree.arrays(list(BRANCHES), entry_start=start,
                                     entry_stop=min(start + step, record["entries"]), library="ak", how=dict)
                for entry, identity in rows:
                    i = entry - start
                    n = int(arrays["jet_nparticles"][i])
                    columns = {f: ak.to_numpy(arrays["part_" + f][i]) for f in JC2_FIELDS}
                    if n < 0 or any(len(v) != n for v in columns.values()):
                        raise ValueError("Offline jagged particle count differs")
                    yield identity, from_columns(columns)
    finally:
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("Training source changed during reading")
