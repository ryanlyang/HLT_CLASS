"""Role-bounded CMS paired reader and a genuinely offline-only JC2 route."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import awkward as ak
import numpy as np
import uproot

from .audit import cms_particle_branches, validate_inventory, latest_tree
from .bridge import Particles, JC2_FIELDS, from_cms, from_jc2
from .contracts import canonical_sha256, safe_relative, sha256_file, validate, validate_compatibility
from .splits import unpack_entries, validate_roles, validate_memberships


@dataclass(frozen=True)
class Pair:
    identity: str
    source_group: str
    offline: Particles
    hlt: Particles | None
    diagnostic_class: int | None = None
    ingestion_audit: dict | None = None


@contextmanager
def authenticated_open(path: Path, digest: str):
    if sha256_file(path) != digest:
        raise ValueError("Response source checksum differs")
    try:
        with uproot.open(path) as handle:
            yield handle
    finally:
        # Also authenticate a bounded consumer that closes the generator early.
        if sha256_file(path) != digest:
            raise ValueError("Response input changed during iteration")


def iter_cms(root: Path, inventory: dict, roles: dict, review: dict, *, role: str,
             fit_role: str | None = None, membership: dict | None = None,
             budget: str = "FULL", file_paths: tuple[str, ...] | None = None,
             confirmation_claim: dict | None = None, selection: dict | None = None,
             chunk_rows: int = 256, diagnostic_labels: bool = False,
             fit_probe_limit: int | None = None) -> Iterator[Pair]:
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    validate_compatibility(review, inventory_hash=inventory["content_hash"])
    if role not in {"response_fit", "response_select", "response_confirm"}:
        raise PermissionError("Forbidden response role")
    if chunk_rows < 1 or fit_role not in {None, "fit_location", "fit_residual"}:
        raise ValueError("Invalid CMS reader parameters")
    if fit_role is not None and role != "response_fit":
        raise PermissionError("Internal fit role cannot alias held-out data")
    if diagnostic_labels and role == "response_fit":
        raise PermissionError("Response fitting has no classification-label capability")
    if fit_probe_limit is not None and (role != "response_fit" or fit_role is None
            or type(fit_probe_limit) is not int or fit_probe_limit < 1):
        raise PermissionError("Bounded worker probes are internal-fit-only")
    if role == "response_confirm":
        if confirmation_claim is None or selection is None:
            raise PermissionError("Locked selection and separate confirmation claim required")
        validate(selection, "SELECTION")
        validate(confirmation_claim, "CONFIRMATION_CLAIM", parents={
            "roles": roles["content_hash"], "selection": selection["content_hash"],
            "compatibility": review["content_hash"],
        })
    rows = [r for r in roles["files"] if r["response_role"] == role
            and (fit_role is None or r["fit_role"] == fit_role)]
    allowed = {r["path"] for r in rows}
    if file_paths is not None:
        if len(set(file_paths)) != len(file_paths) or not set(file_paths) <= allowed:
            raise PermissionError("CMS source subset escapes allowed role")
        rows = [r for r in rows if r["path"] in file_paths]
    masks = {}
    if membership is not None:
        validate_memberships(membership, roles)
        if role != "response_fit" or fit_role is None:
            raise PermissionError("Learning-curve membership is fit-only")
        if budget not in {"250K", "1M", "FULL"}:
            raise ValueError("Unknown fit budget")
        masks = {r["path"]: r for r in membership["budgets"][budget][fit_role]}
    elif budget != "FULL":
        raise ValueError("Smaller fit requires frozen membership")
    particle_branches = cms_particle_branches("offline") + cms_particle_branches("hlt")
    from hlt_classification.scouting.schema import LABEL_BRANCHES
    from hlt_classification.scouting.labels import multiclass_labels
    branches = particle_branches + (tuple(LABEL_BRANCHES) if diagnostic_labels else ())
    for row in rows:
        path = safe_relative(root, row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("CMS source checksum differs")
        declared = masks[row["path"]] if membership is not None else row
        entries = unpack_entries(declared["entry_mask"], row["raw_entries"])
        if len(entries) != declared["selected_entries"]:
            raise ValueError("Membership count differs")
        if fit_probe_limit is not None:
            if fit_probe_limit > len(entries):
                raise ValueError("Probe exceeds authenticated file capacity")
            # Prefixes in file order are not a representative hash-selected probe.
            entries = np.sort(sorted(entries, key=lambda e: canonical_sha256([
                "CMS2JC2_FIT_PROBE/v1", row["sha256"], row["tree_key"], int(e)]))[:fit_probe_limit])
        original = unpack_entries(row["entry_mask"], row["raw_entries"])
        if not np.isin(entries, original, assume_unique=True).all():
            raise ValueError("Fit subset escapes historical eligible rows")
        with authenticated_open(path, row["sha256"]) as handle:
            key, tree = latest_tree(handle)
            if key != row["tree_key"] or tree.num_entries != row["raw_entries"]:
                raise ValueError("CMS tree identity differs")
            for start in range(0, row["raw_entries"], chunk_rows):
                selected = entries[np.searchsorted(entries, start):np.searchsorted(entries, start+chunk_rows)]
                if len(selected) == 0:
                    continue
                arrays = tree.arrays(list(branches), entry_start=start, entry_stop=start+chunk_rows,
                                     library="ak", how=dict)
                labels = (multiclass_labels({k: ak.to_numpy(arrays[k]) for k in LABEL_BRANCHES})
                          if diagnostic_labels else None)
                for entry in selected:
                    columns = {field: ak.to_numpy(arrays[field][int(entry)-start]) for field in particle_branches}
                    label = int(labels[int(entry)-start]) if labels is not None else None
                    if label is not None and label < 0:
                        raise ValueError("Eligible CMS diagnostic label became unmapped")
                    identity = canonical_sha256(["CMS2JC2_ROW/v1", row["sha256"], key, int(entry)])
                    yield Pair(identity, row["sha256"], from_cms(columns, review, side="offline"),
                               from_cms(columns, review, side="hlt"), label,
                               dict(raw_offline_charged=len(columns["cpfcandlt_px"]),
                                    raw_offline_neutral=len(columns["npfcand_px"]),
                                    excluded_lost_tracks=int(np.count_nonzero(columns["cpfcandlt_isLostTrack"])),
                                    raw_hlt_particles=len(columns["scoutpfcand_px"])))


def iter_jc2_offline(root: Path, inventory: dict, profile: dict, review: dict, *, role: str,
                     file_paths: tuple[str, ...] | None = None, chunk_rows: int = 256) -> Iterator[Pair]:
    from hlt_classification.jetclass2_delphes.inventory import validate_inventory as validate_jc2, verify_file
    from hlt_classification.jetclass2_delphes.splits import validate_splits, is_subset_profile
    from hlt_classification.jetclass2_delphes.split_registry import unpack_entries as unpack_jc2
    from hlt_classification.jetclass2_delphes.contracts import row_identity
    if role not in {"train", "validation"}:
        raise PermissionError("JetClass2 final test remains sealed")
    validate_compatibility(review)
    ih = validate_jc2(inventory)
    validate_splits(profile, inventory)
    if (not is_subset_profile(profile) or profile.get("profile") != "TRAIN_500K"
            or chunk_rows < 1):
        raise ValueError("Offline-only transfer requires frozen explicit memberships")
    groups = {r["path"] for r in profile["groups"] if r["role"] == role}
    if file_paths is not None and (len(set(file_paths)) != len(file_paths) or not set(file_paths) <= groups):
        raise PermissionError("JC2 subset escapes role")
    members = {r["path"]: r for r in profile["memberships"][role]["files"]}
    # Do NOT reuse DatasetReader: it reads hlt_matched and native HLT particle arrays.
    branches = ["jet_nparticles"] + ["part_" + f for f in JC2_FIELDS]
    for row in inventory["files"]:
        if row["path"] not in groups or (file_paths is not None and row["path"] not in file_paths):
            continue
        path = verify_file(root, row, inventory)
        entries = unpack_jc2(members[row["path"]]["entry_mask"], row["entries"])
        with authenticated_open(path, row["sha256"]) as handle:
            tree = handle[row["tree_key"]]
            for start in range(0, row["entries"], chunk_rows):
                selected = entries[np.searchsorted(entries, start):np.searchsorted(entries, start+chunk_rows)]
                if len(selected) == 0:
                    continue
                arrays = tree.arrays(branches, entry_start=start, entry_stop=start+chunk_rows,
                                     library="ak", how=dict)
                for entry in selected:
                    i = int(entry) - start
                    cols = {f: ak.to_numpy(arrays["part_"+f][i]) for f in JC2_FIELDS}
                    n = int(arrays["jet_nparticles"][i])
                    if any(len(v) != n for v in cols.values()):
                        raise ValueError("JC2 offline jagged count differs")
                    identity = row_identity(ih, row["path"], row["tree_key"], int(entry))
                    yield Pair(identity, row["sha256"], from_jc2(cols, review,
                               keys=tuple(f"part:{j}" for j in range(n))), None)
