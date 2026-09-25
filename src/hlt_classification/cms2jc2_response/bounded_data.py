"""Separate, selection-claimed bounded confirmation capability.

Reader donor: dev_data.iter_sample at 7d56425f3bb88b6ea36bbc9bd80e81fb166e5bb2.
The original fit-only reader and production confirmation API remain unchanged.
"""
from contextlib import closing
from pathlib import Path

import awkward as ak
import numpy as np

from .audit import cms_particle_branches, latest_tree, validate_inventory
from .bridge import from_cms
from .contracts import artifact, canonical_sha256, load_json, safe_relative, validate, validate_compatibility
from .dev_data import _sample, _hash, checked_file
from .readers import Pair, authenticated_open
from .splits import pack_entries, unpack_entries, validate_roles


def metadata(study):
    return tuple(load_json(checked_file(study["imported"]["files"][name]))
                 for name in ("cms_inventory.json", "response_roles.json"))


def build_membership(study):
    from .bounded_campaign import CONFIRM_JETS
    inventory, roles = metadata(study)
    validate_inventory(inventory)
    validate_roles(roles, inventory)
    fit = [f for f in roles["files"] if f["response_role"] == "response_fit"]
    rows = [f for f in roles["files"] if f["response_role"] == "response_confirm"]
    if not rows or {r["source"] for r in rows} != {r["source"] for r in fit}:
        raise ValueError("Independent confirmation cannot cover fit-source mixture")
    if {r["sha256"] for r in rows} & {r["sha256"] for r in fit}:
        raise PermissionError("Confirmation source bytes overlap fitting")
    capacities = {s: sum(f["selected_entries"] for f in fit if f["source"] == s) for s in {r["source"] for r in fit}}
    members = _sample(rows, CONFIRM_JETS, "CMS2JC2_BOUNDED_CONFIRM/v1", capacities)
    files = {f["path"]: f for f in rows}
    entries = [(r["path"], int(e)) for r in members
               for e in unpack_entries(r["entry_mask"], files[r["path"]]["raw_entries"])]
    if len(entries) != CONFIRM_JETS:
        raise ValueError("Insufficient bounded confirmation capacity")
    entries.sort(key=lambda r: (_hash("CMS2JC2_BOUNDED_SHARD/v1", files[r[0]]["sha256"], r[1]), r))
    shards = []
    for i in range(4):
        subset = entries[i::4]
        shards.append([dict(path=p, selected_entries=sum(a == p for a, _ in subset),
            entry_mask=pack_entries(sorted(e for a, e in subset if a == p), files[p]["raw_entries"]))
            for p in sorted({p for p, _ in subset})])
    return artifact("BOUNDED_MEMBERSHIP", parents={"inventory": inventory["content_hash"], "roles": roles["content_hash"]},
        jets=CONFIRM_JETS, outer_role="response_confirm", members=members, shards=shards,
        source_capacities=capacities, particle_accessed=False, labels_accessed=False,
        larger_production_confirmation_not_performed=True)


def validate_membership(value, study):
    validate(value, "BOUNDED_MEMBERSHIP")
    if value != build_membership(study):
        raise PermissionError("Bounded confirmation membership escapes canonical population")


def claim_value(spec, study):
    from .bounded_campaign import selection
    if spec["stage"] != "bounded_confirm":
        raise PermissionError("No confirmation capability in development stages")
    locked = selection(load_json(checked_file(spec["parent_spec"])))
    validate_membership(spec["membership"], study)
    _, roles = metadata(study)
    if spec["selection_hash"] != locked["content_hash"]:
        raise PermissionError("Independent confirmation selection differs")
    return artifact("BOUNDED_CONFIRMATION_CLAIM", parents={"stage": spec["content_hash"],
        "selection": locked["content_hash"], "membership": spec["membership"]["content_hash"],
        "roles": roles["content_hash"], "compatibility": study["review"]["content_hash"]},
        selected=locked["selected"], selected_response=locked["selected_model"]["content_hash"],
        controls=["B"], no_reselection=True, bounded_confirmation_only=True)


def acquire_claim(spec, study):
    from .dev_campaign import write
    value = claim_value(spec, study)
    write(spec["root"], f"stages/{spec['name']}/confirmation_access.json", value, "BOUNDED_CONFIRMATION_CLAIM")
    return value


def iter_confirmation(spec, study, claim, *, shard):
    from .dev_campaign import stage_dir
    if type(shard) is not int or not 0 <= shard < 4:
        raise PermissionError("Unregistered confirmation shard")
    expected = claim_value(spec, study)
    if claim != expected or load_json(stage_dir(spec)/"confirmation_access.json") != expected:
        raise PermissionError("Separate published confirmation claim required before particle access")
    inventory, roles = metadata(study)
    validate_compatibility(study["review"], inventory_hash=inventory["content_hash"])
    files = {f["path"]: f for f in roles["files"] if f["response_role"] == "response_confirm"}
    branches = cms_particle_branches("offline")+cms_particle_branches("hlt")
    for row in spec["membership"]["shards"][shard]:
        f = files[row["path"]]
        selected = unpack_entries(row["entry_mask"], f["raw_entries"])
        with authenticated_open(safe_relative(Path(study["imported"]["cms_root"]), f["path"]), f["sha256"]) as handle:
            key, tree = latest_tree(handle)
            if key != f["tree_key"] or tree.num_entries != f["raw_entries"]:
                raise ValueError("Confirmation ROOT identity changed")
            for start in sorted(set((selected//256*256).tolist())):
                entries = selected[np.searchsorted(selected, start):np.searchsorted(selected, start+256)]
                arrays = tree.arrays(list(branches), entry_start=start, entry_stop=start+256, library="ak", how=dict)
                for entry in entries:
                    columns = {name: ak.to_numpy(arrays[name][int(entry)-start]) for name in branches}
                    identity = canonical_sha256(["CMS2JC2_ROW/v1", f["sha256"], key, int(entry)])
                    yield Pair(identity, f["sha256"], from_cms(columns, study["review"], side="offline"),
                               from_cms(columns, study["review"], side="hlt"))


def stream(spec, study, claim, *, shard):
    return closing(iter_confirmation(spec, study, claim, shard=shard))
