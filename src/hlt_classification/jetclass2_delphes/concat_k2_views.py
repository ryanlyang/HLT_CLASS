"""Capacity-two assignment and fixed-support, source-anonymous concatenation.

Salience is frozen on native endpoints BEFORE retention and slot duplication.
Only the new D000 endpoint is native HLT x3; historical D000 is unchanged.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib

import numpy as np

from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import CANDIDATES, matcher_spec
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    production_pairing_from_matrices, reference_pairing_from_matrices,
)
from .contracts import artifact
from .reader import Jet, Particles
from .salience_views import pairing_matrices

STRENGTHS = {"D100": Fraction(1), "D075": Fraction(3, 4), "D050": Fraction(1, 2),
             "D025": Fraction(1, 4), "D000": Fraction(0)}


def view_contract(candidate):
    if candidate not in CANDIDATES:
        raise ValueError("Unregistered salience formula")
    return artifact("CONCAT_K2_VIEWS", k=2, total_copies=3, salience=matcher_spec(candidate),
        pool="native_offline_not_persistent_u000", retention="top_frozen_salience_native_index_tie",
        support="three_active_slots_per_native_hlt_original_then_two_rich",
        assignment="exact_global_duplicate_hlt_destinations_frozen_endpoint_salience",
        objectives=["maximum_sum_utility", "minimum_sum_uncapped_qdr",
                    "maximum_sum_occupied_hlt_slot_salience", "minimum_sum_qresponse",
                    "minimum_category_mismatches", "minimum_valid_charge_mismatches",
                    "lexicographic_native_offline_per_destination_unmatched_last"],
        filler="native_hlt_owner_active_from_start", endpoint="hlt_only_repeat_three_rebuild_all_features",
        interpolation="linear_p4_applicable_numeric_atomic_identity_and_uncertainty_v1",
        hash_domain="JC2/CONCAT_K2/v1/measurement", source_features=False,
        model_order="interleaved_owner_original_rich1_rich2_no_position_embedding",
        strengths={k: [v.numerator, v.denominator] for k, v in STRENGTHS.items()},
        cropping_rejects_jets=False, final_test_accessed=False)


def assignment_problem(hlt: Particles, offline: Particles, candidate: str):
    if not len(hlt) or not len(offline):
        raise ValueError("Both native endpoints must be nonempty")
    view_contract(candidate)
    base = pairing_matrices(hlt, offline, candidate)
    return _problem(base, len(hlt), len(offline))


def _problem(base, nh, no):
    retained = np.array(sorted(sorted(range(no),
        key=lambda j: (-int(base["offline_salience"][j]), j))[:2*nh]), np.int32)
    # Reuse the exact integer objective solver, never the old particle maps.
    # Every retained offline particle is on the smaller side. Slot salience
    # counts each occupied HLT destination, not each unique HLT owner.
    arrays = {k: np.repeat(base[k][:, retained], 2, axis=0)
              for k in ("utility", "qdr", "qresponse")}
    arrays.update({k: np.repeat(base[k], 2) for k in
                   ("hlt_salience", "hlt_category", "hlt_charge")})
    arrays.update({k: base[k][retained] for k in
                   ("offline_salience", "offline_category", "offline_charge")})
    arrays["native_offline_index"] = retained.astype(np.int64)
    return arrays, retained


def validate_mapping(mapping, *, nh: int, no: int, retained=None):
    value = np.asarray(mapping)
    if value.dtype != np.int32 or value.shape != (nh, 2) or nh < 1 or no < 1:
        raise ValueError("Capacity-two mapping shape/type differs")
    used = value[value >= 0]
    if (np.any(value < -1) or np.any(used >= no) or len(used) != min(no, 2*nh)
            or len(np.unique(used)) != len(used)):
        raise ValueError("Capacity-two assignment coverage/uniqueness differs")
    ordered = np.where(value < 0, no, value)
    if np.any(ordered[:, 0] > ordered[:, 1]):
        raise ValueError("Exchangeable rich slots must be in canonical native order")
    if retained is not None and not np.array_equal(np.sort(used), np.sort(retained)):
        raise ValueError("Retained offline identity set differs")
    return value


def match_particles(hlt, offline, candidate, *, reference=False, diagnostics=False):
    if not len(hlt) or not len(offline):
        raise ValueError("Both native endpoints must be nonempty")
    view_contract(candidate)
    base = pairing_matrices(hlt, offline, candidate)
    arrays, retained = _problem(base, len(hlt), len(offline))
    solver = reference_pairing_from_matrices if reference else production_pairing_from_matrices
    selected = solver(**arrays)
    mapping = np.full((len(hlt), 2), -1, np.int32)
    flat = mapping.reshape(-1)
    mask = selected >= 0
    flat[mask] = retained[selected[mask]]
    validate_mapping(mapping, nh=len(hlt), no=len(offline), retained=retained)
    return (mapping, base) if diagnostics else mapping


def hlt_x3(hlt):
    if len(hlt) < 1:
        raise ValueError("Empty native HLT")
    return Particles(np.repeat(hlt.values, 3, axis=0))


def deployment_inputs(hlt: Particles, *, copies: int, capacity: int):
    """Standalone adapter: no jet identity, offline view, matches or foundation."""
    from .inputs import build_inputs
    if type(copies) is not int or copies not in (1,3):
        raise ValueError("Only the registered HLT x1/x3 deployments are supported")
    return build_inputs(hlt if copies==1 else hlt_x3(hlt),capacity=capacity)


@lru_cache(maxsize=3)
def _view_digest(candidate):
    return view_contract(candidate)["content_hash"]


def _choice(identity, candidate, owner, slot, group, strength):
    fields = [_view_digest(candidate), identity, str(owner), str(slot), group]
    digest = hashlib.sha256(b"".join(len(s.encode()).to_bytes(4, "little") + s.encode()
                                     for s in fields)).digest()
    return int.from_bytes(digest[:8], "big") * strength.denominator < strength.numerator * (1 << 64)


def build_view(jet: Jet, coordinate: str, candidate: str, mapping=None):
    if coordinate == "HLT_X1":
        return jet.hlt
    if coordinate in {"HLT_X3", "D000"}:
        # Deliberately do not inspect candidate, mapping, offline, or counts.
        return hlt_x3(jet.hlt)
    if jet.offline is None:
        raise PermissionError("Rich view requires explicitly supplied offline particles")
    if coordinate == "OFFLINE":
        return jet.offline
    if coordinate not in STRENGTHS:
        raise ValueError("Unregistered fixed-slot coordinate")
    view_contract(candidate)
    if mapping is None:
        raise ValueError("Rich views require authenticated capacity-two assignments")
    hlt, offline = jet.hlt, jet.offline
    validate_mapping(mapping, nh=len(hlt), no=len(offline))
    output = np.repeat(hlt.values, 3, axis=0)
    a = STRENGTHS[coordinate]
    for owner, slot in zip(*np.nonzero(mapping >= 0)):
        h, o = hlt.values[owner], offline.values[mapping[owner, slot]]
        target = output[3*owner + 1 + slot]
        if a == 1:
            target[:] = o
            continue
        target[:4] = (float(a)*o[:4].astype(np.float64) + float(1-a)*h[:4].astype(np.float64))
        offline_identity = _choice(jet.identity, candidate, owner, slot, "identity", a)
        target[4:10] = (o if offline_identity else h)[4:10]
        for v, e, group in ((10, 11, "d0"), (12, 13, "dz")):
            if (h[4] != 0) != (o[4] != 0):
                target[v:e+1] = (o if offline_identity else h)[v:e+1]
            elif (h[e] > 0) != (o[e] > 0):
                target[v:e+1] = (o if _choice(jet.identity, candidate, owner, slot, group, a) else h)[v:e+1]
            else:
                target[v:e+1] = float(a)*o[v:e+1].astype(np.float64) + float(1-a)*h[v:e+1].astype(np.float64)
    return Particles(output)
