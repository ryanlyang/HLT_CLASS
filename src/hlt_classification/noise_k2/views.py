"""Capacity-two matching on physical proxy particles, preserving validity/mm."""
from fractions import Fraction
from functools import lru_cache
import hashlib

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles, wrap_phi
from hlt_classification.cms_proxy_ladder.inputs import build_inputs
from hlt_classification.cms_proxy_ladder.views import _scouting
from hlt_classification.jetclass2_delphes.concat_k2_views import _problem, validate_mapping
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_matcher import (
    canonical_qdr, canonical_qabs_log_pt_response,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import matcher_spec
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    particle_salience, pair_utility, production_pairing_from_matrices,
    reference_pairing_from_matrices,
)
from .contracts import artifact, require

STRENGTHS = {"D100": Fraction(1), "D075": Fraction(3, 4), "D050": Fraction(1, 2),
             "D025": Fraction(1, 4), "D000": Fraction(0)}


@lru_cache(maxsize=3)
def view_contract(candidate):
    return artifact("VIEWS", matcher=matcher_spec(candidate), k=2, copies=3,
        physical_schema="p4_charge_category_tracking_valid_mm",
        support="original_proxy_then_two_rich_or_active_owner_fillers",
        retention="top_complete_endpoint_salience_native_index_tie",
        assignment="existing_exact_capacity_two_integer_objective_v1",
        interpolation="linear_p4_jointly_valid_numeric_atomic_identity_validity_groups",
        endpoint="proxy_only_x3_rebuild_features", source_features=False)


def match(proxy, offline, candidate, *, reference=False):
    require(len(proxy) > 0 and len(offline) > 0, "Empty matching endpoint")
    qdr = canonical_qdr(np.hypot(proxy.eta[:, None]-offline.eta,
                               wrap_phi(proxy.phi[:, None]-offline.phi)))
    hw = particle_salience(_scouting(proxy, offline=False), candidate)
    ow = particle_salience(_scouting(offline, offline=True), candidate)
    base = dict(qdr=qdr, qresponse=canonical_qabs_log_pt_response(np.log(proxy.pt[:, None]/offline.pt)),
        hlt_salience=hw, offline_salience=ow, utility=pair_utility(qdr, hw, ow),
        hlt_category=proxy.category, offline_category=offline.category,
        hlt_charge=proxy.charge, offline_charge=offline.charge)
    arrays, retained = _problem(base, len(proxy), len(offline))
    solver = reference_pairing_from_matrices if reference else production_pairing_from_matrices
    chosen = solver(**arrays)
    mapping = np.full((len(proxy), 2), -1, np.int32)
    used = chosen >= 0
    mapping.reshape(-1)[used] = retained[chosen[used]]
    validate_mapping(mapping, nh=len(proxy), no=len(offline), retained=retained)
    return mapping, base


def repeat(proxy, copies=3):
    require(type(copies) is int and copies in (1, 3) and len(proxy) > 0, "Invalid deployment copies/support")
    if copies == 1:
        return proxy
    return Particles(*(np.repeat(getattr(proxy, k), copies, axis=0)
        for k in ("p4", "charge", "category", "tracking", "valid")),
        tuple(f"slot:{i}" for i in range(copies*len(proxy))))


def deployment_inputs(proxy, *, copies, capacity):
    return build_inputs(repeat(proxy, copies), capacity=capacity)


def choose(identity, candidate, owner, slot, group, alpha):
    fields = [view_contract(candidate)["content_hash"], identity, str(owner), str(slot), group]
    digest = hashlib.sha256(b"".join(len(s.encode()).to_bytes(4, "little")+s.encode() for s in fields)).digest()
    return int.from_bytes(digest[:8], "big")*alpha.denominator < alpha.numerator*(1 << 64)


def build_view(proxy, *, coordinate, offline=None, identity=None, candidate=None, mapping=None):
    # These branches must never touch identity, candidate, offline or mappings.
    if coordinate == "HLT_X1":
        return repeat(proxy, 1)
    if coordinate in ("HLT_X3", "D000"):
        return repeat(proxy)
    if offline is None:
        raise PermissionError("Rich view requires explicit offline capability")
    if coordinate == "OFFLINE":
        return offline
    require(coordinate in STRENGTHS and isinstance(identity, str) and len(identity) == 64,
            "Unknown rich coordinate/identity")
    view_contract(candidate)
    validate_mapping(mapping, nh=len(proxy), no=len(offline))
    values = {k: np.repeat(getattr(proxy, k), 3, axis=0).copy()
              for k in ("p4", "charge", "category", "tracking", "valid")}
    alpha = STRENGTHS[coordinate]
    for owner, slot in zip(*np.nonzero(mapping >= 0)):
        off, target = int(mapping[owner, slot]), 3*owner+slot+1
        if alpha == 1:
            for name in values:
                values[name][target] = getattr(offline, name)[off]
            continue
        values["p4"][target] = float(alpha)*offline.p4[off]+float(1-alpha)*proxy.p4[owner]
        selected, index = ((offline, off) if choose(identity, candidate, owner, slot, "identity", alpha)
                           else (proxy, owner))
        for name in ("charge", "category"):
            values[name][target] = getattr(selected, name)[index]
        if (proxy.charge[owner] != 0) != (offline.charge[off] != 0):
            for name in ("tracking", "valid"):
                values[name][target] = getattr(selected, name)[index]
            continue
        for val, err, group in ((0, 2, "d0"), (1, 3, "dz")):
            positions = [val, err]
            pv, ov = proxy.valid[owner, positions], offline.valid[off, positions]
            if np.array_equal(pv, ov):
                values["valid"][target, positions] = pv
                values["tracking"][target, positions] = np.where(pv,
                    float(alpha)*offline.tracking[off, positions]+float(1-alpha)*proxy.tracking[owner, positions], 0.)
            else:
                endpoint, j = ((offline, off) if choose(identity, candidate, owner, slot, group, alpha)
                               else (proxy, owner))
                for name in ("tracking", "valid"):
                    values[name][target, positions] = getattr(endpoint, name)[j, positions]
    return Particles(**values, keys=tuple(f"slot:{i}" for i in range(3*len(proxy))))
