"""Full-cardinality paired views for the new 14-raw/17-model field contract.

The matrix solver is reused, but the old 21-field endpoint adapters are not.
With maximum cardinality only one residual side can be nonempty. Therefore U
needs insertion/removal edits, never residual substitution cost calibration.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import math
import numpy as np

from hlt_classification.scouting.hcwdl_fullcard_bottleneck_contracts import matcher_spec
from hlt_classification.scouting.hcwdl_fullcard_bottleneck_matcher import (
    canonical_qdr, canonical_qabs_log_pt_response, production_pairing_from_matrices, validate_pairing,
)
from .contracts import artifact
from .inputs import eta_phi, wrap_phi
from .reader import Jet, Particles


def view_contract() -> dict:
    return artifact(
        "VIEWS", matcher=matcher_spec(), support="replace_source_with_target_v1",
        U="paired_offline_plus_mass_balanced_unmatched_insertion_or_removal",
        D="linear_p4_and_applicable_measurements_atomic_identity_and_validity",
        ordering="hlt_slots_then_offline_native_tail_exact_native_endpoints",
        residual_mass="round_half_up_1e6_times_3_plus_4pt_fraction_plus_2energy_fraction",
        strata="edit_kind_category_charge_applicability_uncertainty_validity",
        switches="hash_order_mass_midpoint_uniform_phase_uint16_half_up",
        d_uncertainty_transition="value_and_error_atomic_when_error_validity_changes",
        d_applicability_transition="all_measurements_follow_atomic_charge_pid_choice",
        learned_calibration="none_no_residual_substitutions_under_full_cardinality",
        source_indices_model_visible=False, label_dependent=False,
    )


@lru_cache(maxsize=1)
def _view_digest() -> str:
    return view_contract()["content_hash"]


def _hash(identity: str, domain: str, *parts) -> bytes:
    # Constant scientific configuration, not per-particle JSON serialization.
    values = [_view_digest(), identity, domain, *map(str, parts)]
    return hashlib.sha256(b"".join(len(x.encode()).to_bytes(4, "little") + x.encode() for x in values)).digest()


def pairing_matrices(hlt: Particles, offline: Particles) -> dict:
    h_eta, h_phi = eta_phi(hlt.p4)
    o_eta, o_phi = eta_phi(offline.p4)
    dr = np.hypot(h_eta[:, None] - o_eta, wrap_phi(h_phi[:, None] - o_phi))
    hp = np.hypot(hlt.p4[:, 0].astype(np.float64), hlt.p4[:, 1].astype(np.float64))
    op = np.hypot(offline.p4[:, 0].astype(np.float64), offline.p4[:, 1].astype(np.float64))
    return dict(
        qdr=canonical_qdr(dr), qresponse=canonical_qabs_log_pt_response(np.log(hp[:, None] / op)),
        hlt_category=np.argmax(hlt.values[:, 5:10], axis=1),
        offline_category=np.argmax(offline.values[:, 5:10], axis=1),
        hlt_charge=hlt.values[:, 4].astype(np.int64), offline_charge=offline.values[:, 4].astype(np.int64),
        native_offline_index=np.arange(len(offline), dtype=np.int64),
    )


def match_particles(hlt: Particles, offline: Particles) -> np.ndarray:
    mapping = production_pairing_from_matrices(**pairing_matrices(hlt, offline))
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    return mapping


def structural_switches(particles: Particles, indices: np.ndarray, *, identity: str, kind: str) -> dict[int, int]:
    if not len(indices):
        return {}
    raw = particles.values[indices].astype(np.float64)
    pts = np.hypot(raw[:, 0], raw[:, 1])
    pt_total, energy_total = math.fsum(pts), math.fsum(raw[:, 3])
    masses = [int(math.floor(1e6 * math.fsum((3., 4. * p / pt_total, 2. * e / energy_total)) + .5))
              for p, e in zip(pts, raw[:, 3])]
    strata = {}
    for index, row, mass in zip(indices, raw, masses):
        key = (kind, int(np.argmax(row[5:10])), int(row[4] != 0), int(row[11] > 0), int(row[13] > 0))
        strata.setdefault(key, []).append((int(index), mass))
    result = {}
    for key, rows in sorted(strata.items()):
        rows.sort(key=lambda r: (_hash(identity, "structural_order", key, r[0]), r[0]))
        total = sum(m for _, m in rows)
        phase = int.from_bytes(_hash(identity, "structural_phase", key)[:8], "big")
        denominator = 2 * (1 << 64) * total
        preceding = 0
        for index, mass in rows:
            numerator = (phase * 2 * total + (2 * preceding + mass) * (1 << 64)) % denominator
            result[index] = (2 * numerator * 65535 + denominator) // (2 * denominator)
            preceding += mass
    return result


def _active(switch: int, fraction: Fraction) -> bool:
    if fraction == 0:
        return False
    if fraction == 1:
        return True
    threshold = (2 * fraction.numerator * 65535 + fraction.denominator) // (2 * fraction.denominator)
    return switch <= threshold


def _d_choice(identity: str, slot: int, group: str, alpha: Fraction) -> bool:
    draw = int.from_bytes(_hash(identity, "measurement", slot, group)[:8], "big")
    return draw * alpha.denominator < alpha.numerator * (1 << 64)


def build_view(jet: Jet, *, u: Fraction, f: Fraction, mapping: np.ndarray | None = None) -> Particles:
    if not isinstance(u, Fraction) or not isinstance(f, Fraction) or not (0 <= u <= 1 and 0 <= f <= 1):
        raise ValueError("View coordinates must be exact fractions in [0,1]")
    if f and u != 1:
        raise ValueError("D only follows completed U support")
    if u == f == 1:
        return jet.hlt  # No offline access, matcher, or reconstruction at D000.
    if jet.offline is None:
        raise PermissionError("This oracle/intermediate view requires declared offline inputs")
    if u == f == 0:
        return jet.offline  # Pure native offline, independent of assignment.
    hlt, offline = jet.hlt, jet.offline
    if mapping is None:
        mapping = match_particles(hlt, offline)
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    common = np.flatnonzero(mapping >= 0)
    residual_h = np.flatnonzero(mapping < 0)
    residual_o = np.setdiff1d(np.arange(len(offline)), mapping[common])
    if len(residual_h) and len(residual_o):
        raise ValueError("Full cardinality cannot leave both sides unmatched")
    shell = np.array(hlt.values, copy=True)
    alpha = 1 - f
    for slot in common:
        h, o = hlt.values[slot], offline.values[mapping[slot]]
        if f == 0:
            shell[slot] = o
            continue
        shell[slot, :4] = (float(f) * h[:4].astype(np.float64) + float(alpha) * o[:4].astype(np.float64)).astype(np.float32)
        use_offline = _d_choice(jet.identity, int(slot), "identity", alpha)
        shell[slot, 4:10] = (o if use_offline else h)[4:10]
        applicability_changes = (h[4] != 0) != (o[4] != 0)
        for value_index, error_index, name in ((10, 11, "d0"), (12, 13, "dz")):
            if applicability_changes:
                shell[slot, value_index:error_index + 1] = (o if use_offline else h)[value_index:error_index + 1]
            elif (h[error_index] > 0) != (o[error_index] > 0):
                chosen = o if _d_choice(jet.identity, int(slot), name, alpha) else h
                shell[slot, value_index:error_index + 1] = chosen[value_index:error_index + 1]
            else:
                shell[slot, value_index:error_index + 1] = (
                    float(f) * h[value_index:error_index + 1].astype(np.float64)
                    + float(alpha) * o[value_index:error_index + 1].astype(np.float64))
    if u == 1:
        return Particles(shell)
    removal = structural_switches(offline, residual_o, identity=jet.identity, kind="remove")
    insertion = structural_switches(hlt, residual_h, identity=jet.identity, kind="insert")
    selected_h = [i for i in range(len(hlt)) if mapping[i] >= 0 or _active(insertion[i], u)]
    selected_o = [i for i in residual_o if not _active(removal[int(i)], u)]
    return Particles(np.concatenate((shell[selected_h], offline.values[selected_o]), axis=0))
