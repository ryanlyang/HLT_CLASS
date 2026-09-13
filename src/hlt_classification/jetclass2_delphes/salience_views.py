"""JetClass2 adapter for exact salience matching and persistent-HLT views."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import math

import numpy as np

from hlt_classification.scouting.highcov_data import Particles as ScoutingParticles
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    CANDIDATES, matcher_spec,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    particle_salience, pair_utility, production_pairing_from_matrices,
    validate_pairing,
)
from .contracts import artifact
from .reader import Jet, Particles
from .views import pairing_matrices as geometric_pairing_matrices


SUPPORT_POLICY = "persistent_hlt_skeleton_offline_tail_v1"


def view_contract(candidate: str) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError("Unknown salience candidate")
    return artifact(
        "SALIENCE_VIEWS", matcher=matcher_spec(candidate),
        support=SUPPORT_POLICY,
        U="persistent_hlt_shell_plus_mass_balanced_source_only_offline_removal",
        D="linear_p4_and_applicable_measurements_atomic_identity_and_validity",
        ordering="persistent_hlt_slots_then_remaining_native_offline_tail",
        endpoints={
            "U000": "offline_on_matched_hlt_slots_native_unmatched_hlt_plus_offline_tail",
            "U100": "hlt_cardinality_offline_on_matched_slots_native_unmatched_hlt",
            "D000": "exact_native_hlt_no_offline_capability",
        },
        residual_mass="round_half_up_1e6_times_3_plus_4pt_fraction_plus_2energy_fraction",
        switches="hash_order_mass_midpoint_uniform_phase_uint16_half_up",
        source_indices_model_visible=False, pairing_validity_model_visible=False,
        correspondence_confidence="absent", label_dependent=False,
    )


def _scouting_particles(value: Particles, *, offline: bool) -> ScoutingParticles:
    count = len(value)
    native = np.arange(count, dtype=np.int64) if offline else None
    return ScoutingParticles(
        p4=np.asarray(value.p4, np.float64),
        category=np.argmax(value.values[:, 5:10], axis=1).astype(np.int8),
        charge=value.values[:, 4].astype(np.float64),
        track=np.zeros((count, 7), np.float64),
        track_valid=np.zeros((count, 7), bool), native_index=native,
    )


def pairing_matrices(hlt: Particles, offline: Particles, candidate: str) -> dict:
    base = geometric_pairing_matrices(hlt, offline)
    hweight = particle_salience(_scouting_particles(hlt, offline=False), candidate)
    oweight = particle_salience(_scouting_particles(offline, offline=True), candidate)
    return {
        **base,
        "hlt_salience": hweight,
        "offline_salience": oweight,
        "utility": pair_utility(base["qdr"], hweight, oweight),
    }


def match_particles(hlt: Particles, offline: Particles, candidate: str) -> np.ndarray:
    mapping = production_pairing_from_matrices(**pairing_matrices(hlt, offline, candidate))
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    return mapping


@lru_cache(maxsize=None)
def _view_digest(candidate: str) -> str:
    return view_contract(candidate)["content_hash"]


def _hash(identity: str, candidate: str, domain: str, *parts: object) -> bytes:
    values = [_view_digest(candidate), identity, domain, *map(str, parts)]
    return hashlib.sha256(b"".join(
        len(value.encode()).to_bytes(4, "little") + value.encode()
        for value in values
    )).digest()


def structural_switches(
    particles: Particles, indices: np.ndarray, *, identity: str, candidate: str,
) -> dict[int, int]:
    if not len(indices):
        return {}
    raw = particles.values[indices].astype(np.float64)
    pts = np.hypot(raw[:, 0], raw[:, 1])
    pt_total, energy_total = math.fsum(pts), math.fsum(raw[:, 3])
    masses = [
        int(math.floor(1e6 * math.fsum((
            3., 4. * pt / pt_total, 2. * energy / energy_total,
        )) + .5))
        for pt, energy in zip(pts, raw[:, 3], strict=True)
    ]
    strata: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for index, row, mass in zip(indices, raw, masses, strict=True):
        key = (
            int(np.argmax(row[5:10])), int(row[4] != 0),
            int(row[11] > 0), int(row[13] > 0),
        )
        strata.setdefault(key, []).append((int(index), mass))
    result: dict[int, int] = {}
    for key, rows in sorted(strata.items()):
        rows.sort(key=lambda item: (
            _hash(identity, candidate, "structural_order", key, item[0]), item[0],
        ))
        total = sum(mass for _, mass in rows)
        phase = int.from_bytes(
            _hash(identity, candidate, "structural_phase", key)[:8], "big",
        )
        denominator = 2 * (1 << 64) * total
        preceding = 0
        for index, mass in rows:
            numerator = (
                phase * 2 * total + (2 * preceding + mass) * (1 << 64)
            ) % denominator
            result[index] = (2 * numerator * 65535 + denominator) // (2 * denominator)
            preceding += mass
    return result


def _active(switch: int, fraction: Fraction) -> bool:
    if fraction == 0:
        return False
    if fraction == 1:
        return True
    threshold = (
        2 * fraction.numerator * 65535 + fraction.denominator
    ) // (2 * fraction.denominator)
    return switch <= threshold


def _d_choice(
    identity: str, candidate: str, slot: int, group: str, alpha: Fraction,
) -> bool:
    draw = int.from_bytes(
        _hash(identity, candidate, "measurement", slot, group)[:8], "big",
    )
    return draw * alpha.denominator < alpha.numerator * (1 << 64)


def build_view(
    jet: Jet, *, u: Fraction, f: Fraction, candidate: str,
    mapping: np.ndarray | None = None,
) -> Particles:
    """Build the persistent-HLT coordinate; only D000 is offline-free."""
    if (
        candidate not in CANDIDATES or not isinstance(u, Fraction)
        or not isinstance(f, Fraction) or not (0 <= u <= 1 and 0 <= f <= 1)
    ):
        raise ValueError("Invalid salience view configuration")
    if f and u != 1:
        raise ValueError("D only follows completed U support")
    if u == f == 1:
        return jet.hlt
    if jet.offline is None:
        raise PermissionError("Salience oracle/intermediate view requires offline inputs")
    hlt, offline = jet.hlt, jet.offline
    if mapping is None:
        mapping = match_particles(hlt, offline, candidate)
    validate_pairing(mapping, nh=len(hlt), no=len(offline))
    common = np.flatnonzero(mapping >= 0)
    residual_offline = np.setdiff1d(
        np.arange(len(offline), dtype=np.int64), mapping[common], assume_unique=True,
    )
    shell = np.array(hlt.values, copy=True)
    alpha = 1 - f
    for slot in common:
        hrow = hlt.values[slot]
        orow = offline.values[mapping[slot]]
        if f == 0:
            shell[slot] = orow
            continue
        shell[slot, :4] = (
            float(f) * hrow[:4].astype(np.float64)
            + float(alpha) * orow[:4].astype(np.float64)
        ).astype(np.float32)
        use_offline = _d_choice(
            jet.identity, candidate, int(slot), "identity", alpha,
        )
        shell[slot, 4:10] = (orow if use_offline else hrow)[4:10]
        applicability_changes = (hrow[4] != 0) != (orow[4] != 0)
        for value_index, error_index, name in ((10, 11, "d0"), (12, 13, "dz")):
            if applicability_changes:
                chosen = orow if use_offline else hrow
                shell[slot, value_index:error_index + 1] = chosen[
                    value_index:error_index + 1
                ]
            elif (hrow[error_index] > 0) != (orow[error_index] > 0):
                chosen = orow if _d_choice(
                    jet.identity, candidate, int(slot), name, alpha,
                ) else hrow
                shell[slot, value_index:error_index + 1] = chosen[
                    value_index:error_index + 1
                ]
            else:
                shell[slot, value_index:error_index + 1] = (
                    float(f) * hrow[value_index:error_index + 1].astype(np.float64)
                    + float(alpha) * orow[value_index:error_index + 1].astype(np.float64)
                )
    if u == 1:
        return Particles(shell)
    removal = structural_switches(
        offline, residual_offline, identity=jet.identity, candidate=candidate,
    )
    selected_offline = [
        index for index in residual_offline
        if not _active(removal[int(index)], u)
    ]
    if not selected_offline:
        return Particles(shell)
    return Particles(np.concatenate((shell, offline.values[selected_offline]), axis=0))


__all__ = [
    "SUPPORT_POLICY", "build_view", "match_particles", "pairing_matrices",
    "structural_switches", "view_contract",
]
