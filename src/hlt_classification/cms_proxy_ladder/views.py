"""Persistent-proxy views over authenticated physical particle endpoints."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import math

import numpy as np

from hlt_classification.cms2jc2_response.bridge import Particles
from hlt_classification.scouting.highcov_data import Particles as ScoutingParticles
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    SALIENCE_PT_LINEAR,
    matcher_spec,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_matcher import (
    FullCardinalitySalienceMatcher,
    validate_pairing,
)

from .contracts import artifact

CANDIDATE = SALIENCE_PT_LINEAR
SUPPORT_POLICY = "cms_proxy_persistent_shell_offline_tail_v1"


def view_contract() -> dict:
    return artifact(
        "VIEWS",
        matcher=matcher_spec(CANDIDATE),
        support=SUPPORT_POLICY,
        U="persistent_proxy_shell_plus_mass_balanced_source_only_offline_removal",
        D="linear_p4_atomic_identity_and_validity_aware_tracking",
        D_coordinate="Dxxx_names_retained_offline_fraction_D000_exact_proxy",
        ordering="persistent_proxy_slots_then_remaining_native_offline_tail",
        endpoints={
            "U000": "offline_on_matched_proxy_slots_unmatched_proxy_plus_offline_tail",
            "U100": "proxy_cardinality_offline_on_matched_slots_unmatched_proxy",
            "D000": "exact_cms_calibrated_proxy_hlt",
            "OFFLINE": "exact_original_dzfix_offline",
        },
        unknown_category="all_zero_model_one_hot",
        invalid_tracking="zero_value_and_zero_error_without_invented_measurement",
        source_indices_model_visible=False,
        pairing_validity_model_visible=False,
        correspondence_confidence="absent",
        label_dependent=False,
    )


@lru_cache(maxsize=1)
def _view_digest() -> str:
    return view_contract()["content_hash"]


def _hash(identity: str, domain: str, *parts: object) -> bytes:
    values = [_view_digest(), identity, domain, *map(str, parts)]
    return hashlib.sha256(b"".join(
        len(value.encode()).to_bytes(4, "little") + value.encode()
        for value in values
    )).digest()


def _scouting(value: Particles, *, offline: bool) -> ScoutingParticles:
    count = len(value)
    track = np.zeros((count, 7), np.float64)
    valid = np.zeros((count, 7), bool)
    track[:, :2] = value.tracking[:, :2]
    valid[:, :2] = value.valid[:, :2]
    return ScoutingParticles(
        p4=np.asarray(value.p4, np.float64),
        category=np.asarray(value.category, np.int8),
        charge=np.asarray(value.charge, np.float64),
        track=track,
        track_valid=valid,
        native_index=np.arange(count, dtype=np.int64) if offline else None,
    )


def match_particles(proxy: Particles, offline: Particles) -> np.ndarray:
    result = FullCardinalitySalienceMatcher(CANDIDATE).match(
        _scouting(proxy, offline=False), _scouting(offline, offline=True),
    )
    mapping = result.concatenated_offline_index
    validate_pairing(mapping, nh=len(proxy), no=len(offline))
    return mapping


def structural_switches(
    offline: Particles, indices: np.ndarray, *, identity: str,
) -> dict[int, int]:
    """Mass-balanced deterministic switches for the removable offline tail."""
    indices = np.asarray(indices, np.int64)
    if not len(indices):
        return {}
    pts = offline.pt[indices]
    energy = offline.p4[indices, 3]
    pt_total, energy_total = math.fsum(pts), math.fsum(energy)
    masses = [
        int(math.floor(1e6 * math.fsum((
            3., 4. * pt / pt_total, 2. * e / energy_total,
        )) + .5))
        for pt, e in zip(pts, energy, strict=True)
    ]
    strata: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for index, mass in zip(indices, masses, strict=True):
        key = (
            int(offline.category[index]), int(offline.charge[index] != 0),
            int(offline.valid[index, 0]), int(offline.valid[index, 1]),
        )
        strata.setdefault(key, []).append((int(index), mass))
    result: dict[int, int] = {}
    for key, rows in sorted(strata.items()):
        rows.sort(key=lambda row: (_hash(identity, "structural_order", key, row[0]), row[0]))
        total = sum(mass for _, mass in rows)
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


def _choose_offline(identity: str, slot: int, group: str, alpha: Fraction) -> bool:
    draw = int.from_bytes(_hash(identity, "measurement", slot, group)[:8], "big")
    return draw * alpha.denominator < alpha.numerator * (1 << 64)


def _particles(
    p4, charge, category, tracking, valid, keys,
) -> Particles:
    return Particles(
        np.asarray(p4, np.float64), np.asarray(charge, np.int8),
        np.asarray(category, np.int8), np.asarray(tracking, np.float64),
        np.asarray(valid, bool), tuple(keys),
    )


def build_view(
    *, identity: str, proxy: Particles, offline: Particles | None,
    coordinate: str, mapping: np.ndarray | None = None,
) -> Particles:
    """Build one registered view without exposing matching metadata to the model."""
    if coordinate == "OFFLINE":
        if offline is None:
            raise PermissionError("Pure offline view requires the offline endpoint")
        return offline
    if coordinate == "D000":
        return proxy
    if len(identity) != 64:
        raise ValueError("Canonical row identity is required")
    if offline is None:
        raise PermissionError("Oracle/intermediate proxy view requires offline particles")
    prefix, value = coordinate[:1], coordinate[1:]
    if prefix not in {"U", "D"} or len(value) != 3 or not value.isdigit():
        raise ValueError("Unregistered proxy-ladder coordinate")
    retained_offline = {"033": Fraction(1, 3), "066": Fraction(2, 3)}.get(
        value, Fraction(int(value), 100),
    )
    if not 0 <= retained_offline <= 1:
        raise ValueError("Coordinate fraction is outside [0,1]")
    mapping = match_particles(proxy, offline) if mapping is None else np.asarray(mapping, np.int32)
    validate_pairing(mapping, nh=len(proxy), no=len(offline))
    common = np.flatnonzero(mapping >= 0)
    residual = np.setdiff1d(
        np.arange(len(offline), dtype=np.int64), mapping[common], assume_unique=True,
    )
    p4 = np.array(proxy.p4, copy=True)
    charge = np.array(proxy.charge, copy=True)
    category = np.array(proxy.category, copy=True)
    tracking = np.array(proxy.tracking, copy=True)
    valid = np.array(proxy.valid, copy=True)
    # Uxxx names the removed offline-tail fraction.  Dxxx names the retained
    # offline feature fraction, so the proxy fraction is 1-Dxxx.
    alpha = Fraction(1) if prefix == "U" else retained_offline
    proxy_fraction = Fraction(0) if prefix == "U" else 1 - retained_offline
    for slot in common:
        source = int(mapping[slot])
        if prefix == "U":
            p4[slot] = offline.p4[source]
            charge[slot] = offline.charge[source]
            category[slot] = offline.category[source]
            tracking[slot] = offline.tracking[source]
            valid[slot] = offline.valid[source]
            continue
        p4[slot] = float(proxy_fraction) * proxy.p4[slot] + float(alpha) * offline.p4[source]
        use_offline = _choose_offline(identity, int(slot), "identity", alpha)
        chosen = offline if use_offline else proxy
        chosen_index = source if use_offline else int(slot)
        charge[slot] = chosen.charge[chosen_index]
        category[slot] = chosen.category[chosen_index]
        # Neutrality is an atomic identity/applicability property.
        if (proxy.charge[slot] != 0) != (offline.charge[source] != 0):
            tracking[slot] = chosen.tracking[chosen_index]
            valid[slot] = chosen.valid[chosen_index]
            continue
        for value_index, error_index, name in ((0, 2, "d0"), (1, 3, "dz")):
            pvalid = proxy.valid[slot, [value_index, error_index]]
            ovalid = offline.valid[source, [value_index, error_index]]
            if np.array_equal(pvalid, ovalid):
                valid[slot, [value_index, error_index]] = pvalid
                for position, index in enumerate((value_index, error_index)):
                    tracking[slot, index] = (
                        float(proxy_fraction) * proxy.tracking[slot, index]
                        + float(alpha) * offline.tracking[source, index]
                    ) if pvalid[position] else 0.
            else:
                take_offline = _choose_offline(identity, int(slot), name, alpha)
                endpoint, index = (offline, source) if take_offline else (proxy, int(slot))
                tracking[slot, [value_index, error_index]] = endpoint.tracking[index, [value_index, error_index]]
                valid[slot, [value_index, error_index]] = endpoint.valid[index, [value_index, error_index]]
    shell = _particles(
        p4, charge, category, tracking, valid,
        tuple(f"proxy:{key}" for key in proxy.keys),
    )
    if prefix == "D" or retained_offline == 1:
        return shell
    switches = structural_switches(offline, residual, identity=identity)
    tail = [
        index for index in residual
        if not _active(switches[int(index)], retained_offline)
    ]
    if not tail:
        return shell
    return _particles(
        np.concatenate((shell.p4, offline.p4[tail])),
        np.concatenate((shell.charge, offline.charge[tail])),
        np.concatenate((shell.category, offline.category[tail])),
        np.concatenate((shell.tracking, offline.tracking[tail])),
        np.concatenate((shell.valid, offline.valid[tail])),
        shell.keys + tuple(f"offline:{offline.keys[index]}" for index in tail),
    )


__all__ = [
    "CANDIDATE", "SUPPORT_POLICY", "build_view", "match_particles",
    "structural_switches", "view_contract",
]
