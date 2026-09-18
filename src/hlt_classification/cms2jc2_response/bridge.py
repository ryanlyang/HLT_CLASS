"""Shared physical particles, separate from normalized model input tensors."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import validate_compatibility

CATEGORIES = ("charged_hadron", "neutral_hadron", "photon", "electron", "muon", "unknown")
TRACKING = ("d0", "dz", "d0err", "dzerr")
JC2_FIELDS = ("px", "py", "pz", "energy", "charge", "isChargedHadron", "isNeutralHadron",
              "isPhoton", "isElectron", "isMuon", "d0val", "d0err", "dzval", "dzerr")


@dataclass(frozen=True)
class Particles:
    p4: np.ndarray
    charge: np.ndarray
    category: np.ndarray
    tracking: np.ndarray
    valid: np.ndarray
    keys: tuple[str, ...]

    def __post_init__(self):
        n = len(self.keys)
        if len(set(self.keys)) != n or any(not isinstance(k, str) or not k for k in self.keys):
            raise ValueError("Particle keys must be unique nonempty strings")
        arrays = {"p4": (self.p4, np.float64, (n, 4)),
                  "charge": (self.charge, np.int8, (n,)),
                  "category": (self.category, np.int8, (n,)),
                  "tracking": (self.tracking, np.float64, (n, 4)),
                  "valid": (self.valid, np.bool_, (n, 4))}
        for name, (raw, dtype, shape) in arrays.items():
            original = np.asarray(raw)
            if original.shape != shape or not np.isfinite(original).all():
                raise ValueError(f"Invalid shared particle {name}")
            arr = np.array(original, dtype=dtype, copy=True)
            if not np.array_equal(original, arr):
                raise ValueError(f"Lossy discrete conversion: {name}")
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        if not np.isin(self.category, range(6)).all() or not np.isin(self.charge, (-1, 0, 1)).all():
            raise ValueError("Invalid identity/charge")
        known = self.category != 5
        charged = self.charge != 0
        if np.any(known & (charged != np.isin(self.category, (0, 3, 4)))):
            raise ValueError("Charge/category inconsistency")
        if np.any(self.valid[~charged]) or np.any(self.tracking[~self.valid] != 0):
            raise ValueError("Tracking applicability/invalid placeholders differ")
        if np.any(self.valid[:, 2:] & ~self.valid[:, :2]):
            raise ValueError("Valid uncertainty without a valid value")
        if np.any((self.tracking[:, 2:] <= 0) & self.valid[:, 2:]):
            raise ValueError("Nonpositive valid uncertainty")
        p2 = (self.p4[:, :3] ** 2).sum(axis=1)
        if np.any(self.p4[:, 3] <= 0) or np.any(self.pt <= 0):
            raise ValueError("Nonpositive pT/energy")
        if np.any(self.p4[:, 3] ** 2 < p2 - 2e-6 * np.maximum(p2, 1.)):
            raise ValueError("Spacelike four-vector beyond source-rounding tolerance")

    def __len__(self):
        return len(self.keys)

    @property
    def pt(self):
        return np.hypot(self.p4[:, 0], self.p4[:, 1])

    @property
    def eta(self):
        return np.arcsinh(self.p4[:, 2] / self.pt)

    @property
    def phi(self):
        return np.arctan2(self.p4[:, 1], self.p4[:, 0])

    @property
    def mass(self):
        return np.sqrt(np.maximum(0., self.p4[:, 3] ** 2 - (self.p4[:, :3] ** 2).sum(axis=1)))

    def take(self, indices) -> "Particles":
        i = np.asarray(indices, dtype=np.int64)
        return Particles(self.p4[i], self.charge[i], self.category[i], self.tracking[i],
                         self.valid[i], tuple(self.keys[k] for k in i))


def wrap_phi(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


def p4_from_coordinates(pt, eta, phi, mass):
    pt, eta, phi, mass = np.broadcast_arrays(pt, eta, phi, mass)
    if np.any(pt <= 0) or np.any(mass < 0):
        raise ValueError("Invalid response physical coordinates")
    with np.errstate(over="raise", invalid="raise"):
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    result = np.stack((px, py, pz, energy), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed four-vector")
    return result


def _identity(flags, charge):
    flags = np.asarray(flags)
    if not np.isin(flags, (0, 1)).all():
        raise ValueError("Nonbinary source PID flag")
    # Ambiguous/absent category is explicit unknown, never argmax-ed into a class.
    return np.where(flags.sum(axis=1) == 1, flags.argmax(axis=1), 5).astype(np.int8)


def from_jc2(columns: dict, review: dict, *, keys: tuple[str, ...]) -> Particles:
    validate_compatibility(review)
    a = np.column_stack([np.asarray(columns[f]) for f in JC2_FIELDS])
    q, category = a[:, 4], _identity(a[:, 5:10], a[:, 4])
    track = a[:, [10, 12, 11, 13]].copy()
    if np.any(np.isfinite(track[:, 2:]) & (track[:, 2:] < 0)):
        raise ValueError("Negative source uncertainty")
    valid = np.isfinite(track) & (q[:, None] != 0)
    valid[:, 2:] &= (track[:, 2:] > 0) & valid[:, :2]
    track[~valid] = 0
    track *= review["jc2_length_to_mm"]
    track[:, 0] *= review["jc2_d0_sign"]
    return Particles(a[:, :4] * review["momentum_to_gev"], q, category, track, valid, keys)


def from_cms(columns: dict, review: dict, *, side: str) -> Particles:
    validate_compatibility(review)
    if side not in {"offline", "hlt"}:
        raise ValueError("Unknown CMS side")
    parts = []
    for prefix in (("cpfcandlt", "npfcand") if side == "offline" else ("scoutpfcand",)):
        get = lambda field: np.asarray(columns[f"{prefix}_{field}"])
        p4 = np.column_stack([get(f) for f in ("px", "py", "pz", "energy")])
        n = len(p4)
        neutral = prefix == "npfcand"
        q = np.zeros(n) if neutral else get("charge")
        flags = np.zeros((n, 5))
        fields = {0: "isChargedHad", 1: "isNeutralHad", 2: "isGamma", 3: "isEl", 4: "isMu"}
        for i, field in fields.items():
            if (neutral and i in (1, 2)) or prefix == "scoutpfcand" or (not neutral and i in (0, 3, 4)):
                flags[:, i] = get(field)
        tr, valid = np.zeros((n, 4)), np.zeros((n, 4), bool)
        if not neutral:
            for i, name in enumerate(("dxy", "dz")):
                val, sig = get(name), get(name + "sig")
                ok = np.isfinite(val) & (q != 0)
                tr[ok, i] = val[ok]
                valid[:, i] = ok
                error_ok = ok & np.isfinite(sig) & (sig != 0) & (val != 0)
                tr[error_ok, i + 2] = np.abs(val[error_ok] / sig[error_ok])
                valid[:, i + 2] = error_ok
        tr *= review["cms_length_to_mm"]
        tr[:, 0] *= review["cms_d0_sign"]
        keep = np.ones(n, bool)
        if prefix == "cpfcandlt":
            lost = get("isLostTrack")
            if not np.isin(lost, (0, 1)).all():
                raise ValueError("Invalid lost-track flag")
            keep = lost == 0
        parts.append(Particles(p4[keep] * review["momentum_to_gev"], q[keep],
                               _identity(flags, q)[keep], tr[keep], valid[keep],
                               tuple(f"{prefix}:{i}" for i in np.flatnonzero(keep))))
    return Particles(*(np.concatenate([getattr(p, key) for p in parts], axis=0)
                       for key in ("p4", "charge", "category", "tracking", "valid")),
                     tuple(k for p in parts for k in p.keys))
