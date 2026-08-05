"""Decode audited raw Scouting collections into matcher particle sets."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .matching import ParticleSet, decode_exclusive_categories


def _row(arrays: Mapping[str, object], branch: str, row: int) -> np.ndarray:
    return np.asarray(arrays[branch][row])


def _p4(arrays: Mapping[str, object], prefix: str, row: int) -> np.ndarray:
    values = [_row(arrays, f"{prefix}_{name}", row) for name in ("px", "py", "pz", "energy")]
    if len({len(item) for item in values}) != 1:
        raise ValueError(f"{prefix} p4 family length mismatch")
    return np.stack(values, axis=1).astype(np.float64, copy=False)


def decode_particle_sets(
    arrays: Mapping[str, object], row: int, *, hlt_max_length: int = 200,
) -> tuple[ParticleSet, ParticleSet, int]:
    hlt_p4 = _p4(arrays, "scoutpfcand", row)
    raw_hlt_count = len(hlt_p4); hlt_p4 = hlt_p4[:hlt_max_length]
    hlt_flags = np.stack([
        _row(arrays, f"scoutpfcand_{name}", row)[:hlt_max_length]
        for name in ("isEl", "isMu", "isChargedHad", "isGamma", "isNeutralHad")
    ], axis=1)
    hlt = ParticleSet(
        hlt_p4, decode_exclusive_categories(hlt_flags),
        _row(arrays, "scoutpfcand_charge", row)[:hlt_max_length].astype(np.float64),
        np.zeros(len(hlt_p4), np.bool_),
    )
    charged_p4 = _p4(arrays, "cpfcandlt", row)
    neutral_p4 = _p4(arrays, "npfcand", row)
    charged_native = np.stack([
        _row(arrays, f"cpfcandlt_{name}", row) for name in ("isEl", "isMu", "isChargedHad")
    ], axis=1)
    neutral_native = np.stack([
        _row(arrays, f"npfcand_{name}", row) for name in ("isGamma", "isNeutralHad")
    ], axis=1)
    charged_flags = np.pad(charged_native, ((0, 0), (0, 2)))
    neutral_flags = np.pad(neutral_native, ((0, 0), (3, 0)))
    offline = ParticleSet(
        np.concatenate((charged_p4, neutral_p4)),
        decode_exclusive_categories(np.concatenate((charged_flags, neutral_flags))),
        np.concatenate((
            _row(arrays, "cpfcandlt_charge", row).astype(np.float64),
            np.zeros(len(neutral_p4), np.float64),
        )),
        np.concatenate((
            _row(arrays, "cpfcandlt_isLostTrack", row).astype(np.bool_),
            np.zeros(len(neutral_p4), np.bool_),
        )),
    )
    return hlt, offline, max(0, raw_hlt_count - hlt_max_length)


__all__ = ["decode_particle_sets"]
