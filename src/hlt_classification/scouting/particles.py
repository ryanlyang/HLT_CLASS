"""Decode audited raw Scouting collections into matcher particle sets."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .matching import ParticleSet, TRACK_FIELDS, decode_exclusive_categories


def _row(arrays: Mapping[str, object], branch: str, row: int) -> np.ndarray:
    return np.asarray(arrays[branch][row])


def _count(arrays: Mapping[str, object], branch: str, row: int) -> int:
    value = np.asarray(arrays[branch][row])
    if value.ndim != 0 or not np.isfinite(value) or int(value) != value or int(value) < 0:
        raise ValueError(f"{branch} must be a nonnegative integer scalar")
    return int(value)


def _p4(arrays: Mapping[str, object], prefix: str, row: int) -> np.ndarray:
    values = [_row(arrays, f"{prefix}_{name}", row) for name in ("px", "py", "pz", "energy")]
    if len({len(item) for item in values}) != 1:
        raise ValueError(f"{prefix} p4 family length mismatch")
    return np.stack(values, axis=1).astype(np.float64, copy=False)


def _measurements(
    arrays: Mapping[str, object], prefix: str, row: int, *, limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    for name in TRACK_FIELDS:
        value = _row(arrays, f"{prefix}_{name}", row)
        columns.append(value if limit is None else value[:limit])
    if len({len(value) for value in columns}) != 1:
        raise ValueError(f"{prefix} matcher measurement family length mismatch")
    values = np.stack(columns, axis=1).astype(np.float64, copy=False)
    validity = np.isfinite(values)
    return np.where(validity, values, 0.0), validity


def decode_particle_sets(
    arrays: Mapping[str, object], row: int, *, hlt_max_length: int = 200,
) -> tuple[ParticleSet, ParticleSet, int]:
    hlt_p4 = _p4(arrays, "scoutpfcand", row)
    raw_hlt_count = _count(arrays, "n_scoutpfcands", row)
    if len(hlt_p4) != raw_hlt_count:
        raise ValueError("scoutpfcand collection length differs from n_scoutpfcands")
    hlt_p4 = hlt_p4[:hlt_max_length]
    hlt_flags = np.stack([
        _row(arrays, f"scoutpfcand_{name}", row)[:hlt_max_length]
        for name in ("isEl", "isMu", "isChargedHad", "isGamma", "isNeutralHad")
    ], axis=1)
    hlt_measurements, hlt_validity = _measurements(
        arrays, "scoutpfcand", row, limit=hlt_max_length,
    )
    hlt = ParticleSet(
        hlt_p4, decode_exclusive_categories(hlt_flags),
        _row(arrays, "scoutpfcand_charge", row)[:hlt_max_length].astype(np.float64),
        np.zeros(len(hlt_p4), np.bool_), hlt_measurements, hlt_validity,
    )
    charged_p4 = _p4(arrays, "cpfcandlt", row)
    neutral_p4 = _p4(arrays, "npfcand", row)
    regular_charged_count = _count(arrays, "n_cpfcands", row)
    lost_track_count = _count(arrays, "n_lts", row)
    neutral_count = _count(arrays, "n_npfcands", row)
    if len(charged_p4) != regular_charged_count + lost_track_count:
        raise ValueError("cpfcandlt collection does not equal n_cpfcands + n_lts")
    if len(neutral_p4) != neutral_count:
        raise ValueError("npfcand collection length differs from n_npfcands")
    charged_native = np.stack([
        _row(arrays, f"cpfcandlt_{name}", row) for name in ("isEl", "isMu", "isChargedHad")
    ], axis=1)
    neutral_native = np.stack([
        _row(arrays, f"npfcand_{name}", row) for name in ("isGamma", "isNeutralHad")
    ], axis=1)
    charged_flags = np.pad(charged_native, ((0, 0), (0, 2)))
    neutral_flags = np.pad(neutral_native, ((0, 0), (3, 0)))
    charged_measurements, charged_validity = _measurements(arrays, "cpfcandlt", row)
    neutral_measurements = np.zeros((len(neutral_p4), len(TRACK_FIELDS)), np.float64)
    neutral_validity = np.zeros_like(neutral_measurements, np.bool_)
    stored_lost_track = _row(arrays, "cpfcandlt_isLostTrack", row).astype(np.bool_)
    boundary_lost_track = np.arange(len(charged_p4)) >= regular_charged_count
    if not np.array_equal(stored_lost_track, boundary_lost_track):
        raise ValueError("cpfcandlt lost-track flags disagree with the declared collection boundary")
    offline = ParticleSet(
        np.concatenate((charged_p4, neutral_p4)),
        decode_exclusive_categories(np.concatenate((charged_flags, neutral_flags))),
        np.concatenate((
            _row(arrays, "cpfcandlt_charge", row).astype(np.float64),
            np.zeros(len(neutral_p4), np.float64),
        )),
        np.concatenate((
            boundary_lost_track,
            np.zeros(len(neutral_p4), np.bool_),
        )),
        np.concatenate((charged_measurements, neutral_measurements)),
        np.concatenate((charged_validity, neutral_validity)),
    )
    return hlt, offline, max(0, raw_hlt_count - hlt_max_length)


__all__ = ["decode_particle_sets"]
