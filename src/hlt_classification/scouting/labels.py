"""Exact PMARD baseline selection and non-overlapping 15-class mapping."""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def _array(value: object) -> np.ndarray:
    return np.asarray(value)


def baseline_mask(arrays: Mapping[str, object]) -> np.ndarray:
    tight = _array(arrays["jet_tightId"])
    jet_no = _array(arrays["jet_no"])
    pt = _array(arrays["scoutfj_pt"])
    mass = _array(arrays["scoutfj_sdmass"])
    count = _array(arrays["n_scoutpfcands"])
    return ((tight == 1) & (jet_no < 2) & (pt > 200) & (pt < 5000)
            & (mass > 0) & (mass < 650) & (count > 0))


def class_membership(arrays: Mapping[str, object]) -> np.ndarray:
    scout = _array(arrays["scoutfj_label"])
    pid = _array(arrays["scoutfj_gen_pid"])
    offline = _array(arrays["fj_label"])
    return np.stack((
        (offline >= 309) & (offline < 314), scout == 17, scout == 18,
        scout == 19, (scout == 20) & (pid == 25), scout == 22, scout == 24,
        scout == 25, scout == 26, scout == 27, scout == 28, scout == 29,
        scout == 21, scout == 23, (scout == 20) & (np.abs(pid) == 37),
    ), axis=1)


def multiclass_labels(arrays: Mapping[str, object]) -> np.ndarray:
    membership = class_membership(arrays)
    multiplicity = membership.sum(axis=1)
    if np.any(multiplicity > 1):
        indexes = np.flatnonzero(multiplicity > 1)[:10].tolist()
        raise ValueError(f"overlapping PMARD labels at chunk indexes {indexes}")
    result = np.full(len(membership), -1, dtype=np.int64)
    mapped = multiplicity == 1
    result[mapped] = membership[mapped].argmax(axis=1)
    return result


__all__ = ["baseline_mask", "class_membership", "multiclass_labels"]
