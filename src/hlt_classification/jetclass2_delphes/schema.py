"""Explicit reduced schema and provisional numeric-to-physics label mapping."""
from __future__ import annotations

import numpy as np

from .contracts import artifact, assumptions, validate

CLASS_NAMES = (
    "QCD", "X_bb", "X_cc", "X_ss", "X_qq", "X_gg", "X_ee", "X_mm",
    "X_tauhtaue", "X_tauhtaum", "X_tauhtauh",
)
SIGNAL_IDS = (0, 1, 2, 3, 9, 10, 11, 12, 13, 14)
QCD_IDS = tuple(range(161, 188))
RAW_TO_CLASS = {raw: i + 1 for i, raw in enumerate(SIGNAL_IDS)}
RAW_TO_CLASS.update({raw: 0 for raw in QCD_IDS})
PARTICLE_FIELDS = (
    "px", "py", "pz", "energy", "charge", "isChargedHadron", "isNeutralHadron",
    "isPhoton", "isElectron", "isMuon", "d0val", "d0err", "dzval", "dzerr",
)
SCALARS = ("jet_label", "hlt_matched", "jet_nparticles", "hlt_jet_nparticles")
PARTICLE_BRANCHES = tuple(
    prefix + field for prefix in ("part_", "hlt_part_") for field in PARTICLE_FIELDS
)


def label_map() -> dict:
    return artifact(
        "LABEL_MAP", class_names=list(CLASS_NAMES),
        raw_to_class={str(k): v for k, v in sorted(RAW_TO_CLASS.items())},
        assumptions_sha256=assumptions()["content_hash"],
        upstream_repository="https://github.com/jet-universe/jetclass2_generation",
        upstream_commit="3a7a1355f4230b5790669286466080d7fa3b6794",
        upstream_file="delphes_analyzers/FatJetMatching.h",
        producer_revision_confirmed=False,
    )


def map_labels(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("Raw labels must be a one-dimensional integer vector")
    unknown = set(map(int, np.unique(raw))) - RAW_TO_CLASS.keys()
    if unknown:
        raise ValueError(f"Unregistered JetClass2 raw labels: {sorted(unknown)}")
    out = np.empty(len(raw), np.int64)
    for value in np.unique(raw):
        out[raw == value] = RAW_TO_CLASS[int(value)]
    return out


def validate_label_map(value: dict) -> None:
    validate(value, "LABEL_MAP")
    if value != label_map():
        raise ValueError("Label-map semantics differ from this reader")


def validate_tree_schema(tree) -> dict:
    types = tree.typenames()
    missing = set(SCALARS + PARTICLE_BRANCHES) - types.keys()
    if missing:
        raise ValueError(f"Missing required ROOT branches: {sorted(missing)}")
    # Freeze all branch types, including unused diagnostic branches, in inventory.
    for name in PARTICLE_BRANCHES:
        if "vector<" not in types[name] and "[]" not in types[name]:
            raise ValueError(f"Expected jagged particle branch: {name}={types[name]}")
    for name in SCALARS:
        expected = "bool" if name == "hlt_matched" else "int"
        if expected not in types[name].lower():
            raise ValueError(f"Invalid scalar type: {name}={types[name]}")
    return dict(sorted(types.items()))
