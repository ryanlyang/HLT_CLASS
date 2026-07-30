"""Frozen JetClass schema and filename-to-class contract."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_DATA_ROOT = Path("/home/ryreu/atlas/PracticeTagging/data")
DEFAULT_TREE_NAME = "tree"
MAX_CONSTITUENTS = 128

CLASS_LABELS = (
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl",
)

# The ordering is part of the contract. Matching itself is longest-prefix first
# so TTBarLep cannot be mistaken for TTBar.
FILENAME_PREFIX_TO_LABEL = (
    ("ZJetsToNuNu", 0),
    ("HToBB", 1),
    ("HToCC", 2),
    ("HToGG", 3),
    ("HToWW4Q", 4),
    ("HToWW2Q1L", 5),
    ("ZToQQ", 6),
    ("WToQQ", 7),
    ("TTBar", 8),
    ("TTBarLep", 9),
)

RAW_FEATURE_BRANCHES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
)
RAW_TOKEN_DIM = len(RAW_FEATURE_BRANCHES)
RAW_TOKEN_FEATURES = (
    "pt",
    "eta",
    "phi",
    "energy",
    "charge",
    "isChargedHadron",
    "isNeutralHadron",
    "isPhoton",
    "isElectron",
    "isMuon",
    "d0val",
    "d0err",
    "dzval",
    "dzerr",
)
LABEL_BRANCHES = tuple(f"label_{name}" for name in CLASS_LABELS)

SCHEMA_CONTRACT = "hlt_classification_jetclass_schema_v1"
SCHEMA_VERSION = 1


def label_from_filename(path: str | Path) -> int:
    """Return the class index encoded by a JetClass filename.

    Prefixes are checked by descending length. This is scientifically
    significant for the overlapping ``TTBar`` and ``TTBarLep`` names.
    """

    name = Path(path).name
    matches = sorted(FILENAME_PREFIX_TO_LABEL, key=lambda item: (-len(item[0]), item[0]))
    for prefix, label in matches:
        if name.startswith(prefix):
            return label
    raise ValueError(f"unrecognized JetClass filename prefix: {name!r}")


def validate_label(label: int) -> int:
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError(f"label must be an integer, got {type(label).__name__}")
    if not 0 <= label < len(CLASS_LABELS):
        raise ValueError(f"label {label} is outside [0, {len(CLASS_LABELS)})")
    return label


def missing_required_branches(branches: Iterable[str]) -> tuple[str, ...]:
    available = set(branches)
    return tuple(name for name in RAW_FEATURE_BRANCHES if name not in available)


def schema_payload() -> dict[str, object]:
    return {
        "contract": SCHEMA_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "data_root": DEFAULT_DATA_ROOT.as_posix(),
        "tree_name": DEFAULT_TREE_NAME,
        "class_labels": list(CLASS_LABELS),
        "filename_prefix_to_label": [
            {"prefix": prefix, "label": label}
            for prefix, label in FILENAME_PREFIX_TO_LABEL
        ],
        "raw_feature_branches": list(RAW_FEATURE_BRANCHES),
        "raw_token_features": list(RAW_TOKEN_FEATURES),
        "label_branches": list(LABEL_BRANCHES),
        "max_constituents": MAX_CONSTITUENTS,
    }
