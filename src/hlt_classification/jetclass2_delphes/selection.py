"""Labels select physics; directory origin only chooses the registered QCD pool."""
from __future__ import annotations

import numpy as np

from .contracts import artifact, validate
from .schema import CLASS_NAMES, label_map, map_labels


def selection_policy(*, include_signal_file_qcd: bool = False) -> dict:
    return artifact(
        "SELECTION", label_map_sha256=label_map()["content_hash"],
        require_hlt_matched=True, include_signal_file_qcd=include_signal_file_qcd,
        source_categories=["train_higgs2p", "train_qcd"],
        extra_kinematic_cuts=[], minimum_particles=1,
        class_names=list(CLASS_NAMES), class_weighting="natural_unweighted",
    )


def validate_policy(policy: dict) -> None:
    validate(policy, "SELECTION")
    flag = policy.get("include_signal_file_qcd")
    if type(flag) is not bool or policy != selection_policy(include_signal_file_qcd=flag):
        raise ValueError("Selection policy differs from supported semantics")


def selected_mask(raw_labels, matched, source: str, policy: dict) -> tuple[np.ndarray, np.ndarray]:
    validate_policy(policy)
    labels = map_labels(raw_labels)
    matched = np.asarray(matched)
    if matched.shape != labels.shape or matched.dtype != np.bool_:
        raise ValueError("hlt_matched must be a boolean vector aligned with labels")
    if source not in policy["source_categories"]:
        raise ValueError(f"Unknown source directory: {source}")
    # Signal labels in nominal QCD files are not silently lost or renamed.
    mask = matched.copy()
    if source == "train_higgs2p" and not policy["include_signal_file_qcd"]:
        mask &= labels != 0
    return mask, labels
