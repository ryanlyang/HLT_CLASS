"""Semantic and byte authentication for vendored Scouting preprocessing data."""

from __future__ import annotations

import json
from pathlib import Path

from hlt_classification.data.cache_contracts import sha256_file, with_content_hash
from .schema import (
    CMSSW_PREPROCESS_COMMIT, CMSSW_PREPROCESS_SHA256, HLT_FEATURE_SPECS,
    HLT_MAX_LENGTH, TOFF_CHARGED_FEATURES, TOFF_NEUTRAL_FEATURES,
    TOFF_PREPROCESS_SHA256, WEAVER_REFERENCE_COMMIT,
)


def validate_vendored_preprocessing(repository_root: str | Path) -> dict[str, object]:
    root = Path(repository_root)
    hlt_path = root / "configs/scouting/GlobalParticleTransformerAK8_V00_preprocess.json"
    toff_path = root / "configs/scouting/toff_v1_contract.json"
    with hlt_path.open(encoding="utf-8") as stream: hlt = json.load(stream)
    with toff_path.open(encoding="utf-8") as stream: toff = json.load(stream)
    if hlt["pf_features"]["var_names"] != [item.output_name for item in HLT_FEATURE_SPECS]:
        raise ValueError("vendored HLT feature order differs")
    if hlt["pf_features"]["max_length"] != HLT_MAX_LENGTH:
        raise ValueError("vendored HLT length differs")
    if CMSSW_PREPROCESS_COMMIT not in hlt["source"]:
        raise ValueError("vendored HLT upstream commit differs")
    if toff["charged"]["features"] != list(TOFF_CHARGED_FEATURES) or toff["neutral"]["features"] != list(TOFF_NEUTRAL_FEATURES):
        raise ValueError("vendored TOFF feature order differs")
    if toff["documented_reference_sha256"] != TOFF_PREPROCESS_SHA256 or toff["weaver_reference_commit"] != WEAVER_REFERENCE_COMMIT:
        raise ValueError("vendored TOFF reference lineage differs")
    return with_content_hash({
        "contract": "hlt_classification_scouting_vendored_preprocessing_v1",
        "schema_version": 1,
        "hlt_file_sha256": sha256_file(hlt_path),
        "hlt_documented_reference_sha256": CMSSW_PREPROCESS_SHA256,
        "hlt_upstream_commit": CMSSW_PREPROCESS_COMMIT,
        "toff_file_sha256": sha256_file(toff_path),
        "toff_documented_reference_sha256": TOFF_PREPROCESS_SHA256,
        "weaver_reference_commit": WEAVER_REFERENCE_COMMIT,
    })


__all__ = ["validate_vendored_preprocessing"]
