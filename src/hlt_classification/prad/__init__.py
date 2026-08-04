"""Privileged Relational Attention Distillation research surfaces."""

from .audit import run_prad_data_audit, summarize_prad_sample
from .contracts import (
    PRAD_CAMPAIGN_NAME,
    PRAD_SPLIT_ROLES,
    PRAD_SPLIT_SEED,
    PRAD_SPLIT_SIZES,
)
from .matching import MatchResult, match_hlt_to_offline, pair_supervision_mask
from .relation import (
    ContextualPairRelation,
    GatedRelationBias,
    RelationBiasProjector,
)
from .splits import (
    PradSplitManifest,
    build_prad_split_manifest,
    load_prad_split_manifest,
    save_prad_split_manifest,
)
from .targets import build_exclusive_ca_assignments, same_cluster_targets

__all__ = [
    "ContextualPairRelation",
    "GatedRelationBias",
    "MatchResult",
    "PRAD_CAMPAIGN_NAME",
    "PRAD_SPLIT_ROLES",
    "PRAD_SPLIT_SEED",
    "PRAD_SPLIT_SIZES",
    "PradSplitManifest",
    "RelationBiasProjector",
    "build_exclusive_ca_assignments",
    "build_prad_split_manifest",
    "load_prad_split_manifest",
    "match_hlt_to_offline",
    "pair_supervision_mask",
    "same_cluster_targets",
    "save_prad_split_manifest",
    "run_prad_data_audit",
    "summarize_prad_sample",
]
