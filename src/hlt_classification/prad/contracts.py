"""Versioned scientific constants for the PRAD campaign."""

from __future__ import annotations

from types import MappingProxyType

PRAD_CAMPAIGN_NAME = "privileged_relational_attention_distillation"

PRAD_SPLIT_CONTRACT = "hlt_classification_prad_split_manifest_v1"
PRAD_SPLIT_SCHEMA_VERSION = 1
PRAD_SPLIT_ROLES = ("train", "val", "test")
PRAD_SPLIT_SIZES = MappingProxyType(
    {"train": 500_000, "val": 150_000, "test": 500_000}
)
PRAD_SPLIT_SEED = 1337
PRAD_SPLIT_ALGORITHM = (
    "capacity_aware_proportional_classwise_without_replacement_then_role_shuffle_v1"
)

PRAD_MATCH_CONTRACT = "hlt_classification_prad_particle_match_v1"
PRAD_MATCH_SCHEMA_VERSION = 1
PRAD_MATCH_ALGORITHM = "charged_neutral_separate_hungarian_v1"

PRAD_CA_TARGET_CONTRACT = "hlt_classification_prad_exclusive_ca_targets_v1"
PRAD_CA_TARGET_SCHEMA_VERSION = 1
PRAD_CA_MULTIPLICITIES = (2, 3, 4)

PRAD_RELATION_CONTRACT = "hlt_classification_prad_relation_v1"
PRAD_RELATION_SCHEMA_VERSION = 1

CURRENT_SOURCE_CAPABILITIES = MappingProxyType(
    {
        "canonical_jet_identity": True,
        "physical_event_id": False,
        "particle_identity": False,
        "track_identity": False,
        "direct_hlt_offline_association": False,
        "offline_vertex_assignment": False,
        "original_jet_radius": False,
        "truth_ancestry": False,
        "fallback_hungarian_matching": True,
        "exclusive_ca_from_four_vectors": True,
    }
)

FORBIDDEN_DEPLOYABLE_FIELDS = frozenset(
    {
        "offline_tokens",
        "offline_target",
        "offline_match",
        "teacher_relation",
        "teacher_bias",
        "teacher_logits",
        "target_mask",
        "class_label",
        "degradation_source_index",
        "construction_index",
    }
)

__all__ = [name for name in globals() if name.startswith("PRAD_")] + [
    "CURRENT_SOURCE_CAPABILITIES",
    "FORBIDDEN_DEPLOYABLE_FIELDS",
]
