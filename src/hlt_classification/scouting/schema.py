"""Immutable ScoutingAK8 source, label, collection, and input contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Final

from hlt_classification.data.cache_contracts import canonical_sha256

SCOUTING_SCHEMA_CONTRACT: Final = "hlt_classification_scouting_schema_v1"
SCOUTING_SCHEMA_VERSION: Final = 1
TREE_NAME: Final = "tree"
DEFAULT_DATA_ROOT: Final = "/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train"

EXPECTED_FILE_COUNT: Final = 53
EXPECTED_RAW_ENTRIES: Final = 5_084_737
EXPECTED_BRANCH_COUNT: Final = 297
EXPECTED_BASELINE_ENTRIES: Final = 4_638_163
EXPECTED_MAPPED_ENTRIES: Final = 4_635_175
EXPECTED_UNMAPPED_ENTRIES: Final = 2_988
EXPECTED_LOGICAL_CONTENT_SHA256: Final = (
    "30ba783bf08be19c65e9dfeb0dcced1b9634ce8a6387bf066e65969393336630"
)

CMSSW_PREPROCESS_COMMIT: Final = "37a104b51c8458bcda97a95ba52d304a5041e922"
CMSSW_PREPROCESS_SHA256: Final = (
    "2e346c5da8491046205f29cba21b0f6e434e273eb89dae068aa5dd40e2ca91f7"
)
WEAVER_REFERENCE_COMMIT: Final = "6a084ec341f5af2e06f6d884e6a4ee4dab8bb4e4"
TOFF_PREPROCESS_SHA256: Final = (
    "32da128ad0e82c6933a36cf409ae766523a116ae66df8d43153c671bc89f7c1d"
)

CLASS_NAMES: Final = (
    "QCD", "Xbb", "Xcc", "Xss", "Xqq", "Xbs", "Xgg", "Xee", "Xmm",
    "Xtauhtaue", "Xtauhtaum", "Xtauhtauh", "Xbc", "Xcs", "Xud",
)
CLASS_TO_INDEX = MappingProxyType({name: index for index, name in enumerate(CLASS_NAMES)})
EXPECTED_CLASS_COUNTS = MappingProxyType({
    "QCD": 3_342_364, "Xbb": 89_046, "Xcc": 88_789, "Xss": 87_849,
    "Xqq": 88_318, "Xbs": 88_310, "Xgg": 89_405, "Xee": 34_295,
    "Xmm": 32_498, "Xtauhtaue": 37_114, "Xtauhtaum": 35_305,
    "Xtauhtauh": 72_757, "Xbc": 184_037, "Xcs": 182_558,
    "Xud": 182_530,
})

BASELINE_BRANCHES: Final = (
    "jet_tightId", "jet_no", "scoutfj_pt", "scoutfj_sdmass", "n_scoutpfcands",
)
LABEL_BRANCHES: Final = ("scoutfj_label", "scoutfj_gen_pid", "fj_label")
OBSERVER_BRANCHES: Final = (
    "event_no", "jet_no", "sample_isQCD", "scoutfj_pt", "scoutfj_eta",
    "scoutfj_phi", "scoutfj_mass", "scoutfj_sdmass", "fj_pt", "fj_eta",
    "fj_phi", "fj_mass", "fj_sdmass", "n_scoutpfcands", "n_cpfcands",
    "n_lts", "n_npfcands", "n_sv",
)

HLT_MAX_LENGTH: Final = 200
HLT_MIN_DEPLOYMENT_LENGTH: Final = 16
OFFLINE_CHARGED_MAX_LENGTH: Final = 90
OFFLINE_NEUTRAL_MAX_LENGTH: Final = 60


@dataclass(frozen=True)
class FeatureSpec:
    output_name: str
    branch: str
    median: float = 0.0
    factor: float = 1.0
    lower: float = -1.0e32
    upper: float = 1.0e32


HLT_FEATURE_SPECS: Final = (
    FeatureSpec("pfcand_quality", "scoutpfcand_quality", 0, .2, -5, 5),
    FeatureSpec("pfcand_charge", "scoutpfcand_charge"),
    FeatureSpec("pfcand_isEl", "scoutpfcand_isEl"),
    FeatureSpec("pfcand_isMu", "scoutpfcand_isMu"),
    FeatureSpec("pfcand_isChargedHad", "scoutpfcand_isChargedHad"),
    FeatureSpec("pfcand_isGamma", "scoutpfcand_isGamma"),
    FeatureSpec("pfcand_isNeutralHad", "scoutpfcand_isNeutralHad"),
    FeatureSpec("pfcand_phirel", "scoutpfcand_phirel"),
    FeatureSpec("pfcand_etarel", "scoutpfcand_etarel"),
    FeatureSpec("pfcand_abseta", "scoutpfcand_abseta", .6, 1.6, -5, 5),
    FeatureSpec("pfcand_pt_log", "scoutpfcand_pt_log", 1, .5, -5, 5),
    FeatureSpec("pfcand_normchi2", "scoutpfcand_normchi2", 5, .2, -5, 5),
    FeatureSpec("pfcand_dz", "scoutpfcand_dz", 0, 180, -5, 5),
    FeatureSpec("pfcand_dxy", "scoutpfcand_dxy", 0, 300, -5, 5),
    FeatureSpec("pfcand_dxysig", "scoutpfcand_dxysig", 0, 1, -5, 5),
    FeatureSpec("pfcand_btagEtaRel", "scoutpfcand_btagEtaRel", 1.5, .5, -5, 5),
    FeatureSpec("pfcand_btagPtRatio", "scoutpfcand_btagPtRatio", 0, 1, -5, 5),
    FeatureSpec("pfcand_btagPParRatio", "scoutpfcand_btagPParRatio", 0, 1, -5, 5),
    FeatureSpec("pfcand_dzsig", "scoutpfcand_dzsig", 0, .9, -5, 5),
    FeatureSpec("pfcand_e_log_nopuppi", "scoutpfcand_e_log", 1.3, .5, -5, 5),
    FeatureSpec("pfcand_lostInnerHits", "scoutpfcand_lostInnerHits"),
)
HLT_VECTOR_BRANCHES: Final = (
    "scoutpfcand_px", "scoutpfcand_py", "scoutpfcand_pz", "scoutpfcand_energy",
)

TOFF_CHARGED_FEATURES: Final = (
    "pt_log_nopuppi", "e_log_nopuppi", "etarel", "phirel", "abseta",
    "charge", "isEl", "isMu", "isChargedHad", "lostInnerHits", "normchi2",
    "quality", "dz", "dzsig", "dxy", "dxysig", "btagEtaRel",
    "btagPtRatio", "btagPParRatio",
)
TOFF_NEUTRAL_FEATURES: Final = (
    "pt_log_nopuppi", "e_log_nopuppi", "etarel", "phirel", "abseta",
    "isGamma", "isNeutralHad",
)
TOFF_VECTOR_FIELDS: Final = ("px", "py", "pz", "energy")

COLLECTION_FIELDS = MappingProxyType({
    "scoutpfcand": 31,
    "cpfcandlt": 44,
    "npfcand": 18,
    "sv": 27,
})
COLLECTION_COUNT_BRANCHES = MappingProxyType({
    "scoutpfcand": "n_scoutpfcands",
    "npfcand": "n_npfcands",
    "sv": "n_sv",
})

MODEL_VISIBLE_BRANCHES = frozenset(
    spec.branch for spec in HLT_FEATURE_SPECS
) | frozenset(HLT_VECTOR_BRANCHES)
FORBIDDEN_DEPLOYABLE_FIELDS = frozenset(LABEL_BRANCHES + OBSERVER_BRANCHES) | frozenset({
    "offline", "match_index", "match_confidence", "match_mask", "alpha",
    "teacher_logits", "teacher_representation", "source_file", "source_entry",
})


def hlt_required_branches() -> frozenset[str]:
    return frozenset(MODEL_VISIBLE_BRANCHES)


def native_offline_required_branches() -> frozenset[str]:
    charged = {f"cpfcandlt_{name}" for name in TOFF_CHARGED_FEATURES + TOFF_VECTOR_FIELDS}
    neutral = {f"npfcand_{name}" for name in TOFF_NEUTRAL_FEATURES + TOFF_VECTOR_FIELDS}
    return frozenset(charged | neutral)


def matching_required_branches() -> frozenset[str]:
    hlt = {
        f"scoutpfcand_{name}" for name in (
            "px", "py", "pz", "energy", "charge", "isEl", "isMu",
            "isChargedHad", "isGamma", "isNeutralHad",
        )
    }
    charged = {
        f"cpfcandlt_{name}" for name in (
            "px", "py", "pz", "energy", "charge", "isEl", "isMu",
            "isChargedHad", "isLostTrack",
        )
    }
    neutral = {
        f"npfcand_{name}" for name in (
            "px", "py", "pz", "energy", "isGamma", "isNeutralHad",
        )
    }
    return frozenset(hlt | charged | neutral)


def schema_payload() -> dict[str, object]:
    return {
        "contract": SCOUTING_SCHEMA_CONTRACT,
        "schema_version": SCOUTING_SCHEMA_VERSION,
        "tree": TREE_NAME,
        "class_names": list(CLASS_NAMES),
        "hlt_features": [asdict(item) for item in HLT_FEATURE_SPECS],
        "hlt_vectors": list(HLT_VECTOR_BRANCHES),
        "hlt_max_length": HLT_MAX_LENGTH,
        "toff_charged_features": list(TOFF_CHARGED_FEATURES),
        "toff_neutral_features": list(TOFF_NEUTRAL_FEATURES),
        "toff_lengths": [OFFLINE_CHARGED_MAX_LENGTH, OFFLINE_NEUTRAL_MAX_LENGTH],
        "source_logical_content_sha256": EXPECTED_LOGICAL_CONTENT_SHA256,
    }


SCOUTING_SCHEMA_SHA256: Final = canonical_sha256(schema_payload())

__all__ = [name for name in globals() if name.isupper()] + [
    "FeatureSpec", "hlt_required_branches", "matching_required_branches", "native_offline_required_branches",
    "schema_payload",
]
