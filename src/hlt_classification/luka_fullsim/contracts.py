"""Separate FullSim identity and selection; no Delphes semantic impersonation."""
from __future__ import annotations

from pathlib import Path

from hlt_classification.data.cache_contracts import (
    canonical_sha256, validate_content_hash, with_content_hash,
)
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, RAW_TO_CLASS

SOURCE = "ryang@lxplus.cern.ch:/eos/user/l/llambrec/jetclass/output_fullsim_630k_trial"
SCALARS = ("jet_label", "hlt_matched", "jet_nparticles", "hlt_jet_nparticles", "jet_pt")
ROLES = ("train", "validation", "final_test")
SEED = 20261008


def artifact(kind: str, **fields) -> dict:
    return with_content_hash(dict(contract=f"LUKA_FULLSIM_{kind}/v1", schema_version=1, **fields))


def validate(value: dict, kind: str) -> str:
    return validate_content_hash(value, expected_contract=f"LUKA_FULLSIM_{kind}/v1")


def selection() -> dict:
    return dict(offline_pt_min_gev=200, comparison="strict_greater_than",
                pt_interpretation="user_agreed_offline_not_producer_confirmed",
                hlt_pt_cut=None, require_hlt_matched=True, require_both_nonempty=True,
                qcd_source="train_qcd_only", other_cuts=[],
                class_names=list(CLASS_NAMES),
                raw_to_class={str(k): v for k, v in sorted(RAW_TO_CLASS.items())},
                label_semantics="provisional_existing_JetClass_codes",
                producer_revision_confirmed=False,
                tracking_units_and_sentinels="unresolved_not_used_in_stage1")


def source_path(raw: Path, relative: str) -> tuple[Path, str]:
    path = relative_file(raw, relative)
    parts = relative.split("/")
    if (len(parts) != 4 or parts[0] != "jetclass2"
            or parts[1] not in ("train_higgs2p", "train_qcd")
            or parts[2] != "fullsim_offline+hlt" or not parts[3].endswith(".root")):
        raise ValueError(f"Unexpected FullSim source path: {relative}")
    # Symlinks are not accepted as independent split groups, even within raw/.
    if any((raw / Path(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1)):
        raise ValueError(f"Symlink source alias: {relative}")
    return path, parts[1]


def row_identity(record: dict, entry: int) -> str:
    return canonical_sha256(dict(domain="LUKA_FULLSIM_ROW/v1", path=record["path"],
                                 sha256=record["sha256"], tree=record["tree_key"], entry=int(entry)))
