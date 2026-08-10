"""Narrow historical TOFF-teacher authority for the dense RKD descent.

This contract deliberately does *not* import the historical HCWDL ladder.
It authenticates one native-offline (TOFF) checkpoint as training-only input
to the two D100 roots.  Every later teacher is produced inside the new dense
campaign.  In particular, this artifact cannot authorize parent M-rungs,
finalists, deployable publication, final-test access, or scientific claims
about a corrected full HCWDL parent.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from hlt_classification.data.cache_contracts import (
    load_json, require_sha256, sha256_file, validate_content_hash,
)
from hlt_classification.provenance import (
    capture_source_snapshot, validate_source_snapshot_payload,
)

from .engine import validate_pmard_training_report
from .hcwdl_recipe import PRIMARY_RECIPE_PROFILE, validate_recipe
from .hcwdl_representation_contracts import (
    DENSE_TEACHER_IMPORT_CONTRACT, build_versioned_artifact,
    validate_parent_hashes, validate_versioned_artifact,
)
from .selective_assignment import validate_row_selection


DENSE_TEACHER_CONTRACT: Final = DENSE_TEACHER_IMPORT_CONTRACT
DENSE_TEACHER_SCHEMA_VERSION: Final = 1
SUPPORTED_HISTORICAL_SOURCE_COMMITS: Final = frozenset({
    "b3154d67c4a7a21d027c3f8b9be5fbcdf885402f",
})
DENSE_TEACHER_FILE_KEYS: Final = frozenset({
    "campaign_spec", "recipe", "source_manifest", "split_manifest",
    "row_selection", "surface_parity", "toff_wrapper_report",
})
_PARENT_KEYS: Final = frozenset({
    "historical_campaign_spec", "historical_recipe",
    "historical_source_manifest", "historical_split_manifest",
    "historical_row_selection", "historical_source_snapshot",
    "surface_parity", "toff_wrapper_report", "toff_engine_report",
    "toff_selected_checkpoint",
})
_PAYLOAD_KEYS: Final = frozenset({
    "authority_derived_from_registered_files", "compatibility_policy",
    "historical_campaign_contract", "historical_source_commit",
    "historical_parent_graph_sha256",
    "historical_recipe_contract", "historical_recipe_profile",
    "historical_class_weight_policy", "teacher_node_id", "teacher_domain",
    "teacher_scope", "wrapper_report_path", "engine_report_path",
    "selected_checkpoint_path", "source_snapshot", "training_only",
    "full_parent_authority", "deployable_publication_authorized",
    "finalist_authority", "final_role_access_authorized",
})


def _absolute_regular(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute() or not candidate.is_file()
        or candidate.is_symlink()
    ):
        raise ValueError(f"dense teacher {name} must be an absolute regular file")
    return candidate.resolve()


def _artifact(path: str | Path, *, name: str) -> tuple[Path, Mapping[str, Any], str]:
    resolved = _absolute_regular(path, name=name)
    value = load_json(resolved)
    if not isinstance(value, Mapping):
        raise TypeError(f"dense teacher {name} artifact is not an object")
    contract = value.get("contract")
    schema = value.get("schema_version")
    if not isinstance(contract, str) or isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError(f"dense teacher {name} artifact is not versioned")
    digest = validate_content_hash(
        value, expected_contract=contract, expected_schema_version=schema,
    )
    return resolved, value, digest


def _validate_historical_campaign(value: Mapping[str, Any]) -> str:
    from .hcwdl_authorization import AUTOMATIC_ENDPOINT_CONTINUATION
    from .hcwdl_campaign import CAMPAIGN_CONTRACT, validate_campaign_spec

    if value.get("contract") != CAMPAIGN_CONTRACT or value.get("schema_version") != 7:
        raise ValueError("dense teacher requires the historical HCWDL v7 campaign")
    digest = validate_campaign_spec(value, executable=True)
    if (
        value.get("mode") == "smoke"
        or value.get("source_commit") not in SUPPORTED_HISTORICAL_SOURCE_COMMITS
        or value.get("endpoint_continuation") != AUTOMATIC_ENDPOINT_CONTINUATION
    ):
        raise PermissionError("historical TOFF campaign is outside the frozen compatibility policy")
    return digest


def _validate_wrapper_engine_checkpoint(
    wrapper_path: Path, *, recipe_sha256: str,
) -> tuple[Mapping[str, Any], str, Path, Mapping[str, Any], str, Path, str]:
    wrapper = load_json(wrapper_path)
    wrapper_sha256 = validate_content_hash(
        wrapper, expected_contract="HCWDL_TRAINING_REPORT/v1",
        expected_schema_version=1,
    )
    parents = wrapper.get("parents")
    if (
        wrapper.get("node_id") != "TOFF" or wrapper.get("complete") is not True
        or wrapper.get("recipe_sha256") != recipe_sha256
        or not isinstance(parents, Mapping) or parents.get("recipe") != recipe_sha256
    ):
        raise ValueError("historical TOFF wrapper lineage differs")
    engine_path = wrapper_path.with_name("training_report.json")
    if not engine_path.is_file() or engine_path.is_symlink():
        raise FileNotFoundError("historical TOFF engine report is absent")
    engine_path = engine_path.resolve()
    engine = load_json(engine_path)
    engine_sha256 = validate_pmard_training_report(engine)
    engine_parents = engine.get("parents")
    scientific = engine.get("scientific_config")
    if (
        wrapper.get("pmard_engine_report_sha256") != engine_sha256
        or wrapper.get("selected_checkpoint_sha256")
        != engine.get("selected_checkpoint_sha256")
        or not isinstance(engine_parents, Mapping)
        or engine_parents.get("recipe") != recipe_sha256
        or not isinstance(scientific, Mapping)
        or scientific.get("recipe_sha256") != recipe_sha256
        or engine.get("complete") is not True
    ):
        raise ValueError("historical TOFF wrapper/engine lineage differs")
    selected_name = engine.get("selected_checkpoint")
    if not isinstance(selected_name, str) or Path(selected_name).name != selected_name:
        raise ValueError("historical TOFF selected checkpoint name differs")
    checkpoint = engine_path.parent / selected_name
    checkpoint_sha256 = require_sha256(
        engine.get("selected_checkpoint_sha256"), name="historical TOFF checkpoint",
    )
    if (
        not checkpoint.is_file() or checkpoint.is_symlink()
        or sha256_file(checkpoint) != checkpoint_sha256
    ):
        raise ValueError("historical TOFF selected checkpoint bytes differ")
    return (
        wrapper, wrapper_sha256, engine_path, engine, engine_sha256,
        checkpoint.resolve(), checkpoint_sha256,
    )


def _derive_authority(
    *, authority_files: Mapping[str, str | Path], historical_project_dir: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not isinstance(authority_files, Mapping) or set(authority_files) != DENSE_TEACHER_FILE_KEYS:
        raise ValueError("dense teacher authority-file registry differs")
    opened = {
        name: _artifact(path, name=name)
        for name, path in sorted(authority_files.items())
    }
    campaign_path, campaign, campaign_sha256 = opened["campaign_spec"]
    del campaign_path
    _validate_historical_campaign(campaign)
    recipe_path, recipe, recipe_sha256 = opened["recipe"]
    del recipe_path
    validate_recipe(
        recipe, require_authorized=True, expected_profile=PRIMARY_RECIPE_PROFILE,
    )
    if (
        recipe.get("contract") != "HCWDL_RECIPE/v4"
        or recipe.get("class_weighting", {}).get("policy")
        != "unweighted_per_jet_population_mean_v1"
        or recipe.get("class_weights") != [1.0] * 15
        or campaign.get("recipe_sha256") != recipe_sha256
    ):
        raise PermissionError("historical TOFF recipe is not the exact unweighted v4 recipe")

    _, source_manifest, source_sha256 = opened["source_manifest"]
    _, split_manifest, split_sha256 = opened["split_manifest"]
    _, row_selection, selection_sha256 = opened["row_selection"]
    validate_row_selection(
        row_selection, split_manifest_sha256=split_sha256,
    )
    if (
        campaign.get("source_manifest_sha256") != source_sha256
        or campaign.get("split_manifest_sha256") != split_sha256
        or recipe.get("class_weighting", {}).get("train_row_selection_sha256")
        != selection_sha256
        or row_selection.get("split_manifest_sha256") != split_sha256
    ):
        raise ValueError("historical TOFF data/selection lineage differs")

    _, parity, parity_sha256 = opened["surface_parity"]
    from hlt_classification.models.hcwdl_surfaces import validate_surface_parity_report
    validate_surface_parity_report(parity)
    if parity.get("authorization_capable") is not True:
        raise PermissionError("installed-Weaver surface parity is not authorizing")

    wrapper_path = opened["toff_wrapper_report"][0]
    (
        _wrapper, wrapper_sha256, engine_path, _engine, engine_sha256,
        checkpoint_path, checkpoint_sha256,
    ) = _validate_wrapper_engine_checkpoint(
        wrapper_path, recipe_sha256=recipe_sha256,
    )

    project = Path(historical_project_dir)
    if not project.is_absolute() or not project.is_dir() or project.is_symlink():
        raise ValueError("historical TOFF project must be an absolute Git worktree")
    snapshot = capture_source_snapshot(project.resolve(), require_clean=True)
    snapshot_sha256 = validate_source_snapshot_payload(snapshot)
    if (
        snapshot.get("git_commit") != campaign.get("source_commit")
        or snapshot.get("git_commit") not in SUPPORTED_HISTORICAL_SOURCE_COMMITS
        or snapshot.get("worktree_clean") is not True
    ):
        raise PermissionError("historical TOFF source checkout differs from its campaign")

    parents = {
        "historical_campaign_spec": campaign_sha256,
        "historical_recipe": recipe_sha256,
        "historical_source_manifest": source_sha256,
        "historical_split_manifest": split_sha256,
        "historical_row_selection": selection_sha256,
        "historical_source_snapshot": snapshot_sha256,
        "surface_parity": parity_sha256,
        "toff_wrapper_report": wrapper_sha256,
        "toff_engine_report": engine_sha256,
        "toff_selected_checkpoint": checkpoint_sha256,
    }
    payload = {
        "authority_derived_from_registered_files": True,
        "compatibility_policy": "exact_b315_unweighted_v4_toff_training_teacher/v1",
        "historical_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v7",
        "historical_parent_graph_sha256": require_sha256(
            campaign.get("graph_sha256"), name="historical HCWDL graph",
        ),
        "historical_source_commit": campaign["source_commit"],
        "historical_recipe_contract": "HCWDL_RECIPE/v4",
        "historical_recipe_profile": PRIMARY_RECIPE_PROFILE,
        "historical_class_weight_policy": "unweighted_per_jet_population_mean_v1",
        "teacher_node_id": "TOFF",
        "teacher_domain": "toff",
        "teacher_scope": ["RSET_D100", "RREL_D100"],
        "wrapper_report_path": str(wrapper_path),
        "engine_report_path": str(engine_path),
        "selected_checkpoint_path": str(checkpoint_path),
        "source_snapshot": snapshot,
        "training_only": True,
        "full_parent_authority": False,
        "deployable_publication_authorized": False,
        "finalist_authority": False,
        "final_role_access_authorized": False,
    }
    return parents, payload


def build_dense_teacher_import_from_files(
    *, authority_files: Mapping[str, str | Path], historical_project_dir: str | Path,
) -> dict[str, Any]:
    parents, payload = _derive_authority(
        authority_files=authority_files,
        historical_project_dir=historical_project_dir,
    )
    result = build_versioned_artifact(
        DENSE_TEACHER_CONTRACT, parents=parents, payload=payload,
    )
    validate_dense_teacher_import(result)
    return result


def validate_dense_teacher_import(value: Mapping[str, Any]) -> str:
    digest = validate_versioned_artifact(
        value, expected_contract=DENSE_TEACHER_CONTRACT,
        required_payload_keys=tuple(sorted(_PAYLOAD_KEYS)),
    )
    parents = validate_parent_hashes(value.get("parents", {}))
    payload = value.get("payload")
    if set(parents) != _PARENT_KEYS or not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("dense teacher import schema differs")
    source = payload.get("source_snapshot")
    if not isinstance(source, Mapping):
        raise ValueError("dense teacher source snapshot differs")
    source_sha256 = validate_source_snapshot_payload(source)
    exact = {
        "authority_derived_from_registered_files": True,
        "compatibility_policy": "exact_b315_unweighted_v4_toff_training_teacher/v1",
        "historical_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v7",
        "historical_recipe_contract": "HCWDL_RECIPE/v4",
        "historical_recipe_profile": PRIMARY_RECIPE_PROFILE,
        "historical_class_weight_policy": "unweighted_per_jet_population_mean_v1",
        "teacher_node_id": "TOFF", "teacher_domain": "toff",
        "teacher_scope": ["RSET_D100", "RREL_D100"],
        "training_only": True, "full_parent_authority": False,
        "deployable_publication_authorized": False,
        "finalist_authority": False, "final_role_access_authorized": False,
    }
    if any(payload.get(name) != expected for name, expected in exact.items()):
        raise PermissionError("dense teacher authority boundary differs")
    if (
        payload.get("historical_source_commit") not in SUPPORTED_HISTORICAL_SOURCE_COMMITS
        or source.get("git_commit") != payload.get("historical_source_commit")
        or source.get("worktree_clean") is not True
        or parents.get("historical_source_snapshot") != source_sha256
    ):
        raise PermissionError("dense teacher historical source binding differs")
    require_sha256(
        payload.get("historical_parent_graph_sha256"),
        name="dense teacher historical graph",
    )
    wrapper_path = _absolute_regular(
        str(payload.get("wrapper_report_path", "")), name="wrapper_report_path",
    )
    wrapper = load_json(wrapper_path)
    wrapper_sha256 = validate_content_hash(
        wrapper, expected_contract="HCWDL_TRAINING_REPORT/v1",
        expected_schema_version=1,
    )
    if wrapper_sha256 != parents["toff_wrapper_report"]:
        raise PermissionError("dense teacher wrapper report content differs")

    engine_path = _absolute_regular(
        str(payload.get("engine_report_path", "")), name="engine_report_path",
    )
    engine_sha256 = validate_pmard_training_report(load_json(engine_path))
    if engine_sha256 != parents["toff_engine_report"]:
        raise PermissionError("dense teacher engine report content differs")

    checkpoint_path = _absolute_regular(
        str(payload.get("selected_checkpoint_path", "")),
        name="selected_checkpoint_path",
    )
    if sha256_file(checkpoint_path) != parents["toff_selected_checkpoint"]:
        raise PermissionError("dense teacher selected checkpoint bytes differ")
    return digest


def validate_dense_teacher_import_against_files(
    value: Mapping[str, Any], *, authority_files: Mapping[str, str | Path],
    historical_project_dir: str | Path,
) -> str:
    digest = validate_dense_teacher_import(value)
    rebuilt = build_dense_teacher_import_from_files(
        authority_files=authority_files,
        historical_project_dir=historical_project_dir,
    )
    if dict(value) != rebuilt:
        raise PermissionError("dense teacher import differs from registered authority files")
    return digest


__all__ = [
    "DENSE_TEACHER_CONTRACT", "DENSE_TEACHER_FILE_KEYS",
    "SUPPORTED_HISTORICAL_SOURCE_COMMITS",
    "build_dense_teacher_import_from_files", "validate_dense_teacher_import",
    "validate_dense_teacher_import_against_files",
]
