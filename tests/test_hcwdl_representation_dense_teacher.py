from __future__ import annotations

from pathlib import Path

from hlt_classification.data.cache_contracts import (
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_dense_teacher as dense
from hlt_classification.scouting.hcwdl_representation_contracts import (
    build_versioned_artifact,
)


H = "1" * 64
COMMIT = "b3154d67c4a7a21d027c3f8b9be5fbcdf885402f"


def test_dense_teacher_reopens_json_content_hashes_and_checkpoint_bytes(
    tmp_path: Path, monkeypatch,
) -> None:
    wrapper = with_content_hash({
        "contract": "HCWDL_TRAINING_REPORT/v1", "schema_version": 1,
    })
    engine = with_content_hash({
        "contract": "TEST_PMARD_REPORT/v1", "schema_version": 1,
    })
    wrapper_path = tmp_path / "hcwdl_training_report.json"
    engine_path = tmp_path / "training_report.json"
    checkpoint_path = tmp_path / "selected.pt"
    write_immutable_json(wrapper_path, wrapper)
    write_immutable_json(engine_path, engine)
    checkpoint_path.write_bytes(b"checkpoint-bytes")
    assert sha256_file(wrapper_path) != wrapper["content_hash"]
    assert sha256_file(engine_path) != engine["content_hash"]

    monkeypatch.setattr(
        dense, "validate_pmard_training_report",
        lambda value: validate_content_hash(
            value, expected_contract="TEST_PMARD_REPORT/v1",
            expected_schema_version=1,
        ),
    )
    monkeypatch.setattr(
        dense, "validate_source_snapshot_payload", lambda _value: H,
    )
    parents = {
        "historical_campaign_spec": H,
        "historical_recipe": H,
        "historical_source_manifest": H,
        "historical_split_manifest": H,
        "historical_row_selection": H,
        "historical_source_snapshot": H,
        "surface_parity": H,
        "toff_wrapper_report": wrapper["content_hash"],
        "toff_engine_report": engine["content_hash"],
        "toff_selected_checkpoint": sha256_file(checkpoint_path),
    }
    payload = {
        "authority_derived_from_registered_files": True,
        "compatibility_policy": "exact_b315_unweighted_v4_toff_training_teacher/v1",
        "historical_campaign_contract": "HCWDL_CAMPAIGN_SPEC/v7",
        "historical_source_commit": COMMIT,
        "historical_parent_graph_sha256": H,
        "historical_recipe_contract": "HCWDL_RECIPE/v4",
        "historical_recipe_profile": dense.PRIMARY_RECIPE_PROFILE,
        "historical_class_weight_policy": "unweighted_per_jet_population_mean_v1",
        "teacher_node_id": "TOFF",
        "teacher_domain": "toff",
        "teacher_scope": ["RSET_D100", "RREL_D100"],
        "wrapper_report_path": str(wrapper_path.resolve()),
        "engine_report_path": str(engine_path.resolve()),
        "selected_checkpoint_path": str(checkpoint_path.resolve()),
        "source_snapshot": {"git_commit": COMMIT, "worktree_clean": True},
        "training_only": True,
        "full_parent_authority": False,
        "deployable_publication_authorized": False,
        "finalist_authority": False,
        "final_role_access_authorized": False,
    }
    artifact = build_versioned_artifact(
        dense.DENSE_TEACHER_CONTRACT, parents=parents, payload=payload,
    )
    assert dense.validate_dense_teacher_import(artifact) == artifact["content_hash"]

    checkpoint_path.write_bytes(b"changed-checkpoint")
    try:
        dense.validate_dense_teacher_import(artifact)
    except PermissionError as error:
        assert "checkpoint bytes differ" in str(error)
    else:
        raise AssertionError("changed checkpoint bytes were accepted")
