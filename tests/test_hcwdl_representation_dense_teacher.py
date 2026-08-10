from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    sha256_file, validate_content_hash, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_dense_teacher as dense
from hlt_classification.scouting.hcwdl_assignment import (
    validate_train_assignment_authority,
)
from hlt_classification.scouting.hcwdl_representation_contracts import (
    build_versioned_artifact,
)
from hlt_classification.scouting.highcov_cache import (
    publish_assignment_manifest, publish_assignment_shard,
)
from hlt_classification.scouting.highcov_matcher import MatchResult
from hlt_classification.scouting.highcov_resources import (
    resource_validation_report,
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


def test_dense_train_assignment_authority_uses_external_lineage_and_shards(
    tmp_path: Path,
) -> None:
    split = "2" * 64
    selection = "3" * 64
    matcher = resource_validation_report()["content_hash"]
    parents = {
        "split_manifest_sha256": split,
        "row_selection_sha256": selection,
        "matcher_resources_sha256": matcher,
    }
    result = MatchResult(
        concatenated_offline_index=np.asarray([0], np.int32),
        native_offline_index=np.asarray([0], np.int32),
        confidence=np.asarray([1.0], np.float32),
        assignment_score=np.asarray([0.0], np.float32),
        accepted=np.asarray([True]),
    )
    publish_assignment_shard(
        tmp_path / "shard_0000", source_path="source.root", role="train",
        source_fold=0, entries=[7], hlt_categories=[np.asarray([0], np.int8)],
        results=[result], parents={**parents, "source_file_sha256": "4" * 64},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = publish_assignment_manifest(
        manifest_path, role="train",
        shard_metadata_paths=[tmp_path / "shard_0000.json"],
        expected_mapped_jets=1, parents=parents,
    )
    assert validate_train_assignment_authority(
        manifest_path, split_manifest_sha256=split,
        row_selection_sha256=selection, expected_mapped_jets=1,
    ) == manifest["content_hash"]

    with pytest.raises(ValueError, match="role or parents"):
        validate_train_assignment_authority(
            manifest_path, split_manifest_sha256=split,
            row_selection_sha256="5" * 64, expected_mapped_jets=1,
        )
    with pytest.raises(ValueError, match="mapped-jet coverage"):
        validate_train_assignment_authority(
            manifest_path, split_manifest_sha256=split,
            row_selection_sha256=selection, expected_mapped_jets=2,
        )
