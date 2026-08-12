from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting import hcwdl_homotopy_representation_prerequisites as prereq


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prerequisites_reject_incompatible_base_policy_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_recipe = tmp_path / "current_recipe.json"
    historical_root = tmp_path / "historical"
    historical_recipe = historical_root / "recipe.json"
    historical_spec = with_content_hash({
        "contract": "TEST_HISTORICAL_SPEC/v1", "schema_version": 1,
        "recipe_path": str(historical_recipe),
    })
    _write(historical_root / "campaign_spec.json", historical_spec)
    _write(current_recipe, {"recipe_id": "current"})
    _write(historical_recipe, {"recipe_id": "historical"})
    monkeypatch.setattr(prereq, "authenticate_parent", lambda _: {
        "spec": {"recipe_path": str(current_recipe)},
        "spec_sha256": "a" * 64,
    })
    monkeypatch.setattr(
        prereq, "capture_source_snapshot",
        lambda *_args, **_kwargs: {"git_commit": "d" * 40, "worktree_clean": True},
    )
    monkeypatch.setattr(
        prereq, "validate_recipe",
        lambda value, **_: "b" * 64 if value["recipe_id"] == "current" else "c" * 64,
    )
    monkeypatch.setattr(
        prereq, "build_recipe_compatibility",
        lambda **_: (_ for _ in ()).throw(ValueError("execution policies differ")),
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="execution policies differ"):
        prereq.prepare_prerequisites(
            parent_homotopy_spec=tmp_path / "parent.json",
            historical_campaign_root=historical_root,
            historical_project_dir=tmp_path,
            project_dir=tmp_path, output_root=output,
            source_commit="d" * 40,
        )
    assert not output.exists()


def test_recipe_compatibility_allows_only_lineage_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {name: f"value:{name}" for name in prereq._EXECUTION_POLICY_FIELDS}
    policy["class_weights"] = [1.0] * 15
    execution = {
        **policy,
        "class_weighting": {
            "policy": "unweighted_per_jet_population_mean_v1",
            "train_class_counts": [10] * 15,
            "train_row_selection_sha256": "1" * 64,
        },
        "evidence": {"execution": "2" * 64}, "content_hash": "e" * 64,
    }
    donor = {
        **policy,
        "class_weighting": {
            "policy": "unweighted_per_jet_population_mean_v1",
            "train_class_counts": [20] * 15,
            "train_row_selection_sha256": "3" * 64,
        },
        "evidence": {"donor": "4" * 64}, "content_hash": "d" * 64,
    }
    execution_path = tmp_path / "execution.json"; donor_path = tmp_path / "donor.json"
    _write(execution_path, execution); _write(donor_path, donor)
    monkeypatch.setattr(
        prereq, "validate_recipe", lambda value, **_: value["content_hash"],
    )
    artifact = prereq.build_recipe_compatibility(
        execution_recipe_path=execution_path, donor_recipe_path=donor_path,
    )
    assert prereq.validate_recipe_compatibility(
        artifact, execution_recipe=execution,
        representation_recipe={"parents": {"parent_recipe": "d" * 64}},
    ) == artifact["content_hash"]

    donor["training_passes"] = "changed"
    _write(donor_path, donor)
    with pytest.raises(ValueError, match="execution policies differ"):
        prereq.build_recipe_compatibility(
            execution_recipe_path=execution_path, donor_recipe_path=donor_path,
        )


def test_historical_report_registry_is_exact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="registry is incomplete"):
        prereq._report_registry(tmp_path)
