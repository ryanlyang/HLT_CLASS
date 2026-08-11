from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting import hcwdl_representation_bootstrap as bootstrap
from hlt_classification.scouting.hcwdl_representation_campaign import create_campaign_spec


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime = with_content_hash({"fixture": "runtime"})
    runtime_path = tmp_path / "runtime.json"
    write_immutable_json(runtime_path, runtime)
    root = tmp_path / "campaign"
    planning = create_campaign_spec(
        mode="smoke", campaign_root=root,
        checkpoint_namespace=tmp_path / "checkpoints",
        project_dir=tmp_path / "project", source_commit="1" * 40,
        source_manifest_sha256="2" * 64, split_manifest_sha256="3" * 64,
        parent_import_sha256="4" * 64,
        representation_recipe_sha256="5" * 64, graph_sha256="6" * 64,
        disposition_sha256="7" * 64,
        disposition="validation_only_parent_claim_consumed",
        role_counts={"train": 512, "validation": 256, "final_test": 256},
        final_source_partitions=1, combined_finalist_count=1,
        runtime_binding_sha256=runtime["content_hash"],
        artifact_paths={
            "source_manifest": root / "inputs/source.json",
            "split_manifest": root / "inputs/split.json",
            "parent_import": root / "import/parent.json",
            "representation_graph": root / "graph/graph.json",
            "representation_recipe": root / "recipes/recipe.json",
            "final_disposition": root / "import/disposition.json",
            "runtime_binding": runtime_path,
        },
    )
    planning_path = root / "campaign_spec.json"
    write_immutable_json(planning_path, planning)
    ordinary = tmp_path / bootstrap.BOOTSTRAP_WORKER_NAMES["ordinary"]
    deterministic = tmp_path / bootstrap.BOOTSTRAP_WORKER_NAMES["deterministic"]
    ordinary.write_text("#!/bin/bash\nordinary\n", encoding="utf-8")
    deterministic.write_text("#!/bin/bash\ndeterministic\n", encoding="utf-8")
    monkeypatch.setattr(
        bootstrap, "validate_runtime_binding",
        lambda value, *, spec: value["content_hash"],
    )
    import hlt_classification.scouting.hcwdl_representation_runtime_rows as runtime_rows
    monkeypatch.setattr(
        runtime_rows, "validate_bound_runtime_task_rows", lambda spec, value: None,
        raising=False,
    )
    return planning, planning_path, runtime, runtime_path, ordinary, deterministic


def test_bootstrap_is_bounded_dependency_closed_and_nonauthorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning, planning_path, _, runtime_path, ordinary, deterministic = _fixture(
        tmp_path, monkeypatch,
    )
    stop = bootstrap.SAFE_BOOTSTRAP_TASK_PREFIX.index("smoke_probe") + 1
    value = bootstrap.build_acceptance_bootstrap(
        planning_spec_path=planning_path,
        runtime_binding_path=runtime_path,
        ordinary_worker_path=ordinary,
        deterministic_worker_path=deterministic,
        authorized_tasks=bootstrap.SAFE_BOOTSTRAP_TASK_PREFIX[:stop],
        authorization_phrase=bootstrap.BOOTSTRAP_AUTHORIZATION_PHRASE,
    )
    bootstrap.validate_acceptance_bootstrap(value)
    assert value["bounded_acceptance_only"] is True
    assert value["final_role_access_authorized"] is False
    assert value["pilot_submission_authorized"] is False
    assert value["scheduler_mutated"] is False
    assert set(value["resources"]) == {
        "cpu_small", "cpu_io", "gpu_target", "gpu_representation",
        "gpu_final_prediction",
    }
    bootstrap.validate_acceptance_bootstrap_task(
        value, planning_spec=planning, task_key="smoke_probe",
        deterministic_worker=False,
    )

    deterministic.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="worker bytes differ"):
        bootstrap.validate_acceptance_bootstrap(value)


def test_bootstrap_rejects_wrong_phrase_training_and_worker_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning, planning_path, _, runtime_path, ordinary, deterministic = _fixture(
        tmp_path, monkeypatch,
    )
    common = dict(
        planning_spec_path=planning_path,
        runtime_binding_path=runtime_path,
        ordinary_worker_path=ordinary,
        deterministic_worker_path=deterministic,
    )
    with pytest.raises(PermissionError, match="phrase differs"):
        bootstrap.build_acceptance_bootstrap(
            **common, authorized_tasks=("tap_schema",), authorization_phrase="wrong",
        )
    noncanonical = tmp_path / "review-copy.json"
    write_immutable_json(noncanonical, planning)
    with pytest.raises(PermissionError, match="canonical smoke root"):
        bootstrap.build_acceptance_bootstrap(
            **{**common, "planning_spec_path": noncanonical},
            authorized_tasks=("tap_schema",),
            authorization_phrase=bootstrap.BOOTSTRAP_AUTHORIZATION_PHRASE,
        )
    with pytest.raises(PermissionError, match="frozen safe prefix"):
        bootstrap.build_acceptance_bootstrap(
            **common,
            authorized_tasks=(*bootstrap.SAFE_BOOTSTRAP_TASK_PREFIX, "train_RSET_M1c"),
            authorization_phrase=bootstrap.BOOTSTRAP_AUTHORIZATION_PHRASE,
        )

    stop = bootstrap.SAFE_BOOTSTRAP_TASK_PREFIX.index("miniature_D100_build") + 1
    value = bootstrap.build_acceptance_bootstrap(
        **common,
        authorized_tasks=bootstrap.SAFE_BOOTSTRAP_TASK_PREFIX[:stop],
        authorization_phrase=bootstrap.BOOTSTRAP_AUTHORIZATION_PHRASE,
    )
    with pytest.raises(PermissionError, match="worker role differs"):
        bootstrap.validate_acceptance_bootstrap_task(
            value, planning_spec=planning, task_key="miniature_D100_build",
            deterministic_worker=False,
        )


def test_bootstrap_workers_are_isolated_and_never_name_final_roles() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in bootstrap.BOOTSTRAP_WORKER_NAMES.values():
        text = (root / "sbatch" / name).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "PYTHONNOUSERSITE=1" in text
        assert "exec python -s" in text
        assert "final" not in text.lower()
