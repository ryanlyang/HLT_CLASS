from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.scouting import hcwdl_representation_campaign as campaign
from hlt_classification.scouting.hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION, create_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE,
    build_dense_resource_probe_authorization,
    build_dense_resource_probe_ledger,
    build_dense_resource_probe_plan,
    validate_dense_resource_probe_authorization,
    validate_dense_resource_probe_ledger,
    validate_dense_resource_probe_plan,
)
from hlt_classification.scouting.hcwdl_representation_resources import (
    DENSE_RESOURCE_CLASSES,
    artifact_reference,
    build_dense_storage_estimate,
    build_dense_storage_template,
    validate_dense_storage_estimate,
    validate_dense_storage_availability,
    validate_dense_storage_template,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, check=True,
    capture_output=True, text=True,
).stdout.strip()


def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(campaign, "validate_source_checkout", lambda *a, **k: None)
    root = tmp_path / "dense"
    spec = create_campaign_spec(
        mode="smoke", campaign_root=root,
        checkpoint_namespace=tmp_path / "checkpoints", project_dir=REPOSITORY,
        source_commit=SOURCE_COMMIT, source_manifest_sha256="1" * 64,
        split_manifest_sha256="2" * 64, parent_import_sha256="3" * 64,
        representation_recipe_sha256="4" * 64, graph_sha256="5" * 64,
        disposition_sha256="6" * 64,
        disposition=DENSE_TRAINING_DISPOSITION,
        role_counts={"train": 512, "validation": 256, "final_test": 0},
        final_source_partitions=0, combined_finalist_count=0,
    )
    spec_path = root / "planning" / "campaign_spec.json"
    write_immutable_json(spec_path, spec)
    spec = load_json(spec_path)
    data_root = tmp_path / "data"
    data_root.mkdir()
    value = build_dense_resource_probe_plan(
        planning_spec_path=spec_path, planning_spec=spec,
        data_root=data_root, conda_environment="atlas_kd_tigris",
    )
    return value


def test_dense_resource_probe_plan_is_four_scalar_measurements_plus_one_collector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, monkeypatch)
    assert validate_dense_resource_probe_plan(plan) == plan["content_hash"]
    assert [row["resource_class"] for row in plan["rows"]] == list(
        DENSE_RESOURCE_CLASSES
    )
    assert all(row["command"][0:2] == ["sbatch", "--parsable"] for row in plan["rows"])
    assert all(row["command"][-1] == row["worker_path"] for row in plan["rows"])
    assert all("--array" not in token for row in plan["rows"] for token in row["command"])
    representation = next(
        row for row in plan["rows"] if row["resource_class"] == "gpu_representation"
    )
    assert Path(representation["result_path"]).parts[-2:] == (
        "resources", "dense_storage_template.json",
    )
    assert Path(representation["runtime_measurement_path"]).parts[-4:] == (
        "review", "resource_probes", "gpu_representation",
        "worker_runtime_measurement.json",
    )
    assert plan["collector"]["worker_path"].endswith(
        "collect_hcwdl_representation_resource_probes.sh"
    )
    assert plan["authorizes_dense_graph_submission"] is False
    assert plan["final_role_access_authorized"] is False


def test_dense_resource_probe_authority_and_ledger_are_exact_and_nonpromoting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="phrase differs"):
        build_dense_resource_probe_authorization(
            plan=plan, authorization_phrase="submit the graph",
        )
    authority = build_dense_resource_probe_authorization(
        plan=plan,
        authorization_phrase=DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE,
    )
    assert validate_dense_resource_probe_authorization(
        authority, plan=plan,
    ) == authority["content_hash"]
    assert authority["dense_graph_submission_authorized"] is False
    assert authority["pilot_submission_authorized"] is False
    assert authority["measurement_probe_job_count"] == 4
    assert authority["collector_job_count"] == 1
    assert authority["scheduler_job_count"] == 5
    jobs = {name: str(10_000 + index) for index, name in enumerate(
        DENSE_RESOURCE_CLASSES,
    )}
    ledger = build_dense_resource_probe_ledger(
        plan=plan, authorization=authority, job_ids=jobs,
        collector_job_id="20000",
    )
    assert validate_dense_resource_probe_ledger(
        ledger, plan=plan, authorization=authority,
    ) == ledger["content_hash"]
    assert ledger["dense_graph_submitted"] is False
    assert ledger["final_role_accessed"] is False
    with pytest.raises(ValueError, match="distinct"):
        build_dense_resource_probe_ledger(
            plan=plan, authorization=authority, job_ids=jobs,
            collector_job_id=next(iter(jobs.values())),
        )


def test_dense_storage_uses_measured_templates_and_scales_all_86_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {}
    for name, size in {
        "resume_state": 101, "selected_checkpoint": 103,
        "deployable_checkpoint": 107,
    }.items():
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode("ascii") + b"x" * size)
        files[name] = artifact_reference(path)
    template = build_dense_storage_template(
        source_commit=SOURCE_COMMIT, planning_spec_sha256="1" * 64,
        representation_recipe_sha256="2" * 64, graph_sha256="3" * 64,
        dense_teacher_import_sha256="4" * 64,
        resume_state_template=files["resume_state"],
        selected_checkpoint_template=files["selected_checkpoint"],
        deployable_checkpoint_template=files["deployable_checkpoint"],
    )
    template_path = tmp_path / "dense_storage_template.json"
    write_immutable_json(template_path, template)
    template_ref = artifact_reference(template_path)
    assert validate_dense_storage_template(
        template, expected_source_commit=SOURCE_COMMIT,
        expected_recipe_sha256="2" * 64, expected_graph_sha256="3" * 64,
        expected_dense_teacher_import_sha256="4" * 64,
    ) == template["content_hash"]
    estimate = build_dense_storage_estimate(
        train_rows=300_000, validation_rows=100_000,
        dense_teacher_import_sha256="4" * 64,
        storage_template=template_ref,
    )
    assert validate_dense_storage_estimate(
        estimate, storage_template=template_ref,
        expected_source_commit=SOURCE_COMMIT,
        expected_recipe_sha256="2" * 64, expected_graph_sha256="3" * 64,
        expected_dense_teacher_import_sha256="4" * 64,
    ) == estimate["content_hash"]
    assert estimate["training_node_count"] == 86
    assert estimate["final_role_storage_bytes"] == 0
    assert estimate["filesystem_headroom_bytes"] * 2 >= estimate[
        "subtotal_before_filesystem_headroom_bytes"
    ]
    from hlt_classification.scouting import hcwdl_representation_resources as resources
    usage = type("Usage", (), {
        "free": estimate["minimum_free_bytes_at_submission"] - 1,
    })()
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(PermissionError, match="requires .* free bytes"):
        validate_dense_storage_availability(estimate, campaign_root=tmp_path)
    changed = dict(template)
    changed["source_commit"] = "5" * 40
    from hlt_classification.data.cache_contracts import with_content_hash
    changed = with_content_hash({
        key: value for key, value in changed.items() if key != "content_hash"
    })
    with pytest.raises(PermissionError, match="lineage differs"):
        validate_dense_storage_template(
            changed, expected_source_commit=SOURCE_COMMIT,
        )
