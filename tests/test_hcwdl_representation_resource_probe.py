from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_campaign as campaign
from hlt_classification.scouting.hcwdl_representation_campaign import (
    DENSE_TRAINING_DISPOSITION, create_campaign_spec,
)
from hlt_classification.scouting.hcwdl_representation_resource_probe import (
    DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE,
    DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_PHRASE,
    build_dense_resource_probe_authorization,
    build_dense_resource_probe_collector_recovery_authorization,
    build_dense_resource_probe_collector_recovery_ledger,
    build_dense_resource_probe_ledger,
    build_dense_resource_probe_plan,
    validate_dense_resource_probe_authorization,
    validate_dense_resource_probe_collector_recovery_authorization,
    validate_dense_resource_probe_collector_recovery_ledger,
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


def test_collector_compatibility_is_one_clean_direct_operational_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_resource_probe as probe

    expected = "1" * 40
    actual = "2" * 40
    responses = {
        ("status", "--porcelain"): "",
        ("rev-parse", "HEAD"): actual + "\n",
        ("merge-base", "--is-ancestor", expected, actual): "",
        ("rev-list", "--count", f"{expected}..{actual}"): "1\n",
        ("diff", "--name-only", f"{expected}..{actual}"): (
            "\n".join(sorted(probe._COLLECTOR_COMPATIBILITY_BASE_PATHS)) + "\n"
        ),
    }

    def fake_run(command, *, cwd, check, capture_output, text):
        assert command[0] == "git" and cwd == tmp_path
        key = tuple(command[1:])
        return subprocess.CompletedProcess(command, 0, responses[key], "")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    probe._validate_collector_compatible_checkout(
        tmp_path, expected_commit=expected,
    )
    responses[("rev-list", "--count", f"{expected}..{actual}")] = "2\n"
    responses[("diff", "--name-only", f"{expected}..{actual}")] = (
        "\n".join(sorted(probe._COLLECTOR_RECOVERY_PATHS)) + "\n"
    )
    assert probe._validate_collector_compatible_checkout(
        tmp_path, expected_commit=expected,
    ) == actual
    responses[("rev-list", "--count", f"{expected}..{actual}")] = "3\n"
    assert probe._validate_collector_compatible_checkout(
        tmp_path, expected_commit=expected,
    ) == actual
    responses[("rev-parse", "HEAD^")] = expected + "\n"
    responses[("diff", "--name-only", f"{expected}..{actual}")] = (
        "\n".join(sorted(probe._POST_RECOVERY_COMPATIBILITY_PATHS)) + "\n"
    )
    assert probe._validate_post_recovery_compatible_checkout(
        tmp_path, authorized_commit=expected,
    ) == actual
    responses[("rev-list", "--count", f"{expected}..{actual}")] = "4\n"
    with pytest.raises(PermissionError, match="compatibility successor"):
        probe._validate_collector_compatible_checkout(
            tmp_path, expected_commit=expected,
        )


def test_one_replacement_collector_is_bound_without_probe_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_resource_probe as probe

    plan = _plan(tmp_path, monkeypatch)
    authority = build_dense_resource_probe_authorization(
        plan=plan,
        authorization_phrase=DENSE_RESOURCE_PROBE_AUTHORIZATION_PHRASE,
    )
    jobs = {name: str(80_190 + index) for index, name in enumerate(
        DENSE_RESOURCE_CLASSES,
    )}
    ledger = build_dense_resource_probe_ledger(
        plan=plan, authorization=authority, job_ids=jobs,
        collector_job_id="80194",
    )
    ledger_path = tmp_path / "dense_resource_probe_ledger.json"
    write_immutable_json(ledger_path, ledger)
    failed_log = tmp_path / "slurm-80194.out"
    failed_log.write_text(
        "collect_hcwdl_representation_dense_resource_probes.py ReqGRES "
        "returned non-zero exit status 1",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        probe, "_validate_collector_compatible_checkout", lambda *a, **k: "9" * 40,
    )
    recovery = build_dense_resource_probe_collector_recovery_authorization(
        plan=plan, authorization=authority, ledger=ledger,
        ledger_path=ledger_path, failed_collector_log=artifact_reference(failed_log),
        authorization_phrase=(
            DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_PHRASE
        ),
    )
    assert validate_dense_resource_probe_collector_recovery_authorization(
        recovery, plan=plan, authorization=authority, ledger=ledger,
    ) == recovery["content_hash"]
    assert recovery["probe_job_ids"] == jobs
    assert recovery["measurement_probe_job_count"] == 0
    assert recovery["probe_jobs_rerun_authorized"] is False
    assert recovery["dense_graph_submission_authorized"] is False
    replacement = build_dense_resource_probe_collector_recovery_ledger(
        plan=plan, authorization=authority, ledger=ledger,
        recovery_authorization=recovery, replacement_collector_job_id="80200",
    )
    assert validate_dense_resource_probe_collector_recovery_ledger(
        replacement, plan=plan, authorization=authority, ledger=ledger,
        recovery_authorization=recovery,
    ) == replacement["content_hash"]
    assert replacement["probe_jobs_rerun"] is False
    with pytest.raises(ValueError, match="replacement dense collector"):
        build_dense_resource_probe_collector_recovery_ledger(
            plan=plan, authorization=authority, ledger=ledger,
            recovery_authorization=recovery, replacement_collector_job_id="80194",
        )


def test_collector_reuses_only_byte_identical_partial_raw_capture(
    tmp_path: Path,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_resource_probe as probe

    path = tmp_path / "sacct.psv"
    probe._publish_or_match_raw_accounting(path, b"exact accounting\n")
    probe._publish_or_match_raw_accounting(path, b"exact accounting\n")
    with pytest.raises(PermissionError, match="differs from prior immutable capture"):
        probe._publish_or_match_raw_accounting(path, b"changed accounting\n")


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


def test_dense_measurements_cross_only_the_exact_accounting_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_resources as resources
    from hlt_classification.scouting.hcwdl_representation_contracts import (
        DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_CONTRACT,
        DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_LEDGER_CONTRACT,
    )

    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=project, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "dense-test@example.invalid"],
        cwd=project, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Dense Test"], cwd=project, check=True,
    )
    handoff = project / "docs" / "HANDOFF.md"
    handoff.write_text("measured\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/HANDOFF.md"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "measured"], cwd=project, check=True)
    measured = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    handoff.write_text("collector compatibility\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "collector"], cwd=project, check=True)
    campaign_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    monkeypatch.setattr(
        resources, "_DENSE_RESOURCE_COMPATIBILITY_CHANGED_PATHS",
        frozenset({"docs/HANDOFF.md"}),
    )
    jobs = {name: str(80190 + index) for index, name in enumerate(
        resources.DENSE_RESOURCE_CLASSES,
    )}
    authorization = with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_CONTRACT,
        "schema_version": 1,
        "measured_source_commit": measured,
        "compatibility_source_commit": campaign_commit,
        "probe_job_ids": jobs,
        "probe_jobs_rerun_authorized": False,
        "dense_graph_submission_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
    })
    ledger = with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_LEDGER_CONTRACT,
        "schema_version": 1,
        "recovery_authorization_sha256": authorization["content_hash"],
        "measured_source_commit": measured,
        "compatibility_source_commit": campaign_commit,
        "probe_job_ids": jobs,
        "probe_jobs_rerun": False,
        "dense_graph_submitted": False,
        "final_role_accessed": False,
    })
    authorization_path = tmp_path / "authorization.json"
    ledger_path = tmp_path / "ledger.json"
    write_immutable_json(authorization_path, authorization)
    write_immutable_json(ledger_path, ledger)
    projection = resources._dense_resource_source_compatibility(
        project_dir=project, measured_source_commit=measured,
        campaign_source_commit=campaign_commit,
        representation_recipe_sha256="a" * 64,
        recipe_producer_source_sha256="b" * 64,
        recovery_authorization=authorization,
        recovery_authorization_reference=artifact_reference(authorization_path),
        recovery_ledger=ledger,
        recovery_ledger_reference=artifact_reference(ledger_path),
    )
    assert projection["measurement_jobs_rerun"] is False
    assert projection["training_code_changed"] is False
    assert projection["changed_paths"] == ["docs/HANDOFF.md"]

    training = project / "src" / "training.py"
    training.parent.mkdir()
    training.write_text("changed = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/training.py"], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "training change"], cwd=project, check=True)
    changed_campaign = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    with pytest.raises(PermissionError, match="accounting boundary"):
        resources._dense_resource_source_compatibility(
            project_dir=project, measured_source_commit=measured,
            campaign_source_commit=changed_campaign,
            representation_recipe_sha256="a" * 64,
            recipe_producer_source_sha256="b" * 64,
            recovery_authorization=authorization,
            recovery_authorization_reference=artifact_reference(authorization_path),
            recovery_ledger=ledger,
            recovery_ledger_reference=artifact_reference(ledger_path),
        )


def test_compatible_profile_preserves_measurements_and_old_recipe_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_representation_resources as resources
    from hlt_classification.scouting.hcwdl_representation_contracts import (
        DENSE_RESOURCE_PROFILE_CONTRACT,
        DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_CONTRACT,
        DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_LEDGER_CONTRACT,
    )

    measured = "1" * 40
    campaign_commit = "2" * 40
    recipe = "3" * 64
    producer_source = "4" * 64
    base = with_content_hash({
        "contract": DENSE_RESOURCE_PROFILE_CONTRACT,
        "schema_version": 1,
        "disposition": "dense_training_only",
        "requests": {"cpu_small": {"cpus": 2}},
        "measurements": {"cpu_small": {"measured": True}},
        "array_concurrency_limits": {},
        "measurement_environment": {
            "source_commit": measured,
            "production_workers": {},
        },
    })
    authorization = with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_AUTHORIZATION_CONTRACT,
        "schema_version": 1,
    })
    ledger = with_content_hash({
        "contract": DENSE_RESOURCE_PROBE_COLLECTOR_RECOVERY_LEDGER_CONTRACT,
        "schema_version": 1,
    })
    paths = {}
    for name, value in (
        ("base", base), ("authorization", authorization), ("ledger", ledger),
    ):
        path = tmp_path / f"{name}.json"
        write_immutable_json(path, value)
        paths[name] = path
    projection = {
        "compatibility_class": "slurm_accounting_only/v1",
        "project_dir": str(tmp_path.resolve()),
        "measured_source_commit": measured,
        "campaign_source_commit": campaign_commit,
        "changed_paths": ["collector.py"],
        "changed_paths_sha256": "5" * 64,
        "representation_recipe_sha256": recipe,
        "recipe_producer_source_sha256": producer_source,
        "collector_recovery_authorization": artifact_reference(
            paths["authorization"]
        ),
        "collector_recovery_ledger": artifact_reference(paths["ledger"]),
        "measurement_jobs_rerun": False,
        "training_code_changed": False,
        "dense_graph_submission_authorized": False,
        "pilot_submission_authorized": False,
        "final_role_access_authorized": False,
    }
    monkeypatch.setattr(
        resources, "_validate_dense_exact_measured_profile",
        lambda value, *, expected_source_commit: value["content_hash"],
    )
    monkeypatch.setattr(
        resources, "_dense_resource_source_compatibility",
        lambda **_kwargs: dict(projection),
    )
    compatible = resources.build_dense_compatible_measured_profile(
        base_profile=base,
        base_profile_reference=artifact_reference(paths["base"]),
        project_dir=tmp_path,
        campaign_source_commit=campaign_commit,
        representation_recipe_sha256=recipe,
        recipe_producer_source_sha256=producer_source,
        recovery_authorization=authorization,
        recovery_authorization_reference=artifact_reference(paths["authorization"]),
        recovery_ledger=ledger,
        recovery_ledger_reference=artifact_reference(paths["ledger"]),
    )
    assert resources.validate_dense_measured_profile(
        compatible, expected_source_commit=campaign_commit,
        expected_recipe_sha256=recipe,
    ) == compatible["content_hash"]
    assert resources.dense_resource_measurement_source_commit(
        compatible
    ) == measured
    assert resources.dense_resource_recipe_producer_source_sha256(
        compatible, runtime_source_sha256="6" * 64,
    ) == producer_source

    changed = dict(compatible)
    changed["measurements"] = {"cpu_small": {"measured": False}}
    changed = with_content_hash({
        key: value for key, value in changed.items() if key != "content_hash"
    })
    with pytest.raises(PermissionError, match="changes measured resources"):
        resources.validate_dense_measured_profile(
            changed, expected_source_commit=campaign_commit,
            expected_recipe_sha256=recipe,
        )
