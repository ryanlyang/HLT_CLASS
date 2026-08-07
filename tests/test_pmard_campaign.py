from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from hlt_classification.data.cache_contracts import canonical_sha256, with_content_hash
from hlt_classification.provenance import SOURCE_SNAPSHOT_CONTRACT
from hlt_classification.scouting.campaign import (
    create_pmard_campaign_spec, create_pmard_production_dry_run,
    experiment_registry, pmard_tasks, submit_pmard_campaign,
    validate_pmard_campaign_spec,
)
from hlt_classification.scouting.evidence import (
    build_miniature_report, build_resource_evidence, build_storage_evidence,
)
from hlt_classification.scouting.locks import (
    create_full_endpoint_authorization, create_lock,
    validate_full_endpoint_authorization, validate_lock,
)
from hlt_classification.scouting.training import MATCHER_FOLD_SEED
from hlt_classification.scouting.workflow import Workflow


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source():
    commit = "a" * 40; tree = "b" * 40; tracked = _digest("tracked")
    return with_content_hash({
        "contract": SOURCE_SNAPSHOT_CONTRACT, "schema_version": 1,
        "git_commit": commit, "git_tree": tree, "tracked_files_sha256": tracked,
        "tracked_file_count": 10, "worktree_clean": True,
        "source_snapshot_sha256": canonical_sha256({
            "git_commit": commit, "git_tree": tree, "tracked_files_sha256": tracked,
        }),
    })


def test_pmard_dry_run_is_complete_topological_and_nonnumeric_ids_rejected():
    spec = create_pmard_campaign_spec(
        source_snapshot=_source(), source_manifest_sha256=_digest("source"),
        split_manifest_sha256=_digest("split"), campaign_root="/tmp/pmard", mode="smoke",
    )
    assert validate_pmard_campaign_spec(spec) == spec["content_hash"]
    ledger = submit_pmard_campaign(spec, spec_path="/tmp/spec.json", dry_run=True)
    assert set(ledger["jobs"]) == {task["name"] for task in spec["tasks"]}
    assert all(command[0:2] == ["sbatch", "--parsable"] for command in ledger["commands"])
    commands = {
        task["name"]: command for task, command in zip(spec["tasks"], ledger["commands"], strict=True)
    }
    assert "--signal=B:USR1@120" in commands["representation"]
    assert "--signal=B:USR1@120" not in commands["source_audit"]
    assert ledger["mutated"] is False
    assert "final_test" not in ledger["jobs"] and "miniature_summary" in ledger["jobs"]
    representation = next(task for task in spec["tasks"] if task["name"] == "representation")
    assert representation["array"] == "0-12%2"
    assert spec["registry"]["representation_arms"][-3:] == ["R4_PAIR", "R4_GRAM", "R5"]
    assert spec["registry"]["matcher"]["variant"] == "fitted_strict"
    assert spec["registry"]["primary_repair_family"] == "SELECTIVE_FULL_PARTICLE_ENDPOINT/v1"
    assert spec["contract"] == "hlt_classification_pmard_campaign_spec_v10"
    assignment = next(task for task in spec["tasks"] if task["name"] == "assignment_cache")
    full_lock = next(task for task in spec["tasks"] if task["name"] == "full_endpoint_lock")
    training_lock = next(task for task in spec["tasks"] if task["name"] == "training_lock")
    assert assignment["dependencies"] == ["matcher_result_lock"]
    assert assignment["array"] == "0-42%4"
    assert spec["registry"]["row_budgets"]["smoke"]["train"] == 4096
    assert spec["registry"]["assignment_cache"]["unmatched_policy"] == "retain_exact_hlt_token_v1"
    assert spec["registry"]["ram_cache_miniature"] == {
        "smoke_only": True, "max_rows_per_role": 4096,
        "arm": "K2", "alpha": .25, "cache_teacher_targets": True,
        "expected_training_stream": "hlt_only_cached_logits",
    }
    assert full_lock["dependencies"] == ["assignment_manifest"]
    assert "full_endpoint_lock" in training_lock["dependencies"]
    cache_miniature = next(
        task for task in spec["tasks"] if task["name"] == "ram_cache_miniature"
    )
    alpha_sweep = next(task for task in spec["tasks"] if task["name"] == "k2_alpha_sweep")
    assert cache_miniature["dependencies"] == ["oracle_validation"]
    assert cache_miniature["checkpointable"] is True
    assert alpha_sweep["dependencies"] == ["ram_cache_miniature"]
    with pytest.raises(PermissionError):
        create_pmard_campaign_spec(
            source_snapshot=_source(), source_manifest_sha256=_digest("source"),
            split_manifest_sha256=_digest("split"), campaign_root="/tmp/pmard",
            mode="production",
        )


def test_pmard_batch_worker_execs_python_for_batch_shell_signal_delivery():
    worker = Path(__file__).resolve().parents[1] / "sbatch" / "run_pmard_task.sh"
    commands = [
        line.strip() for line in worker.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert commands[-1].startswith("exec python -s ")


def test_every_smoke_student_command_uses_shared_row_selection():
    workflow = object.__new__(Workflow)
    workflow.spec = {
        "mode": "smoke",
        "source_snapshot": {"source_snapshot_sha256": "a" * 64},
    }
    workflow.repository = Path("/repository")
    workflow.root = Path("/campaign")
    workflow.data = Path("/data")
    workflow.split = Path("/campaign/split.json")
    workflow.audit = Path("/campaign/audit.json")
    workflow.row_selection = Path("/campaign/row_selection.json")
    workflow.assignment_manifest = Path("/campaign/assignments.json")
    workflow.full_endpoint_lock = Path("/campaign/full_endpoint.json")
    workflow.full_matcher = Path("/campaign/matcher.json")
    workflow._matcher_settings = lambda: {
        "selected_variant": "fitted_strict", "threshold": 0.9828147479721088,
    }
    workflow._locked_training = lambda: {
        "temperature": 2, "total_updates": 2, "batch_size": 64,
        "peak_learning_rate": 1e-3, "screen_seed": 1337,
    }
    for arm in (f"K{index}" for index in range(7)):
        command = workflow._student_command(
            output=Path(f"/output/{arm}"), arm=arm, alpha=0,
            hlt_teacher=None, privileged_teacher=None,
        )
        assert command.count("--row-selection") == 1
        assert command[command.index("--row-selection") + 1] == str(workflow.row_selection)
        assert "--max-rows-per-role" not in command
    cache_command = workflow._student_command(
        output=Path("/output/cache"), arm="K2", alpha=0,
        hlt_teacher=None, privileged_teacher=None,
        extra=("--cache-teacher-targets",),
    )
    assert "--cache-teacher-targets" in cache_command
    assert cache_command[cache_command.index("--row-selection") + 1] == str(workflow.row_selection)
    alpha_command = workflow._student_command(
        output=Path("/output/alpha"), arm="K2", alpha=.25,
        hlt_teacher=Path("/teacher/hlt.json"),
        privileged_teacher=Path("/teacher/privileged.json"),
    )
    assert all(isinstance(value, str) for value in alpha_command)
    assert alpha_command[alpha_command.index("--matcher-variant") + 1] == "fitted_strict"
    assert alpha_command[alpha_command.index("--matcher-threshold") + 1] == "0.9828147479721088"
    assert alpha_command[alpha_command.index("--repair-family") + 1] == "SELECTIVE_FULL_PARTICLE_ENDPOINT"
    assert alpha_command[alpha_command.index("--assignment-manifest") + 1] == str(workflow.assignment_manifest)
    assert "--matcher-report" not in alpha_command
    alpha_teacher_command = workflow._teacher_command(
        output=Path("/output/T25"), experiment="T25", alpha=.25,
    )
    assert all(isinstance(value, str) for value in alpha_teacher_command)
    assert alpha_teacher_command[
        alpha_teacher_command.index("--matcher-threshold") + 1
    ] == "0.9828147479721088"
    assert "--cache-privileged-views" not in alpha_teacher_command
    workflow.spec = {
        **workflow.spec, "mode": "pilot",
        "registry": {"privileged_view_cache": {
            "enabled_modes": ["pilot", "production"], "max_gib": 320.0,
        }},
    }
    cached_teacher = workflow._teacher_command(
        output=Path("/output/T25_cached"), experiment="T25", alpha=.25,
    )
    cached_student = workflow._student_command(
        output=Path("/output/R3_cached"), arm="K2", alpha=.25,
        hlt_teacher=Path("/teacher/hlt.json"),
        privileged_teacher=Path("/teacher/privileged.json"),
    )
    for command in (cached_teacher, cached_student):
        assert command.count("--cache-privileged-views") == 1
        assert command[command.index("--view-cache-max-gib") + 1] == "320.0"


def test_pilot_registers_exact_300k_100k_100k_and_sealed_final_assignments():
    spec = create_pmard_campaign_spec(
        source_snapshot=_source(), source_manifest_sha256=_digest("source"),
        split_manifest_sha256=_digest("split"), campaign_root="/tmp/pilot", mode="pilot",
    )
    assert validate_pmard_campaign_spec(spec) == spec["content_hash"]
    assert spec["registry"]["row_budgets"]["pilot"] == {
        "train": 300_000, "validation": 100_000, "final_test": 100_000,
    }
    tasks = {task["name"]: task for task in spec["tasks"]}
    assert spec["registry"]["privileged_view_cache"] == {
        "storage": "process_local_ram_float32_particle_views_v1",
        "enabled_modes": ["smoke", "pilot", "production"], "max_gib": 320.0,
        "sampler_replay": "exact_chunk_file_buffer_schedule_v1",
        "durable_artifact_published": False,
    }
    assert tasks["teachers"]["memory"] == "192G"
    assert tasks["representation"]["memory"] == "192G"
    assert tasks["confirmation"]["memory"] == "192G"
    assert tasks["final_row_selection"]["dependencies"] == ["execution_lock"]
    assert tasks["final_assignment_cache"]["dependencies"] == ["final_row_selection"]
    assert tasks["final_test"]["dependencies"] == ["final_assignment_manifest"]


def test_campaign_v9_remains_valid_for_the_active_immutable_pilot():
    current = create_pmard_campaign_spec(
        source_snapshot=_source(), source_manifest_sha256=_digest("source"),
        split_manifest_sha256=_digest("split"), campaign_root="/tmp/legacy", mode="pilot",
    )
    registry = experiment_registry(campaign_version=9)
    evidence = current["evidence"]
    identity = canonical_sha256({
        "source_snapshot_sha256": current["source_snapshot"]["source_snapshot_sha256"],
        "source_manifest_sha256": current["source_manifest_sha256"],
        "split_manifest_sha256": current["split_manifest_sha256"],
        "mode": "pilot", "registry": registry, "evidence": evidence,
    })
    legacy = {
        key: value for key, value in current.items()
        if key != "content_hash"
    }
    legacy.update({
        "contract": "hlt_classification_pmard_campaign_spec_v9",
        "schema_version": 9, "campaign_id": f"pmard_pilot_{identity[:16]}",
        "registry": registry,
        "tasks": [task.to_dict() for task in pmard_tasks(
            smoke=False, pilot=True, campaign_version=9,
        )],
    })
    legacy = with_content_hash(legacy)
    assert validate_pmard_campaign_spec(legacy) == legacy["content_hash"]


def test_production_requires_validated_complete_evidence_bundle():
    source = _source(); source_hash = _digest("source"); split_hash = _digest("split")
    smoke = create_pmard_campaign_spec(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/smoke", mode="smoke",
    )
    live = submit_pmard_campaign(
        smoke, spec_path="/tmp/smoke.json", dry_run=False,
        runner=lambda command, counter=iter(range(1000, 2000)): str(next(counter)),
    )
    monitor = with_content_hash({
        "contract": "hlt_classification_pmard_monitor_v1", "schema_version": 1,
        "campaign_spec_sha256": smoke["content_hash"],
        "jobs": [{"task": task, "job_id": job, "state": "COMPLETED", "reusable": True}
                 for task, job in live["jobs"].items()],
    })
    usage = {job: {
        "elapsed_seconds": 1, "max_rss_bytes": 1024, "allocated_cpus": 1,
        "max_gpu_memory_bytes": 0, "root_bytes_read": 1,
        "root_wait_milliseconds": 0, "peak_ram_tmp_bytes": 0,
    } for job in live["jobs"].values()}
    resource = build_resource_evidence(
        smoke_spec=smoke, live_ledger=live, monitor=monitor,
        usage_by_job_id=usage, measurement_host="tigris", campaign_artifact_bytes=100,
    )
    storage = build_storage_evidence(
        resource_evidence=resource, measurement_host="tigris", measurement_path="/scratch",
        available_bytes=10_000, peak_durable_bytes=100, peak_ram_tmp_bytes=200,
    )
    miniature = build_miniature_report(
        smoke_spec=smoke, monitor=monitor, resource_evidence=resource,
        storage_evidence=storage,
    )
    dry_run = create_pmard_production_dry_run(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/production",
        spec_path="/tmp/production.json",
    )
    production = create_pmard_campaign_spec(
        source_snapshot=source, source_manifest_sha256=source_hash,
        split_manifest_sha256=split_hash, campaign_root="/tmp/production",
        mode="production", production_authorized=True,
        evidence_artifacts={"miniature_report": miniature, "dry_run_report": dry_run,
                            "resource_evidence": resource, "storage_evidence": storage},
    )
    assert validate_pmard_campaign_spec(production) == production["content_hash"]
    assert "final_test" in {task["name"] for task in production["tasks"]}
    production_tasks = {task["name"]: task for task in production["tasks"]}
    assert production_tasks["teachers"]["memory"] == "384G"
    assert production_tasks["representation"]["memory"] == "384G"


def test_lock_chain_requires_exact_predecessor():
    spec_hash = _digest("spec")
    parent = create_lock("data", payload={"ok": True}, campaign_spec_sha256=spec_hash)
    child = create_lock("matcher_design", payload={"frozen": True}, campaign_spec_sha256=spec_hash, parent_lock=parent)
    assert validate_lock(child, expected_level="matcher_design") == child["content_hash"]
    with pytest.raises(ValueError):
        create_lock("training", payload={}, campaign_spec_sha256=spec_hash, parent_lock=parent)
    with pytest.raises(ValueError, match="different campaign spec"):
        create_lock(
            "matcher_design", payload={}, campaign_spec_sha256=_digest("other-spec"),
            parent_lock=parent,
        )


def test_full_endpoint_lock_requires_all_categories_and_exact_complete_coverage():
    spec_hash = _digest("spec")
    split_hash = _digest("split")
    fold_hashes = [_digest(f"fold-{fold}") for fold in range(5)]
    full_matcher_hash = "f" * 64
    data = create_lock("data", payload={}, campaign_spec_sha256=spec_hash)
    design = create_lock("matcher_design", payload={}, campaign_spec_sha256=spec_hash, parent_lock=data)
    validation = with_content_hash({
        "contract": "hlt_classification_pmard_matcher_validation_v2", "schema_version": 2,
        "threshold": .99,
        "parents": {
            "split_manifest_sha256": split_hash,
            "matcher_report_sha256": full_matcher_hash,
        },
        "variants": {"M5": {
            "native_coverage": .95, "passes_initial_99pct_lcb": True,
        }},
    })
    matcher = create_lock(
        "matcher_result", campaign_spec_sha256=spec_hash, parent_lock=design,
        payload={
            "selected_variant": "M5", "meets_initial_precision_target": True,
            "threshold": .99, "matcher_fold_seed": MATCHER_FOLD_SEED,
            "split_manifest_sha256": split_hash,
            "fold_matcher_report_sha256": fold_hashes,
            "category_eligibility": {str(index): True for index in range(5)},
            "matching_only_selection": [{"variant": "M5", "native_coverage": .94}],
            "validation_report_sha256": validation["content_hash"],
            "full_matcher_report_sha256": full_matcher_hash,
        },
    )
    coverage_parents = {
        "split_manifest_sha256": split_hash,
        "matcher_result_lock_sha256": matcher["content_hash"],
        "full_matcher_report_sha256": full_matcher_hash,
        **{
            f"matcher_fold_{fold}_report_sha256": fold_hashes[fold]
            for fold in range(5)
        },
    }
    role_counts = {
        "expected_mapped_jets": 10, "scanned_mapped_jets": 10,
        "visible_hlt_tokens": 20, "assigned_hlt_tokens": 20,
        "unassigned_hlt_tokens": 0, "invalid_assignment_tokens": 0,
        "duplicate_assignment_tokens": 0, "unknown_category_tokens": 0,
        "visible_by_category": [4] * 5, "assigned_by_category": [4] * 5,
        "coverage": 1.0, "complete": True,
    }
    coverage = with_content_hash({
        "contract": "hlt_classification_pmard_full_role_coverage_v2",
        "schema_version": 2,
        "scope": "all_mapped_train_and_validation_rows_v1",
        "selected_variant": "M5", "threshold": .99,
        "matcher_fold_seed": MATCHER_FOLD_SEED,
        "parents": dict(sorted(coverage_parents.items())),
        "roles": {"train": dict(role_counts), "validation": dict(role_counts)},
        "complete": True, "assignment_artifact_published": False,
        "downstream_classifier_or_label_used_for_matching": False,
    })
    authorization = create_full_endpoint_authorization(
        matcher_result_lock=matcher, full_validation=validation,
        full_role_coverage=coverage,
        campaign_spec_sha256=spec_hash,
    )
    assert validate_full_endpoint_authorization(
        authorization, matcher_report_sha256=full_matcher_hash,
        matcher_variant="M5", matcher_threshold=.99,
        split_manifest_sha256=split_hash,
        fold_matcher_report_sha256=fold_hashes,
    ) == authorization["content_hash"]
    shortened_coverage = dict(coverage)
    shortened_coverage["roles"] = {
        "train": dict(role_counts), "validation": dict(role_counts),
    }
    shortened_coverage["roles"]["train"].update({
        "scanned_mapped_jets": 9,
        "visible_hlt_tokens": 18,
        "assigned_hlt_tokens": 18,
        "visible_by_category": [4, 4, 4, 3, 3],
        "assigned_by_category": [4, 4, 4, 3, 3],
    })
    # Even a forged `complete` flag and perfect observed-token coverage cannot
    # authorize a scan that processed only 9 of 10 manifest-declared jets.
    shortened_coverage = with_content_hash({
        key: value for key, value in shortened_coverage.items() if key != "content_hash"
    })
    with pytest.raises(PermissionError, match="jet scan is incomplete"):
        create_full_endpoint_authorization(
            matcher_result_lock=matcher, full_validation=validation,
            full_role_coverage=shortened_coverage,
            campaign_spec_sha256=spec_hash,
        )
    incomplete_coverage = dict(coverage)
    incomplete_coverage["roles"] = {
        "train": dict(role_counts), "validation": dict(role_counts),
    }
    incomplete_coverage["roles"]["train"]["assigned_hlt_tokens"] = 19
    incomplete_coverage["roles"]["train"]["unassigned_hlt_tokens"] = 1
    incomplete_coverage["roles"]["train"]["coverage"] = .95
    incomplete_coverage["roles"]["train"]["complete"] = False
    incomplete_coverage["complete"] = False
    incomplete_coverage = with_content_hash({
        key: value for key, value in incomplete_coverage.items() if key != "content_hash"
    })
    with pytest.raises(PermissionError):
        create_full_endpoint_authorization(
            matcher_result_lock=matcher, full_validation=validation,
            full_role_coverage=incomplete_coverage,
            campaign_spec_sha256=spec_hash,
        )
    partial = dict(matcher)
    partial["payload"] = dict(matcher["payload"])
    partial["payload"]["category_eligibility"] = dict(
        matcher["payload"]["category_eligibility"]
    )
    partial["payload"]["category_eligibility"]["4"] = False
    partial = with_content_hash({key: value for key, value in partial.items() if key != "content_hash"})
    with pytest.raises(PermissionError):
        create_full_endpoint_authorization(
            matcher_result_lock=partial, full_validation=validation,
            full_role_coverage=coverage,
            campaign_spec_sha256=spec_hash,
        )
    with pytest.raises(PermissionError):
        validate_full_endpoint_authorization(
            authorization, matcher_report_sha256=full_matcher_hash,
            matcher_variant="M0", matcher_threshold=.99,
            split_manifest_sha256=split_hash,
            fold_matcher_report_sha256=fold_hashes,
        )
