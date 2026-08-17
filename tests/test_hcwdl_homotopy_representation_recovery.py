from __future__ import annotations

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import _task_registry
from hlt_classification.scouting.hcwdl_homotopy_representation_contracts import (
    MONITOR_REPORT_CONTRACT, RESOURCE_RECOVERY_CONTRACT,
    CAMPAIGN_SPEC_CONTRACT, SCHEMA_VERSION, SOURCE_RECOVERY_CONTRACT,
    SUBMISSION_LEDGER_CONTRACT,
    build_artifact,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_recovery import (
    build_monitor_report, build_recovery, exact_cancellation_commands,
    failed_downstream_closure, recovery_command_plan, validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_campaign import (
    SMOKE_RESOURCES,
)
from hlt_classification.scouting.hcwdl_homotopy_representation_graph import GRAPH_SHA256


def _spec():
    return with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION, "tasks": _task_registry(),
        "final_test_accessed": False,
    })


def test_failed_closure_is_exact_strategy_suffix_plus_aggregate():
    spec = _spec()
    monitor = build_artifact(
        MONITOR_REPORT_CONTRACT,
        parents={"campaign_spec": spec["content_hash"], "submission_ledger": "b" * 64},
        rows=[
            {"task_id": row["task_id"], "classification": (
                "retryable_failure" if row["task_id"] == "train_F_RSET_U060" else "complete"
            )}
            for row in spec["tasks"]
        ],
    )
    # Pure closure only needs the graph registry and monitor contract.  Patch
    # campaign validation is unnecessary here because this fixture omits the
    # filesystem-bound campaign fields.
    import hlt_classification.scouting.hcwdl_homotopy_representation_recovery as module
    original = module.validate_campaign
    module.validate_campaign = lambda *_args, **_kwargs: spec["content_hash"]
    try:
        closure = failed_downstream_closure(spec, monitor)
    finally:
        module.validate_campaign = original
    assert "train_F_RSET_U060" in closure
    assert "train_F_RSET_M1" in closure
    assert "train_F_RREL_U060" not in closure
    assert closure[-2:] == ("aggregate", "campaign_complete")


def test_failed_closure_allows_independent_live_track(monkeypatch):
    import hlt_classification.scouting.hcwdl_homotopy_representation_recovery as module

    spec = _spec()
    monitor = build_artifact(
        MONITOR_REPORT_CONTRACT,
        parents={"campaign_spec": spec["content_hash"], "submission_ledger": "b" * 64},
        rows=[
            {
                "task_id": row["task_id"],
                "classification": (
                    "retryable_failure" if row["task_id"] == "train_F_RREL_U100"
                    else "live" if row["task_id"] == "train_F_RSET_D80"
                    else "complete"
                ),
            }
            for row in spec["tasks"]
        ],
    )
    monkeypatch.setattr(
        module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"],
    )
    closure = failed_downstream_closure(spec, monitor)
    assert "train_F_RREL_U100" in closure
    assert "train_F_RSET_D80" not in closure


def test_ledger_and_cancellation_are_exact_ids_only():
    ledger = build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={"campaign_spec": "a" * 64, "command_plan": "b" * 64},
        jobs={"a": "12", "b": "13"}, submission_phrase="x",
    )
    validate_submission_ledger(ledger)
    assert exact_cancellation_commands(ledger) == (("scancel", "12"), ("scancel", "13"))


def test_partial_prefix_monitor_marks_only_unsubmitted_suffix_for_recovery(monkeypatch):
    import hlt_classification.scouting.hcwdl_homotopy_representation_recovery as module

    spec = _spec()
    first_two = [row["task_id"] for row in spec["tasks"][:2]]
    ledger = build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={"campaign_spec": spec["content_hash"], "command_plan": "b" * 64},
        jobs=dict(zip(first_two, ("12", "13"), strict=True)),
        submission_phrase="x", submitted_task_count=2,
        complete_submission=False,
    )
    monkeypatch.setattr(
        module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"],
    )
    report = build_monitor_report(
        spec=spec, ledger=ledger,
        scheduler_states={
            "12": {"state": "COMPLETED", "reason": "None"},
            "13": {"state": "COMPLETED", "reason": "None"},
        },
    )
    assert [row["classification"] for row in report["rows"][:2]] == [
        "complete", "complete",
    ]
    assert all(
        row["state"] == "NOT_SUBMITTED"
        and row["classification"] == "retryable_failure"
        and row["job_id"] is None
        for row in report["rows"][2:]
    )
    assert failed_downstream_closure(spec, report) == tuple(
        row["task_id"] for row in spec["tasks"][2:]
    )


def test_resource_recovery_rewrites_gpu_request_without_scientific_change(tmp_path):
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "old"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    resources = {key: dict(value) for key, value in SMOKE_RESOURCES.items()}
    resources["training"]["gpu"] = "gpu:h100:1"
    recovery = build_artifact(
        RESOURCE_RECOVERY_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"],
            "submission_ledger": "d" * 64, "monitor_report": "e" * 64,
        },
        kind="resource", closure=["train_F_RSET_U020"],
        project_dir=str(tmp_path / "old"), source_commit="a" * 40,
        resources=resources, recovery_path=str(tmp_path / "recovery.json"),
    )
    plan = recovery_command_plan(spec, recovery)
    command = plan["commands"][0]["command"]
    assert "--gres=gpu:h100:1" in command
    assert command.count("--signal=B:USR1@120") == 1


def test_recovery_plan_retains_authenticated_external_live_join(tmp_path):
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(tmp_path / "old"), "source_commit": "a" * 40,
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    recovery = build_artifact(
        RESOURCE_RECOVERY_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"],
            "submission_ledger": "d" * 64, "monitor_report": "e" * 64,
        },
        kind="resource", closure=["train_F_RREL_M1", "aggregate", "campaign_complete"],
        project_dir=str(tmp_path / "old"), source_commit="a" * 40,
        resources=SMOKE_RESOURCES, recovery_path=str(tmp_path / "recovery.json"),
        external_live_dependencies={"train_F_RSET_M1": "9876"},
    )
    plan = recovery_command_plan(spec, recovery)
    aggregate = next(row for row in plan["commands"] if row["task_id"] == "aggregate")
    assert aggregate["dependencies"] == ["train_F_RSET_M1", "train_F_RREL_M1"]
    dependency = next(
        token for token in aggregate["command"] if token.startswith("--dependency=")
    )
    assert "${JOB_train_F_RSET_M1}" in dependency
    assert "${JOB_train_F_RREL_M1}" in dependency


def test_resource_recovery_preserves_effective_source_recovery(monkeypatch, tmp_path):
    import hlt_classification.scouting.hcwdl_homotopy_representation_recovery as module

    original_project = tmp_path / "original"
    effective_project = tmp_path / "effective"
    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "campaign_root": str(tmp_path / "campaign"),
        "project_dir": str(original_project), "source_commit": "a" * 40,
        "semantic_source_sha256": {"science.py": "1" * 64},
        "parent_homotopy_spec_sha256": "b" * 64,
        "graph_sha256": GRAPH_SHA256, "combined_recipe_sha256": "c" * 64,
        "resources": SMOKE_RESOURCES, "tasks": _task_registry(),
        "final_test_accessed": False,
    })
    prior = build_artifact(
        SOURCE_RECOVERY_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"],
            "submission_ledger": "2" * 64, "monitor_report": "3" * 64,
        },
        kind="source", closure=["train_F_RREL_U100"],
        project_dir=str(effective_project), source_commit="d" * 40,
        semantic_source_sha256={"science.py": "4" * 64},
        resources=SMOKE_RESOURCES, recovery_path=str(tmp_path / "prior.json"),
        graph_sha256=GRAPH_SHA256, combined_recipe_sha256="c" * 64,
        output_root=str(tmp_path / "campaign"), authorization_phrase="source",
    )
    ledger = build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"], "command_plan": "5" * 64,
            "prior_submission_ledger": "6" * 64,
            "recovery": prior["content_hash"],
        },
        jobs={row["task_id"]: str(index + 10)
              for index, row in enumerate(spec["tasks"])},
        submission_phrase="source", complete_submission=True,
    )
    monitor = build_artifact(
        MONITOR_REPORT_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"],
            "submission_ledger": ledger["content_hash"],
        },
        rows=[{
            "task_id": row["task_id"], "classification": "complete",
            "job_id": ledger["jobs"][row["task_id"]],
        } for row in spec["tasks"]],
    )
    resources = {key: dict(value) for key, value in SMOKE_RESOURCES.items()}
    resources["training"]["walltime"] = "10:00:00"
    monkeypatch.setattr(module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"])
    monkeypatch.setattr(
        module, "failed_downstream_closure",
        lambda *_args, **_kwargs: ("train_F_RREL_U100",),
    )

    recovery = build_recovery(
        spec=spec, ledger=ledger, monitor=monitor, kind="resource",
        project_dir=effective_project, source_commit="d" * 40,
        resources=resources,
        authorization_phrase="AUTHORIZE HCWDL U RKD RESOURCE ONLY RECOVERY",
        recovery_path=tmp_path / "resource.json", prior_recovery=prior,
    )
    assert recovery["source_commit"] == "d" * 40
    assert recovery["project_dir"] == str(effective_project.resolve())
    assert recovery["semantic_source_sha256"] == {"science.py": "4" * 64}
    assert recovery["parents"]["effective_recovery"] == prior["content_hash"]
    assert recovery["resources"]["training"]["walltime"] == "10:00:00"


def test_chained_resource_recovery_requires_exact_prior_recovery(monkeypatch, tmp_path):
    import hlt_classification.scouting.hcwdl_homotopy_representation_recovery as module

    spec = with_content_hash({
        "contract": CAMPAIGN_SPEC_CONTRACT, "schema_version": SCHEMA_VERSION,
        "project_dir": str(tmp_path / "original"), "source_commit": "a" * 40,
        "semantic_source_sha256": {}, "resources": SMOKE_RESOURCES,
        "tasks": _task_registry(), "final_test_accessed": False,
    })
    ledger = build_artifact(
        SUBMISSION_LEDGER_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"], "command_plan": "5" * 64,
            "recovery": "6" * 64,
        },
        jobs={row["task_id"]: str(index + 10)
              for index, row in enumerate(spec["tasks"])},
        submission_phrase="source", complete_submission=True,
    )
    monitor = build_artifact(
        MONITOR_REPORT_CONTRACT,
        parents={
            "campaign_spec": spec["content_hash"],
            "submission_ledger": ledger["content_hash"],
        },
        rows=[{
            "task_id": row["task_id"], "classification": "complete",
            "job_id": ledger["jobs"][row["task_id"]],
        } for row in spec["tasks"]],
    )
    monkeypatch.setattr(module, "validate_campaign", lambda *_args, **_kwargs: spec["content_hash"])
    monkeypatch.setattr(module, "failed_downstream_closure", lambda *_args: ("aggregate",))
    try:
        build_recovery(
            spec=spec, ledger=ledger, monitor=monitor, kind="resource",
            project_dir=tmp_path / "original", source_commit="a" * 40,
            resources=SMOKE_RESOURCES,
            authorization_phrase="AUTHORIZE HCWDL U RKD RESOURCE ONLY RECOVERY",
            recovery_path=tmp_path / "resource.json",
        )
    except ValueError as error:
        assert "requires its prior recovery" in str(error)
    else:
        raise AssertionError("chained recovery accepted no prior recovery artifact")
