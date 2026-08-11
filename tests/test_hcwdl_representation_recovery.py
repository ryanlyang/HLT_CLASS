from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_representation_campaign import (
    COMMAND_PLAN_CONTRACT,
    SUBMISSION_LEDGER_CONTRACT,
)
from hlt_classification.scouting.hcwdl_representation_recovery import (
    build_monitor_report,
    build_recovery_plan,
    build_recovery_submission_ledger,
    exact_cancellation_commands,
    validate_ledger_chain,
)
from hlt_classification.scouting.hcwdl_representation_runtime_binding import (
    build_runtime_binding,
)


def _posix_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return value.split(":", 1)[-1] if ":" in value else value


def _runtime_fixture(tmp_path: Path):
    root = _posix_path(tmp_path)
    tasks = [
        {
            "task_key": key, "kind": "tap_schema", "dependencies": [],
            "resource_class": "cpu_small", "registered_inputs": [],
            "registered_outputs": [f"artifacts/{key}.json"],
        }
        for key in ("one", "two")
    ]
    spec = with_content_hash({
        "mode": "pilot", "campaign_root": root,
        "checkpoint_namespace": f"{root}/checkpoints",
        "project_dir": "/project", "source_commit": "1" * 40,
        "source_manifest_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "parent_import_sha256": "4" * 64,
        "representation_recipe_sha256": "5" * 64,
        "graph_sha256": "6" * 64, "disposition_sha256": "7" * 64,
        "disposition": "combined_confirmatory", "role_counts": {},
        "final_source_partitions": 0, "combined_finalist_count": 0,
        "artifact_paths": {"runtime_binding": f"{root}/runtime_binding.json"},
        "resources": {"cpu_small": {}}, "array_concurrency_limits": {},
        "resource_request_sha256": "8" * 64, "tasks": tasks,
    })
    facts = {
        "conda_environment": "atlas_kd_tigris", "data_root": "/data",
        "device": "cpu", "project_dir": "/project",
        "python_no_user_site": True, "source_snapshot_sha256": "9" * 64,
        "weaver_runtime_sha256": "a" * 64,
    }
    rows = {
        key: {"single": {
            "array_index": None, "device": "cpu", "inputs": {},
            "outputs": {f"artifacts/{key}.json": f"{root}/artifacts/{key}.json"},
            "parameters": {}, "runtime_signature_sha256": "b" * 64,
        }}
        for key in ("one", "two")
    }
    return spec, build_runtime_binding(spec=spec, runtime_facts=facts, task_rows=rows)


def _publish_output(tmp_path: Path, task: str) -> None:
    write_immutable_json(
        tmp_path / "artifacts" / f"{task}.json",
        with_content_hash({
            "contract": "HCWDL_REPRESENTATION_TAP/v1",
            "schema_version": 1, "task": task,
        }),
    )


def _ledger(*, campaign: str, command_plan: str = "b" * 64):
    return with_content_hash({
        "contract": SUBMISSION_LEDGER_CONTRACT, "schema_version": 1,
        "campaign_spec_sha256": campaign, "command_plan_sha256": command_plan,
        "jobs": {"one": "101", "two": "102"}, "materialized_commands": [],
    })


def _monitor(ledger, *, states=None):
    return build_monitor_report(
        original_ledger=ledger, recovery_ledgers=(),
        scheduler_states=states or {"101": "COMPLETED", "102": "FAILED"},
        previous_report_sha256=None, sequence=0,
    )


def _command_plan():
    return with_content_hash({
        "contract": COMMAND_PLAN_CONTRACT, "schema_version": 1,
        "campaign_identity": {}, "worktree_clean_required": True,
        "commands": [
            {"task_key": "one", "dependencies": [], "command": ["sbatch", "one"]},
            {"task_key": "two", "dependencies": ["one"],
             "command": ["sbatch", "--dependency", "${afterok:one}", "two"]},
        ],
    })


def test_monitor_recovery_audits_filesystem_and_exact_cancellation(tmp_path) -> None:
    spec, binding = _runtime_fixture(tmp_path)
    _publish_output(tmp_path, "one")
    ledger = _ledger(campaign=spec["content_hash"])
    plan = build_recovery_plan(
        monitor_report=_monitor(ledger), spec=spec, runtime_binding=binding,
    )
    assert [row["task_key"] for row in plan["retry_rows"]] == ["two"]
    assert [row["status"] for row in plan["output_audit"]["rows"]] == [
        "valid", "absent",
    ]
    assert exact_cancellation_commands(ledger, ()) == [
        ["scancel", "101"], ["scancel", "102"],
    ]


def test_dispatch_reaudits_filesystem_and_rejects_stale_plan(tmp_path) -> None:
    spec, binding = _runtime_fixture(tmp_path)
    _publish_output(tmp_path, "one")
    command_plan = _command_plan()
    original = _ledger(
        campaign=spec["content_hash"], command_plan=command_plan["content_hash"],
    )
    plan = build_recovery_plan(
        monitor_report=_monitor(original), spec=spec, runtime_binding=binding,
    )
    _publish_output(tmp_path, "two")
    with pytest.raises(ValueError, match="fresh filesystem"):
        build_recovery_submission_ledger(
            recovery_plan=plan, command_plan=command_plan,
            original_ledger=original, prior_recovery_ledgers=(),
            spec=spec, runtime_binding=binding,
            scheduler=lambda _command: "201", execute=True,
        )


def test_recovery_dependencies_use_latest_ids_and_monitor_keeps_union(tmp_path) -> None:
    spec, binding = _runtime_fixture(tmp_path)
    command_plan = _command_plan()
    original = _ledger(
        campaign=spec["content_hash"], command_plan=command_plan["content_hash"],
    )
    _publish_output(tmp_path, "two")
    first_plan = build_recovery_plan(
        monitor_report=_monitor(
            original, states={"101": "FAILED", "102": "COMPLETED"},
        ),
        spec=spec, runtime_binding=binding,
    )
    commands = []
    first = build_recovery_submission_ledger(
        recovery_plan=first_plan, command_plan=command_plan,
        original_ledger=original, prior_recovery_ledgers=(),
        spec=spec, runtime_binding=binding,
        scheduler=lambda command: commands.append(command) or "201", execute=True,
    )

    _publish_output(tmp_path, "one")
    (tmp_path / "artifacts" / "two.json").unlink()
    second_plan = build_recovery_plan(
        monitor_report=build_monitor_report(
            original_ledger=original, recovery_ledgers=(first,),
            scheduler_states={"101": "FAILED", "102": "FAILED", "201": "COMPLETED"},
            previous_report_sha256=None, sequence=0,
        ),
        spec=spec, runtime_binding=binding,
    )
    second = build_recovery_submission_ledger(
        recovery_plan=second_plan, command_plan=command_plan,
        original_ledger=original, prior_recovery_ledgers=(first,),
        spec=spec, runtime_binding=binding,
        scheduler=lambda command: commands.append(command) or "202", execute=True,
    )
    assert commands[-1] == ["sbatch", "--dependency", "afterok:201", "two"]
    assert validate_ledger_chain(original, (first, second)) == {
        "one": "201", "two": "202",
    }
    monitor = build_monitor_report(
        original_ledger=original, recovery_ledgers=(first, second),
        scheduler_states={
            "101": "FAILED", "102": "FAILED", "201": "COMPLETED",
            "202": "RUNNING",
        },
        previous_report_sha256=None, sequence=0,
    )
    assert sum(bool(row["superseded"]) for row in monitor["rows"]) == 2
    assert exact_cancellation_commands(original, (first, second)) == [
        ["scancel", "101"], ["scancel", "102"],
        ["scancel", "201"], ["scancel", "202"],
    ]
