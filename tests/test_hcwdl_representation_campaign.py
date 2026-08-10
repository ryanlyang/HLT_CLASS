from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import hlt_classification.scouting.hcwdl_representation_campaign as campaign_module

from hlt_classification.scouting.hcwdl_representation_campaign import (
    AUTHORIZATION_PHRASE,
    CONTROLS,
    adapter_registered_input_requirements,
    assemble_submission_ledger_from_events,
    build_command_plan,
    build_submission_authorization,
    build_task_registry,
    create_campaign_spec,
    materialize_command,
    primary_node_ids,
    submit_command_plan,
    validate_campaign_spec,
    validate_submission_authorization,
    validate_submission_event_chain,
    validate_task_registry,
)
from hlt_classification.data.cache_contracts import with_content_hash


def _spec(tmp_path, disposition="combined_confirmatory"):
    return create_campaign_spec(
        mode="pilot", campaign_root=tmp_path / "campaign",
        checkpoint_namespace=tmp_path / "checkpoints", project_dir="/project",
        source_commit="1" * 40, source_manifest_sha256="2" * 64,
        split_manifest_sha256="3" * 64, parent_import_sha256="4" * 64,
        representation_recipe_sha256="5" * 64, graph_sha256="6" * 64,
        disposition_sha256="7" * 64, disposition=disposition,
        role_counts={"train": 300_000, "validation": 100_000, "final_test": 100_000},
        final_source_partitions=3, combined_finalist_count=12,
    )


def test_registry_contains_exact_primary_and_controls_and_serial_banks() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=3,
        combined_finalist_count=12,
    )
    assert {row.graph_node for row in tasks if row.kind == "train_node"} == set(primary_node_ids())
    assert {row.graph_node for row in tasks if row.kind == "train_control"} == set(CONTROLS)
    assert len(primary_node_ids()) == 86
    assert len(tasks) == 295
    assert all(row.array is None or "%" not in row.array for row in tasks)


def test_command_plan_identity_is_acyclic_with_runtime_binding() -> None:
    spec = _spec(Path("/tmp"))
    unbound = build_command_plan(spec)
    bound = build_command_plan({
        **spec,
        "runtime_binding_sha256": "a" * 64,
        "runtime_status": "immutable",
    })
    assert bound == unbound


def test_registry_executes_bounded_cache_control_and_exact_final_producers() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=3,
        combined_finalist_count=12,
    )
    by_key = {row.task_key: row for row in tasks}
    assert [
        by_key[name].kind for name in (
            "miniature_D100_build", "miniature_D100_verify_cleanup",
            "miniature_TOFF_build", "miniature_TOFF_verify_cleanup",
            "cache_miniature",
        )
    ] == [
        "target_build", "cache_miniature_bank", "target_build",
        "cache_miniature_bank", "cache_miniature",
    ]
    assert by_key["miniature_TOFF_build"].dependencies == (
        "miniature_D100_verify_cleanup",
    )
    assert by_key["cache_miniature"].dependencies == (
        "miniature_D100_verify_cleanup", "miniature_TOFF_verify_cleanup",
    )
    assert {
        f"${{cache_miniature:{bank}:{kind}}}"
        for bank in ("D100", "TOFF")
        for kind in (
            "evidence", "cleanup_authorization", "cleanup_completion",
        )
    } <= set(by_key["cache_miniature"].registered_inputs)
    assert by_key["zero_coefficient_acceptance"].dependencies == ("smoke_probe",)
    assert {
        "${parent_import}", "${task_output:architecture_attestation}",
        "${task_output:parent_loss_attestation}",
    } <= set(by_key["zero_coefficient_acceptance"].registered_inputs)
    assert "${representation_recipe}" not in by_key[
        "control_registry"
    ].registered_inputs
    assert "shuffle_map" not in by_key
    assert by_key["target_TOFF_screen"].dependencies[-1] == "pretraining_reservation"
    assert "final/selection/branch_access.json" in by_key[
        "final_selection"
    ].registered_outputs
    assert "final/capabilities/selection.json" in by_key[
        "final_selection"
    ].registered_outputs
    for key in (
        "final_assignment_shards", "final_prediction_shards",
        "locked_metric_join",
    ):
        assert "final/capabilities/${task_id}.json" in by_key[key].registered_outputs
    assert by_key["representation_execution_lock"].registered_outputs == (
        "locks/07_execution.json", "final/prediction_spec.json",
    )
    assert by_key["locked_metric_join"].registered_outputs == (
        "final/metric_join.json", "final/evaluations",
        "reports/paired_bootstrap", "final/capabilities/${task_id}.json",
    )
    assert by_key["final_aggregate"].dependencies == ("locked_metric_join",)
    assert by_key["final_aggregate"].registered_outputs == (
        "reports/final_aggregate.json",
    )


def test_confirmation_rows_publish_registered_run_pointers() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=2,
        combined_finalist_count=3,
    )
    confirmation = [row for row in tasks if row.kind == "confirmation"]
    assert len(confirmation) == 4
    assert all(
        "confirmation/runs/${execution_id}.json" in row.registered_outputs
        for row in confirmation
    )


def test_warm_m1_binds_immediate_same_strategy_predecessor() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=2,
        combined_finalist_count=3,
    )
    by_key = {row.task_key: row for row in tasks}
    for strategy in ("RSET", "RREL"):
        warm = by_key[f"train_{strategy}_M1w"]
        assert f"${{task_output:train_{strategy}_D0w:3}}" in warm.registered_inputs
        assert "${parent_model_sources}" not in warm.registered_inputs
        assert "${parent_reports}" not in warm.registered_inputs
        cold = by_key[f"train_{strategy}_M1c"]
        assert "${parent_model_sources}" not in cold.registered_inputs
        assert not any(
            value.startswith("${task_output:train_") and value.endswith(":3}")
            for value in cold.registered_inputs
        )


def test_parent_import_and_downstream_parent_consumers_bind_fresh_evidence() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=2,
        combined_finalist_count=17,
    )
    by_key = {row.task_key: row for row in tasks}
    parent_import = by_key["parent_import"]
    assert {
        "${prebuilt_parent_import}", "${parent_reports}",
        "${parent_model_sources}",
        "${parent_confirmation_reports}",
        "${task_output:architecture_attestation:0}",
        "${task_output:parent_loss_attestation:0}",
    } <= set(parent_import.registered_inputs)
    for task in tasks:
        if task.kind == "target_build":
            assert "${task_output:architecture_attestation:0}" in task.registered_inputs
            assert "${parent_import}" in task.registered_inputs
    screen = by_key["screen_aggregate"]
    assert "${parent_reports}" in screen.registered_inputs
    assert "${task_output:architecture_attestation:0}" in screen.registered_inputs


def test_finalist_lock_registers_every_artifact_it_freezes() -> None:
    tasks = build_task_registry(
        disposition="combined_confirmatory", final_source_partitions=2,
        combined_finalist_count=5,
    )
    finalist_lock = next(task for task in tasks if task.task_key == "finalist_lock")
    required = {
        "${task_output:architecture_attestation:0}",
        "${task_output:parent_loss_attestation:0}",
    }
    for node in ("RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w"):
        required.update(
            f"${{task_output:train_{node}:{output_index}}}"
            for output_index in (0, 1, 2)
        )
    assert required <= set(finalist_lock.registered_inputs)

    execution_lock = next(
        task for task in tasks if task.task_key == "representation_execution_lock"
    )
    assert "${prediction_runtime_signature}" in execution_lock.registered_inputs

    shared_claim = next(task for task in tasks if task.task_key == "shared_claim_gate")
    prediction = next(task for task in tasks if task.task_key == "final_prediction_shards")
    prediction_finalize = next(
        task for task in tasks if task.task_key == "final_prediction_manifests"
    )
    for node in ("RSET_M1c", "RSET_M1w", "RREL_M1c", "RREL_M1w"):
        assert f"${{task_output:train_{node}:2}}" in shared_claim.registered_inputs
        assert f"${{task_output:train_{node}:3}}" in prediction.registered_inputs
        for output_index in (0, 1, 2):
            assert (
                f"${{task_output:train_{node}:{output_index}}}"
                in prediction_finalize.registered_inputs
            )


def test_validation_only_registry_contains_no_final_role_tasks(tmp_path) -> None:
    spec = _spec(tmp_path, "validation_only_parent_claim_consumed")
    validate_campaign_spec(spec)
    forbidden = {"final_selection", "assignment_shard", "prediction_shard", "metric_join"}
    assert not forbidden.intersection(row["kind"] for row in spec["tasks"])
    aggregate = next(row for row in spec["tasks"] if row["kind"] == "validation_only_aggregate")
    assert aggregate["registered_outputs"] == (
        "reports/validation_only_aggregate.json",
    )


def test_symbolic_command_plan_and_materialization(tmp_path) -> None:
    spec = _spec(tmp_path)
    plan = build_command_plan(spec)
    assert all(
        f"--job-name=hcwdlr_{row['task_key']}" in row["command"]
        for row in plan["commands"]
    )
    comments = [
        next(token for token in command["command"] if token.startswith("--comment="))
        for command in plan["commands"]
    ]
    assert len(comments) == len(set(comments)) == len(plan["commands"])
    assert all(
        comment == f"--comment=hcwdl-rkd-{row['scheduler_reconciliation_token']}"
        for comment, row in zip(comments, plan["commands"], strict=True)
    )
    row = next(value for value in plan["commands"] if value["dependencies"])
    assert any("${afterok:" in token for token in row["command"])
    ids = {dependency: str(index + 100) for index, dependency in enumerate(row["dependencies"])}
    materialized = materialize_command(row, job_ids=ids)
    assert not any("${afterok:" in token for token in materialized)
    assert all(token in materialized for token in (f"afterok:{value}" for value in ids.values()))


def _submission_test_context(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_command_plan(spec)
    monkeypatch.setattr(
        campaign_module, "validate_campaign_spec", lambda value, executable=False: value["content_hash"],
    )
    return spec, plan


def test_initial_submission_resumes_from_durable_event_prefix_without_duplicates(
    tmp_path, monkeypatch,
) -> None:
    spec, plan = _submission_test_context(tmp_path, monkeypatch)
    events = []
    scheduled = []

    def scheduler(command):
        scheduled.append(list(command))
        return str(1000 + len(scheduled))

    def crashing_writer(event):
        events.append(dict(event))
        if event["phase"] == "submitted" and event["task_sequence"] == 1:
            raise RuntimeError("injected after durable submission event")

    with pytest.raises(RuntimeError, match="injected"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
            checkout_validator=lambda _: None, event_writer=crashing_writer,
        )
    assert len(scheduled) == 2
    ledger = submit_command_plan(
        spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
        checkout_validator=lambda _: None, event_writer=lambda row: events.append(dict(row)),
        prior_events=events,
    )
    assert len(scheduled) == len(plan["commands"])
    validate_submission_event_chain(
        events, spec=spec, command_plan=plan, require_complete=True,
    )
    assert ledger == assemble_submission_ledger_from_events(
        events, spec=spec, command_plan=plan,
    )

    tampered = dict(events[0]); tampered.pop("content_hash")
    tampered["command"] = [*tampered["command"], "--forged"]
    forged = [with_content_hash(tampered), *events[1:]]
    with pytest.raises(ValueError, match="intent differs|chain differs"):
        validate_submission_event_chain(
            forged, spec=spec, command_plan=plan,
        )


def test_unresolved_submission_intent_requires_exact_reconciliation(
    tmp_path, monkeypatch,
) -> None:
    spec, plan = _submission_test_context(tmp_path, monkeypatch)
    events = []
    scheduled_ids = []

    def scheduler(_command):
        job_id = str(2000 + len(scheduled_ids))
        scheduled_ids.append(job_id)
        return job_id

    def lose_first_result(event):
        if event["phase"] == "submitted":
            raise RuntimeError("injected before result publication")
        events.append(dict(event))

    with pytest.raises(RuntimeError, match="injected"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
            checkout_validator=lambda _: None, event_writer=lose_first_result,
        )
    assert len(events) == len(scheduled_ids) == 1
    with pytest.raises(PermissionError, match="reconciliation"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
            checkout_validator=lambda _: None,
            event_writer=lambda row: events.append(dict(row)), prior_events=events,
        )
    assert len(scheduled_ids) == 1
    with pytest.raises(PermissionError, match="live scheduler"):
        submit_command_plan(
            spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
            checkout_validator=lambda _: None,
            event_writer=lambda row: events.append(dict(row)), prior_events=events,
            reconciled_job_ids={events[0]["content_hash"]: scheduled_ids[0]},
        )
    assert len(events) == len(scheduled_ids) == 1
    ledger = submit_command_plan(
        spec=spec, command_plan=plan, scheduler=scheduler, execute=True,
        checkout_validator=lambda _: None,
        event_writer=lambda row: events.append(dict(row)), prior_events=events,
        reconciled_job_ids={events[0]["content_hash"]: scheduled_ids[0]},
        reconciliation_validator=lambda _intent, _job_id: None,
    )
    assert len(scheduled_ids) == len(plan["commands"])
    assert ledger["jobs"][plan["commands"][0]["task_key"]] == scheduled_ids[0]


def test_submission_authorization_binds_strict_candidate_audit() -> None:
    hashes = {
        "command_plan_sha256": "2" * 64,
        "executable_candidate_audit_sha256": "3" * 64,
        "resource_profile_sha256": "4" * 64,
        "storage_estimate_sha256": "5" * 64,
        "tigris_acceptance_sha256": "6" * 64,
        "parent_import_sha256": "7" * 64,
        "representation_recipe_sha256": "8" * 64,
        "disposition_sha256": "9" * 64,
    }
    authorization = build_submission_authorization(
        mode="pilot",
        source_commit="1" * 40,
        authorization_phrase=AUTHORIZATION_PHRASE,
        **hashes,
    )
    validate_submission_authorization(
        authorization, mode="pilot", source_commit="1" * 40, **hashes,
    )
    with pytest.raises(PermissionError, match="lineage differs"):
        validate_submission_authorization(
            authorization,
            mode="pilot",
            source_commit="1" * 40,
            **{**hashes, "executable_candidate_audit_sha256": "a" * 64},
        )


def test_planning_spec_cannot_embed_candidate_or_authorization(tmp_path) -> None:
    arguments = {
        "mode": "pilot",
        "campaign_root": tmp_path / "campaign",
        "checkpoint_namespace": tmp_path / "checkpoints",
        "project_dir": "/project",
        "source_commit": "1" * 40,
        "source_manifest_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "parent_import_sha256": "4" * 64,
        "representation_recipe_sha256": "5" * 64,
        "graph_sha256": "6" * 64,
        "disposition_sha256": "7" * 64,
        "disposition": "combined_confirmatory",
        "role_counts": {
            "train": 300_000, "validation": 100_000, "final_test": 100_000,
        },
        "final_source_partitions": 3,
        "combined_finalist_count": 12,
    }
    with pytest.raises(ValueError, match="candidate or submission authority"):
        create_campaign_spec(
            **arguments,
            executable_candidate_audit_sha256="8" * 64,
        )

    forged = _spec(tmp_path / "forged")
    forged = with_content_hash({
        **forged,
        "planning_only": False,
        "live_submission_authorized": True,
    })
    with pytest.raises(PermissionError, match="planning flags differ"):
        validate_campaign_spec(forged, executable=False)


def test_every_adapter_artifact_requirement_has_an_exact_registered_route() -> None:
    partitions = 3
    finalists = 12
    tasks = build_task_registry(
        disposition="combined_confirmatory",
        final_source_partitions=partitions,
        combined_finalist_count=finalists,
    )
    by_key = {row.task_key: row for row in tasks}
    for row in tasks:
        required = set(adapter_registered_input_requirements(
            row,
            final_source_partitions=partitions,
            combined_finalist_count=finalists,
        ))
        assert required <= set(row.registered_inputs), row.task_key
        for dependency in row.dependencies:
            producer = by_key[dependency]
            if producer.array is not None or len(producer.registered_outputs) != 1:
                assert f"${{task_output:{dependency}}}" not in row.registered_inputs

    screen_index = next(
        index for index, row in enumerate(tasks) if row.kind == "screen_aggregate"
    )
    screen = tasks[screen_index]
    required_screen_report = "${task_output:train_RSET_M1c:0}"
    forged_screen = replace(
        screen,
        registered_inputs=tuple(
            value for value in screen.registered_inputs
            if value != required_screen_report
        ),
    )
    forged = (*tasks[:screen_index], forged_screen, *tasks[screen_index + 1:])
    with pytest.raises(ValueError, match="production-adapter artifact routes"):
        validate_task_registry(forged, disposition="combined_confirmatory")
