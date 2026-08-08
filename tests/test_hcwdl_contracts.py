from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash, write_immutable_json
from hlt_classification.scouting.hcwdl_campaign import (
    LEGACY_CAMPAIGN_CONTRACT, PREVIOUS_CAMPAIGN_CONTRACT,
    PRIOR_CAMPAIGN_CONTRACT, ROLE_COUNTS, build_command_plan,
    create_campaign_spec, slurm_commands, split_submission_commands,
    validate_campaign_spec,
)
from hlt_classification.scouting.hcwdl_authorization import (
    AUTHORIZATION_PHRASE, LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT,
    PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT,
    PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT, build_submission_authorization,
    require_canonical_campaign_spec_path, validate_submission_authorization,
)
from hlt_classification.scouting.hcwdl_contracts import (
    authenticate_source_files, require_role_access,
)
from hlt_classification.data.cache_contracts import sha256_file
from hlt_classification.scouting.hcwdl_ladder import NODE_REGISTRY
from hlt_classification.scouting.hcwdl_ladder import GRAPH_SHA256
from hlt_classification.scouting.hcwdl_locks import (
    claim_final_execution, create_execution_lock, create_lock, validate_lock,
)
from hlt_classification.scouting.hcwdl_recipe import example_recipe, validate_recipe
from hlt_classification.scouting.hcwdl_qualification import (
    DIAGNOSTIC_ACK_PHRASE, QUALIFIERS, build_diagnostic_acknowledgement,
    validate_diagnostic_acknowledgement,
)
from hlt_classification.scouting.hcwdl_recovery import (
    assemble_submission_ledger, build_monitor_report, build_submission_event,
    build_submission_ledger, build_task_attestation, exact_cancel_ids, resume_tasks,
    validate_submission_ledger, validate_task_attestation,
)
from hlt_classification.scouting.hcwdl_resources import (
    build_resource_profile, estimate_storage, validate_resource_profile,
)
from hlt_classification.scouting.hcwdl_reporting import (
    build_confirmation_registry, build_screen_aggregate, result_key,
    select_declared_candidate,
)
from hlt_classification.scouting.hcwdl_training import TRAINING_REPORT_CONTRACT
from hlt_classification.scouting.engine import PMARD_TRAINING_REPORT_CONTRACT
from hlt_classification.scouting.hcwdl_workflow import HcwdlWorkflow
from hlt_classification.scouting.hcwdl_views import EphemeralHcwdlTargetBank, EphemeralHcwdlViewBank
from hlt_classification.scouting.inputs import ParticleInputs


H = "a" * 64
G = "b" * 64


def _spec(mode: str, *, planning: bool = True):
    return create_campaign_spec(
        mode=mode, campaign_root="/campaign", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=None if planning else H,
        recipe_path=None if planning else "/campaign/recipe.json",
        planning_only=planning, source_manifest_path="/source.json",
        split_manifest_path="/split.json", data_root="/data",
    )


def test_campaign_modes_are_complete_uncapped_and_test_sealed():
    smoke = _spec("smoke")
    pilot = _spec("pilot")
    midscale500k = _spec("midscale500k")
    midscale1m = _spec("midscale1m")
    midscale2m = _spec("midscale2m")
    assert smoke["role_counts"] == ROLE_COUNTS["smoke"]
    assert pilot["role_counts"] == {"train": 300_000, "validation": 100_000, "final_test": 100_000}
    assert midscale500k["role_counts"] == {
        "train": 500_000, "validation": 250_000, "final_test": 250_000,
    }
    assert midscale1m["role_counts"] == {
        "train": 1_000_000, "validation": 400_000, "final_test": 400_000,
    }
    assert midscale2m["role_counts"] == {
        "train": 2_000_000, "validation": 500_000, "final_test": 500_000,
    }
    smoke_tasks = {row["task_id"]: row for row in smoke["tasks"]}
    pilot_tasks = {row["task_id"]: row for row in pilot["tasks"]}
    assert len([row for row in smoke["tasks"] if row["graph_node"]]) == 23
    assert {row["graph_node"] for row in smoke["tasks"] if row["graph_node"]} == set(NODE_REGISTRY)
    assert "test_row_selection" not in smoke_tasks
    assert pilot_tasks["test_row_selection"]["dependencies"] == ("execution_lock",)
    assert all("%" not in row["array"] for row in pilot["tasks"] if row["array"])
    assert pilot_tasks["assign_train"]["array"] == "0-3"
    assert pilot_tasks["assign_validation"]["array"] == "0-1"
    validate_campaign_spec(smoke)
    validate_campaign_spec(midscale500k)
    validate_campaign_spec(midscale1m)
    validate_campaign_spec(midscale2m)
    with pytest.raises(PermissionError):
        validate_campaign_spec(pilot, executable=True)


def test_campaign_mode_counts_are_exact_and_v3_through_v5_remain_readable():
    midscale500k = _spec("midscale500k")
    midscale1m = _spec("midscale1m")
    midscale2m = _spec("midscale2m")
    forged = dict(midscale2m)
    forged["role_counts"] = dict(ROLE_COUNTS["pilot"])
    forged = with_content_hash(forged)
    with pytest.raises(ValueError, match="registered mode"):
        validate_campaign_spec(forged)

    legacy = dict(_spec("pilot"))
    legacy["contract"] = LEGACY_CAMPAIGN_CONTRACT
    legacy["schema_version"] = 3
    legacy = with_content_hash(legacy)
    assert validate_campaign_spec(legacy) == legacy["content_hash"]

    previous = dict(midscale500k)
    previous["contract"] = PREVIOUS_CAMPAIGN_CONTRACT
    previous["schema_version"] = 4
    previous = with_content_hash(previous)
    assert validate_campaign_spec(previous) == previous["content_hash"]

    prior = dict(midscale1m)
    prior["contract"] = PRIOR_CAMPAIGN_CONTRACT
    prior["schema_version"] = 5
    prior = with_content_hash(prior)
    assert validate_campaign_spec(prior) == prior["content_hash"]

    invalid_legacy = dict(midscale500k)
    invalid_legacy["contract"] = LEGACY_CAMPAIGN_CONTRACT
    invalid_legacy["schema_version"] = 3
    invalid_legacy = with_content_hash(invalid_legacy)
    with pytest.raises(ValueError, match="mode or graph"):
        validate_campaign_spec(invalid_legacy)

    invalid_previous = dict(midscale1m)
    invalid_previous["contract"] = PREVIOUS_CAMPAIGN_CONTRACT
    invalid_previous["schema_version"] = 4
    invalid_previous = with_content_hash(invalid_previous)
    with pytest.raises(ValueError, match="mode or graph"):
        validate_campaign_spec(invalid_previous)

    invalid_prior = dict(midscale2m)
    invalid_prior["contract"] = PRIOR_CAMPAIGN_CONTRACT
    invalid_prior["schema_version"] = 5
    invalid_prior = with_content_hash(invalid_prior)
    with pytest.raises(ValueError, match="mode or graph"):
        validate_campaign_spec(invalid_prior)


def test_midscale2m_authorization_is_v6_and_v3_through_v5_remain_readable():
    authorization = build_submission_authorization(
        mode="midscale2m", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H, resource_request_sha256=G,
        command_plan_sha256=H, authorization_phrase=AUTHORIZATION_PHRASE,
    )
    assert authorization["contract"] == "HCWDL_SUBMISSION_AUTHORIZATION/v6"
    assert authorization["schema_version"] == 6
    assert validate_submission_authorization(
        authorization, mode="midscale2m", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H, resource_request_sha256=G,
        command_plan_sha256=H, production_authorization_sha256=None,
    ) == authorization["content_hash"]

    prior = dict(authorization)
    prior["contract"] = PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT
    prior["schema_version"] = 5
    prior["mode"] = "midscale1m"
    prior = with_content_hash(prior)
    assert validate_submission_authorization(
        prior, mode="midscale1m", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H, resource_request_sha256=G,
        command_plan_sha256=H, production_authorization_sha256=None,
    ) == prior["content_hash"]

    invalid_prior = dict(authorization)
    invalid_prior["contract"] = PRIOR_SUBMISSION_AUTHORIZATION_CONTRACT
    invalid_prior["schema_version"] = 5
    invalid_prior = with_content_hash(invalid_prior)
    with pytest.raises(ValueError, match="authorization mode"):
        validate_submission_authorization(
            invalid_prior, mode="midscale2m", source_commit="c" * 40,
            source_manifest_sha256=H, split_manifest_sha256=G,
            recipe_sha256=H, resource_request_sha256=G,
            command_plan_sha256=H, production_authorization_sha256=None,
        )

    previous = dict(authorization)
    previous["contract"] = PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT
    previous["schema_version"] = 4
    previous["mode"] = "midscale500k"
    previous = with_content_hash(previous)
    assert validate_submission_authorization(
        previous, mode="midscale500k", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H, resource_request_sha256=G,
        command_plan_sha256=H, production_authorization_sha256=None,
    ) == previous["content_hash"]

    invalid_previous = dict(authorization)
    invalid_previous["contract"] = PREVIOUS_SUBMISSION_AUTHORIZATION_CONTRACT
    invalid_previous["schema_version"] = 4
    invalid_previous = with_content_hash(invalid_previous)
    with pytest.raises(ValueError, match="authorization mode"):
        validate_submission_authorization(
            invalid_previous, mode="midscale2m", source_commit="c" * 40,
            source_manifest_sha256=H, split_manifest_sha256=G,
            recipe_sha256=H, resource_request_sha256=G,
            command_plan_sha256=H, production_authorization_sha256=None,
        )

    legacy = dict(authorization)
    legacy["contract"] = LEGACY_SUBMISSION_AUTHORIZATION_CONTRACT
    legacy["schema_version"] = 3
    legacy["mode"] = "pilot"
    legacy = with_content_hash(legacy)
    assert validate_submission_authorization(
        legacy, mode="pilot", source_commit="c" * 40,
        source_manifest_sha256=H, split_manifest_sha256=G,
        recipe_sha256=H, resource_request_sha256=G,
        command_plan_sha256=H, production_authorization_sha256=None,
    ) == legacy["content_hash"]


def test_executable_campaign_spec_path_is_not_redirectable(tmp_path: Path):
    root = tmp_path / "campaign"
    canonical = root / "campaign_spec.json"
    assert require_canonical_campaign_spec_path(canonical, campaign_root=root) == canonical.resolve()
    with pytest.raises(PermissionError, match="canonical campaign path"):
        require_canonical_campaign_spec_path(tmp_path / "copy.json", campaign_root=root)


def test_source_authentication_rehashes_bytes_and_rejects_drift(tmp_path: Path):
    source = tmp_path / "sample.root"; source.write_bytes(b"immutable fixture")
    record = {"path": source.name, "sha256": sha256_file(source)}
    report = authenticate_source_files(tmp_path, (record,))
    assert report["files"] == 1 and report["all_source_bytes_reauthenticated"]
    source.write_bytes(b"drifted fixture")
    with pytest.raises(ValueError, match="source bytes differ"):
        authenticate_source_files(tmp_path, (record,))


def test_slurm_commands_are_topological_absolute_and_checkpoint_signaled():
    commands = slurm_commands(_spec("pilot"))
    positions = {row["task_id"]: index for index, row in enumerate(commands)}
    for row in commands:
        assert all(positions[parent] < positions[row["task_id"]] for parent in row["dependencies"])
        command = row["command"]
        assert "--account=reu-aisocial" in command
        assert "--partition=tigris" in command
        assert command[-1] == "/home/ryreu/atlas/HLT_Classification/sbatch/run_hcwdl_task.sh"
    train = next(row for row in commands if row["task_id"] == "train_M0")
    assert "--signal=B:USR1@120" in train["command"]
    endpoint_gate = next(
        row for row in commands if row["task_id"] == "shell_endpoint_qualification_lock"
    )
    assert "--hold" not in endpoint_gate["command"]
    qualification, ladder = split_submission_commands(_spec("pilot"))
    assert qualification[-1]["task_id"] == "endpoint_qualification"
    assert ladder[0]["task_id"] == "shell_endpoint_qualification_lock"


def test_recipe_contains_explicit_control_policy_and_remains_unauthorized():
    recipe = example_recipe()
    assert recipe["controls"]["predecessor_only_coefficients"] == {
        "ce": 0.25, "predecessor_kd": 0.75,
    }
    with pytest.raises(PermissionError):
        validate_recipe(recipe, require_authorized=True)


def test_lock_chain_and_final_claim_are_fail_closed(tmp_path: Path):
    with pytest.raises(PermissionError):
        require_role_access("final_test", branch_read=True, completed_locks=("finalist",))
    require_role_access(
        "final_test", branch_read=True, completed_locks=("finalist", "execution"),
    )
    assignment = create_lock("assignment", campaign_spec_sha256=H, payload={"authorized": True})
    recipe = create_lock("recipe", campaign_spec_sha256=H, parent_lock=assignment, payload={})
    qualification = create_lock(
        "shell_endpoint_qualification", campaign_spec_sha256=H,
        parent_lock=recipe, payload={},
    )
    confirmation = create_lock(
        "confirmation_registry", campaign_spec_sha256=H,
        parent_lock=qualification, payload={},
    )
    finalist = create_lock("finalist", campaign_spec_sha256=H, parent_lock=confirmation, payload={})
    execution = create_lock("execution", campaign_spec_sha256=H, parent_lock=finalist, payload={})
    validate_lock(execution, expected_level="execution")
    claim_path = tmp_path / "claim.json"
    claim_final_execution(claim_path, execution_lock=execution, test_assignment_manifest_sha256=G)
    with pytest.raises(FileExistsError):
        claim_final_execution(claim_path, execution_lock=execution, test_assignment_manifest_sha256=G)
    bound_execution = create_execution_lock(
        campaign_spec_sha256=H, finalist_lock=finalist,
        split_manifest_sha256=G, final_test_selection_rule_sha256=H,
        matcher_resources_sha256=G, recipe_sha256=H, source_commit="c" * 40,
    )
    assert bound_execution["payload"]["final_test_selection_rule_sha256"] == H


def test_endpoint_acknowledgement_is_explicit_and_binds_every_diagnostic():
    qualifier_hashes = {name: H for name in QUALIFIERS}
    with pytest.raises(PermissionError):
        build_diagnostic_acknowledgement(
            campaign_spec_sha256=H, assignment_manifest_sha256=G,
            recipe_sha256=H, cache_miniature_sha256=G,
            qualifier_report_sha256=qualifier_hashes,
            acknowledgement_phrase="yes",
        )
    acknowledgement = build_diagnostic_acknowledgement(
        campaign_spec_sha256=H, assignment_manifest_sha256=G,
        recipe_sha256=H, cache_miniature_sha256=G,
        qualifier_report_sha256=qualifier_hashes,
        acknowledgement_phrase=DIAGNOSTIC_ACK_PHRASE,
    )
    assert validate_diagnostic_acknowledgement(
        acknowledgement, campaign_spec_sha256=H, assignment_manifest_sha256=G,
        recipe_sha256=H, cache_miniature_sha256=G,
        qualifier_report_sha256=qualifier_hashes,
    ) == acknowledgement["content_hash"]
    with pytest.raises(ValueError, match="qualifier lineage"):
        validate_diagnostic_acknowledgement(
            acknowledgement, campaign_spec_sha256=H, assignment_manifest_sha256=G,
            recipe_sha256=H, cache_miniature_sha256=G,
            qualifier_report_sha256={**qualifier_hashes, "T0": G},
        )


def test_screen_order_and_confirmation_registry_are_frozen():
    rows = [
        {"node_id": "M2c", "validation": {"macro_ovr_auc": .9, "cross_entropy": .5,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 3.0}},
        {"node_id": "M3c", "validation": {"macro_ovr_auc": .9, "cross_entropy": .5,
         "macro_mean_log_qcd_rejection_at_50pct_signal": 3.0}},
    ]
    assert result_key(rows[0]) < result_key(rows[1])
    assert select_declared_candidate(rows, allowed_nodes=("M2c", "M3c"))["selected_node_id"] == "M2c"
    screen = with_content_hash({
        "contract": "HCWDL_SCREEN_AGGREGATE/v1", "schema_version": 1,
        "selected_intermediate_cold": {"selected_node_id": "M3c"},
        "selected_intermediate_warm": {"selected_node_id": "M4w"},
    })
    registry = build_confirmation_registry(screen, seeds=(11, 22, 33, 44, 55))
    assert len(registry) == 55
    assert len({(row["node_id"], row["seed"]) for row in registry}) == 55
    extended = build_confirmation_registry(
        screen, seeds=(11, 22, 33, 44, 55),
        include_label_only_warm_continuation=True,
    )
    assert len(extended) == 60
    assert sum(row["node_id"] == "NULL_WARM_LABEL_ONLY" for row in extended) == 5
    spec = create_campaign_spec(
        mode="pilot", campaign_root="/campaign", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=None, recipe_path=None, planning_only=True,
        source_manifest_path="/source.json", split_manifest_path="/split.json",
        data_root="/data", include_label_only_warm_continuation=True,
    )
    confirmation = next(row for row in spec["tasks"] if row["task_id"] == "confirmation")
    assert confirmation["array"] == "0-59"


def _screen_training_pair(node_id: str):
    checkpoint = (node_id.encode().hex() + "0" * 64)[:64]
    final_checkpoint = (node_id.encode().hex() + "1" * 64)[:64]
    engine = with_content_hash({
        "contract": PMARD_TRAINING_REPORT_CONTRACT,
        "schema_version": 6,
        "scientific_config": {
            "campaign": "HCWDL", "graph_sha256": GRAPH_SHA256,
            "recipe_sha256": H, "node": {"node_id": node_id},
        },
        "validation": {
            "macro_ovr_auc": .9, "cross_entropy": .5,
            "macro_mean_log_qcd_rejection_at_50pct_signal": 3.0,
        },
        "selected_checkpoint_sha256": checkpoint,
        "final_checkpoint_sha256": final_checkpoint,
    })
    node = with_content_hash({
        "contract": TRAINING_REPORT_CONTRACT, "schema_version": 1,
        "node_id": node_id, "graph_sha256": GRAPH_SHA256,
        "recipe_sha256": H, "parents": {"recipe": H},
        "pmard_engine_report_sha256": engine["content_hash"],
        "selected_checkpoint_sha256": checkpoint,
        "final_checkpoint_sha256": final_checkpoint,
        "selection": {}, "complete": True,
    })
    return engine, node


def test_screen_aggregate_pairs_hcwdl_identity_with_authenticated_engine_metrics():
    pairs = [_screen_training_pair(node_id) for node_id in NODE_REGISTRY]
    aggregate = build_screen_aggregate(
        [pair[0] for pair in pairs], node_reports=[pair[1] for pair in pairs],
        campaign_spec_sha256=G, recipe_sha256=H, assignment_lock_sha256=G,
    )
    assert len(aggregate["rows"]) == len(NODE_REGISTRY)
    assert {row["node_id"] for row in aggregate["rows"]} == set(NODE_REGISTRY)
    assert aggregate["rows"][0]["report_sha256"] in {
        pair[1]["content_hash"] for pair in pairs
    }

    forged = dict(pairs[0][1])
    forged["pmard_engine_report_sha256"] = G
    forged = with_content_hash({key: value for key, value in forged.items() if key != "content_hash"})
    with pytest.raises(ValueError, match="authenticated PMARD engine report"):
        build_screen_aggregate(
            [pair[0] for pair in pairs],
            node_reports=[forged, *[pair[1] for pair in pairs[1:]]],
            campaign_spec_sha256=G, recipe_sha256=H, assignment_lock_sha256=G,
        )


def test_screen_workflow_reads_engine_and_hcwdl_node_reports(tmp_path: Path):
    workflow = object.__new__(HcwdlWorkflow)
    workflow.root = tmp_path
    workflow.spec = {
        "tasks": [{"task_id": "screen_aggregate"}],
        "content_hash": G, "recipe_sha256": H,
    }
    workflow.locks = {"assignment": tmp_path / "locks/assignment.json"}
    write_immutable_json(workflow.locks["assignment"], with_content_hash({
        "contract": "fixture", "schema_version": 1,
    }))
    for node_id in NODE_REGISTRY:
        engine, node = _screen_training_pair(node_id)
        output = tmp_path / "training" / node_id
        write_immutable_json(output / "training_report.json", engine)
        write_immutable_json(output / "hcwdl_training_report.json", node)

    outputs = workflow.run("screen_aggregate")
    aggregate = json.loads(outputs[0].read_text())
    assert aggregate["all_registered_nodes_completed"] is True
    assert len(aggregate["rows"]) == len(NODE_REGISTRY)


def _view(rows: int = 3) -> ParticleInputs:
    return ParticleInputs(
        np.zeros((rows, 21, 2), np.float32), np.zeros((rows, 4, 2), np.float32),
        np.ones((rows, 1, 2), np.bool_), np.full(rows, 2, np.int32),
    )


def test_ram_view_and_target_banks_construct_once_and_identity_join():
    calls = {"hlt": 0}
    def builder():
        calls["hlt"] += 1
        yield {"identity_keys": ["a", "b", "c"], "labels": np.array([0, 1, 2]), "view": _view()}
    bank = EphemeralHcwdlViewBank.build(
        role="train", domain_builders={"hlt": builder},
        assignment_manifest_sha256=H, split_manifest_sha256=G,
    )
    assert calls == {"hlt": 1}
    assert sum(len(batch["labels"]) for batch in bank.batches("hlt", batch_size=2, epoch=0, shuffle=False, seed=1)) == 3
    targets = EphemeralHcwdlTargetBank(split_manifest_sha256=G)
    target = targets.build_once(
        teacher_id="D0c", domain="hlt", teacher_report_sha256=H,
        batches=bank.batches("hlt", batch_size=2, epoch=0, shuffle=False, seed=1),
        forward=lambda value: np.tile(np.arange(15, dtype=np.float32), (len(value.raw_lengths), 1)),
    )
    assert target.join(("c", "a")).shape == (2, 15)
    with pytest.raises(RuntimeError):
        targets.build_once(
            teacher_id="D0c", domain="hlt", teacher_report_sha256=H,
            batches=(), forward=lambda value: np.empty((0, 15), np.float32),
        )


def test_exact_id_monitor_resume_and_cancel():
    ledger = build_submission_ledger(
        campaign_spec_sha256=H, jobs={"a": "101", "b": "102", "c": "103"},
        commands={"a": ["x"], "b": ["y"], "c": ["z"]}, dry_run=False,
    )
    monitor = build_monitor_report(
        ledger, states_by_job_id={"101": "COMPLETED", "102": "FAILED", "103": "PENDING"},
    )
    assert exact_cancel_ids(ledger) == ("101", "102", "103")
    assert resume_tasks(monitor, dependency_graph={"a": (), "b": ("a",), "c": ("b",)}) == ("b", "c")
    recovery = build_submission_ledger(
        campaign_spec_sha256=H, jobs={"b": "202", "c": "203"},
        commands={"b": ["y2"], "c": ["z2"]}, dry_run=False,
        parent_ledger_sha256=ledger["content_hash"],
        monitor_report_sha256=monitor["content_hash"],
        superseded_jobs={"b": "102", "c": "103"},
    )
    assert recovery["superseded_jobs"] == {"b": "102", "c": "103"}
    assert validate_submission_ledger(recovery) == recovery["content_hash"]


def test_submission_journal_and_task_attestations_are_recoverable_and_fail_closed(tmp_path: Path):
    events = [
        build_submission_event(
            campaign_spec_sha256=H, task_id=task, job_id=str(200 + index),
            command=("sbatch", task), sequence=index,
        )
        for index, task in enumerate(("a", "b"))
    ]
    ledger = assemble_submission_ledger(events, campaign_spec_sha256=H)
    assert ledger["jobs"] == {"a": "200", "b": "201"}
    monitor = build_monitor_report(
        ledger, states_by_job_id={"200": "COMPLETED", "201": "COMPLETED"},
        artifact_validity={"a": True, "b": False},
    )
    assert [row["disposition"] for row in monitor["rows"]] == [
        "complete", "retryable_failure",
    ]
    assert resume_tasks(
        monitor, dependency_graph={"a": (), "b": ("a",), "c": ("b",)},
    ) == ("b", "c")

    output = tmp_path / "result.bin"; output.write_bytes(b"result")
    attestation = build_task_attestation(
        campaign_spec_sha256=H, task_id="a", array_index=None, outputs=(output,),
    )
    validate_task_attestation(
        attestation, campaign_spec_sha256=H, task_id="a", array_index=None,
    )
    output.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="output bytes differ"):
        validate_task_attestation(
            attestation, campaign_spec_sha256=H, task_id="a", array_index=None,
        )


def test_storage_estimate_and_measured_profile_are_explicit():
    storage = estimate_storage(
        visible_tokens_by_role={"train": 100, "validation": 20, "final_test": 20},
        selected_checkpoint_bytes=1000, rolling_checkpoint_bytes=2000,
        concurrent_training_jobs=3,
    )
    assert storage["assignment_bytes"] == 560
    assert storage["durable_repaired_dataset_bytes"] == 0
    requests = {
        "cpu_small": {"cpus": 2, "memory": "8G", "walltime": "00:10:00", "gpu": None},
        "cpu_assignment": {"cpus": 8, "memory": "192G", "walltime": "24:00:00", "gpu": None},
        "gpu_root": {"cpus": 8, "memory": "320G", "walltime": "24:00:00", "gpu": "gpu:gh200:1"},
        "gpu_single": {"cpus": 8, "memory": "320G", "walltime": "48:00:00", "gpu": "gpu:gh200:1"},
        "gpu_dual": {"cpus": 8, "memory": "320G", "walltime": "48:00:00", "gpu": "gpu:gh200:1"},
    }
    profile = build_resource_profile(
        requests=requests, miniature_report_sha256=H,
        storage_estimate_sha256=storage["content_hash"], measurement_report_sha256=G,
        safety_factor=1.5,
    )
    assert validate_resource_profile(profile) == profile["content_hash"]
    candidate = create_campaign_spec(
        mode="pilot", campaign_root="/campaign", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=H, recipe_path="/recipe.json", planning_only=False,
        source_manifest_path="/source.json", split_manifest_path="/split.json",
        data_root="/data", resource_measurement_sha256=profile["content_hash"],
        resource_profile=profile,
    )
    assert candidate["resource_profile_status"] == "measured_prelaunch_candidate"
    assert candidate["resources"] == profile["requests"]
    assert build_command_plan(candidate)["content_hash"] == candidate["command_plan_sha256"]
    authorization = build_submission_authorization(
        mode="pilot", source_commit="c" * 40, source_manifest_sha256=H,
        split_manifest_sha256=G, recipe_sha256=H,
        resource_request_sha256=candidate["resource_request_sha256"],
        command_plan_sha256=candidate["command_plan_sha256"],
        authorization_phrase=AUTHORIZATION_PHRASE,
    )
    executable = create_campaign_spec(
        mode="pilot", campaign_root="/campaign", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 4, "validation": 2, "final_test": 2},
        recipe_sha256=H, recipe_path="/recipe.json", planning_only=False,
        source_manifest_path="/source.json", split_manifest_path="/split.json",
        data_root="/data", live_submission_authorized=True,
        resource_measurement_sha256=profile["content_hash"], resource_profile=profile,
        submission_authorization=authorization,
    )
    assert executable["command_plan_sha256"] == candidate["command_plan_sha256"]
    validate_campaign_spec(executable, executable=True)


def test_submission_authorization_cannot_be_reused_for_a_changed_command_plan():
    requests = {
        name: {
            "cpus": 2, "memory": "8G", "walltime": "00:10:00",
            "gpu": None if name.startswith("cpu_") else "gpu:gh200:1",
        }
        for name in ("cpu_small", "cpu_assignment", "gpu_root", "gpu_single", "gpu_dual")
    }
    profile = build_resource_profile(
        requests=requests, miniature_report_sha256=H,
        storage_estimate_sha256=G, measurement_report_sha256=H, safety_factor=1.0,
    )
    candidate = create_campaign_spec(
        mode="pilot", campaign_root="/campaign-a", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 1, "validation": 1, "final_test": 1},
        recipe_sha256=H, recipe_path="/recipe.json", planning_only=False,
        source_manifest_path="/source.json", split_manifest_path="/split.json",
        data_root="/data", resource_measurement_sha256=profile["content_hash"],
        resource_profile=profile,
    )
    authorization = build_submission_authorization(
        mode="pilot", source_commit="c" * 40, source_manifest_sha256=H,
        split_manifest_sha256=G, recipe_sha256=H,
        resource_request_sha256=candidate["resource_request_sha256"],
        command_plan_sha256=candidate["command_plan_sha256"],
        authorization_phrase=AUTHORIZATION_PHRASE,
    )
    with pytest.raises(PermissionError, match="authorization lineage differs"):
        create_campaign_spec(
            mode="pilot", campaign_root="/campaign-b", source_manifest_sha256=H,
            split_manifest_sha256=G, source_commit="c" * 40,
            role_source_counts={"train": 1, "validation": 1, "final_test": 1},
            recipe_sha256=H, recipe_path="/recipe.json", planning_only=False,
            source_manifest_path="/source.json", split_manifest_path="/split.json",
            data_root="/data", live_submission_authorized=True,
            resource_measurement_sha256=profile["content_hash"], resource_profile=profile,
            submission_authorization=authorization,
        )


def test_first_tigris_smoke_is_authorizable_without_fabricated_measurements():
    common = dict(
        mode="smoke", campaign_root="/campaign", source_manifest_sha256=H,
        split_manifest_sha256=G, source_commit="c" * 40,
        role_source_counts={"train": 1, "validation": 1, "final_test": 1},
        recipe_sha256=H, recipe_path="/recipe.json", planning_only=False,
        source_manifest_path="/source.json", split_manifest_path="/split.json",
        data_root="/data",
    )
    candidate = create_campaign_spec(**common)
    assert candidate["resource_profile"] is None
    assert candidate["resource_profile_status"] == "smoke_test_only"
    authorization = build_submission_authorization(
        mode="smoke", source_commit="c" * 40, source_manifest_sha256=H,
        split_manifest_sha256=G, recipe_sha256=H,
        resource_request_sha256=candidate["resource_request_sha256"],
        command_plan_sha256=candidate["command_plan_sha256"],
        authorization_phrase=AUTHORIZATION_PHRASE,
    )
    executable = create_campaign_spec(
        **common, live_submission_authorized=True,
        submission_authorization=authorization,
    )
    assert executable["resource_profile_status"] == "bootstrap_miniature_authorized"
    assert executable["command_plan_sha256"] == candidate["command_plan_sha256"]
    validate_campaign_spec(executable, executable=True)
