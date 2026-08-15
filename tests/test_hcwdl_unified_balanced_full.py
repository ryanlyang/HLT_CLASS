from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_recipe import (
    CLASS_WEIGHT_POLICY, FULL_DATA_RECIPE_CONTRACT,
    PRIMARY_RECIPE_DECISION, build_recipe, validate_recipe,
)
from hlt_classification.scouting.hcwdl_recovery import (
    build_monitor_report, build_submission_ledger, validate_submission_ledger,
)
from hlt_classification.scouting.hcwdl_training import node_training_config
from hlt_classification.scouting.hcwdl_unified_balanced_full_campaign import (
    ARM_RESOURCES, ARMS_CREATION_PHRASE, FOUNDATION_RESOURCES,
    create_arm_specs, create_foundation, validate_arm_campaign,
    validate_foundation_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_contracts import (
    BALANCED_WIRING_RECOVERY_SPEC_CONTRACT,
    MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT,
    assignment_lock_payload, foundation_lock_payload, validate_assignment_lock,
    validate_campaign_submission, validate_foundation_lock, validate_graph,
)
from hlt_classification.scouting import hcwdl_unified_balanced_full_recovery as full_recovery
from hlt_classification.scouting import hcwdl_unified_balanced_builder as balanced_builder
from hlt_classification.scouting.hcwdl_unified_balanced_full_graph import (
    ARM_IDS, ARM_WEIGHTS, FACTORIZED_NODES, META_REGISTRY, arm_registry,
    idealized_u000_ancestry, training_registry_for_arm,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_recovery import (
    build_recovery_spec, recovery_command_plan, validate_recovery_command_plan,
    validate_recovery_spec,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_workflow import (
    _full_recipe,
)
from hlt_classification.scouting.hcwdl_unified_balanced_runner import DOMAINS
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (
    balanced_switch_config_payload,
)
from hlt_classification.scouting.hcwdl_upper_cache import build_coupling_lock
from hlt_classification.scouting.highcov_resources import (
    resource_validation_report,
)
from hlt_classification.scouting.splits import SourceFileRecord


H = "a" * 64


def test_balanced_sidecar_stream_receives_authenticated_assignment_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the external caller of the assignment-locked row iterator."""

    source_path = "sample.root"
    store = SimpleNamespace(
        join=lambda path, entries: (
            np.full((len(entries), 200), -1, np.int16),
            np.zeros((len(entries), 200), np.float32),
        ),
    )
    monkeypatch.setattr(
        balanced_builder, "load_base_shard", lambda path: (
            {"role": "train", "source_path": source_path},
            {
                "entries": np.asarray([7], np.int64),
                "row_offsets": np.asarray([0, 0], np.uint64),
                **{
                    name: np.empty(0, np.int64)
                    for name in (
                        "edit_kind", "source_native_offline_index",
                        "target_hlt_slot", "target_kind",
                        "target_native_offline_index", "cost_q", "mass_q",
                    )
                },
            },
        ),
    )
    monkeypatch.setattr(
        balanced_builder, "role_records",
        lambda manifest, role: [SimpleNamespace(path=source_path)],
    )
    monkeypatch.setattr(balanced_builder, "RowSelection", lambda *args, **kwargs: object())
    monkeypatch.setattr(balanced_builder, "DenseAssignmentStore", lambda path: store)

    def selected(*, assignments, **kwargs):
        assert assignments is store
        yield None, source_path, np.asarray([7], np.int64), {}

    monkeypatch.setattr(balanced_builder, "_selected_source_chunks", selected)
    monkeypatch.setattr(
        balanced_builder, "_prepared_partitions",
        lambda arrays, mapping: ([object()], None, None),
    )
    monkeypatch.setattr(
        balanced_builder, "balanced_switch_placements",
        lambda edits, **kwargs: (),
    )
    published = object()
    monkeypatch.setattr(
        balanced_builder, "publish_balanced_sidecar",
        lambda *args, **kwargs: published,
    )

    assert balanced_builder.build_balanced_sidecar_for_source(
        split_manifest={"content_hash": H}, selection_manifest={},
        assignment_manifest=tmp_path / "assignments.json", data_root=tmp_path,
        role="train", source_index=0,
        base_metadata_path=tmp_path / "base.json",
        switch_config_sha256=H, output_base=tmp_path / "balanced",
        producer_commit="a" * 40,
    ) is published


def _primary_recipe() -> dict:
    return build_recipe({
        "recipe_profile": "primary_ladder", "purpose": "hcwdl_primary_ladder",
        "repair_family": "HIGHCOV_SHELL_EXACT/v1",
        "training_passes": 60, "validation_every_passes": 1,
        **PRIMARY_RECIPE_DECISION,
        "class_weighting": {
            "policy": CLASS_WEIGHT_POLICY, "train_class_counts": [20] * 15,
            "train_row_selection_sha256": H,
        },
        "class_weights": [1.0] * 15,
        "evidence": {"test": H},
    }, authorized=True)


def test_full3_graph_is_exact_factorized_38_fit_registry() -> None:
    assert len(META_REGISTRY) == 38
    assert tuple(ARM_IDS) == ("C25P75", "C10P90", "C10P75G15")
    for arm_id in ARM_IDS:
        registry = arm_registry(arm_id)
        assert tuple(registry) == (*FACTORIZED_NODES, "D100direct")
        assert len(registry) == 12
        first = registry["U020"]
        assert first.parent_id == "shared/U000"
        assert first.grandparent_id is None
        assert first.parent_kd_weight == ARM_WEIGHTS[arm_id][1] + ARM_WEIGHTS[arm_id][2]
        assert registry["D0F"].input_domain == "hlt"
        assert registry["M1F"].input_domain == "hlt"
        assert (registry["M1F"].ce_weight, registry["M1F"].parent_kd_weight) == (.25, .75)
        ancestry = idealized_u000_ancestry(arm_id)
        assert set(ancestry) == set(registry)
        assert all(0 <= value <= 1 for value in ancestry.values())


def test_full_data_recipe_is_a_distinct_authorized_20_pass_contract() -> None:
    selection = with_content_hash({
        "contract": "hlt_classification_pmard_row_selection_v1", "schema_version": 1,
        "split_manifest_sha256": H, "seed": 1337,
        "roles": {
            "train": {
                "all_rows": True, "rows": 300,
                "class_counts": [20] * 15, "sources": [],
            },
            "validation": {
                "all_rows": True, "rows": 150,
                "class_counts": [10] * 15, "sources": [],
            },
        },
    })
    recipe = _full_recipe(
        parent=_primary_recipe(), selection=selection,
        selection_sha256=selection["content_hash"], overlay_sha256=H,
    )
    assert recipe["contract"] == FULL_DATA_RECIPE_CONTRACT
    assert recipe["training_passes"] == 20
    assert recipe["validation_every_passes"] == 1
    assert recipe["class_weights"] == [1.0] * 15
    assert validate_recipe(
        recipe, require_authorized=True, expected_profile="full_data_scaleup",
    ) == recipe["content_hash"]
    tampered = dict(recipe); tampered["training_passes"] = 60
    tampered = with_content_hash({k: v for k, v in tampered.items() if k != "content_hash"})
    with pytest.raises(ValueError, match="budget"):
        validate_recipe(tampered)

    rows = 2_600_001
    config = node_training_config(
        "U020", recipe, train_rows=rows, replicate_seed=1337,
        registry=training_registry_for_arm("C25P75"), domains=DOMAINS,
    )
    expected_updates = 20 * math.ceil(
        rows / int(recipe["batching"]["effective_batch_size"])
    )
    assert config.total_updates == expected_updates
    assert config.validation_interval == expected_updates // 20
    assert config.validation_checks == 20


def test_full_assignment_lock_is_role_complete_and_fail_closed() -> None:
    lock = assignment_lock_payload(
        foundation_spec_sha256=H,
        role_rows={"train": 2_600_000, "validation": 1_000_000},
        parents={"split": H}, manifests={"train": H, "validation": H},
        recomputation_audits={"train": H, "validation": H},
        dustbin_fractions={"train": .09, "validation": .08},
    )
    assert validate_assignment_lock(lock) == lock["content_hash"]
    with pytest.raises(ValueError, match="dustbin"):
        assignment_lock_payload(
            foundation_spec_sha256=H,
            role_rows={"train": 1, "validation": 1}, parents={"split": H},
            manifests={"train": H, "validation": H},
            recomputation_audits={"train": H, "validation": H},
            dustbin_fractions={"train": .10, "validation": .08},
        )


def test_full3_creation_dry_run_and_recovery_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_unified_balanced_full_campaign as campaign

    project = Path(__file__).resolve().parents[1]
    primary_root = tmp_path / "primary"
    (primary_root / "matcher").mkdir(parents=True)
    (primary_root / "locks").mkdir(parents=True)
    split_path = tmp_path / "split.json"
    def record(role: str, rows: int) -> SourceFileRecord:
        base, remainder = divmod(rows, 15)
        counts = tuple(base + (index < remainder) for index in range(15))
        return SourceFileRecord(
            f"{role}.root", role, rows, H, rows, counts,
        )

    split_records = {
        "train": record("train", 2_600_000),
        "validation": record("validation", 1_000_000),
        "final_test": record("final_test", 1_000_000),
    }
    split = with_content_hash({
        "contract": "TEST_SPLIT/v1", "schema_version": 1,
        "roles": {
            role: {"files": [asdict(row)]}
            for role, row in split_records.items()
        },
        "final_test_accessed": False,
    })
    write_immutable_json(split_path, split)
    recipe_path = tmp_path / "recipe.json"
    write_immutable_json(recipe_path, _primary_recipe())
    write_immutable_json(
        primary_root / "matcher/resources_validation.json",
        resource_validation_report(),
    )
    for name in ("shell_endpoint_qualification", "assignment"):
        write_immutable_json(
            primary_root / f"locks/{name}.json",
            with_content_hash({
                "contract": f"TEST_{name}/v1", "schema_version": 1,
            }),
        )
    records = {
        "train": [SimpleNamespace(mapped_entries=2_600_000, path="train.root")],
        "validation": [SimpleNamespace(mapped_entries=1_000_000, path="validation.root")],
        "final_test": [SimpleNamespace(mapped_entries=1_000_000, path="test.root")],
    }
    evidence = {
        "spec": {
            "split_manifest_path": str(split_path),
            "data_root": str(tmp_path / "data"),
        },
        "spec_path": tmp_path / "template_spec.json", "spec_hash": H,
        "root": tmp_path / "template", "preparation_lock_hash": H,
        "primary": {"recipe_path": str(recipe_path)},
        "primary_root": primary_root, "split": split,
        "split_hash": split["content_hash"],
    }
    monkeypatch.setattr(campaign, "authenticate_parent_homotopy", lambda _path: evidence)
    monkeypatch.setattr(campaign, "role_records", lambda _split, role: records[role])
    source_commit = "d" * 40
    foundation_root = tmp_path / "foundation"
    spec = create_foundation(
        parent_homotopy_spec=tmp_path / "template_spec.json",
        campaign_root=foundation_root, project_dir=project,
        source_commit=source_commit, publish=True,
    )
    assert spec["role_counts"] == {
        "train": 2_600_000, "validation": 1_000_000,
        "final_test": 1_000_000,
    }
    assert validate_foundation_campaign(spec) == spec["content_hash"]
    assert load_json(foundation_root / "graph.json")["fit_count"] == 38
    dry_campaign = with_content_hash({
        "contract": "HCWDL_UNIFIED_BALANCED_FULL_CAMPAIGN_SUBMISSION/v1",
        "schema_version": 1, "dry_run": True,
        "foundation_spec_sha256": spec["content_hash"],
        "foundation_submission_ledger_sha256": H,
        "autolaunch_deferred_until_foundation_lock": True,
        "arm_order": list(ARM_IDS), "final_test_accessed": False,
    })
    assert validate_campaign_submission(dry_campaign) == dry_campaign["content_hash"]
    dry_campaign_path = tmp_path / "campaign_dry.json"
    subprocess.run([
        sys.executable, "-s",
        str(project / "scripts/submit_hcwdl_unified_balanced_full_campaign.py"),
        "--foundation-spec", str(foundation_root / "foundation_spec.json"),
        "--foundation-ledger", str(tmp_path / "foundation_dry.json"),
        "--arms-root", str(tmp_path / "deferred_arms"),
        "--arm-ledger-root", str(tmp_path / "deferred_ledgers"),
        "--autolaunch-receipt", str(tmp_path / "autolaunch_receipt.json"),
        "--output", str(dry_campaign_path),
    ], cwd=project, env={**os.environ, "PYTHONPATH": str(project / "src")},
       check=True, capture_output=True, text=True)
    assert validate_campaign_submission(
        load_json(dry_campaign_path)
    ) == load_json(dry_campaign_path)["content_hash"]
    lock = foundation_lock_payload(
        foundation_spec_sha256=spec["content_hash"],
        role_counts=spec["role_counts"], parents={"endpoint": H},
        u000_report_sha256=H, m0paired_report_sha256=H,
        u000_checkpoint_sha256=H, m0paired_checkpoint_sha256=H,
        u000_target_manifest_sha256=H, recipe_sha256=H,
    )
    assert validate_foundation_lock(lock) == lock["content_hash"]
    lock_path = foundation_root / "locks/foundation.json"
    write_immutable_json(lock_path, lock)
    arms_root = tmp_path / "arms"
    arms = create_arm_specs(
        foundation_lock=lock_path, arms_root=arms_root,
        project_dir=project, source_commit=source_commit,
        authorize_live_submission=True,
        authorization_phrase=ARMS_CREATION_PHRASE, publish=True,
    )
    assert tuple(arms) == ARM_IDS
    for arm, arm_spec in arms.items():
        assert validate_arm_campaign(arm_spec) == arm_spec["content_hash"], arm
    (arms_root / ARM_IDS[0] / "training").mkdir()
    reused_arms = create_arm_specs(
        foundation_lock=lock_path, arms_root=arms_root,
        project_dir=project, source_commit=source_commit,
        authorize_live_submission=True,
        authorization_phrase=ARMS_CREATION_PHRASE, publish=True,
    )
    assert reused_arms == arms
    dry_root = tmp_path / "dry_ledgers"
    environment = dict(os.environ); environment["PYTHONPATH"] = str(project / "src")
    subprocess.run([
        sys.executable, "-s",
        str(project / "scripts/submit_hcwdl_unified_balanced_full_arms.py"),
        "--arms-root", str(arms_root), "--output-root", str(dry_root),
    ], cwd=project, env=environment, check=True, capture_output=True, text=True)
    for arm in ARM_IDS:
        ledger = load_json(dry_root / arm / "submission_ledger.json")
        validate_submission_ledger(ledger)
        assert ledger["dry_run"] is True
        assert len(ledger["jobs"]) == 14
    arm = ARM_IDS[0]; arm_spec = arms[arm]
    dry = load_json(dry_root / arm / "submission_ledger.json")
    live = build_submission_ledger(
        campaign_spec_sha256=arm_spec["content_hash"],
        jobs={task: str(90000 + index) for index, task in enumerate(dry["jobs"])},
        commands=dry["commands"], dry_run=False,
    )
    live_path = tmp_path / "live.json"; write_immutable_json(live_path, live)
    states = {job: "COMPLETED" for job in live["jobs"].values()}
    states[live["jobs"]["train_U060"]] = "TIMEOUT"
    monitor = build_monitor_report(live, states_by_job_id=states)
    monitor_path = tmp_path / "monitor.json"; write_immutable_json(monitor_path, monitor)
    recovery = build_recovery_spec(
        scope_spec_path=arms_root / arm / "arm_spec.json",
        submission_ledger_path=live_path, monitor_report_path=monitor_path,
        recovery_root=tmp_path / "recovery", project_dir=project,
        source_commit=source_commit,
        resource_overrides={"gpu_training": {"walltime": "30:00:00"}},
    )
    assert validate_recovery_spec(recovery) == recovery["content_hash"]
    assert recovery["task_ids"][0] == "train_U060"
    assert "train_M1F" in recovery["task_ids"]
    assert "train_D100direct" not in recovery["task_ids"]
    plan = recovery_command_plan(recovery)
    assert validate_recovery_command_plan(plan, recovery_spec=recovery) == plan["content_hash"]

    selection = with_content_hash({
        "contract": "hlt_classification_pmard_row_selection_v1",
        "schema_version": 1,
        "split_manifest_sha256": spec["parents"]["split_manifest_sha256"],
        "seed": 1337,
        "roles": {
            role: {
                "all_rows": True,
                "rows": int(spec["role_counts"][role]),
                "class_counts": list(split_records[role].class_counts),
                "sources": [{
                    "path": split_records[role].path,
                    "rows": split_records[role].mapped_entries,
                }],
            }
            for role in ("train", "validation")
        },
        "selection_rule": "per_class_smallest_identity_sha256_rank_v1",
        "access_lock_sha256": {},
    })
    selection_path = Path(spec["artifact_paths"]["selection_manifest"])
    write_immutable_json(selection_path, selection)
    mapped_lock = assignment_lock_payload(
        foundation_spec_sha256=spec["content_hash"],
        role_rows={
            role: int(spec["role_counts"][role])
            for role in ("train", "validation")
        },
        parents={"row_selection_sha256": selection["content_hash"]},
        manifests={"train": H, "validation": H},
        recomputation_audits={"train": H, "validation": H},
        dustbin_fractions={"train": .08, "validation": .08},
    )
    write_immutable_json(foundation_root / "locks/assignment.json", mapped_lock)

    foundation_dry = load_json(tmp_path / "foundation_dry.json")
    foundation_live = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"],
        jobs={
            task: str(91000 + index)
            for index, task in enumerate(foundation_dry["jobs"])
        },
        commands=foundation_dry["commands"], dry_run=False,
    )
    foundation_live_path = tmp_path / "foundation_live.json"
    write_immutable_json(foundation_live_path, foundation_live)
    foundation_states = {
        job: "COMPLETED" for job in foundation_live["jobs"].values()
    }
    foundation_states[foundation_live["jobs"]["scale_calibration"]] = "FAILED"
    foundation_monitor = build_monitor_report(
        foundation_live, states_by_job_id=foundation_states,
    )
    foundation_monitor_path = tmp_path / "foundation_monitor.json"
    write_immutable_json(foundation_monitor_path, foundation_monitor)

    repaired_semantic = dict(spec["semantic_source_sha256"])
    for name in full_recovery.MAPPED_IDENTITY_REPAIR_SEMANTIC_FILES:
        repaired_semantic[name] = "b" * 64
    monkeypatch.setattr(
        full_recovery, "semantic_source_hashes", lambda _project: repaired_semantic,
    )
    execution_recovery = build_recovery_spec(
        scope_spec_path=foundation_root / "foundation_spec.json",
        submission_ledger_path=foundation_live_path,
        monitor_report_path=foundation_monitor_path,
        recovery_root=tmp_path / "mapped_identity_recovery",
        project_dir=project, source_commit="e" * 40,
        execution_repair=full_recovery.MAPPED_IDENTITY_REPAIR,
        authorization_phrase=full_recovery.MAPPED_IDENTITY_REPAIR_PHRASE,
    )
    assert execution_recovery["contract"] == MAPPED_IDENTITY_RECOVERY_SPEC_CONTRACT
    assert execution_recovery["task_ids"][0] == "scale_calibration"
    assert execution_recovery["mapped_identity_repair_evidence"][
        "assignment_lock_sha256"
    ] == mapped_lock["content_hash"]
    assert validate_recovery_spec(execution_recovery) == execution_recovery["content_hash"]
    execution_plan = recovery_command_plan(execution_recovery)
    assert validate_recovery_command_plan(
        execution_plan, recovery_spec=execution_recovery,
    ) == execution_plan["content_hash"]

    # A later integration failure may be repaired from this recovery ledger
    # without republishing the already authenticated coupling prefix.
    parent_recovery_path = (
        Path(execution_recovery["recovery_root"]) / "recovery_spec.json"
    )
    write_immutable_json(parent_recovery_path, execution_recovery)
    execution_commands = {
        row["task_id"]: row["command"]
        for row in execution_plan["commands"]
    }
    execution_live = build_submission_ledger(
        campaign_spec_sha256=execution_recovery["content_hash"],
        jobs={
            task: str(92000 + index)
            for index, task in enumerate(execution_commands)
        },
        commands=execution_commands, dry_run=False,
    )
    execution_live_path = tmp_path / "mapped_identity_live.json"
    write_immutable_json(execution_live_path, execution_live)
    execution_states = {
        job: "COMPLETED" for job in execution_live["jobs"].values()
    }
    for task in ("train_balanced", "validation_balanced"):
        execution_states[execution_live["jobs"][task]] = "FAILED"
    execution_monitor = build_monitor_report(
        execution_live, states_by_job_id=execution_states,
    )
    execution_monitor_path = tmp_path / "mapped_identity_monitor.json"
    write_immutable_json(execution_monitor_path, execution_monitor)

    coupling = build_coupling_lock(
        campaign_spec_sha256=spec["content_hash"],
        coupling_config_sha256=H, scale_calibration_sha256=H,
        switch_calibration_sha256=H, train_manifest_sha256=H,
        validation_manifest_sha256=H, audit_sha256=H,
    )
    write_immutable_json(foundation_root / "locks/coupling.json", coupling)
    balanced = balanced_switch_config_payload(
        base_coupling_lock_sha256=coupling["content_hash"],
    )
    write_immutable_json(foundation_root / "balanced/config.json", balanced)

    balanced_semantic = dict(repaired_semantic)
    for name in full_recovery.BALANCED_WIRING_REPAIR_SEMANTIC_FILES:
        balanced_semantic[name] = "d" * 64
    monkeypatch.setattr(
        full_recovery, "semantic_source_hashes", lambda _project: balanced_semantic,
    )
    balanced_recovery = build_recovery_spec(
        scope_spec_path=foundation_root / "foundation_spec.json",
        parent_recovery_spec_path=parent_recovery_path,
        submission_ledger_path=execution_live_path,
        monitor_report_path=execution_monitor_path,
        recovery_root=tmp_path / "balanced_wiring_recovery",
        project_dir=project, source_commit="f" * 40,
        execution_repair=full_recovery.BALANCED_WIRING_REPAIR,
        authorization_phrase=full_recovery.BALANCED_WIRING_REPAIR_PHRASE,
    )
    assert balanced_recovery["contract"] == BALANCED_WIRING_RECOVERY_SPEC_CONTRACT
    assert balanced_recovery["task_ids"][:2] == [
        "train_balanced", "validation_balanced",
    ]
    assert "scale_calibration" not in balanced_recovery["task_ids"]
    assert balanced_recovery["balanced_wiring_repair_evidence"][
        "coupling_lock_sha256"
    ] == coupling["content_hash"]
    assert balanced_recovery["balanced_wiring_repair_evidence"][
        "balanced_switch_config_sha256"
    ] == balanced["content_hash"]
    assert validate_recovery_spec(balanced_recovery) == balanced_recovery["content_hash"]
    balanced_plan = recovery_command_plan(balanced_recovery)
    assert validate_recovery_command_plan(
        balanced_plan, recovery_spec=balanced_recovery,
    ) == balanced_plan["content_hash"]

    unexpected_balanced = dict(balanced_semantic)
    unexpected_balanced[
        "src/hlt_classification/scouting/engine.py"
    ] = "e" * 64
    monkeypatch.setattr(
        full_recovery, "semantic_source_hashes", lambda _project: unexpected_balanced,
    )
    with pytest.raises(ValueError, match="unexpected source"):
        build_recovery_spec(
            scope_spec_path=foundation_root / "foundation_spec.json",
            parent_recovery_spec_path=parent_recovery_path,
            submission_ledger_path=execution_live_path,
            monitor_report_path=execution_monitor_path,
            recovery_root=tmp_path / "invalid_balanced_wiring_recovery",
            project_dir=project, source_commit="f" * 40,
            execution_repair=full_recovery.BALANCED_WIRING_REPAIR,
            authorization_phrase=full_recovery.BALANCED_WIRING_REPAIR_PHRASE,
        )

    unexpected_semantic = dict(repaired_semantic)
    unexpected_semantic[
        "src/hlt_classification/scouting/engine.py"
    ] = "c" * 64
    monkeypatch.setattr(
        full_recovery, "semantic_source_hashes", lambda _project: unexpected_semantic,
    )
    with pytest.raises(ValueError, match="unexpected source"):
        build_recovery_spec(
            scope_spec_path=foundation_root / "foundation_spec.json",
            submission_ledger_path=foundation_live_path,
            monitor_report_path=foundation_monitor_path,
            recovery_root=tmp_path / "invalid_mapped_identity_recovery",
            project_dir=project, source_commit="e" * 40,
            execution_repair=full_recovery.MAPPED_IDENTITY_REPAIR,
            authorization_phrase=full_recovery.MAPPED_IDENTITY_REPAIR_PHRASE,
        )
