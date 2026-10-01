from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_contracts import (
    ASSIGNMENT_LOCK_CONTRACT, CANDIDATES, DIAGNOSTIC_REPORT_CONTRACT,
    FOUNDATION_LOCK_CONTRACT, MATCHER_ACCEPTANCE_CONTRACT, SCHEMA_VERSION as FOUNDATION_SCHEMA_VERSION,
    SALIENCE_PT_QUADRATIC, U000_EQUIVALENCE_LOCK_CONTRACT,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_screen import (
    select_candidate,
)
from hlt_classification.scouting.hcwdl_fullcard_salience_screen_graph import (
    BOTTLENECK_REFERENCE, NODE_REGISTRY as SCREEN_NODES, SCREEN_ORDER,
    recipe_payload as screen_recipe, validate_graph as validate_screen_graph,
)
from hlt_classification.scouting.hcwdl_homotopy import (
    PERSISTENT_HLT_SUPPORT_POLICY,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from hlt_classification.scouting.hcwdl_tri100_spine4_salience_contracts import (
    SCHEMA_VERSION, SOURCE_LOCK_CONTRACT, SPEC_CONTRACT, artifact,
)
from hlt_classification.scouting.hcwdl_tri100_spine4_salience_graph import (
    ANCHOR_NODE_ID, BRANCH_NODES, BRANCH_ORDER, FIT_ORDER, NODE_REGISTRY,
    REDUCER_ORDER, SOURCE_DISTRIBUTION, recipe_payload, validate_graph,
)
from hlt_classification.scouting.splits import SourceFileRecord


def test_screen_is_four_matched_ce_only_u100_fits() -> None:
    assert validate_screen_graph()
    assert SCREEN_ORDER == (BOTTLENECK_REFERENCE, *CANDIDATES)
    assert len(SCREEN_NODES) == 4
    assert len({node.seed_alias for node in SCREEN_NODES.values()}) == 1
    assert all(node.coordinate_name == "U100" for node in SCREEN_NODES.values())
    assert all((node.ce_weight, node.kd_weight) == (1.0, 0.0) for node in SCREEN_NODES.values())
    recipe = screen_recipe()
    assert recipe["training"]["maximum_passes"] == 100
    assert recipe["training"]["minimum_passes"] == 60
    assert recipe["execution"]["world_size"] == 1


def test_production_graph_is_selected_salience_four_spine() -> None:
    assert validate_graph()
    assert len(FIT_ORDER) == 30 and len(REDUCER_ORDER) == 26
    assert FIT_ORDER[0] == ANCHOR_NODE_ID
    assert NODE_REGISTRY[ANCHOR_NODE_ID].training_passes == 100
    assert tuple(len(BRANCH_NODES[name]) for name in BRANCH_ORDER) == (1, 5, 8, 15)
    recipe = recipe_payload()
    assert recipe["anchor_training"]["maximum_passes"] == 100
    assert recipe["training"]["maximum_passes"] == 100
    assert recipe["training"]["effective_batch_size"] == 256
    assert recipe["loss"] == {
        "kind": "constant_ce_kd_v1", "ce_weight": .25,
        "kd_weight": .75, "temperature": 2.0,
    }
    assert recipe["support_policy"] == PERSISTENT_HLT_SUPPORT_POLICY
    assert recipe["execution"]["world_size"] == 1


def test_source_lock_authenticates_screen_result_and_selected_foundation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_tri100_spine4_salience_source as source

    monkeypatch.setattr(
        source, "validate_foundation", lambda value: value["content_hash"],
    )
    monkeypatch.setattr(
        source, "validate_established_source", lambda value: value["content_hash"],
    )
    matcher_hash = "a" * 64
    foundation_root = tmp_path / "foundation"
    (foundation_root / "locks").mkdir(parents=True)
    (foundation_root / "matcher").mkdir()
    foundation_path = foundation_root / "foundation_spec.json"
    foundation = with_content_hash({
        "candidate": SALIENCE_PT_QUADRATIC,
        "campaign_root": str(foundation_root),
        "parents": {"matcher_spec": matcher_hash},
        "artifact_paths": {
            "foundation_lock": str(foundation_root / "locks/foundation.json"),
            "u000_equivalence_lock": str(foundation_root / "locks/u000.json"),
            "source_lock": str(foundation_root / "locks/source.json"),
        },
        "replicate_seed": 1337,
        "role_counts": {"train": 30, "validation": 15, "final_test": 15},
    })
    write_immutable_json(foundation_path, foundation)
    diagnostic_hashes = {}
    for role in ("train", "validation"):
        diagnostic = with_content_hash({
            "contract": DIAGNOSTIC_REPORT_CONTRACT,
            "schema_version": FOUNDATION_SCHEMA_VERSION,
        })
        write_immutable_json(
            foundation_root / f"matcher/{role}_diagnostics.json", diagnostic,
        )
        diagnostic_hashes[role] = diagnostic["content_hash"]
    assignment = with_content_hash({
        "contract": ASSIGNMENT_LOCK_CONTRACT,
        "schema_version": FOUNDATION_SCHEMA_VERSION,
        "foundation_spec_sha256": foundation["content_hash"],
        "matcher_spec_sha256": matcher_hash,
        "role_diagnostics": diagnostic_hashes,
        "complete_smaller_side_coverage": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "final_test_accessed": False,
    })
    write_immutable_json(foundation_root / "locks/assignment.json", assignment)
    acceptance = with_content_hash({
        "contract": MATCHER_ACCEPTANCE_CONTRACT,
        "schema_version": FOUNDATION_SCHEMA_VERSION,
    })
    write_immutable_json(foundation_root / "locks/matcher_acceptance.json", acceptance)
    equivalence = with_content_hash({
        "contract": U000_EQUIVALENCE_LOCK_CONTRACT,
        "schema_version": FOUNDATION_SCHEMA_VERSION,
        "foundation_spec_sha256": foundation["content_hash"],
        "parents": {"new_assignment_lock": assignment["content_hash"]},
        "role_rows": {"train": 30, "validation": 15},
        "identical_p0_tensors_all_rows": True,
        "identical_labels_and_identity_order": True,
        "u000_checkpoint_reused_read_only": True,
        "u000_probability_bank_reused_read_only": True,
        "u000_retrained": False,
        "final_test_accessed": False,
    })
    write_immutable_json(foundation_root / "locks/u000.json", equivalence)
    established = with_content_hash({
        "parents": {"source_campaign": "b" * 64},
        "u000": {"report_sha256": "c" * 64},
        "u000_probability": {"lock_sha256": "d" * 64},
    })
    write_immutable_json(foundation_root / "locks/source.json", established)
    foundation_lock = with_content_hash({
        "contract": FOUNDATION_LOCK_CONTRACT,
        "schema_version": FOUNDATION_SCHEMA_VERSION,
        "foundation_spec_sha256": foundation["content_hash"],
        "parents": {
            "assignment_lock": assignment["content_hash"],
            "u000_equivalence_lock": equivalence["content_hash"],
            "matcher_acceptance": acceptance["content_hash"],
        },
        "role_counts": foundation["role_counts"],
        "u000_reused_read_only": True,
        "assignment_dependent_descendants_rebuilt": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "rolling_resume_persisted": False,
        "optimizer_state_persisted": False,
        "ordinary_final_test_capability": False,
        "final_test_accessed": False,
    })
    write_immutable_json(foundation_root / "locks/foundation.json", foundation_lock)

    def row(candidate: str, index: int) -> dict[str, object]:
        return {
            "candidate": candidate,
            "macro_ovr_auc": (
                .951 if candidate == SALIENCE_PT_QUADRATIC else .950
            ),
            "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
            "accuracy": .8, "pt_weighted_validation_delta_r": .1,
            "foundation_spec_sha256": (
                foundation["content_hash"]
                if candidate == SALIENCE_PT_QUADRATIC else str(index + 1) * 64
            ),
            "matcher_spec_sha256": (
                matcher_hash if candidate == SALIENCE_PT_QUADRATIC
                else str(index + 4) * 64
            ),
            "fit_report_sha256": "789a"[index] * 64,
        }

    rows = {candidate: row(candidate, index) for index, candidate in enumerate(CANDIDATES)}
    bottleneck = row("BOTTLENECK_REFERENCE", 3)
    screen_path = tmp_path / "screen_report.json"
    parents = {
        "campaign_spec": "e" * 64, "screen_split": "f" * 64,
        "fit_BOTTLENECK_REFERENCE": bottleneck["fit_report_sha256"],
        **{f"fit_{name}": value["fit_report_sha256"] for name, value in rows.items()},
    }
    report, selection = select_candidate(
        candidate_rows=rows, bottleneck_row=bottleneck, parents=parents,
        screen_report_path=str(screen_path),
    )
    assert report["selected_candidate"] == SALIENCE_PT_QUADRATIC
    write_immutable_json(screen_path, report)
    selection_path = tmp_path / "selection.json"
    write_immutable_json(selection_path, selection)
    locked = source.build_source_lock(foundation_path, selection_path)
    assert locked["selected_candidate"] == SALIENCE_PT_QUADRATIC
    assert locked["parents"]["screen_report"] == report["content_hash"]

    corrupt = dict(report)
    corrupt.pop("content_hash")
    corrupt["selected_candidate"] = CANDIDATES[0]
    corrupt = with_content_hash(corrupt)
    corrupt_path = tmp_path / "corrupt_screen_report.json"
    write_immutable_json(corrupt_path, corrupt)
    corrupt_selection = dict(selection)
    corrupt_selection.pop("content_hash")
    corrupt_selection["screen_report_path"] = str(corrupt_path)
    corrupt_selection["parents"] = {
        **dict(corrupt["parents"]), "screen_report": corrupt["content_hash"],
    }
    corrupt_selection = with_content_hash(corrupt_selection)
    corrupt_selection_path = tmp_path / "corrupt_selection.json"
    write_immutable_json(corrupt_selection_path, corrupt_selection)
    with pytest.raises(ValueError):
        source.build_source_lock(foundation_path, corrupt_selection_path)


def test_screen_campaign_is_isolated_four_fit_firewalled_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_fullcard_salience_screen_campaign as campaign

    paths = {
        name: str((tmp_path / f"{name}.json").resolve()) for name in SCREEN_ORDER
    }
    hashes = {name: str(index + 1) * 64 for index, name in enumerate(SCREEN_ORDER)}
    reference = {
        "replicate_seed": 1337,
        "role_counts": {"train": 100, "validation": 60, "final_test": 40},
    }
    monkeypatch.setattr(
        campaign, "_foundations", lambda **kwargs: (paths, hashes, reference),
    )
    root = tmp_path / "screen"
    spec = campaign.create_screen_campaign(
        bottleneck_foundation_spec=paths[BOTTLENECK_REFERENCE],
        salience_foundation_specs=[paths[name] for name in CANDIDATES],
        campaign_root=root, project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_screen_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 9 and spec["fit_count"] == 4
    assert spec["selection_multiplicity"] == 3
    assert spec["v_select_visible_during_training"] is False
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    assert tasks["select"]["dependencies"] == [
        f"fit_{name}" for name in SCREEN_ORDER
    ]
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 9
    assert all(row["existing_campaign_dependencies"] == [] for row in [spec])
    fit_commands = [
        row["command"] for row in plan["commands"] if row["task_id"].startswith("fit_")
    ]
    assert len(fit_commands) == 4
    assert all("--gres=gpu:gh200:1" in command for command in fit_commands)
    assert all(any("HCWDL_SALIENCE_SCREEN_SPEC=" in item for item in command)
               for command in fit_commands)


def test_candidate_foundation_rebuilds_assignment_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_fullcard_salience_foundation_campaign as campaign

    source = with_content_hash({
        "parents": {"source_campaign": "1" * 64},
        "foundation_spec_path": str(tmp_path / "old/foundation_spec.json"),
        "replicate_seed": 1337,
        "role_counts": {"train": 30, "validation": 15, "final_test": 15},
    })
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    monkeypatch.setattr(
        campaign, "validate_foundation_campaign",
        lambda value, **kwargs: value["content_hash"],
    )
    monkeypatch.setattr(
        campaign, "validate_bottleneck_foundation",
        lambda value: value["content_hash"],
    )
    old_root = tmp_path / "old"; (old_root / "coupling").mkdir(parents=True)
    split = with_content_hash({
        "contract": "TEST_SPLIT/v1", "schema_version": 1,
        "roles": {
            role: {"files": [asdict(SourceFileRecord(
                f"{role}.root", role, 10, "f" * 64, 10,
                tuple([10] + [0] * 14),
            ))]}
            for role in ("train", "validation", "final_test")
        }, "final_test_accessed": False,
    })
    selection = with_content_hash({"contract": "TEST_SELECTION/v1", "schema_version": 1})
    recipe = with_content_hash({"contract": "TEST_RECIPE/v1", "schema_version": 1})
    for name, value in (("split.json", split), ("selection.json", selection), ("recipe.json", recipe)):
        write_immutable_json(old_root / name, value)
    write_immutable_json(
        old_root / "coupling/config.json",
        with_content_hash({"contract": "TEST_CONFIG/v1", "schema_version": 1}),
    )
    old = with_content_hash({
        "data_root": str(tmp_path / "data"),
        "artifact_paths": {
            "split_manifest": str(old_root / "split.json"),
            "selection_manifest": str(old_root / "selection.json"),
            "recipe": str(old_root / "recipe.json"),
        },
    })
    write_immutable_json(old_root / "foundation_spec.json", old)
    old_train = old_root / "old_train_assignment.json"
    old_validation = old_root / "old_validation_assignment.json"
    write_immutable_json(old_train, with_content_hash({"contract": "OLD_ASSIGN/v1", "schema_version": 1}))
    write_immutable_json(old_validation, with_content_hash({"contract": "OLD_ASSIGN/v1", "schema_version": 1}))
    bottleneck = with_content_hash({
        "artifact_paths": {
            "train_assignment_manifest": str(old_train),
            "validation_assignment_manifest": str(old_validation),
        },
    })
    bottleneck_path = tmp_path / "bottleneck.json"
    write_immutable_json(bottleneck_path, bottleneck)
    root = tmp_path / "candidate"
    spec = campaign.create_foundation(
        source_campaign_spec=tmp_path / "source.json",
        bottleneck_foundation_spec=bottleneck_path,
        candidate=CANDIDATES[0], foundation_root=root,
        project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_foundation(spec, executable=True) == spec["content_hash"]
    assert spec["candidate"] == CANDIDATES[0]
    assert spec["assignment_dependent_descendants_rebuilt"] is True
    assert spec["old_assignment_reused_for_views"] is False
    assert spec["u000_retrained"] is False
    assert len(spec["tasks"]) == 19
    assert len(load_json(root / "command_plan.json")["commands"]) == 19


def _fake_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hlt_classification.scouting import hcwdl_tri100_spine4_salience_campaign as campaign
    from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import (
        TRAINING_REPORT_CONTRACT as CE_CONTRACT,
    )

    selection_hash = "6" * 64
    source = artifact({
        "parents": {
            "source_campaign": "1" * 64, "foundation_lock": "2" * 64,
            "foundation_spec": "3" * 64, "assignment_lock": "4" * 64,
            "matcher_spec": "5" * 64, "selection_lock": selection_hash,
            "screen_report": "7" * 64,
        },
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "selection_lock_path": str(tmp_path / "selection.json"),
        "screen_report_path": str(tmp_path / "screen_report.json"),
        "selected_candidate": SALIENCE_PT_QUADRATIC,
        "selection_multiplicity": 3,
        "replicate_seed": 1337,
        "role_counts": {
            "train": 2_777_855, "validation": 957_541,
            "final_test": 899_779,
        },
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda foundation, selection: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
    established_root = tmp_path / "established"; established_root.mkdir()
    established = with_content_hash({
        "campaign_root": str(established_root), "final_test_accessed": False,
    })
    established_path = tmp_path / "established.json"
    write_immutable_json(established_path, established)
    monkeypatch.setattr(
        campaign, "validate_established_campaign", lambda value: value["content_hash"],
    )
    baseline = with_content_hash({
        "contract": CE_CONTRACT, "schema_version": 1, "node_id": "M0CE60",
        "validation": {"accuracy": .8, "macro_ovr_auc": .94},
        "final_test_accessed": False,
    })
    baseline_path = tmp_path / "m0.json"
    write_immutable_json(baseline_path, baseline)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        foundation_spec=tmp_path / "foundation.json",
        selection_lock=tmp_path / "selection.json",
        established_campaign_spec=established_path,
        m0ce60_report=baseline_path, campaign_root=root,
        project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    return campaign, spec, root


def test_campaign_is_isolated_61_task_selected_winner_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, spec, root = _fake_campaign(tmp_path, monkeypatch)
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert spec["contract"] == SPEC_CONTRACT and spec["schema_version"] == SCHEMA_VERSION
    assert len(spec["tasks"]) == 61
    assert spec["fresh_fit_count"] == 30 and spec["reducer_count"] == 26
    assert spec["selected_candidate"] == SALIENCE_PT_QUADRATIC
    assert spec["selection_multiplicity"] == 3
    assert spec["only_changed_variable"] == "selected_salience_matching_objective"
    assert spec["existing_campaign_dependencies"] == []
    assert spec["existing_campaign_outputs_mutated"] is False
    assert spec["ordinary_final_test_capability"] is False
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    assert tasks[f"train_{ANCHOR_NODE_ID}"]["dependencies"] == ["preflight"]
    anchor_reducer = f"reduce_{SOURCE_DISTRIBUTION}"
    assert tasks[anchor_reducer]["dependencies"] == [f"train_{ANCHOR_NODE_ID}"]
    for branch in BRANCH_ORDER:
        assert tasks[f"train_{BRANCH_NODES[branch][0]}"]["dependencies"] == [anchor_reducer]
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 61
    fit = next(row["command"] for row in plan["commands"] if row["task_id"].startswith("train_"))
    assert "--gres=gpu:gh200:1" in fit
    assert "--cpus-per-task=72" in fit and "--mem=320G" in fit
    assert any("HCWDL_SPINE4S_SPEC=" in item for item in fit)
    assert all("spine4_bottleneck" not in item for item in fit)


def test_recovery_closes_over_selected_winner_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, spec, root = _fake_campaign(tmp_path, monkeypatch)
    from hlt_classification.scouting import hcwdl_tri100_spine4_salience_recovery as recovery
    plan = load_json(root / "command_plan.json")
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    jobs = {name: str(20_000 + index) for index, name in enumerate(commands)}
    ledger = build_submission_ledger(
        campaign_spec_sha256=spec["content_hash"], jobs=jobs,
        commands=commands, dry_run=False,
    )
    ledger_path = tmp_path / "ledger.json"; write_immutable_json(ledger_path, ledger)
    monitor = recovery.build_monitor(
        spec=spec, ledger=ledger,
        states_by_job_id={job: "FAILED" for job in jobs.values()},
    )
    monitor_path = tmp_path / "monitor.json"; write_immutable_json(monitor_path, monitor)
    value = recovery.create_recovery(
        subject_spec=root / "campaign_spec.json", subject_ledger=ledger_path,
        monitor_report=monitor_path, recovery_root=tmp_path / "recovery",
        project_dir=tmp_path, source_commit="b" * 40,
    )
    assert recovery.validate_recovery(value) == value["content_hash"]
    assert len(value["retry_tasks"]) == 61
    recovery_plan = load_json(tmp_path / "recovery/command_plan.json")
    assert len(recovery_plan["commands"]) == 61


def test_salience_worker_shell_contracts() -> None:
    workers = (
        "sbatch/run_hcwdl_fullcard_salience_foundation_task.sh",
        "sbatch/run_hcwdl_fullcard_salience_screen_task.sh",
        "sbatch/run_hcwdl_tri100_spine4_salience_task.sh",
        "sbatch/run_hcwdl_tri100_spine4_salience_recovery_task.sh",
    )
    for worker in workers:
        text = Path(worker).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "PYTHONNOUSERSITE=1" in text
        assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert 'exec python -s "${PROJECT_DIR}/scripts/' in text
