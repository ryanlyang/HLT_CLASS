from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import (
    load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_graph import (
    BRANCH_NODES, BRANCH_ORDER, BRANCH_PATHS, EXECUTION, FIT_ORDER,
    NODE_REGISTRY, PROBABILITY_COMPONENTS, REDUCER_ORDER, recipe_payload,
    validate_graph,
)
from hlt_classification.scouting.splits import SourceFileRecord


H = "a" * 64


def test_bottleneck_graph_is_exact_controlled_clone():
    assert validate_graph()
    assert tuple(len(BRANCH_NODES[name]) for name in BRANCH_ORDER) == (1, 5, 8, 15)
    assert BRANCH_PATHS["COARSE"] == ("U050", "U100", "D066", "D033", "D000")
    assert len(FIT_ORDER) == 29 and len(REDUCER_ORDER) == 25
    assert all(len(value) == 1 for value in PROBABILITY_COMPONENTS.values())
    assert dict(EXECUTION)["world_size"] == 1
    recipe = recipe_payload()
    assert recipe["loss"] == {
        "kind": "constant_ce_kd_v1", "ce_weight": .25,
        "kd_weight": .75, "temperature": 2.0,
    }
    assert recipe["training"]["effective_batch_size"] == 256
    assert recipe["training"]["maximum_passes"] == 100
    assert recipe["training"]["minimum_passes"] == 60
    assert recipe["only_changed_variable"] == "particle_pairing_foundation_lineage"
    for branch in BRANCH_ORDER:
        previous = None
        for node_id in BRANCH_NODES[branch]:
            node = NODE_REGISTRY[node_id]
            assert node.parent_node_id == previous
            assert node.initialization == "fresh" and node.auxiliary == "none"
            previous = node_id


def test_bottleneck_probability_contract_cannot_authenticate_established_bank(tmp_path: Path):
    from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_probability import (
        BottleneckProbabilityTargets, publish_probability_lock,
        publish_probability_role, validate_probability_lock,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_probability import (
        publish_probability_role as publish_old_role,
    )

    distribution = REDUCER_ORDER[0]
    component = PROBABILITY_COMPONENTS[distribution][0]
    identities = np.zeros((4, 32), np.uint8); identities[:, 0] = np.arange(4)
    logits = np.arange(60, dtype=np.float32).reshape(4, 15) / 30
    lineage = {component: {
        "report_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
        "logits_sha256": "3" * 64,
    }}
    parents = {
        "campaign_spec": "4" * 64, "foundation": "5" * 64,
        "assignment_lock": "6" * 64,
    }
    train = publish_probability_role(
        tmp_path / "new", distribution_id=distribution, role="train",
        identity_digests=identities, component_logits={component: logits},
        component_lineage=lineage, parents=parents, producer_commit="a" * 40,
    )
    validation = publish_probability_role(
        tmp_path / "new", distribution_id=distribution, role="validation",
        identity_digests=identities, component_logits={component: logits},
        component_lineage=lineage, parents=parents, producer_commit="a" * 40,
    )
    lock = publish_probability_lock(
        tmp_path / "new/lock.json", distribution_id=distribution,
        train_manifest=train, validation_manifest=validation, parents=parents,
    )
    assert validate_probability_lock(
        tmp_path / "new/lock.json", distribution_id=distribution,
    )[0] == lock
    target = BottleneckProbabilityTargets.load(
        tmp_path / "new/train_manifest.json", distribution_id=distribution,
    )
    assert np.array_equal(target.join(identities[[2, 0]]), target.probabilities[[2, 0]])
    publish_old_role(
        tmp_path / "old", distribution_id=distribution, role="train",
        identity_digests=identities, component_logits={component: logits},
        component_lineage=lineage, parents=parents, producer_commit="a" * 40,
    )
    with pytest.raises(ValueError):
        BottleneckProbabilityTargets.load(
            tmp_path / "old/train_manifest.json", distribution_id=distribution,
        )


def _fake_science_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from hlt_classification.scouting import hcwdl_tri100_spine4_bottleneck_campaign as campaign
    from hlt_classification.scouting.hcwdl_mhpe_tri60_ce_control_contracts import (
        TRAINING_REPORT_CONTRACT as CE_CONTRACT,
    )
    from hlt_classification.scouting.hcwdl_tri100_spine4_bottleneck_contracts import (
        SOURCE_LOCK_CONTRACT, artifact,
    )
    source = artifact({
        "parents": {
            "source_campaign": "1" * 64, "foundation_lock": "2" * 64,
            "foundation_spec": "3" * 64, "assignment_lock": "4" * 64,
            "matcher_spec": "5" * 64,
        },
        "foundation_spec_path": str(tmp_path / "foundation.json"),
        "replicate_seed": 1337,
        "role_counts": {"train": 2_777_855, "validation": 957_541, "final_test": 899_779},
    }, contract=SOURCE_LOCK_CONTRACT)
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
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
    baseline_path = tmp_path / "m0.json"; write_immutable_json(baseline_path, baseline)
    root = tmp_path / "campaign"
    spec = campaign.create_campaign(
        foundation_spec=tmp_path / "foundation.json",
        established_campaign_spec=established_path, m0ce60_report=baseline_path,
        campaign_root=root, project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    return campaign, spec, root


def test_campaign_is_exact_isolated_58_task_dag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    campaign, spec, root = _fake_science_creation(tmp_path, monkeypatch)
    assert campaign.validate_campaign(spec, executable=True) == spec["content_hash"]
    assert len(spec["tasks"]) == 58
    assert spec["fresh_fit_count"] == 29 and spec["reducer_count"] == 25
    assert spec["established_campaign_completion_required"] is False
    assert spec["existing_campaign_dependencies"] == []
    assert spec["existing_campaign_outputs_mutated"] is False
    assert spec["ensembles"] is False and spec["weight_continuation"] is False
    assert spec["rolling_resume"] is False and spec["partial_checkpoint_reuse"] is False
    assert spec["ordinary_final_test_capability"] is False
    tasks = {row["task_id"]: row for row in spec["tasks"]}
    for branch in BRANCH_ORDER:
        assert tasks[f"train_{BRANCH_NODES[branch][0]}"]["dependencies"] == ["preflight"]
    plan = load_json(root / "command_plan.json")
    assert len(plan["commands"]) == 58
    assert all(not row["external_dependencies"] for row in plan["commands"])
    fit = next(row["command"] for row in plan["commands"] if row["task_id"].startswith("train_"))
    assert "--cpus-per-task=72" in fit and "--mem=320G" in fit
    assert "--time=3-00:00:00" in fit and "--gres=gpu:gh200:1" in fit
    assert all("HCWDL_SPINE4B_" in item or "HCWDL" not in item for item in fit)


def test_campaign_recovery_closes_over_exact_failed_dag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    campaign, spec, root = _fake_science_creation(tmp_path, monkeypatch)
    from hlt_classification.scouting import hcwdl_tri100_spine4_bottleneck_recovery as recovery
    plan = load_json(root / "command_plan.json")
    commands = {row["task_id"]: row["command"] for row in plan["commands"]}
    jobs = {name: str(10000 + index) for index, name in enumerate(commands)}
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
    assert len(value["retry_tasks"]) == 58
    recovery_plan = load_json(tmp_path / "recovery/command_plan.json")
    assert len(recovery_plan["commands"]) == 58


def test_foundation_dag_rebuilds_every_assignment_dependent_descendant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from hlt_classification.scouting import hcwdl_fullcard_bottleneck_foundation_campaign as campaign
    source = with_content_hash({
        "parents": {"source_campaign": "1" * 64},
        "foundation_spec_path": str(tmp_path / "old/foundation_spec.json"),
        "replicate_seed": 1337,
        "role_counts": {"train": 30, "validation": 15, "final_test": 15},
    })
    monkeypatch.setattr(campaign, "build_source_lock", lambda path: source)
    monkeypatch.setattr(campaign, "validate_source_lock", lambda value: value["content_hash"])
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
    train_assignment = with_content_hash({"contract": "OLD_ASSIGN/v1", "schema_version": 1})
    validation_assignment = with_content_hash({"contract": "OLD_ASSIGN/v1", "schema_version": 1})
    write_immutable_json(old_root / "train_assignment.json", train_assignment)
    write_immutable_json(old_root / "validation_assignment.json", validation_assignment)
    write_immutable_json(old_root / "coupling/config.json", with_content_hash({"contract":"CFG/v1","schema_version":1}))
    old = with_content_hash({
        "data_root": str(tmp_path / "data"),
        "artifact_paths": {
            "split_manifest": str(old_root / "split.json"),
            "selection_manifest": str(old_root / "selection.json"),
            "recipe": str(old_root / "recipe.json"),
            "train_assignment_manifest": str(old_root / "train_assignment.json"),
            "validation_assignment_manifest": str(old_root / "validation_assignment.json"),
        },
    })
    write_immutable_json(old_root / "foundation_spec.json", old)
    monkeypatch.setattr(
        campaign, "validate_foundation_campaign", lambda value, **kwargs: value["content_hash"],
    )
    root = tmp_path / "new"
    spec = campaign.create_foundation(
        source_campaign_spec=tmp_path / "source.json", foundation_root=root,
        project_dir=tmp_path, source_commit="a" * 40,
        authorize_live_submission=True,
        authorization_phrase=campaign.CREATION_PHRASE,
    )
    assert campaign.validate_foundation(spec, executable=True) == spec["content_hash"]
    kinds = [row["kind"] for row in spec["tasks"]]
    for expected in (
        "assignment", "assignment_manifest", "assignment_lock",
        "scale_calibration", "coupling_base", "coupling_lock",
        "balanced_sidecar", "balanced_manifest", "u000_equivalence",
        "foundation_lock",
    ):
        assert expected in kinds
    assert spec["old_assignment_reused_for_views"] is False
    assert spec["u000_retrained"] is False
    assert spec["pairing_provenance"] == "validity_only_not_correspondence_confidence"
    assert len(load_json(root / "command_plan.json")["commands"]) == 19


def test_completed_foundation_source_authentication_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hlt_classification.scouting import hcwdl_tri100_spine4_bottleneck_source as source
    from hlt_classification.scouting.hcwdl_fullcard_bottleneck_contracts import (
        ASSIGNMENT_LOCK_CONTRACT, DIAGNOSTIC_REPORT_CONTRACT,
        FOUNDATION_LOCK_CONTRACT, MATCHER_ACCEPTANCE_CONTRACT,
        U000_EQUIVALENCE_LOCK_CONTRACT,
    )

    root = tmp_path / "foundation"
    (root / "locks").mkdir(parents=True)
    (root / "matcher").mkdir()
    diagnostics = {}
    for role in ("train", "validation"):
        value = with_content_hash({
            "contract": DIAGNOSTIC_REPORT_CONTRACT, "schema_version": 1,
            "role": role,
        })
        write_immutable_json(root / f"matcher/{role}_diagnostics.json", value)
        diagnostics[role] = value["content_hash"]
    assignment = with_content_hash({
        "contract": ASSIGNMENT_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": "f" * 64,
        "matcher_spec_sha256": "m" * 64,
        "role_diagnostics": diagnostics,
        "complete_smaller_side_coverage": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "final_test_accessed": False,
    })
    write_immutable_json(root / "locks/assignment.json", assignment)
    acceptance = with_content_hash({
        "contract": MATCHER_ACCEPTANCE_CONTRACT, "schema_version": 1,
    })
    write_immutable_json(root / "locks/matcher_acceptance.json", acceptance)
    equivalence = with_content_hash({
        "contract": U000_EQUIVALENCE_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": "f" * 64,
        "parents": {"new_assignment_lock": assignment["content_hash"]},
        "role_rows": {"train": 30, "validation": 15},
        "identical_p0_tensors_all_rows": True,
        "identical_labels_and_identity_order": True,
        "u000_checkpoint_reused_read_only": True,
        "u000_probability_bank_reused_read_only": True,
        "u000_retrained": False, "final_test_accessed": False,
    })
    write_immutable_json(root / "locks/u000_equivalence.json", equivalence)
    foundation_lock = with_content_hash({
        "contract": FOUNDATION_LOCK_CONTRACT, "schema_version": 1,
        "foundation_spec_sha256": "f" * 64,
        "parents": {
            "assignment_lock": assignment["content_hash"],
            "u000_equivalence_lock": equivalence["content_hash"],
            "matcher_acceptance": acceptance["content_hash"],
        },
        "role_counts": {"train": 30, "validation": 15, "final_test": 15},
        "u000_reused_read_only": True,
        "assignment_dependent_descendants_rebuilt": True,
        "pairing_provenance": "validity_only_not_correspondence_confidence",
        "rolling_resume_persisted": False, "optimizer_state_persisted": False,
        "ordinary_final_test_capability": False, "final_test_accessed": False,
    })
    write_immutable_json(root / "locks/foundation.json", foundation_lock)
    established = with_content_hash({
        "parents": {"source_campaign": "s" * 64},
        "u000": {"report_sha256": "r" * 64},
        "u000_probability": {"lock_sha256": "p" * 64},
    })
    write_immutable_json(root / "locks/established.json", established)
    foundation = {
        "content_hash": "f" * 64, "campaign_root": str(root),
        "parents": {"matcher_spec": "m" * 64},
        "artifact_paths": {
            "foundation_lock": str(root / "locks/foundation.json"),
            "u000_equivalence_lock": str(root / "locks/u000_equivalence.json"),
            "source_lock": str(root / "locks/established.json"),
        },
        "replicate_seed": 7,
        "role_counts": {"train": 30, "validation": 15, "final_test": 15},
    }
    foundation_path = root / "foundation_spec.json"
    write_immutable_json(foundation_path, foundation)
    monkeypatch.setattr(source, "validate_foundation", lambda value: "f" * 64)
    monkeypatch.setattr(
        source, "validate_established_source",
        lambda value: value["content_hash"],
    )
    value = source.build_source_lock(foundation_path)
    assert value["parents"]["assignment_lock"] == assignment["content_hash"]

    bad = dict(assignment)
    bad["complete_smaller_side_coverage"] = False
    bad.pop("content_hash")
    bad = with_content_hash(bad)
    actual_load_json = source.load_json
    monkeypatch.setattr(
        source, "load_json",
        lambda path: (
            bad if Path(path) == root / "locks/assignment.json"
            else actual_load_json(path)
        ),
    )
    with pytest.raises(ValueError, match="completion lineage"):
        source.build_source_lock(foundation_path)


def test_bottleneck_worker_shell_contracts() -> None:
    workers = (
        "sbatch/run_hcwdl_fullcard_bottleneck_foundation_task.sh",
        "sbatch/run_hcwdl_tri100_spine4_bottleneck_task.sh",
        "sbatch/run_hcwdl_tri100_spine4_bottleneck_recovery_task.sh",
    )
    for worker in workers:
        text = Path(worker).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert 'source "${PROJECT_DIR}/sbatch/common.sh"' in text
        assert "PYTHONNOUSERSITE=1" in text
        assert 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in text
        assert 'exec python -s "${PROJECT_DIR}/scripts/' in text
