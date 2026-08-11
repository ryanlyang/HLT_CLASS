from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256, load_json, with_content_hash, write_immutable_json,
)
from hlt_classification.scouting import hcwdl_final as final_module
from hlt_classification.scouting import loaders
from hlt_classification.scouting.hcwdl_final_recovery import (
    FINAL_RECOVERY_PLAN_CONTRACT, build_final_recovery_plan,
    validate_final_recovery_spec,
)
from hlt_classification.scouting.hcwdl_locks import (
    create_lock, recover_or_claim_final_execution,
)


H = "a" * 64
G = "b" * 64


class _FakeModel:
    def __init__(self):
        self.loaded = None
        self.device = None

    def load_state_dict(self, value, strict=True):
        self.loaded = (value, strict)

    def to(self, device):
        self.device = device
        return self


def test_pmard_loader_moves_constructed_model_to_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "training_report.json"
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = {
        "selected_checkpoint": checkpoint.name,
        "selected_checkpoint_sha256": H,
    }
    model = _FakeModel()
    monkeypatch.setattr(loaders, "load_json", lambda _: report)
    monkeypatch.setattr(loaders, "validate_pmard_training_report", lambda _: H)
    monkeypatch.setattr(loaders, "sha256_file", lambda _: H)
    import torch
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"model": {"weight": 1}},
    )
    loaded, returned = loaders.load_pmard_model(
        report_path, model_factory=lambda: model, device="cuda:0",
    )
    assert loaded is model and returned is report
    assert model.loaded == ({"weight": 1}, True)
    assert model.device == "cuda:0"


def test_exact_existing_final_claim_is_recoverable_but_cannot_be_rebound(
    tmp_path: Path,
) -> None:
    assignment = create_lock(
        "assignment", campaign_spec_sha256=H, payload={"authorized": True},
    )
    recipe = create_lock(
        "recipe", campaign_spec_sha256=H, parent_lock=assignment, payload={},
    )
    qualification = create_lock(
        "shell_endpoint_qualification", campaign_spec_sha256=H,
        parent_lock=recipe, payload={},
    )
    confirmation = create_lock(
        "confirmation_registry", campaign_spec_sha256=H,
        parent_lock=qualification, payload={},
    )
    finalist = create_lock(
        "finalist", campaign_spec_sha256=H,
        parent_lock=confirmation, payload={},
    )
    execution = create_lock(
        "execution", campaign_spec_sha256=H, parent_lock=finalist, payload={},
    )
    path = tmp_path / "execution_claim.json"
    first, first_disposition = recover_or_claim_final_execution(
        path, execution_lock=execution, test_assignment_manifest_sha256=G,
    )
    second, second_disposition = recover_or_claim_final_execution(
        path, execution_lock=execution, test_assignment_manifest_sha256=G,
    )
    assert first == second
    assert first_disposition == "created_new_claim"
    assert second_disposition == "reused_existing_exact_claim"
    with pytest.raises(PermissionError, match="claim lineage differs"):
        recover_or_claim_final_execution(
            path, execution_lock=execution,
            test_assignment_manifest_sha256="c" * 64,
        )


def _metrics(auc: float) -> dict[str, float]:
    return {
        "cross_entropy": .5, "accuracy": .8, "macro_ovr_auc": auc,
        "macro_mean_log_qcd_rejection_at_50pct_signal": 7.0,
        "top_label_ece_15_bin": .01,
    }


def test_final_evaluation_resumes_frozen_registry_after_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = tmp_path / "split.json"
    selection_path = tmp_path / "selection.json"
    finalist_path = tmp_path / "finalist.json"
    execution_path = tmp_path / "execution.json"
    assignment_path = tmp_path / "assignment.json"
    output_root = tmp_path / "evaluation"
    write_immutable_json(split_path, with_content_hash({
        "contract": "fixture", "schema_version": 1, "content": "split",
    }))
    write_immutable_json(selection_path, with_content_hash({
        "contract": "fixture", "schema_version": 1, "content": "selection",
    }))
    training_paths = []
    rows = []
    for index, node in enumerate(("M1c", "D100")):
        path = tmp_path / node / "training_report.json"
        report = with_content_hash({
            "contract": "fixture", "schema_version": 1,
            "selected_checkpoint_sha256": chr(99 + index) * 64,
        })
        write_immutable_json(path, report)
        training_paths.append(path)
        rows.append({
            "node_id": node, "seed": 10 + index, "report_path": str(path),
            "report_sha256": report["content_hash"],
            "checkpoint_sha256": report["selected_checkpoint_sha256"],
        })
    finalist = with_content_hash({
        "contract": "fixture", "schema_version": 1,
        "payload": {"finalists": rows},
    })
    execution = with_content_hash({
        "contract": "fixture", "schema_version": 1,
        "parent_lock_sha256": finalist["content_hash"],
    })
    write_immutable_json(finalist_path, finalist)
    write_immutable_json(execution_path, execution)
    write_immutable_json(assignment_path, with_content_hash({
        "contract": "fixture", "schema_version": 1,
    }))

    class Store:
        manifest = {"content_hash": G}

        def __init__(self, _):
            pass

    claim_calls = {"count": 0}

    def recover(*args, **kwargs):
        claim_calls["count"] += 1
        return {"content_hash": H}, (
            "created_new_claim" if claim_calls["count"] == 1
            else "reused_existing_exact_claim"
        )

    evaluations = {"count": 0}

    def evaluate(*args, **kwargs):
        evaluations["count"] += 1
        if evaluations["count"] == 2:
            raise RuntimeError("injected interruption")
        return _metrics(.9 + evaluations["count"] * .001)

    monkeypatch.setattr(final_module, "DenseAssignmentStore", Store)
    monkeypatch.setattr(final_module, "RowSelection", lambda *a, **k: object())
    monkeypatch.setattr(
        final_module, "validate_lock",
        lambda value, expected_level: (
            finalist["content_hash"] if expected_level == "finalist"
            else execution["content_hash"]
        ),
    )
    monkeypatch.setattr(final_module, "recover_or_claim_final_execution", recover)
    monkeypatch.setattr(final_module, "load_pmard_model", lambda *a, **k: (object(), load_json(a[0])))
    monkeypatch.setattr(final_module, "scouting_model_factory_for_report", lambda _: lambda: object())
    monkeypatch.setattr(final_module, "evaluate_model", evaluate)
    monkeypatch.setattr(final_module, "iterate_model_batches", lambda *a, **k: ())
    monkeypatch.setattr(final_module, "iterate_pmard_batches", lambda *a, **k: ())

    arguments = dict(
        split_manifest_path=split_path, selection_manifest_path=selection_path,
        test_assignment_manifest_path=assignment_path,
        finalist_lock_path=finalist_path, execution_lock_path=execution_path,
        data_root=tmp_path, output_root=output_root,
        checkpoint_namespace_path=tmp_path / "checkpoint_namespace",
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        final_module.run_final_evaluation(**arguments)
    first_report = output_root / "000_M1c_seed10.json"
    assert first_report.is_file()
    first_bytes = first_report.read_bytes()

    manifest = final_module.run_final_evaluation(**arguments)
    assert first_report.read_bytes() == first_bytes
    assert evaluations["count"] == 3
    assert manifest["claim_disposition"] == "reused_existing_exact_claim"
    assert len(manifest["reports"]) == 2
    assert manifest["test_used_for_selection"] is False


def _recovery_spec() -> dict[str, object]:
    resources = {
        "evaluation": {
            "cpus": 8, "memory": "320G", "walltime": "24:00:00",
            "gpu": "gpu:gh200:1",
        },
        "aggregate": {
            "cpus": 8, "memory": "32G", "walltime": "02:00:00",
            "gpu": None,
        },
    }
    references = {
        name: {"path": f"/{name}.json", "content_hash": H}
        for name in (
            "parent_campaign_spec", "parent_submission_ledger", "failure_monitor",
            "finalist_lock", "execution_lock", "test_assignment_manifest",
            "execution_claim",
        )
    }
    payload = {
        "contract": "HCWDL_FINAL_RECOVERY_SPEC/v1", "schema_version": 1,
        "campaign": "HCWDL_INTERRUPTED_FINAL_RECOVERY",
        "recovery_root": "/recovery", "project_dir": "/project",
        "source_commit": "c" * 40, "live_submission_authorized": True,
        **references, "failed_job_id": "60360", "failed_state": "FAILED",
        "execution_claim_sha256": G, "frozen_finalist_count": 10,
        "tasks": [
            {"task_id": "sealed_final_evaluation", "dependencies": [],
             "resource": "evaluation"},
            {"task_id": "aggregate_report",
             "dependencies": ["sealed_final_evaluation"],
             "resource": "aggregate"},
        ],
        "resources": resources,
        "resource_request_sha256": canonical_sha256(resources),
        "final_test_selection_performed": False,
        "existing_exact_claim_reused": True,
        "command_plan_sha256": None,
    }
    provisional = with_content_hash(payload)
    payload["command_plan_sha256"] = build_final_recovery_plan(provisional)[
        "content_hash"
    ]
    return with_content_hash(payload)


def test_final_recovery_spec_has_exactly_two_chained_jobs() -> None:
    spec = _recovery_spec()
    assert validate_final_recovery_spec(spec, executable=True) == spec["content_hash"]
    plan = build_final_recovery_plan(spec)
    assert plan["contract"] == FINAL_RECOVERY_PLAN_CONTRACT
    assert len(plan["commands"]) == 2
    assert plan["commands"][1]["dependencies"] == ["sealed_final_evaluation"]
    assert "--signal=B:USR1@120" in plan["commands"][0]["command"]
    assert all(
        "--array" not in argument
        for row in plan["commands"] for argument in row["command"]
    )
    worker = Path("sbatch/run_hcwdl_final_recovery.sh").read_text()
    assert "exec python -s" in worker
