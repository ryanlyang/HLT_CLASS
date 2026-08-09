from __future__ import annotations

from pathlib import Path

import pytest

from hlt_classification.data.cache_contracts import (
    canonical_sha256,
    write_immutable_json,
)
from hlt_classification.scouting import hcwdl_representation_candidate as candidate


def _summary() -> dict:
    identity = {"campaign": "reviewed"}
    return {
        "campaign_identity": identity,
        "campaign_identity_sha256": canonical_sha256(identity),
        "source_commit": "1" * 40,
        "mode": "pilot",
        "command_plan_sha256": "2" * 64,
        "runtime_binding_sha256": "3" * 64,
        "resource_profile_sha256": "4" * 64,
        "storage_estimate_sha256": "5" * 64,
        "fixed_size_inventory_sha256": "6" * 64,
        "tigris_acceptance_sha256": "7" * 64,
        "task_count": 9,
        "task_array_row_count": 17,
    }


def test_candidate_audit_binds_exact_input_bytes_and_never_authorizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {}
    for name in ("planning", "plan", "runtime"):
        path = tmp_path / f"{name}.json"
        write_immutable_json(path, {"name": name})
        paths[name] = path
    monkeypatch.setattr(candidate, "_strict_gate_summary", lambda **_: _summary())

    audit = candidate.build_executable_candidate_audit(
        planning_spec_path=paths["planning"],
        command_plan_path=paths["plan"],
        runtime_binding_path=paths["runtime"],
    )
    candidate.validate_executable_candidate_audit(audit)
    assert audit["all_machine_execution_gates_passed"] is True
    assert audit["human_submission_authorization_present"] is False
    assert audit["scheduler_mutated"] is False
    assert audit["authorizes_submission"] is False

    paths["runtime"].write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        candidate.validate_executable_candidate_audit(audit)


def test_candidate_audit_rejects_authority_flag_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"input-{index}.json"
        write_immutable_json(path, {"index": index})
        paths.append(path)
    monkeypatch.setattr(candidate, "_strict_gate_summary", lambda **_: _summary())
    audit = candidate.build_executable_candidate_audit(
        planning_spec_path=paths[0], command_plan_path=paths[1],
        runtime_binding_path=paths[2],
    )
    forged = {**audit, "authorizes_submission": True}
    from hlt_classification.data.cache_contracts import with_content_hash

    with pytest.raises(PermissionError, match="authority boundary"):
        candidate.validate_executable_candidate_audit(with_content_hash(forged))


def test_strict_candidate_checks_clean_source_and_every_bound_runtime_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hlt_classification.scouting.hcwdl_representation_campaign as campaign
    import hlt_classification.scouting.hcwdl_representation_resources as resources
    import hlt_classification.scouting.hcwdl_representation_runtime_binding as binding
    import hlt_classification.scouting.hcwdl_representation_runtime_rows as runtime_rows

    hashes = {
        "resource_profile_sha256": "4" * 64,
        "storage_estimate_sha256": "5" * 64,
        "fixed_size_inventory_sha256": "6" * 64,
        "tigris_acceptance_sha256": "7" * 64,
    }
    planning = {
        "planning_only": True,
        "live_submission_authorized": False,
        "submission_authorization": None,
        "submission_authorization_sha256": None,
        "runtime_status": "immutable",
        "runtime_binding_sha256": "3" * 64,
        "command_plan_sha256": "2" * 64,
        "source_commit": "1" * 40,
        "project_dir": str(tmp_path),
        "mode": "pilot",
        "representation_recipe_sha256": "8" * 64,
        "resource_profile": {},
        "storage_estimate": {},
        "fixed_size_inventory": {},
        "tigris_acceptance": {},
        "tasks": [{"array": None}],
        **hashes,
    }
    plan = {
        "content_hash": "2" * 64,
        "campaign_identity": {"campaign": "strict"},
        "campaign_identity_sha256": "9" * 64,
    }
    runtime = {"tasks": [{"rows": [{"array_index": None}]}]}
    observed: list[str] = []
    monkeypatch.setattr(campaign, "validate_campaign_spec", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "validate_command_plan", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "build_command_plan", lambda spec: dict(plan))
    monkeypatch.setattr(
        campaign, "validate_source_checkout",
        lambda repository, *, expected_commit: observed.append(
            f"source:{repository}:{expected_commit}"
        ),
    )
    monkeypatch.setattr(
        campaign, "validate_tigris_acceptance", lambda *a, **k: "7" * 64,
    )
    monkeypatch.setattr(
        resources, "validate_measured_profile", lambda *a, **k: "4" * 64,
    )
    monkeypatch.setattr(
        resources, "validate_storage_estimate", lambda *a, **k: "5" * 64,
    )
    monkeypatch.setattr(
        binding, "validate_runtime_binding", lambda *a, **k: "3" * 64,
    )
    monkeypatch.setattr(
        runtime_rows, "validate_bound_runtime_task_rows",
        lambda spec, value: observed.append("all-runtime-rows"),
    )

    summary = candidate._strict_gate_summary(
        planning_spec=planning, command_plan=plan, runtime_binding=runtime,
    )
    assert summary["task_array_row_count"] == 1
    assert observed == [
        "all-runtime-rows",
        f"source:{tmp_path}:{'1' * 40}",
    ]
