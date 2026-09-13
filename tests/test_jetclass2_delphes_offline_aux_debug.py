"""Synthetic orchestration evidence, not genuine Weaver/A100 acceptance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.offline_aux import campaign, submission, execution, workflow
from hlt_classification.jetclass2_delphes.offline_aux import preparation_import as reuse
from hlt_classification.jetclass2_delphes.offline_aux.contracts import (
    artifact, load_json, write_immutable_json, sha256_file, recipe,
)
from hlt_classification.jetclass2_delphes.offline_aux.model import model_contract


def rehash(value):
    return artifact(value["contract"].split("OFFLINE_AUX_")[1].split("/")[0],
                    **{k: v for k, v in value.items() if k not in
                       {"contract", "schema_version", "content_hash", "final_test_accessed"}})


def receipt(stage, task, result):
    root = Path(stage["root"])
    output = root / "attempts/initial" / task
    write_immutable_json(output / "result.json", result)
    descriptor = dict(path=(output / "result.json").relative_to(root).as_posix(),
                      bytes=(output / "result.json").stat().st_size,
                      sha256=sha256_file(output / "result.json"))
    record = artifact("TASK_RECEIPT", task_id=task, stage_sha256=stage["content_hash"],
                      output_directory=output.relative_to(root).as_posix(), outputs=[descriptor], result=result)
    write_immutable_json(root / "receipts" / (task + ".json"), record)


@pytest.fixture
def original(tmp_path, monkeypatch):
    # Mock only source checkout and raw snapshot admission. All stage/receipt,
    # payload checksum, continuation, plan and acceptance validators remain real.
    monkeypatch.setattr(campaign, "_source", lambda *a, **k: None)
    monkeypatch.setattr(campaign, "authenticate_profile", lambda *a, **k: None)
    monkeypatch.setattr(reuse, "_code_hashes", lambda project, commit: (("fixture.py", "a"*64),))
    raw = tmp_path / "raw"
    raw.mkdir()
    source = campaign.create_study(root=tmp_path / "original", data_root=raw,
        inventory={"fixture": "inventory"}, profile={"fixture": "profile"},
        project_dir=tmp_path / "old_code", source_commit="1"*40)
    gate = campaign.create_stage(source, "GATE")
    sample = artifact("SAMPLE_PROFILE", study_sha256=source["content_hash"],
                      role_split_sha256="b"*64, target_minutes=30)
    receipt(gate, "sample", sample)
    prepare = campaign.create_stage(source, "PREPARE")
    for task in prepare["tasks"]:
        if task["kind"] == "target":
            receipt(prepare, task["task_id"], artifact("TARGET_SHARD", role=task["role"], shard=task["shard"]))
    normalizer = artifact("NORMALIZATION", fixture="original calibration")
    receipt(prepare, "normalize", artifact("PREPARATION_LOCK", study_sha256=source["content_hash"],
        role_split_sha256="b"*64, train_bank={"fixture": "train"}, selection_bank={"fixture": "select"},
        normalizer=normalizer))
    return source


def continuation(original, tmp_path):
    return reuse.create_debug_continuation(source_study=Path(original["root"]) / "study_spec.json",
        root=tmp_path / "debug_study", project_dir=tmp_path / "new_code", source_commit="2"*40)


def toy_acceptance(study):
    return artifact("EXECUTION_ACCEPTANCE_DEBUG", study_sha256=study["content_hash"],
        source_commit=study["source_commit"], site=study["site"], model=model_contract(),
        recipe_sha256=recipe()["content_hash"], full_train_rows=500000, full_select_rows=200000,
        passed=True, selected_restore_proved=True, hlt_only_deployment_proved=True,
        no_persisted_probe_weights=True, rolling_resume=False,
        fp32_parity=[dict(training=x, passed=True) for x in (False, True)],
        bf16_probes=[dict(arm=x, passed=True) for x in ("CE", "COMP", "STRUCT", "BOTH")],
        gpu_peak_bytes=1, gpu=dict(total_memory_bytes=40*2**30), host_peak_bytes=1,
        projected_bytes=2**30, resources=dict(cpus=8, workers=8, memory_mb=73728,
            train_minutes=800, evaluation_minutes=60, bootstrap_minutes=60, target_minutes=30),
        measurement_site=execution_site("sporc_a100_debug"), production_site=study["site"],
        transfer_policy=reuse.TRANSFER)


def test_debug_only_profile_reuses_cpu_preparation_without_writing_source(original, tmp_path):
    old_root = Path(original["root"])
    before = {p.relative_to(old_root): p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    study = continuation(original, tmp_path)
    assert len(study["preparation_import"]["completed"]["receipts"]) == 9
    assert study["site"] == execution_site("sporc_a100")
    assert campaign.prior(study, "GATE", "sample") == campaign.prior(original, "GATE", "sample")
    assert campaign.prior(study, "PREPARE", "normalize") == campaign.prior(original, "PREPARE", "normalize")
    with pytest.raises(ValueError, match="already imported"):
        campaign.create_stage(study, "GATE")
    stage = campaign.create_stage(study, "PREPARE")
    assert stage["tasks"] == [dict(task_id="profile", kind="profile", dependencies=[])]
    attempt = Path(stage["root"]) / "attempts/initial"
    dry = submission.submit(stage, study, attempt)
    assert set(dry["jobs"]) == {"profile"}
    command = dry["commands"]["profile"]
    for option in ("--partition=debug", "--gres=gpu:a100:1", "--cpus-per-task=8",
                   "--mem=73728M", "--time=240", "--no-requeue"):
        assert option in command
    assert not any(arg.startswith("--dependency") for arg in command)
    assert not list(Path(study["root"]).rglob("*.npz"))
    assert not list(Path(study["root"]).rglob("receipts/*.json"))
    assert before == {p.relative_to(old_root): p.read_bytes() for p in old_root.rglob("*") if p.is_file()}
    with pytest.raises(FileNotFoundError):
        campaign.create_stage(study, "DISCOVERY")  # Real acceptance still required.


def test_profile_then_scientific_plans_stay_tier3(original, tmp_path):
    study = continuation(original, tmp_path)
    stage = campaign.create_stage(study, "PREPARE")
    accepted = toy_acceptance(study)
    execution.validate_acceptance(accepted, study)
    receipt(stage, "profile", accepted)
    discovery = campaign.create_stage(study, "DISCOVERY")
    plan = submission.command_plan(discovery, study, Path(discovery["root"]) / "attempts/initial")
    assert len(plan["commands"]) == 11
    assert sum(t["kind"] == "train" for t in discovery["tasks"]) == 10
    assert all("--partition=tier3" in row["command"] for row in plan["commands"])
    # Site selection cannot send any other task kind to debug.
    for kind in ("sample", "target", "normalize", "train", "evaluate", "bootstrap", "complete"):
        assert campaign.task_site(study, {"kind": kind}) == execution_site("sporc_a100")


@pytest.mark.parametrize("field", ["measurement_site", "production_site", "transfer_policy", "source_commit"])
def test_debug_acceptance_binding_cannot_be_relabelled(original, tmp_path, field):
    study = continuation(original, tmp_path)
    value = toy_acceptance(study)
    value[field] = "wrong"
    with pytest.raises(ValueError):
        execution.validate_acceptance(rehash(value), study)


def test_debug_acceptance_requires_same_resources_and_new_contract(original, tmp_path):
    study = continuation(original, tmp_path)
    value = toy_acceptance(study)
    value["resources"]["cpus"] = 16
    with pytest.raises(ValueError, match="resources differ"):
        execution.validate_acceptance(rehash(value), study)
    with pytest.raises(ValueError):
        execution.validate_acceptance(toy_acceptance(study), original)
    legacy = toy_acceptance(study)
    legacy["contract"] = legacy["contract"].replace("ACCEPTANCE_DEBUG", "ACCEPTANCE")
    with pytest.raises(ValueError):
        execution.validate_acceptance(rehash(legacy), study)


@pytest.mark.parametrize("problem", ["missing_receipt", "corrupt_payload", "missing_payload"])
def test_import_requires_authentic_completed_outputs(original, tmp_path, problem):
    root = Path(original["root"]) / "stages/PREPARE"
    if problem == "missing_receipt":
        (root / "receipts/targets_VAL_SELECT_01.json").unlink()
    elif problem == "corrupt_payload":
        (root / "attempts/initial/targets_VAL_SELECT_01/result.json").write_bytes(b"corrupt")
    else:
        (root / "attempts/initial/targets_VAL_SELECT_01/result.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        continuation(original, tmp_path)
    assert not (tmp_path / "debug_study").exists()


def test_changed_preparation_code_and_science_are_rejected(original, tmp_path, monkeypatch):
    study = continuation(original, tmp_path)
    changed = copy.deepcopy(study)
    changed["seed_register"]["DISCOVERY"]["shared_init"] += 1
    with pytest.raises(ValueError):
        campaign.validate_study(rehash(changed))
    monkeypatch.setattr(reuse, "_code_hashes", lambda project, commit: (("fixture.py", commit),))
    with pytest.raises(ValueError, match="Preparation source changed"):
        campaign.validate_study(study)


def test_import_receipt_drift_and_cross_task_receipts_fail(original, tmp_path):
    study = continuation(original, tmp_path)
    path = Path(original["root"]) / "stages/PREPARE/receipts/targets_TRAIN_00.json"
    value = load_json(path)
    value["task_id"] = "targets_TRAIN_01"
    path.write_text(json.dumps(rehash(value)))
    with pytest.raises(ValueError, match="another stage/task"):
        campaign.validate_study(study)


def test_import_is_not_an_arbitrary_profile_or_report_capability(original, tmp_path):
    study = continuation(original, tmp_path)
    with pytest.raises(PermissionError, match="Only CPU"):
        reuse.imported_stage(study, "CONFIRMATION")
    with pytest.raises(FileNotFoundError):
        campaign.prior(study, "PREPARE", "profile")
    with pytest.raises(ValueError, match="original prepared study"):
        reuse.create_debug_continuation(source_study=Path(study["root"]) / "study_spec.json",
            root=tmp_path / "chained", project_dir=tmp_path / "another_code", source_commit="3"*40)
    with pytest.raises(ValueError, match="disjoint"):
        reuse.create_debug_continuation(source_study=Path(original["root"]) / "study_spec.json",
            root=Path(original["root"]) / "nested", project_dir=tmp_path / "new_code", source_commit="2"*40)


def test_allocation_checks_actual_profile_and_production_sites(original, tmp_path, monkeypatch):
    study = continuation(original, tmp_path)
    stage = campaign.create_stage(study, "PREPARE")
    observed = []
    def allocation(site):
        observed.append(site["partition"])
        return "123", 8, 73728
    monkeypatch.setattr(submission, "allocation", allocation)
    submission.verify_allocation(study, stage, stage["tasks"][0])
    assert observed == ["debug"]
    task = {"kind": "train"}
    submission.verify_allocation(study, dict(resources=dict(cpus=8, memory_mb=73728, train_minutes=800)), task)
    assert observed == ["debug", "tier3"]


def test_profile_dispatch_uses_original_target_bank_and_normalizer(original, tmp_path, monkeypatch):
    study = continuation(original, tmp_path)
    stage = campaign.create_stage(study, "PREPARE")
    prepared, _ = campaign.prior(original, "PREPARE", "normalize")
    monkeypatch.setattr(workflow, "context", lambda s: ({"content_hash": "b"*64}, tmp_path))
    monkeypatch.setattr(workflow, "load_role", lambda *a: {})
    visited = []
    def collection(source_stage, split, metadata, role):
        visited.append(source_stage["study_sha256"])
        return prepared["train_bank" if role == "TRAIN" else "selection_bank"], "values", "mask", {}
    monkeypatch.setattr(workflow, "target_collection", collection)
    monkeypatch.setattr(workflow, "make_caches", lambda *a: ({"TRAIN": "train", "VAL_SELECT": "val"}, 123))
    def measure(current, split, tc, vc, values, mask, normalizer, **kwargs):
        assert current == study and normalizer == prepared["normalizer"]
        assert (tc, vc, values, mask) == ("train", "val", "values", "mask")
        return {"synthetic_dispatch_only": True}
    monkeypatch.setattr(execution, "measure", measure)
    result = workflow.dispatch(study, stage, stage["tasks"][0], tmp_path / "out")
    assert result == {"synthetic_dispatch_only": True}
    assert visited == [original["content_hash"]]*2


def test_imported_storage_remains_in_total_budget(original, tmp_path, monkeypatch):
    study = continuation(original, tmp_path)
    own = reuse.storage_audit(study["root"])
    imported = study["preparation_import"]["completed"]["payload_bytes"]
    assert reuse.audit_study_storage(study)["bytes"] == own["bytes"] + imported
    monkeypatch.setattr(reuse, "MAX_TOTAL", own["bytes"] + imported - 1)
    with pytest.raises(ValueError, match="including imported"):
        reuse.audit_study_storage(study)


def test_live_debug_submission_is_exact_and_revalidates_imports(original, tmp_path, monkeypatch):
    study = continuation(original, tmp_path)
    stage = campaign.create_stage(study, "PREPARE")
    attempt = Path(stage["root"]) / "attempts/initial"
    submission.submit(stage, study, attempt)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch" and "--partition=debug" in command
        return SimpleNamespace(stdout="123456\n")
    monkeypatch.setattr(submission.subprocess, "run", run)
    monkeypatch.setattr(submission.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*2**30))
    phrase = "AUTHORIZE JC2 OFFLINE AUX PREPARE EXACT STAGE"
    with pytest.raises(PermissionError):
        submission.submit(stage, study, attempt, execute=True, phrase="wrong")
    assert not calls
    result = submission.submit(stage, study, attempt, execute=True, phrase=phrase)
    assert result["jobs"] == {"profile": "123456"} and len(calls) == 1
    assert submission.submit(stage, study, attempt, execute=True, phrase=phrase) == result
    assert len(calls) == 1
    (Path(original["root"]) / "stages/PREPARE/attempts/initial/targets_TRAIN_00/result.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum/size"):
        submission.submit(stage, study, attempt, execute=True, phrase=phrase)
    assert len(calls) == 1


def test_debug_stage_cannot_change_resources_or_add_scientific_jobs(original, tmp_path):
    study = continuation(original, tmp_path)
    stage = campaign.create_stage(study, "PREPARE")
    bad = copy.deepcopy(stage)
    bad["resources"]["memory_mb"] = 8192
    with pytest.raises(ValueError, match="resources differ"):
        campaign.validate_stage(rehash(bad), study)
    bad = copy.deepcopy(stage)
    bad["tasks"].append(dict(task_id="train", kind="train", dependencies=[]))
    with pytest.raises(ValueError, match="task graph differs"):
        campaign.validate_stage(rehash(bad), study)


def test_continue_debug_cli_uses_new_study_without_auto_submitting(original, tmp_path, monkeypatch, capsys):
    import importlib.util
    import sys
    path = Path(__file__).resolve().parents[1] / "scripts/jetclass2_delphes_offline_aux.py"
    module = importlib.util.spec_from_file_location("aux_debug_cli", path)
    cli = importlib.util.module_from_spec(module)
    module.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", [str(path), "continue-debug", "--source-study",
        str(Path(original["root"]) / "study_spec.json"), "--root", str(tmp_path / "cli_study"),
        "--project-dir", str(tmp_path / "new_code"), "--source-commit", "2"*40])
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["site"]["partition"] == "tier3"
    assert result["profile_measurement_site"]["partition"] == "debug"
    assert list((tmp_path / "cli_study").iterdir()) == [tmp_path / "cli_study/study_spec.json"]
