"""Synthetic orchestration tests; these are NEVER real Tigris/Weaver evidence."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_jetclass2_delphes import snapshot  # fixture, synthetic ROOT bytes only
from test_jetclass2_delphes_training import TinyModel, toy_cache
from hlt_classification.data.cache_contracts import load_json, write_immutable_json, with_content_hash
from hlt_classification.jetclass2_delphes import production as prod, submission as sub
from hlt_classification.jetclass2_delphes.contracts import artifact
from hlt_classification.jetclass2_delphes.foundation import (
    build_foundation_spec, build_assignment_shard, audit_sample,
)
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile
from hlt_classification.jetclass2_delphes.execution import execution_site
from hlt_classification.jetclass2_delphes.cache import cache_budgets


@pytest.fixture
def campaign(snapshot, tmp_path, monkeypatch):
    data, inventory, splits = snapshot
    registry = build_registry(data, inventory, splits, training_sizes=(11, 22), validation_size=11, test_size=11)
    split = select_profile(registry, inventory, "TRAIN_22")
    foundation = build_foundation_spec(inventory, split)
    foundation_root = tmp_path / "foundation"
    write_immutable_json(foundation_root / "foundation_spec.json", foundation)
    for task in foundation["assignment_tasks"]:
        build_assignment_shard(foundation, data_root=data, output_root=foundation_root, file_index=task["file_index"])
    write_immutable_json(foundation_root / "sample_audit.json", audit_sample(foundation, data_root=data, rows_per_file=1))
    miniature = artifact("LOCAL_OR_REMOTE_ACCEPTANCE", foundation_sha256=foundation["content_hash"],
                         passed=True, scientific_results=False, final_test_accessed=False, test_only=True)
    resource_report = artifact("KERNEL_TRAINING_REPORT", foundation_sha256=foundation["content_hash"],
                               scientific_fit=False, acceptance_only=True, passes=1,
                               final_test_accessed=False, test_only=True)
    profile = artifact("RUNTIME_PROFILE", version=2, foundation_sha256=foundation["content_hash"],
                       source_commit="d" * 40, model=prod.model_contract(), passed=True,
                       installed_environment=artifact("INSTALLED_ENVIRONMENT", version=2, test_only=True),
                       miniature=miniature, miniature_sha256=miniature["content_hash"], resource_training_report=resource_report,
                       measured_full_population=True, ram_only_views=True, rolling_resume=False,
                       final_test_accessed=False, workers=1, cpus=2, memory_mb=4096,
                       train_minutes=120, reduce_minutes=60, max_train_minutes=2880, selected_state_bytes=1024,
                       execution_site=execution_site("sporc_a100"), slurm_job_id="123",
                       gpu=dict(name="NVIDIA A100-SXM4-40GB", total_memory_bytes=40*2**30, compute_capability=[8, 0]),
                       gpu_peak_bytes=2**30, cache_budgets=cache_budgets(foundation, 4096, 1))
    # Every call to a remote acceptance/source assertion remains explicitly
    # mocked. No synthetic profile ever leaves pytest's disposable directory.
    monkeypatch.setattr(prod, "_source", lambda *args: None)
    spec = prod.create_campaign(foundation, foundation_root=foundation_root, data_root=data,
                                campaign_root=tmp_path / "campaign", project=tmp_path / "pinned-project",
                                source_commit="d" * 40, profile=profile)
    return spec


def fake_slurm(monkeypatch, *, states=None, fail_sbatch=False):
    calls = []
    states = {} if states is None else states
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "sbatch":
            if fail_sbatch:
                raise OSError("simulated lost acknowledgement")
            return SimpleNamespace(stdout=str(10000 + sum(c[0] == "sbatch" for c in calls)))
        if command[0] == "sacct":
            jobs = command[command.index("-j") + 1].split(",")
            return SimpleNamespace(stdout="\n".join(f"{job}|{states.get(job, 'FAILED')}|" for job in jobs))
        raise AssertionError(f"Unexpected remote command: {command}")
    monkeypatch.setattr(sub.subprocess, "run", run)
    monkeypatch.setattr(sub.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**15))
    return calls, states


def test_production_graph_dry_run_and_exact_authorization(campaign, monkeypatch):
    spec = campaign
    root = Path(spec["campaign_root"])
    calls, _ = fake_slurm(monkeypatch)
    plan = sub.command_plan(spec)
    assert len(plan["commands"]) == 59
    assert spec["contract"] == "JETCLASS2_DELPHES_CAMPAIGN_SPEC/v3"
    scientific = spec["scientific_plan"]
    assert scientific["split_profile"] == "TRAIN_22"
    assert scientific["role_counts"] == {"train": 22, "validation": 11, "final_test": 11}
    assert sum(t["kind"] == "train" for t in spec["tasks"]) == 31
    assert sum(t["kind"] == "reduce" for t in spec["tasks"]) == 26
    for row in plan["commands"]:
        command = row["command"]
        assert "--nodes=1" in command and "--ntasks=1" in command
        assert not any("final_test" in arg for arg in command)
        if row["task_id"].startswith(("train_", "reduce_")):
            assert "--gres=gpu:a100:1" in command
            assert "--cpus-per-task=2" in command
        assert "--partition=tier3" in command and "--qos=qos_tier3" in command
        assert command[-1] == "sporc_a100"
        assert "--no-requeue" in command
    with pytest.raises(PermissionError, match="authorization"):
        sub.submit(spec, bookkeeping_root=root, execute=True)
    with pytest.raises(ValueError, match="dry run"):
        sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    assert calls == []
    dry = sub.submit(spec, bookkeeping_root=root, execute=False)
    assert dry["dry_run"] is True and len(dry["jobs"]) == 59 and not calls
    live = sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    assert not live["dry_run"] and len(live["jobs"]) == 59
    assert all("${JOB_" not in arg for cmd in calls for arg in cmd)
    assert sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE) == live
    assert len(calls) == 59  # idempotent, never resubmits an already journalled graph
    bad = with_content_hash({**spec, "final_test_accessed": True})
    with pytest.raises(ValueError, match="graph/model"):
        prod.validate_campaign(bad)
    with pytest.raises(ValueError, match="escape"):
        sub.submit(spec, bookkeeping_root=root.parent, execute=False)


def test_checkpoint_teacher_bank_and_restart_zero_publication(campaign, monkeypatch):
    torch.set_num_threads(1)
    spec = campaign
    root = Path(spec["campaign_root"])
    monkeypatch.setattr(prod, "execution_gate", lambda *args: None)
    monkeypatch.setattr(prod, "DelphesParticleTransformer", TinyModel)
    def cache(foundation, *, role, coordinate_name, **kwargs):
        result = toy_cache(role)
        result.foundation_sha256 = foundation["content_hash"]
        result.coordinate_name = coordinate_name
        return result
    monkeypatch.setattr(prod, "prepare_cache", cache)
    direct = "train_JC2_DIRECT_D000_from_U000"
    with pytest.raises(ValueError, match="Missing authenticated parent"):
        prod.run_task(spec, direct, attempt="early", device="cpu")
    for task in ("train_M0HLT", "train_U000", "reduce_U000", direct):
        result = prod.run_task(spec, task, attempt="first", device="cpu")
        assert prod.completed_task(spec, task) == result
        assert prod.run_task(spec, task, attempt="retry", device="cpu") == result
        assert not (root / "attempts" / task / "retry").exists()
        if task.startswith("train"):
            report = load_json(root / result["result"]["training_report"])
            assert report["scientific_fit"] and 60 <= report["passes"] <= 100
            assert report["selected_weights_restored"] and not report["rolling_resume_written"]
    checkpoints = list(root.rglob("*.pt"))
    assert len(checkpoints) == 3
    assert not list(root.rglob("*.npy")) and not list(root.rglob("*resume*"))
    rows = prod.result_rows(spec)
    assert sum(r["state"] == "COMPLETE" for r in rows) == 3
    assert rows[0]["recovery"]["macro_ovr_auc"] == 0
    calls, _ = fake_slurm(monkeypatch)
    sub.submit(spec, bookkeeping_root=root, execute=False)
    ledger = sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    recovered = sub.prepare_recovery(spec, ledger, output_root=root / "recovery")
    assert len(recovered["remaining_tasks"]) == 55
    assert "train_M0HLT" not in recovered["remaining_tasks"]
    assert "reduce_U000" not in recovered["remaining_tasks"]
    assert "train_JC2_COARSE_U050_from_U000" in recovered["remaining_tasks"]
    # Traverse every branch with the same tiny synthetic model and the actual
    # full-budget kernel. This exercises all nested teacher/aggregate edges,
    # while mocking ONLY data volume, installed factory and external gates.
    for task in spec["tasks"]:
        prod.run_task(spec, task["task_id"], attempt="full_fixture", device="cpu")
    assert len(list(root.rglob("*.pt"))) == 31
    assert all(r["state"] == "COMPLETE" for r in prod.result_rows(spec))
    completed = prod.completed_task(spec, "campaign_complete")
    assert completed["result"]["fresh_fit_count"] == 31
    assert completed["result"]["scientific_result_does_not_control_completion"]
    checkpoint = root / prod.completed_task(spec, "train_M0HLT")["result"]["checkpoint"]
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        prod.completed_task(spec, "train_M0HLT")


def test_recovery_refuses_active_unknown_and_other_newer_attempts(campaign, monkeypatch):
    spec = campaign
    root = Path(spec["campaign_root"])
    calls, states = fake_slurm(monkeypatch)
    sub.submit(spec, bookkeeping_root=root, execute=False)
    old = sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    job = next(iter(old["jobs"].values()))
    for unsafe in ("RUNNING", "PENDING", "UNKNOWN"):
        states[job] = unsafe
        with pytest.raises(PermissionError, match="terminal"):
            sub.prepare_recovery(spec, old, output_root=root / "recovery")
    states[job] = "CANCELLED"
    report = sub.prepare_recovery(spec, old, output_root=root / "recovery")
    assert len(report["remaining_tasks"]) == 59
    assert report["jobs_cancelled"] == [] and not report["resume_optimizer"]
    sub.submit(spec, bookkeeping_root=root / "recovery", execute=False)
    newer = sub.submit(spec, bookkeeping_root=root / "recovery", execute=True, authorization_phrase=prod.AUTHORIZE)
    states[next(iter(newer["jobs"].values()))] = "RUNNING"
    # An old, terminal ledger cannot hide a later running recovery.
    with pytest.raises(PermissionError, match="Another campaign attempt"):
        sub.prepare_recovery(spec, old, output_root=root / "unsafe_second_recovery")
    assert not any(c[0] == "scancel" for c in calls)


def test_lost_submission_acknowledgement_is_not_blindly_retried(campaign, monkeypatch):
    spec = campaign
    root = Path(spec["campaign_root"])
    calls, _ = fake_slurm(monkeypatch, fail_sbatch=True)
    sub.submit(spec, bookkeeping_root=root, execute=False)
    with pytest.raises(OSError, match="acknowledgement"):
        sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    with pytest.raises(PermissionError, match="Ambiguous"):
        sub.submit(spec, bookkeeping_root=root, execute=True, authorization_phrase=prod.AUTHORIZE)
    assert len(calls) == 1


def test_profile_and_live_worker_gates(campaign, monkeypatch):
    profile = campaign["runtime_profile"]
    with pytest.raises(PermissionError, match="GPU"):
        prod.execution_gate(profile, "cpu")
    monkeypatch.setattr(prod, "allocation", lambda site: ("123", 2, 4096))
    monkeypatch.setattr(prod, "gpu_identity", lambda: profile["gpu"])
    monkeypatch.setattr(prod, "installed_environment", lambda: profile["installed_environment"])
    prod.execution_gate(profile, "cuda")
    monkeypatch.setattr(prod, "allocation", lambda site: ("123", 1, 4096))
    with pytest.raises(ValueError, match="allocation"):
        prod.execution_gate(profile, "cuda")
    bad = with_content_hash({**profile, "measured_full_population": False})
    with pytest.raises(ValueError, match="runtime evidence"):
        prod.validate_profile(bad, campaign["foundation"], campaign["source_commit"])
    final_count = len(campaign["foundation"]["assignment_tasks"])
    with pytest.raises(ValueError, match="ordinary-role"):
        prod.run_preparation(campaign["foundation"], foundation_root=Path(campaign["foundation_root"]),
                             data_root=Path(campaign["data_root"]), project=Path(campaign["project_dir"]),
                             source_commit=campaign["source_commit"], task="assign", array_index=final_count)
