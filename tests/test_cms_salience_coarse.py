"""Coarse migration: scientific parity, read-only imports and exact Slurm scope."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.cms_salience_learned import (
    campaign, coarse_submission as scheduler, contracts, data,
    preparation_import as preparation, production, shared_import as shared,
)
from hlt_classification.cms_salience_learned.storage import fingerprint, load_receipt, publish_receipt
from hlt_classification.data.cache_contracts import canonical_sha256, load_json, sha256_file, write_immutable_json
from hlt_classification.scouting.hcwdl_exact_dag_submission import _resolved
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_cms_salience_learned import (
    acceptance_resource_evidence, create_from, rehash, tiny_campaign,  # noqa: F401
)
from test_hcwdl_offline_hlt_fusion import fake_weaver  # noqa: F401


def gate_fixture(spec):
    """Synthetic evidence tests validators, never claims real GPU acceptance."""
    proofs = [contracts.artifact("TRAINING_REPORT", node={"role": r}, scientific_fit=False, passes=1)
              for r in ("reference_ce", "direct_kd", "fusion_acquisition", "fusion_withdrawal")]
    evidence = contracts.artifact("EXECUTION_ACCEPTANCE", contract_version=2,
        campaign_spec_sha256=spec["content_hash"], site=spec["site"], source_commit=spec["source_commit"],
        genuine_allocation=True, installed_weaver_forward_backward=True, exact_extraction=True, slurm_job_id="21720795",
        endpoint_parity=True, full_population_cache_rows={r: contracts.BUDGETS[r] for r in ("train", "validation")},
        final_test_accessed=False, peak_rss_bytes=1, peak_cuda_bytes=1, total_cuda_bytes=100,
        miniature_reports=proofs, **acceptance_resource_evidence(spec))
    path = Path(spec["campaign_root"]) / "execution_acceptance.json"
    write_immutable_json(path, evidence)
    publish_receipt(spec, "preflight", [path])


@pytest.fixture
def coarse_pair(tiny_campaign, monkeypatch):
    source = tiny_campaign
    for task in campaign.tasks(source)["prepare"]:
        production.run_task(source, task["task_id"], device="cpu")
    gate_fixture(source)
    rows = campaign.command_plan(source, "science")["commands"]
    jobs = {r["task_id"]: str(21721546 + i) for i, r in enumerate(rows)}
    ledger = build_submission_ledger(campaign_spec_sha256=source["content_hash"], jobs=jobs,
        commands={r["task_id"]: _resolved(r, jobs) for r in rows}, dry_run=False)
    write_immutable_json(Path(source["campaign_root"]) / "science_submission_ledger.json", ledger)
    monkeypatch.setattr(preparation, "preparation_code", lambda *args: {"fixture": "same"})
    monkeypatch.setattr(shared, "reference_code", lambda *args: {"fixture": "same"})
    path = Path(source["campaign_root"]) / "campaign_spec.json"
    coarse = create_from(source, "coarse", ladder="coarse", reuse_preparation_spec=path, reuse_shared_spec=path)
    return source, coarse


def test_graph_versions_counts_exact_fractions_and_legacy_hash():
    dense, coarse = contracts.graph(), contracts.graph("coarse")
    assert dense["content_hash"] == "cf594cd57048ed01d90d6ef1c7a0f7088cb7bed91fbf9d9673c455e41ba4adda"
    assert coarse["schema_version"] == 2 and dense["schema_version"] == 1
    assert coarse["rung_order"] == ["U000", "U050", "U100", "D066", "D033", "D000"]
    assert (coarse["fit_count"], coarse["extraction_count"], coarse["reducer_count"], len(coarse["tasks"])) == (14, 5, 10, 31)
    seen = set()
    for task in coarse["tasks"]:
        assert set(task["dependencies"]) <= seen
        seen.add(task["task_id"])
    for name in ("M0HLT", "OFFLINE", "U000", "DIRECT_D000"):
        assert next(n for n in coarse["nodes"] if n["node_id"] == name) == next(n for n in dense["nodes"] if n["node_id"] == name)
    assert coarse["training"] == dense["training"]
    from hlt_classification.scouting.hcwdl_homotopy import HomotopyCoordinate
    assert contracts.coordinate("U050") == HomotopyCoordinate(1, 2, 0, 1)
    assert contracts.coordinate("D066") == HomotopyCoordinate(1, 1, 1, 3)
    assert contracts.coordinate("D033") == HomotopyCoordinate(1, 1, 2, 3)
    with pytest.raises(ValueError):
        contracts.graph("arbitrary")


@pytest.mark.parametrize("old_hash,new_hash", sorted(preparation._REVIEWED_COORDINATE_AST_PAIRS))
def test_reviewed_preparation_ast_migration_is_closed(old_hash, new_hash):
    old = {p: "a" * 40 for p in preparation.PREPARATION_CODE}
    old["coordinate_ast_sha256"] = old_hash
    new = dict(old, coordinate_ast_sha256=new_hash)
    assert preparation.compatible_preparation_code(old, new)
    assert preparation.compatible_preparation_code(old, old)
    assert not preparation.compatible_preparation_code(new, old)
    assert not preparation.compatible_preparation_code(old, dict(new, coordinate_ast_sha256="f" * 64))
    assert not preparation.compatible_preparation_code(old, dict(new, **{preparation.PREPARATION_CODE[0]: "c" * 40}))
    other_new = next(b for a, b in preparation._REVIEWED_COORDINATE_AST_PAIRS if a != old_hash)
    assert not preparation.compatible_preparation_code(old, dict(new, coordinate_ast_sha256=other_new))


def test_real_git_preparation_migration_on_running_python():
    tree = ast.parse(Path(contracts.__file__).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "coordinate")
    root = Path(__file__).resolve().parents[1]
    donor = "7bb171382b7206013bc5d9308a4c22b2929bc7f4"
    previous = preparation.preparation_code(root, donor)
    consumer = preparation.preparation_code(root, "4f862c11045943f1237d1cef166d32a85c341ed3")
    assert consumer["coordinate_ast_sha256"] == canonical_sha256(ast.dump(fn, include_attributes=False))
    assert (previous["coordinate_ast_sha256"], consumer["coordinate_ast_sha256"]) in preparation._REVIEWED_COORDINATE_AST_PAIRS
    assert preparation.compatible_preparation_code(previous, consumer)
    producer = preparation.preparation_code(root, "f2e8a374f522a39c7f3a6331f0ec77ae12cabaea")
    assert preparation.compatible_preparation_code(producer, consumer)
    fingerprints = shared.reference_code(root, donor)
    assert "production.run_fit:ast" in fingerprints and "contracts.node:ast" in fingerprints


def test_coarse_dry_graph_preserves_shared_jobs_and_requires_fresh_gate(coarse_pair, monkeypatch):
    source, coarse = coarse_pair
    assert coarse["schema_version"] == 4 and coarse["preparation_import"]["schema_version"] == 2
    assert campaign.validate_campaign(coarse)
    assert campaign.gate_check(source)
    rows = campaign.tasks(coarse)["science"]
    assert len(rows) == 31
    assert sum(t["kind"] == "train" for t in rows) == 10
    assert [r["task_id"] for r in rows if r["kind"] == "import_shared"] == list(contracts.SHARED_TASKS)
    plan = campaign.command_plan(coarse, "science")
    for row in plan["commands"]:
        if row["task_id"] in contracts.SHARED_TASKS:
            assert not any("--gres" in c for c in row["command"])
            assert any("${SOURCE_JOB_" in c for c in row["command"])
    assert campaign.submit(coarse, "science")["dry_run"]
    with pytest.raises(PermissionError):
        campaign.submit(coarse, "science", execute=True, authorization_phrase=contracts.AUTHORIZATION)
    with pytest.raises(FileNotFoundError):
        campaign.submit(coarse, "science", execute=True, authorization_phrase=contracts.COARSE_AUTHORIZATION)
    monkeypatch.setattr(scheduler, "queue_rows", lambda *args: {})
    with pytest.raises(FileNotFoundError):
        scheduler.retire_dense(coarse, execute=True, authorization_phrase=scheduler.RETIRE_AUTHORIZATION)


def test_coarse_views_share_authentic_preparation_and_sealed_test(coarse_pair):
    source, coarse = coarse_pair
    before = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    production.run_task(coarse, "foundation", device="cpu")
    for lower, higher in zip(contracts.COARSE_RUNG_ORDER[1:], contracts.COARSE_RUNG_ORDER):
        original = data.build_cache(source, "train", lower, higher)
        actual = production.build_cache(coarse, "train", lower, higher)
        assert actual.foundation_sha256 == original.foundation_sha256
        np.testing.assert_array_equal(actual.identities, original.identities)
        for view in actual.views:
            for a, b in zip(actual.views[view], original.views[view]):
                np.testing.assert_array_equal(a, b)
        if lower.startswith("D"):
            np.testing.assert_array_equal(actual.views[lower][2], actual.views[higher][2])
    with pytest.raises(PermissionError):
        production.build_cache(coarse, "final_test", "D000")
    assert before == {p: sha256_file(p) for p in before}


def test_shared_source_rejects_changed_jobs_recipe_and_implementation(coarse_pair, monkeypatch):
    source, coarse = coarse_pair
    descriptor = coarse["shared_source"]
    altered = rehash(descriptor, jobs=dict(descriptor["jobs"], train_U000="999999"))
    with pytest.raises(ValueError, match="provenance"):
        shared.validate_shared_source(dict(coarse, shared_source=altered))
    with pytest.raises(ValueError, match="resources"):
        shared.validate_shared_source(dict(coarse, resources=dict(coarse["resources"], cpus=3)))
    with pytest.raises(ValueError, match="implementation"):
        monkeypatch.setattr(shared, "reference_code", lambda project, commit: {"fixture": commit})
        shared.validate_shared_source(dict(coarse, source_commit="c" * 40))


def test_shared_import_and_full_coarse_miniature_chain(coarse_pair, fake_weaver, monkeypatch):
    source, coarse = coarse_pair
    torch.set_num_threads(1)
    kernel = production.train
    monkeypatch.setattr(production, "train", lambda *a, **kw: kernel(*a, **kw, acceptance_passes=1))
    monkeypatch.setattr(production, "validate_gpu_allocation", lambda *a: None)
    for task in contracts.SHARED_TASKS:
        production.run_task(source, task, device="cpu")
    before = {p: sha256_file(p) for p in Path(source["campaign_root"]).rglob("*") if p.is_file()}
    production.run_task(coarse, "foundation", device="cpu")
    gate_fixture(coarse)
    # Every import, acquisition, reduction, withdrawal, extraction and aggregate
    # uses the real production IO boundaries. Only the optimizer loop is short.
    for task in campaign.tasks(coarse)["science"]:
        production.run_task(coarse, task["task_id"], device="cpu")
        load_receipt(coarse, task["task_id"])
    for task in contracts.SHARED_TASKS:
        name = task.removeprefix("train_").removeprefix("reduce_")
        relative = f"probabilities/{name}/bank.json" if task.startswith("reduce_") else f"training/{name}/report.json"
        old = load_json(Path(source["campaign_root"]) / relative)
        new = load_json(Path(coarse["campaign_root"]) / relative)
        assert new["campaign_spec_sha256"] == coarse["content_hash"]
        assert new["imported_from"]["artifact_sha256"] == old["content_hash"]
        if task.startswith("train_"):
            assert new["checkpoint"]["sha256"] == old["checkpoint"]["sha256"]
    cache = production.build_cache(coarse, "train", "U050")
    bank, probabilities = production.load_bank(coarse, "U000", cache)
    assert probabilities["probabilities"].shape == (90, 15)
    carrier, report = production.load_model(coarse, "CARRIER_D000", "cpu")
    assert report["exact_extraction"] and report["context_parameters_removed"]
    assert not any("context" in name for name in carrier.state_dict())
    assert before == {p: sha256_file(p) for p in before}
    assert load_json(Path(coarse["campaign_root"]) / "campaign_complete.json")["final_test_accessed"] is False
    # A real receipt does not license a later corrupt checkpoint.
    target = Path(source["campaign_root"]) / "training/U000/selected_model.pt"
    with target.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="payload changed"):
        shared.source_task_outputs(source, "reduce_U000")


@pytest.fixture
def submission_fixture(tmp_path, monkeypatch):
    # Scheduler-only fixture isolates actual external dependency resolution and
    # journal replay without constructing large training artifacts.
    source = dict(campaign_root=str(tmp_path / "source"), site={"account": "reu-aisocial"})
    spec = dict(campaign_root=str(tmp_path / "coarse"), content_hash="a" * 64,
                shared_source={"jobs": {"train_U000": "101", "reduce_U000": "102"}})
    plan = {"commands": [
        dict(task_id="train_U000", dependencies=[], command=["sbatch", "--dependency=afterok:${SOURCE_JOB_train_U000}", "train_U000"]),
        dict(task_id="reduce_U000", dependencies=["train_U000"], command=["sbatch", "--dependency=afterok:${JOB_train_U000}:${SOURCE_JOB_reduce_U000}", "reduce_U000"]),
        dict(task_id="train_ACQUIRE_U050", dependencies=["reduce_U000"], command=["sbatch", "--dependency=afterok:${JOB_reduce_U000}", "train_ACQUIRE_U050"]),
    ]}
    monkeypatch.setattr(scheduler, "validate_shared_source", lambda s: source)
    scheduler.submit_shared_dag(spec, plan, execute=False)
    return spec, source, plan


@pytest.mark.parametrize("finished", [False, True])
def test_submission_uses_pending_source_or_verified_completed_receipt(submission_fixture, monkeypatch, finished):
    spec, source, plan = submission_fixture
    commands = []
    monkeypatch.setattr(scheduler, "_completed", lambda *args: finished)
    monkeypatch.setattr(scheduler, "queue_rows", lambda s, jobs: {j: dict(state="RUNNING", reason="node") for j in jobs.values()})
    def run(command, **kwargs):
        assert command[0] == "sbatch"
        commands.append(command)
        return SimpleNamespace(stdout=str(200 + len(commands)))
    monkeypatch.setattr(scheduler.subprocess, "run", run)
    ledger = scheduler.submit_shared_dag(spec, plan, execute=True)
    assert ledger["jobs"] == {"train_U000": "201", "reduce_U000": "202", "train_ACQUIRE_U050": "203"}
    assert commands[0] == (["sbatch", "train_U000"] if finished else ["sbatch", "--dependency=afterok:101", "train_U000"])
    assert f"--dependency=afterok:201{'' if finished else ':102'}" in commands[1]
    assert "--dependency=afterok:202" in commands[2]
    assert scheduler.submit_shared_dag(spec, plan, execute=True) == ledger
    assert len(commands) == 3


def test_submission_resumes_journal_and_rejects_tampering(submission_fixture, monkeypatch):
    spec, source, plan = submission_fixture
    monkeypatch.setattr(scheduler, "_external_job", lambda *args: args[-1])
    monkeypatch.setattr(scheduler, "_completed", lambda *args: False)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command, stderr="quota")
        return SimpleNamespace(stdout=str(300 + len(calls)))
    monkeypatch.setattr(scheduler.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        scheduler.submit_shared_dag(spec, plan, execute=True)
    ledger = scheduler.submit_shared_dag(spec, plan, execute=True)
    assert ledger["jobs"]["train_U000"] == "301"
    assert len(calls) == 4 and sum(c[-1] == "train_U000" for c in calls) == 1
    altered = dict(plan, commands=[dict(plan["commands"][0], command=["sbatch", "wrong"])] + plan["commands"][1:])
    with pytest.raises(FileExistsError, match="dry ledger"):
        scheduler.submit_shared_dag(spec, altered, execute=True)


def test_only_dependency_failure_with_completion_allows_retry(submission_fixture, monkeypatch):
    spec, source, plan = submission_fixture
    monkeypatch.setattr(scheduler, "_external_job", lambda *args: args[-1])
    monkeypatch.setattr(scheduler, "_completed", lambda *args: True)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command, stderr="Job dependency problem")
        return SimpleNamespace(stdout=str(400 + len(calls)))
    monkeypatch.setattr(scheduler.subprocess, "run", run)
    scheduler.submit_shared_dag(spec, plan, execute=True)
    assert calls[1] == ["sbatch", "train_U000"]
    assert len(calls) == 4


@pytest.mark.parametrize("reason", ["DependencyNeverSatisfied", "JobHeldAdmin"])
def test_source_dependency_never_satisfied_rejected_not_bypassed(submission_fixture, monkeypatch, reason):
    spec, source, plan = submission_fixture
    monkeypatch.setattr(scheduler, "_completed", lambda *args: False)
    monkeypatch.setattr(scheduler, "queue_rows", lambda s, j: {"101": dict(state="PENDING", reason=reason)})
    if reason == "DependencyNeverSatisfied":
        with pytest.raises(ValueError, match="cannot satisfy"):
            scheduler._external_job(source, "train_U000", "101")
    else:
        assert scheduler._external_job(source, "train_U000", "101") == "101"


def test_missing_source_job_without_receipt_fails(submission_fixture, monkeypatch):
    _, source, _ = submission_fixture
    monkeypatch.setattr(scheduler, "_completed", lambda *args: False)
    monkeypatch.setattr(scheduler, "queue_rows", lambda *args: {})
    with pytest.raises(ValueError, match="absent"):
        scheduler._external_job(source, "train_U000", "101")


def test_squeue_completion_race_requires_verified_receipt(submission_fixture, monkeypatch):
    _, source, _ = submission_fixture
    calls = iter((False, True))
    monkeypatch.setattr(scheduler, "_completed", lambda *a: next(calls))
    def missing(*args):
        raise subprocess.CalledProcessError(1, ["squeue"], stderr="Invalid job id specified")
    monkeypatch.setattr(scheduler, "queue_rows", missing)
    assert scheduler._external_job(source, "train_U000", "101") is None
    monkeypatch.setattr(scheduler, "_completed", lambda *a: False)
    with pytest.raises(subprocess.CalledProcessError):
        scheduler._external_job(source, "train_U000", "101")


def test_live_submission_needs_preexisting_dry_evidence(submission_fixture):
    spec, _, plan = submission_fixture
    # New isolated root has no evidence. Do not delete the original fixture.
    other = dict(spec, campaign_root=str(Path(spec["campaign_root"]).parent / "no_dry"))
    with pytest.raises(ValueError, match="dry-run evidence"):
        scheduler.submit_shared_dag(other, plan, execute=True)


def test_replacement_creator_infers_same_science_resources(coarse_pair):
    source, _ = coarse_pair
    replacement = campaign.create_coarse_from_dense(
        source_spec=Path(source["campaign_root"]) / "campaign_spec.json",
        campaign_root=Path(source["campaign_root"]).parent / "coarse_inferred",
        project_dir=source["project_dir"], source_commit=source["source_commit"])
    assert replacement["graph"]["rung_order"] == list(contracts.COARSE_RUNG_ORDER)
    assert replacement["resources"] == source["resources"]
    assert replacement["site"] == source["site"]
    assert replacement["split_manifest"] == source["split_manifest"]


def test_retirement_is_exact_nonshared_only_and_requires_new_gate(coarse_pair, monkeypatch):
    source, coarse = coarse_pair
    calls = []
    monkeypatch.setattr(scheduler.getpass, "getuser", lambda: "ryreu")
    ledger = shared.dense_ledger(source)
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            assert "--user=ryreu" in command
            return SimpleNamespace(stdout="\n".join(f"{j}|PENDING|ryreu|cmslfh_{t}|reu-aisocial|Dependency"
                for t, j in ledger["jobs"].items()) + "\n999|RUNNING|ryreu|unrelated|reu-aisocial|node")
        assert command[0] == "scancel"
        return SimpleNamespace(stdout="")
    # Patch queue-only subprocess adapter; validation's Git helpers are fixture mocks.
    monkeypatch.setattr(scheduler.subprocess, "run", run)
    preview = scheduler.retire_dense(coarse)
    assert len(preview["dense_jobs"]) == 41
    assert set(preview["preserved_jobs"]) == set(contracts.SHARED_TASKS)
    with pytest.raises(PermissionError):
        scheduler.retire_dense(coarse, execute=True)
    with pytest.raises(FileNotFoundError):
        scheduler.retire_dense(coarse, execute=True, authorization_phrase=scheduler.RETIRE_AUTHORIZATION)
    assert all(c[0] == "squeue" for c in calls)
    production.run_task(coarse, "foundation", device="cpu")
    gate_fixture(coarse)
    parents_checked = []
    monkeypatch.setattr(scheduler, "_external_job", lambda s, t, j: parents_checked.append(t))
    scheduler.retire_dense(coarse, execute=True, authorization_phrase=scheduler.RETIRE_AUTHORIZATION)
    cancel = next(c for c in calls if c[0] == "scancel")
    assert set(cancel[1:]) == set(preview["dense_jobs"])
    assert not set(cancel[1:]).intersection(preview["preserved_jobs"].values())
    assert parents_checked == list(contracts.SHARED_TASKS)


@pytest.mark.parametrize("field,value", [(2, "someone_else"), (3, "other_job"), (4, "other_account")])
def test_cancellation_live_identity_fail_closed(monkeypatch, field, value):
    monkeypatch.setattr(scheduler.getpass, "getuser", lambda: "ryreu")
    row = ["123", "PENDING", "ryreu", "cmslfh_aggregate", "reu-aisocial", "Dependency"]
    row[field] = value
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="|".join(row)))
    with pytest.raises(ValueError, match="identity"):
        scheduler.queue_rows({"site": {"account": "reu-aisocial"}}, {"aggregate": "123"})


def test_completed_retirement_with_no_active_jobs_is_a_noop(monkeypatch):
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    assert scheduler.queue_rows({"site": {"account": "reu-aisocial"}}, {"aggregate": "123"}) == {}
