"""Partition-only K2 moves preserve science, resources, IDs and attestations."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from hlt_classification.data.cache_contracts import load_json, write_immutable_json
from hlt_classification.jetclass2_delphes import (
    concat_k2_campaign as campaign, concat_k2_execution as execution,
    concat_k2_runtime as runtime, concat_k2_submit as scheduler,
)
from hlt_classification.scouting.hcwdl_recovery import build_submission_ledger
from test_jetclass2_concat_k2 import spec_at
from test_jetclass2_concat_k2_source import imported_source
from test_jetclass2_dzfix_fusion_chain import rehash


def test_partition_is_not_a_scientific_choice_and_old_versions_fail_closed(tmp_path):
    tier3, debug = campaign.registration(), campaign.registration("debug")
    assert tier3["execution_site"]["partition"] == "tier3"
    assert {k: v for k, v in tier3.items() if k != "execution_site"} == {
        k: v for k, v in debug.items() if k != "execution_site"}
    assert tier3["execution_policy"]["mutable_scheduler_fields"] == ["Partition"]
    assert max(r["minutes"] for r in tier3["resources"].values()) <= 1440
    for kind in ("LAUNCH_SPEC", "CAMPAIGN_SPEC", "ACCEPTANCE"):
        value = campaign.artifact(kind, test_only=True)
        assert value["schema_version"] == 3
        for version in (1, 2):
            old = rehash(value, contract=value["contract"].replace("/v3", f"/v{version}"), schema_version=version)
            with pytest.raises(ValueError, match="contract mismatch"):
                campaign.validate(old, kind)
    # Matching and view formats/seed domains did not change.
    assert campaign.artifact("FOUNDATION_SPEC")["schema_version"] == 1


@pytest.mark.parametrize("partition", ["gpu", "tier2", "tigris", "debug,tier3", "", None])
def test_only_two_explicit_single_partitions_allowed(partition):
    with pytest.raises(ValueError, match="partition"):
        execution.site_for_partition(partition)


def test_allowlist_cannot_be_expanded_even_with_rehashed_payload():
    spec = campaign.registration()
    spec["execution_policy"]["allowed_sites"].append({"partition": "tier2"})
    spec["execution_policy"] = rehash(spec["execution_policy"])
    with pytest.raises(ValueError, match="policy"):
        execution.admit_site(spec, "tier3")


@pytest.mark.parametrize("partition", ["tier3", "debug"])
def test_explicit_launch_partition_survives_materialization(imported_source, monkeypatch, partition):
    from hlt_classification.jetclass2_delphes import concat_k2_source as source
    launch, _, root = imported_source
    monkeypatch.setattr(campaign, "_source", lambda *a: None)
    changed = source.create_launch(screen_spec=root / "screen_spec.json",
        inventory_path=root / "inventory.json", launch_root=root.parent / (partition + "_launch"),
        campaign_root=root.parent / (partition + "_campaign"),
        project=Path(launch["project_dir"]), source_commit=launch["source_commit"], partition=partition)
    source.validate_launch(changed)
    spec = campaign.create(launch=changed)
    campaign.validate_campaign(spec)
    assert spec["execution_site"]["partition"] == partition
    assert all("--partition=" + partition in r["command"]
               for r in scheduler.plan(spec, "full")["commands"])


def worker(tmp_path, monkeypatch, requested, actual, name):
    spec = rehash(spec_at(tmp_path), **campaign.registration(requested))
    launch = name in {"after_matching", "after_gate"}
    if launch:
        spec = campaign.artifact("LAUNCH_SPEC", registration=campaign.registration(requested),
            launch_root=str(tmp_path / "launch"), project_dir=str(tmp_path / "project"))
        resource = spec["registration"]["resources"]["metadata"]
        stage = name
    else:
        stage = "gate" if name in campaign.gates(spec) else "science"
        resource = spec["resources"][next(r["resource"] for r in spec["tasks"] if r["task_id"] == name)]
    root = Path(spec["launch_root"] if launch else spec["campaign_root"])
    ledger = build_submission_ledger(campaign_spec_sha256=spec["content_hash"], dry_run=False,
        jobs={name: "456"}, commands={name: scheduler.command(spec, name, resource, launch=launch)})
    path = root / ("submissions_" + stage) / "submission_ledger.json"
    write_immutable_json(path, ledger)
    for key, value in dict(SLURM_JOB_ID="456", SLURM_CLUSTER_NAME="sporc",
        SLURM_JOB_PARTITION=actual, SLURM_MEM_PER_NODE=str(resource["memory_mb"]),
        PYTHONNOUSERSITE="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(scheduler.sys, "prefix", "/home/ryreu/miniconda3/envs/atlas_kd_sporc")
    fields = dict(Account="reu-aisocial", Partition=actual, QOS="qos_tier3",
        NumNodes="1", NumTasks="1", NumCPUs=str(resource["cpus"]), NodeList="skl-a-48",
        TimeLimit=f"{resource['minutes']//60:02d}:{resource['minutes']%60:02d}:00")
    def show(command, **kw):
        assert command == ["scontrol", "show", "job", "-o", "456"]
        return SimpleNamespace(stdout=" ".join(f"{k}={v}" for k, v in fields.items()))
    monkeypatch.setattr(scheduler.subprocess, "run", show)
    return spec, fields, path, root, launch


@pytest.mark.parametrize("requested,actual", [("tier3", "debug"), ("debug", "tier3"),
                                             ("tier3", "tier3"), ("debug", "debug")])
@pytest.mark.parametrize("name", ["after_matching", "after_gate", "authenticate", "assign_0000",
                                  "preflight", "train_CONCAT_K2_D100", "reduce_CONCAT_K2_D100"])
def test_workers_allow_partition_only_move_and_record_actual_site(tmp_path, monkeypatch, requested, actual, name):
    spec, fields, path, root, launch = worker(tmp_path, monkeypatch, requested, actual, name)
    original = path.read_bytes()
    assert scheduler.authenticate_job(spec, name, launch=launch) == "456"
    assert scheduler.authenticate_job(spec, name, launch=launch) == "456"
    assert path.read_bytes() == original  # Original command remains the submission intent.
    record = load_json(root / "execution" / name / "456.json")
    assert record["actual_site"]["partition"] == actual
    assert record["requested_site"]["partition"] == requested
    assert record["subject_sha256"] == spec["content_hash"] and record["job_id"] == "456"
    if requested != actual:
        assert "--partition=" + requested in load_json(path)["commands"][name]


@pytest.mark.parametrize("field,value", [("Account", "wrong"), ("QOS", "different"),
    ("NumNodes", "2"), ("NumTasks", "2"), ("NumCPUs", "8"),
    ("TimeLimit", "12:00:00"), ("TimeLimit", "UNLIMITED"), ("Partition", "tier2")])
def test_scheduler_changes_other_than_partition_still_fail(tmp_path, monkeypatch, field, value):
    spec, fields, _, root, _ = worker(tmp_path, monkeypatch, "tier3", "debug", "preflight")
    fields[field] = value
    with pytest.raises((PermissionError, ValueError)):
        scheduler.authenticate_job(spec, "preflight")
    assert not (root / "execution").exists()


@pytest.mark.parametrize("key,value", [("SLURM_JOB_PARTITION", "tier3"),
    ("SLURM_CLUSTER_NAME", "tigris"), ("SLURM_MEM_PER_NODE", "192000"),
    ("SLURM_JOB_ID", "999"), ("PYTHONNOUSERSITE", "0")])
def test_environment_spoofing_or_wrong_job_is_rejected(tmp_path, monkeypatch, key, value):
    spec, _, _, _, _ = worker(tmp_path, monkeypatch, "tier3", "debug", "preflight")
    monkeypatch.setenv(key, value)
    with pytest.raises((PermissionError, ValueError)):
        scheduler.authenticate_job(spec, "preflight")


@pytest.mark.parametrize("accepted,actual", [("tier3", "debug"), ("debug", "tier3")])
def test_gpu_gate_checks_actual_partition_and_identical_accepted_hardware(tmp_path, monkeypatch, accepted, actual):
    spec = spec_at(tmp_path)
    monkeypatch.setenv("SLURM_JOB_PARTITION", actual)
    seen = []
    def allocated(site):
        seen.append(site)
        return "456", 4, 320000
    monkeypatch.setattr(runtime, "allocation", allocated)
    evidence = dict(site=execution.site_for_partition(accepted), requested_site=spec["execution_site"],
        execution_policy_sha256=spec["execution_policy"]["content_hash"],
        gpu={"name": "A100", "total_memory_bytes": 40*1024**3}, environment={"weaver": "same"})
    execution.validate_acceptance_site(spec, evidence)
    monkeypatch.setattr(runtime, "science_gate", lambda s: evidence)
    monkeypatch.setattr(runtime, "gpu_identity", lambda: dict(evidence["gpu"]))
    monkeypatch.setattr(runtime, "installed_environment", lambda: dict(evidence["environment"]))
    assert runtime.execution_gate(spec, science=True) == "456"
    assert seen == [execution.site_for_partition(actual)]
    monkeypatch.setattr(runtime, "gpu_identity", lambda: {"name": "H100"})
    with pytest.raises(ValueError, match="GPU/software"):
        runtime.execution_gate(spec, science=True)
    monkeypatch.setattr(runtime, "gpu_identity", lambda: dict(evidence["gpu"]))
    monkeypatch.setattr(runtime, "installed_environment", lambda: {"weaver": "changed"})
    with pytest.raises(ValueError, match="GPU/software"):
        runtime.execution_gate(spec, science=True)


@pytest.mark.parametrize("fault", ["site", "policy", "requested"])
def test_acceptance_partition_is_not_an_unchecked_exception(fault):
    spec = campaign.registration()
    evidence = dict(site=execution.site_for_partition("debug"), requested_site=spec["execution_site"],
        execution_policy_sha256=spec["execution_policy"]["content_hash"])
    if fault == "site":
        evidence["site"] = rehash(evidence["site"], gpu_family="H100")
    elif fault == "policy":
        evidence["execution_policy_sha256"] = "f"*64
    else:
        evidence["requested_site"] = execution.site_for_partition("debug")
    with pytest.raises(ValueError, match="acceptance"):
        execution.validate_acceptance_site(spec, evidence)


def test_worker_and_queue_shell_expose_portability_without_mutating_jobs():
    root = Path(__file__).resolve().parents[1]
    helper = (root / "scripts/queue_jetclass2_concat_k2.sh").read_text()
    shell = (root / "sbatch/run_jetclass2_concat_k2.sh").read_text()
    assert '${K2_PARTITION:-tier3}' in helper and '--partition "${K2_PARTITION}"' in helper
    assert 'SLURM_JOB_PARTITION' in shell
    assert 'tier3) export JC2_SITE=sporc_a100' in shell
    assert 'debug) export JC2_SITE=sporc_a100_debug' in shell
    assert "scontrol update" not in shell + helper and "scancel" not in shell + helper
