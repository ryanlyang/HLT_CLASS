"""Local synthetic recovery tests; not a substitute for OSCAR hardware evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hlt_classification.data.cache_contracts import load_json, sha256_file
from hlt_classification.jetclass2_delphes.salience_learned_contracts import artifact as kernel_artifact
from hlt_classification.noise_k2 import campaign as original, contracts as c, data, runtime as science
from hlt_classification.noise_k2 import recovery as r, recovery_runtime as worker
from test_noise_k2 import toy_spec, rehash, make_cache

REUSED_SCIENCE = {
    "train_HLT_X1_CE", "train_CONCAT_K2_D100", "reduce_CONCAT_K2_D100",
    "train_DIRECT_HLT_X3_KD", "train_CONCAT_K2_D075", "reduce_CONCAT_K2_D075",
    "train_CONCAT_K2_D050",
}
RETRY = ["train_HLT_X3_CE", "train_OFFLINE_CE", "reduce_CONCAT_K2_D050",
         "train_CONCAT_K2_D025", "reduce_CONCAT_K2_D025", "train_CONCAT_K2_D000",
         "reduce_CONCAT_K2_D000", "train_HLT_X1_COMPRESSED", "aggregate", "complete"]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(17, 11)

    def forward(self, features, vectors, mask):
        return self.fc((features*mask).sum(-1)/mask.sum(-1))


def tiny_model(node):
    torch.manual_seed(node["initialization_seed"])
    return Tiny()


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Real hashes, receipts, journals and banks; synthetic scientific evidence."""
    spec = rehash(toy_spec(tmp_path), dataset_root=str(tmp_path/"dataset"))
    root = Path(spec["campaign_root"])
    c.publish(root/"campaign_spec.json", spec)
    monkeypatch.setattr(original, "validate_campaign", lambda *a, **k: spec["content_hash"])
    monkeypatch.setattr(r, "source_compatibility", lambda *a: {"unchanged.py": "a"*64})
    acceptance = c.artifact("ACCEPTANCE", environment={"toy": "local-only"}, gpu={"name": "toy GPU"})
    monkeypatch.setattr(science, "science_gate", lambda s: acceptance)
    monkeypatch.setattr(science, "installed_environment", lambda: acceptance["environment"])
    monkeypatch.setattr(science, "gpu_identity", lambda: acceptance["gpu"])
    jobs, count = {}, []
    def fake_submit(command, **kw):
        count.append(command)
        return SimpleNamespace(stdout=str(1000+len(count)), returncode=0, stderr="")
    monkeypatch.setattr(r.subprocess, "run", fake_submit)
    for stage in ("gate", "science"):
        plan = original.plan(spec, stage)
        dry = root/(stage+"_dry_run.json")
        r.submit_exact_dag(identity=spec["content_hash"], plan=plan, output=dry, canonical_dry_run=dry, execute=False)
        ledger = r.submit_exact_dag(identity=spec["content_hash"], plan=plan,
            output=root/(stage+"_submission_ledger.json"), canonical_dry_run=dry, execute=True)
        jobs.update(ledger["jobs"])
    train, val = make_cache("train"), make_cache("validation")
    from hlt_classification.jetclass2_delphes.salience_learned_data import IndexedRamCache
    codes = data.partition_codes(val.identities, val.labels)
    subsets = {n: IndexedRamCache(val, np.flatnonzero(codes == i), role="validation")
               for i, n in enumerate(("checkpoint", "diagnostic", "report"))}
    monkeypatch.setattr(science, "new_model", tiny_model)
    monkeypatch.setattr(science, "caches", lambda *a: (train, subsets, {"content_hash": "b"*64}))
    def cache(s, role, coordinate):
        assert s == spec and role in ("train", "validation")  # No recovery root or final test.
        return train if role == "train" else val
    monkeypatch.setattr(data, "cache", cache)
    monkeypatch.setattr(data, "get_foundation", lambda s: {"capacity": 32})
    torch.set_num_threads(1)
    probabilities = np.full((len(train), 11), 1/11, np.float32)
    metrics = science.evaluate_probabilities(train.labels, probabilities)
    done = {}
    for row in spec["tasks"]:
        name = row["task_id"]
        if row["kind"] in original.SCIENCE and name not in REUSED_SCIENCE:
            continue
        directory = root/"outputs"/name
        directory.mkdir(parents=True)
        result = {}
        if row["kind"] == "train":
            node = science.node_for(spec, row["node_id"])
            checkpoint = directory/"selected.pt"
            science.save_state(checkpoint, tiny_model(node).state_dict())
            kernel = kernel_artifact("TRAINING_REPORT", node=node, scientific_fit=True, acceptance_only=False,
                selected_pass=1, passes=75, validation=metrics, final_test_accessed=False)
            report = c.artifact("TRAINING_REPORT", campaign_sha256=spec["content_hash"], node=node,
                kernel_report=kernel, training=spec["training"], report_validation=metrics,
                selected_checkpoint_sha256=sha256_file(checkpoint), report_is_validation_not_final_test=True)
            path = directory/"training_report.json"
            c.publish(path, report)
            result = dict(training_report=path, checkpoint=checkpoint, training_report_sha256=report["content_hash"])
        elif row["kind"] == "reduce":
            report_hash = done["train_"+row["node_id"]]["result"]["training_report_sha256"]
            bank = science.publish_bank(directory/"bank", foundation_sha256=train.foundation_sha256,
                teacher_report_sha256=report_hash, teacher_node=row["node_id"], role="train",
                identities=train.identities, probabilities=probabilities)
            result = dict(bank=directory/"bank", bank_sha256=bank["content_hash"], training_report_sha256=report_hash)
        result = {k: v.relative_to(root).as_posix() if isinstance(v, Path) else v for k, v in result.items()}
        c.publish(directory/"result.json", c.artifact("RESULT", result=result))
        receipt = c.artifact("TASK", campaign_sha256=spec["content_hash"], task_id=name,
            source_commit=spec["source_commit"], job_id=jobs[name], result=result,
            parents={p: done[p]["content_hash"] for p in row["dependencies"]},
            outputs=[c.reference(root, p) for p in sorted(directory.rglob("*")) if p.is_file()])
        c.publish(root/"tasks"/(name+".json"), receipt)
        done[name] = receipt
    # Failed output files must remain byte-for-byte intact throughout recovery.
    c.publish(root/"outputs/train_HLT_X3_CE/failure.json", c.artifact("FAILURE", message="CUDA ECC"))
    states = {job: dict(state="COMPLETED" if name in done else "CANCELLED",
        exit_code="0:0", name="noisek2_"+name, nodes="gpu3001") for name, job in jobs.items()}
    for name in ("train_HLT_X3_CE", "train_OFFLINE_CE", "reduce_CONCAT_K2_D050"):
        states[jobs[name]].update(state="FAILED", exit_code="1:0")
    monkeypatch.setattr(r, "accounting", lambda bound: deepcopy(states))
    return SimpleNamespace(spec=spec, root=root, jobs=jobs, states=states, done=done,
                           train=train, validation=subsets, acceptance=acceptance)


def create(source, tmp_path):
    return r.create(source_spec=source.root/"campaign_spec.json", recovery_root=tmp_path/"recovery",
                    project_dir=tmp_path/"new-code", source_commit="f"*40)


def tree_hash(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def live(spec, monkeypatch):
    commands = []
    def run(command, **kw):
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kw.get("env", {}))
        commands.append(command)
        return SimpleNamespace(stdout=str(2000+len(commands)), stderr="", returncode=0)
    monkeypatch.setattr(r.subprocess, "run", run)
    ledger = r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE)
    return ledger, commands


def test_create_full_ten_job_dry_plan_reuses_all_completed_bytes(source, tmp_path):
    before = tree_hash(source.root)
    spec = create(source, tmp_path)
    assert r.validate_recovery(spec) == source.spec
    assert spec["retry_tasks"] == RETRY
    assert set(spec["reused_tasks"]) & {t["task_id"] for t in source.spec["tasks"] if t["kind"] in original.SCIENCE} == REUSED_SCIENCE
    dry = load_json(tmp_path/"recovery/dry_run.json")
    assert dry["dry_run"] is True and len(dry["jobs"]) == 10
    assert not (tmp_path/".noise_k2_recovery_claims").exists()
    rows = r.plan(spec, source.spec)["commands"]
    originals = {row["task_id"]: row for row in original.plan(source.spec, "science")["commands"]}
    for row in rows:
        name, command = row["task_id"], row["command"]
        assert not any(str(job) in value for job in source.jobs.values() for value in command if value.startswith("--dependency="))
        assert set(row["dependencies"]) <= set(RETRY)
        for prefix in ("--mem=", "--time=", "--cpus-per-task=", "--qos=", "--partition=", "--gres="):
            assert [v for v in command if v.startswith(prefix)] == [v for v in originals[name]["command"] if v.startswith(prefix)]
        assert ("--exclude=gpu3001" in command) == (name not in ("aggregate", "complete"))
    assert rows[2]["dependencies"] == []  # D050 reducer reads the original selected checkpoint.
    assert rows[3]["dependencies"] == ["reduce_CONCAT_K2_D050"]
    assert tree_hash(source.root) == before


@pytest.mark.parametrize("change", ["missing_gate", "successful_no_receipt", "failed_with_receipt", "environment", "all_complete"])
def test_create_fail_closed_before_publishing(source, tmp_path, monkeypatch, change):
    if change == "missing_gate":
        monkeypatch.setattr(original, "completed", lambda *a, **k: None)
    elif change == "successful_no_receipt":
        source.states[source.jobs["train_HLT_X3_CE"]].update(state="COMPLETED", exit_code="0:0")
    elif change == "failed_with_receipt":
        source.states[source.jobs["train_HLT_X1_CE"]].update(state="FAILED", exit_code="1:0")
    elif change == "environment":
        monkeypatch.setattr(science, "installed_environment", lambda: {"different": "version"})
    else:
        monkeypatch.setattr(r, "inventory", lambda *a: (source.done, []))
    with pytest.raises(ValueError):
        create(source, tmp_path)
    assert not (tmp_path/"recovery").exists()


def test_corrupt_completed_file_is_not_silently_retrained(source, tmp_path):
    (source.root/"outputs/train_CONCAT_K2_D050/selected.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        create(source, tmp_path)
    assert not (tmp_path/"recovery").exists()


def test_dry_authorization_live_idempotence_and_unchanged_original(source, tmp_path, monkeypatch):
    spec = create(source, tmp_path)
    before = tree_hash(source.root)
    with pytest.raises(ValueError, match="phrase"):
        r.submit(spec, execute=True)
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.setenv("SBATCH_PARTITION", "debug")
    ledger, calls = live(spec, monkeypatch)
    assert len(ledger["jobs"]) == 10 and not ledger["dry_run"]
    assert sum("--test-only" not in cmd for cmd in calls) == 10
    assert r.replacement_jobs(spec, source.spec) == ledger["jobs"]
    for name, command in ledger["commands"].items():
        assert "${JOB_" not in " ".join(command)
    count = len(calls)
    assert r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE) == ledger
    assert len(calls) == count and tree_hash(source.root) == before


def test_pretest_failure_and_ambiguous_sbatch_cannot_duplicate(source, tmp_path, monkeypatch):
    spec = create(source, tmp_path)
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="No allocation"))
    with pytest.raises(ValueError, match="resource test"):
        r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE)
    assert not (tmp_path/"recovery/submission.claim").exists()
    def ambiguous(command, **kw):
        if "--test-only" in command:
            return SimpleNamespace(returncode=0, stderr="")
        raise OSError("sbatch acknowledgement lost")
    monkeypatch.setattr(r.subprocess, "run", ambiguous)
    with pytest.raises(OSError, match="acknowledgement"):
        r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE)
    with pytest.raises(ValueError, match="ambiguous"):
        r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE)
    second = r.create(source_spec=source.root/"campaign_spec.json", recovery_root=tmp_path/"another",
                      project_dir=tmp_path/"new-code", source_commit="f"*40)
    with pytest.raises(FileExistsError):
        r.submit(second, execute=True, authorization_phrase=r.AUTHORIZE)


@pytest.mark.parametrize("case", ["active", "missing", "wrong_name", "nonterminal", "duplicate"])
def test_exact_accounting_rejects_unknown_and_live_jobs(tmp_path, monkeypatch, case):
    def run(cmd, **kw):
        if cmd[0] == "squeue":
            return SimpleNamespace(stdout="12|PENDING\n" if case == "active" else "99|RUNNING\n")
        line = "12|CANCELLED by 100|0:0|noisek2_train_A|gpu3001\n"
        if case == "missing": line = "13|COMPLETED|0:0|other|host\n"
        if case == "wrong_name": line = line.replace("noisek2_train_A", "someone_else")
        if case == "nonterminal": line = line.replace("CANCELLED by 100", "RUNNING")
        if case == "duplicate": line += line
        return SimpleNamespace(stdout=line)
    monkeypatch.setattr(r.subprocess, "run", run)
    with pytest.raises(ValueError):
        r.accounting({"train_A": "12"})


def test_accounting_ignores_unrelated_jobs_and_normalizes_cancel(monkeypatch):
    monkeypatch.setattr(r.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout=
        "99|RUNNING\n" if cmd[0] == "squeue" else "12|CANCELLED by 100|0:0|noisek2_A|gpu3001\n12.batch|FAILED|1:0|batch|gpu3001\n"))
    assert r.accounting({"A": "12"})["12"]["state"] == "CANCELLED"


def test_new_scientific_source_cannot_reuse_old_acceptance(tmp_path, monkeypatch):
    old, new = tmp_path/"old", tmp_path/"new"
    path = "src/hlt_classification/noise_k2/runtime.py"
    for root in (old, new):
        (root/Path(path).parent).mkdir(parents=True)
        (root/path).write_text("original", encoding="utf8")
    monkeypatch.setattr(original, "_source", lambda *a: None)
    monkeypatch.setattr(r, "__file__", str(new/"src/hlt_classification/noise_k2/recovery.py"))
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=path+"\n"))
    source = dict(project_dir=str(old), source_commit="a"*40)
    assert r.source_compatibility(source, new, "b"*40)[path] == sha256_file(old/path)
    (new/path).write_text("changed loss", encoding="utf8")
    with pytest.raises(ValueError, match="scientific source changed"):
        r.source_compatibility(source, new, "b"*40)


def test_source_journal_and_receipt_job_ids_are_not_interchangeable(source, tmp_path):
    gate = source.root/"gate_submission_ledger.json"
    ledger = load_json(gate)
    ledger["jobs"]["authenticate"] = "88888"
    import json
    gate.write_text(json.dumps(ledger), encoding="utf8")
    with pytest.raises(ValueError, match="ledger/journal"):
        create(source, tmp_path)


@pytest.mark.parametrize("problem", ["job", "excluded", "cpu", "account", "partition", "memory", "prefix"])
def test_worker_rejects_wrong_id_node_or_allocation(source, tmp_path, monkeypatch, problem):
    spec = create(source, tmp_path)
    ledger, _ = live(spec, monkeypatch)
    name = "train_HLT_X3_CE"
    site, res = source.spec["execution_site"], source.spec["resources"]["train"]
    prefix = site["conda_base"]+"/envs/"+site["conda_env"]
    env = dict(SLURM_JOB_ID=ledger["jobs"][name], SLURM_CLUSTER_NAME=site["cluster"],
        SLURM_CPUS_PER_TASK=str(res["cpus"]), SLURM_MEM_PER_NODE=str(res["memory_mb"]),
        CONDA_PREFIX=prefix, PYTHONNOUSERSITE="1")
    for k, v in env.items(): monkeypatch.setenv(k, v)
    monkeypatch.setattr(r.sys, "prefix", prefix)
    monkeypatch.setattr(r.platform, "machine", lambda: site["architecture"])
    monkeypatch.setattr(r.platform, "node", lambda: "gpu2708")
    fields = dict(Account=site["account"], Partition=site["partition"], NumNodes="1",
                  NumTasks="1", NumCPUs=str(res["cpus"]), NodeList="gpu2708")
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout=" ".join(k+"="+v for k, v in fields.items())))
    from hlt_classification.jetclass2_delphes import execution
    calls = []
    monkeypatch.setattr(execution, "allocation", lambda s: calls.append(s))
    assert r.authenticate_job(spec, source.spec, name) == ledger["jobs"][name]
    assert calls == [site]
    if problem == "job": monkeypatch.setenv("SLURM_JOB_ID", "99999")
    elif problem == "excluded": fields["NodeList"] = "gpu3001"
    elif problem == "cpu": fields["NumCPUs"] = "1"
    elif problem == "account": fields["Account"] = "other"
    elif problem == "partition": fields["Partition"] = "batch"
    elif problem == "memory": monkeypatch.setenv("SLURM_MEM_PER_NODE", "1")
    else: monkeypatch.setenv("CONDA_PREFIX", "/different/environment")
    with pytest.raises(ValueError): r.authenticate_job(spec, source.spec, name)


def test_recorded_state_change_blocks_live_submit(source, tmp_path, monkeypatch):
    spec = create(source, tmp_path)
    source.states[source.jobs["train_HLT_X3_CE"]]["nodes"] = "different"
    with pytest.raises(ValueError, match="scheduler state changed"):
        r.submit(spec, execute=True, authorization_phrase=r.AUTHORIZE)
    assert not (tmp_path/"recovery/submission.claim").exists()


def test_unapproved_roots_nodes_and_rehashed_task_sets_fail(source, tmp_path):
    for root in (source.root, source.root/"child", tmp_path/"other-parent/recovery"):
        with pytest.raises(ValueError):
            r.create(source_spec=source.root/"campaign_spec.json", recovery_root=root,
                     project_dir=tmp_path/"new-code", source_commit="f"*40)
    with pytest.raises(ValueError, match="gpu3001"):
        r.create(source_spec=source.root/"campaign_spec.json", recovery_root=tmp_path/"recovery",
                 project_dir=tmp_path/"new-code", source_commit="f"*40, excluded_nodes=[])
    spec = create(source, tmp_path)
    with pytest.raises(ValueError, match="completion state"):
        r.validate_recovery(rehash(spec, retry_tasks=spec["retry_tasks"][:-1]))


def test_result_pointer_cannot_escape_authenticated_outputs(source, tmp_path):
    spec = create(source, tmp_path)
    context = r.Context(spec, source.spec)
    done, owner = context.completed("train_CONCAT_K2_D050")
    bad = deepcopy(done)
    bad["result"]["checkpoint"] = "outputs/train_CONCAT_K2_D100/selected.pt"
    with pytest.raises(ValueError, match="outside receipt"):
        context.result_path((bad, owner), "checkpoint")
    bad["result"]["checkpoint"] = "../elsewhere.pt"
    with pytest.raises(ValueError):
        context.result_path((bad, owner), "checkpoint")


def test_new_worker_runs_native_kernel_mixed_lineage_no_original_writes(source, tmp_path, monkeypatch):
    spec = create(source, tmp_path)
    ledger, _ = live(spec, monkeypatch)
    monkeypatch.setattr(worker, "authenticate_job", lambda s, old, name: ledger["jobs"][name])
    smoke = []
    monkeypatch.setattr(worker, "gpu_smoke", lambda device: smoke.append(device) or {"passed": True})
    before = tree_hash(source.root)
    for name in RETRY:
        receipt = worker.run_task(spec, name, device="cpu")
        assert receipt["job_id"] == ledger["jobs"][name]
        assert r.Context(spec, source.spec).completed(name)[0] == receipt
    assert len(smoke) == 8
    rows = worker.result_rows(r.Context(spec, source.spec))
    assert len(rows) == 10 and all(row["validation"] is not None for row in rows)
    assert sum(row["provenance"] == "original" for row in rows) == 5
    assert all(row["recovery"] is not None for row in rows)
    compressed = r.load(tmp_path/"recovery/outputs/train_HLT_X1_COMPRESSED/deployment.json", "DEPLOYMENT")
    assert compressed["physical_proxy_copies"] == 1
    assert not compressed["offline_inputs"] and not compressed["assignment_inputs"]
    assert tree_hash(source.root) == before
    with pytest.raises(ValueError, match="read-only"):
        worker.run_task(spec, "train_CONCAT_K2_D050", device="cpu")


def test_gpu_health_failure_precedes_cache_and_preserves_failure(source, tmp_path, monkeypatch):
    spec = create(source, tmp_path)
    ledger, _ = live(spec, monkeypatch)
    monkeypatch.setattr(worker, "authenticate_job", lambda s, old, name: ledger["jobs"][name])
    def fail(device):
        raise RuntimeError("CUDA ECC")
    monkeypatch.setattr(worker, "gpu_smoke", fail)
    monkeypatch.setattr(science, "caches", lambda *a: pytest.fail("cache before GPU health"))
    before = tree_hash(source.root)
    with pytest.raises(RuntimeError, match="ECC"):
        worker.run_task(spec, "train_HLT_X3_CE", device="cpu")
    failure = tmp_path/"recovery/outputs/train_HLT_X3_CE/failure.json"
    assert r.load(failure, "FAILURE")["error_type"] == "RuntimeError"
    assert not (tmp_path/"recovery/tasks/train_HLT_X3_CE.json").exists()
    with pytest.raises(FileExistsError):
        worker.run_task(spec, "train_HLT_X3_CE", device="cpu")
    assert tree_hash(source.root) == before


@pytest.mark.parametrize("name", ["OFFLINE_CE", "CONCAT_K2_D025"])
def test_fit_adapter_selected_weights_predictions_match_original(source, tmp_path, monkeypatch, name):
    spec = create(source, tmp_path)
    context = r.Context(spec, source.spec)
    # D025 uses the already imported D075 bank for this isolated adapter parity
    # case; the actual registered DAG is separately enforced in end-to-end tests.
    node = deepcopy(science.node_for(source.spec, name))
    if node["teacher_distribution"] is not None:
        node["teacher_distribution"] = "CONCAT_K2_D075"
    a, b = tmp_path/"fit-original", tmp_path/"fit-replacement"
    a.mkdir(); b.mkdir()
    expected = science.fit(source.spec, node, a, "cpu")
    actual = worker.fit(context, node, b, "cpu")
    e, v = load_json(expected["training_report"]), load_json(actual["training_report"])
    assert e["report_validation"] == v["report_validation"]
    assert e["checkpoint_validation"] == v["checkpoint_validation"]
    assert e["kernel_report"]["selected_pass"] == v["kernel_report"]["selected_pass"]
    assert e["kernel_report"]["passes"] == v["kernel_report"]["passes"]
    se = torch.load(expected["checkpoint"], weights_only=True)
    sv = torch.load(actual["checkpoint"], weights_only=True)
    assert se.keys() == sv.keys()
    assert all(torch.equal(se[k], sv[k]) for k in se)
