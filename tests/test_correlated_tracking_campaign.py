"""Authentic synthetic-parent integration and guarded single-job orchestration."""
import subprocess
from pathlib import Path

import numpy as np
import pytest
import uproot

from test_literature_proxy_v2_campaign import parent
from hlt_classification.correlated_tracking import campaign as c, worker as w
from hlt_classification.correlated_tracking.contracts import load_json, with_content_hash


@pytest.fixture
def pilot(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "PROJECT", tmp_path / "new-code")
    monkeypatch.setattr(c, "source", lambda *a, **kw: dict(commit="c"*40, file_sha256={}))
    return c.create(project=c.PROJECT, commit="c"*40, parent_spec=parent, root=tmp_path / "correlated")


def test_full_cpu_pilot_replay_banks_and_corruption(pilot, monkeypatch, capsys):
    def forbidden(*a, **kw):
        raise AssertionError("New pilot must not read ROOT/labels/test/native HLT")
    monkeypatch.setattr(uproot, "open", forbidden)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    report = w.run(pilot, workers=2)
    assert report["jets"] == 12 and report["replay"]["exact"] and report["structure_exact"]
    assert report["no_classifier_fit"] and not report["final_test_accessed"]
    assert w.results(pilot) == w.run(pilot, workers=2) == report
    root = Path(pilot["root"])
    for name in ("overlays.pdf", "mechanism.pdf"):
        assert (root / name).read_bytes().startswith(b"%PDF")
    assert "CORR_HIGH/response/all/dz/pull" in (root / "statistics.csv").read_text()
    identities = []
    for path in sorted((root / "blocks").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            identities.extend(data["identity"].tolist())
            assert len(data.files) == 2+7*5
            assert not any("label" in key or "latent" in key or "ancestry" in key for key in data.files)
            for side in w.SIDES:
                assert len(data[side+"_tracking"]) == data["offsets"][-1]
                for field in ("p4", "charge", "category", "valid"):
                    np.testing.assert_array_equal(data[side+"_"+field], data["OFFLINE_"+field])
    assert set(identities) == {i for r in pilot["inputs"] for i in r["selected"]["identities"]}
    w.print_results(report)
    assert "No accuracy-gap" in capsys.readouterr().out
    with (root / "statistics.csv").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Output bytes"):
        w.results(pilot)


def test_dry_authorization_clean_env_and_idempotency(pilot, monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kw["env"])
        return subprocess.CompletedProcess(argv, 0, "98765;tigris\n", "")
    monkeypatch.setattr(c.subprocess, "run", fake)
    monkeypatch.setenv("SBATCH_GRES", "gpu:1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "72")
    plan = c.submit(pilot)["plan"]
    assert not calls and plan["jobs"] == 1 and not plan["automatic_followup"]
    assert plan["resources"] == c.RESOURCES
    with pytest.raises(PermissionError):
        c.submit(pilot, execute=True, reviewed_hash="bad", authorization=c.AUTHORIZATION)
    kw = dict(execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION)
    ledger = c.submit(pilot, **kw)
    assert ledger["job"] == "98765" and len(calls) == 2 and "--test-only" in calls[0]
    assert c.submit(pilot, **kw) == ledger and len(calls) == 2


def test_ambiguous_submission_keeps_lock(pilot, monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, "unknown", ""))
    args = dict(execute=True, reviewed_hash=c.plan(pilot)["content_hash"], authorization=c.AUTHORIZATION)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit(pilot, **args)
    with pytest.raises(FileExistsError):
        c.submit(pilot, **args)


def test_scope_source_and_allocation_guards(pilot, monkeypatch):
    for field in c.FLAGS:
        with pytest.raises(PermissionError):
            c.validate_spec(with_content_hash(dict(pilot, **{field: True})))
    for field in ("recipe", "resources", "population", "inputs"):
        with pytest.raises(ValueError):
            c.validate_spec(with_content_hash(dict(pilot, **{field: {}})))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="allocated"):
        w.run(pilot, workers=3)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        w.run(pilot, workers=1)
    monkeypatch.setattr(c, "source", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="Source"):
        c.validate_spec(pilot)


def test_saved_plan_cannot_be_silently_changed(pilot, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("No scheduler call with changed plan")
    monkeypatch.setattr(c.subprocess, "run", forbidden)
    path = Path(pilot["root"])/"command_plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Saved command plan"):
        c.submit(pilot)


def test_insufficient_space_never_generates_particles(pilot, monkeypatch):
    from types import SimpleNamespace
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setattr(w.shutil, "disk_usage", lambda root: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Insufficient output space"):
        w.run(pilot, workers=1)
    assert not (Path(pilot["root"])/"receipt.json").exists()
    assert not (Path(pilot["root"])/"blocks").exists()


def test_fresh_roots_and_parent_reference(pilot, parent):
    for root in (Path(parent).parent/"inside", Path(pilot["root"])):
        with pytest.raises(ValueError, match="Fresh"):
            c.create(project=c.PROJECT, commit="c"*40, parent_spec=parent, root=root)
    altered = dict(pilot, parent_spec=dict(pilot["parent_spec"], sha256="0"*64))
    with pytest.raises(ValueError, match="bytes changed"):
        c.validate_spec(with_content_hash(altered))


def test_complete_source_pin_and_worker_static(monkeypatch):
    project = c.PROJECT
    required = []
    def git(_, *args):
        if args[0] == "rev-parse":
            return "a"*40
        if args[0] == "ls-files":
            required.extend(args)
        if args[0] == "branch":
            return "origin/main"
        return ""
    monkeypatch.setattr(c, "git", git)
    pin = c.source(project, "a"*40, pushed=True)
    assert "src/hlt_classification/correlated_tracking/kernel.py" in pin["file_sha256"]
    assert "src/hlt_classification/literature_proxy_v2/inputs.py" in required
    shell = (project/"sbatch/run_jetclass2_correlated_tracking.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_correlated_tracking.py"' in shell
    assert "PYTHONNOUSERSITE=1" in shell and "LD_LIBRARY_PATH" in shell
    helper = (project/"scripts/queue_jetclass2_correlated_tracking.sh").read_text()
    assert c.AUTHORIZATION in helper and "--reviewed-hash" in helper
    assert "!= 20000" in helper
