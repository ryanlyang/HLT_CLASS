"""Real synthetic parent chain, paired pilot outputs and safe queue controls."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from test_literature_proxy_v3_campaign import parent, parent_v2
from hlt_classification.literature_context import campaign as c, worker as w
from hlt_classification.literature_context.contracts import load_json, with_content_hash


@pytest.fixture
def pilot(parent_v2, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "PROJECT", tmp_path/"context-code")
    monkeypatch.setattr(c, "source", lambda *a, **k: dict(commit="f"*40, file_sha256={}))
    return c.create(project=c.PROJECT, commit="f"*40, parent_spec=parent_v2, root=tmp_path/"context")


def test_full_pilot_exact_replay_outputs_and_no_root(pilot, monkeypatch):
    monkeypatch.setattr(uproot, "open", lambda *a, **k: pytest.fail("Context pilot opened ROOT"))
    for name in w.THREADS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    result = w.run(pilot, workers=2)
    assert result["replay"]["exact"] and result["structure_exact"]
    assert not result["calibration_refitted"] and not result["classifier_performance_measured"]
    assert result["jets"] == pilot["population"]["jets"]
    assert result == w.results(pilot) == w.run(pilot, workers=2)
    root = Path(pilot["root"])
    for path in (root/"blocks").glob("*.npz"):
        with np.load(path, allow_pickle=False) as b:
            for field in ("p4", "charge", "category", "valid", "offsets"):
                np.testing.assert_array_equal(b[f"LOW_NOISE_{field}"], b[f"CONTEXT_{field}"])
    for side in w.SIDES:
        assert result["metrics"][f"{side}/jet/multiplicity"]["count"] == result["jets"]
    assert result["audit"]["inverse_max_scaled_error"] < 1e-10
    assert result["audit"]["new_frontend_inverse_max_scaled_error"] < 1e-4
    w.print_results(result)
    (root/"statistics.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        w.results(pilot)


def test_dry_exact_submit_and_no_ambiguous_retry(pilot, monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kw["env"])
        return subprocess.CompletedProcess(argv, 0, "987654\n", "")
    monkeypatch.setattr(c.subprocess, "run", fake)
    plan = c.submit(pilot)["plan"]
    assert not calls and plan["jobs"] == 1 and not plan["automatic_followup"]
    assert plan["resources"]["cpus"] == 16 and plan["resources"]["gpus"] == 0
    with pytest.raises(PermissionError):
        c.submit(pilot, execute=True, reviewed_hash="wrong", authorization=c.AUTHORIZATION)
    args = dict(execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION)
    ledger = c.submit(pilot, **args)
    assert ledger["job"] == "987654" and len(calls) == 2
    assert c.submit(pilot, **args) == ledger and len(calls) == 2


def test_ambiguous_submission_keeps_lock(pilot, monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, "unclear", ""))
    args = dict(execute=True, reviewed_hash=c.plan(pilot)["content_hash"], authorization=c.AUTHORIZATION)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit(pilot, **args)
    with pytest.raises(FileExistsError):
        c.submit(pilot, **args)


def test_scope_source_parent_and_allocation_guards(pilot, monkeypatch):
    for field in ("validation_accessed", "final_test_accessed", "production_qualified"):
        with pytest.raises(PermissionError):
            c.validate_spec(with_content_hash(dict(pilot, **{field: True})))
    for field in ("recipe", "resources", "calibration", "population", "inputs"):
        with pytest.raises(ValueError):
            c.validate_spec(with_content_hash(dict(pilot, **{field: {}})))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="allocated"):
        w.run(pilot, workers=3)
    monkeypatch.setattr(c, "source", lambda *a, **k: {})
    with pytest.raises(ValueError, match="Source"):
        c.validate_spec(pilot)


def test_parent_and_existing_roots_never_overwritten(pilot, parent_v2):
    for root in (Path(parent_v2).parent/"inside", Path(pilot["root"])):
        with pytest.raises(ValueError, match="Fresh"):
            c.create(project=c.PROJECT, commit="f"*40, parent_spec=parent_v2, root=root)


def test_source_closure_and_worker_static(tmp_path, monkeypatch):
    project = c.PROJECT
    shell = (project/"sbatch/run_jetclass2_literature_context.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_literature_context.py"' in shell
    assert "PYTHONNOUSERSITE=1" in shell and "LD_LIBRARY_PATH" in shell
    helper = (project/"scripts/queue_jetclass2_literature_context.sh").read_text()
    assert c.AUTHORIZATION in helper and "--reviewed-hash" in helper
    monkeypatch.setattr(c.old, "source", lambda *a, **k: dict(commit="f"*40, file_sha256={"v3": "pinned"}))
    required = []
    monkeypatch.setattr(c, "git", lambda project, *args: required.extend(args))
    # Source closure uses real current paths, not a scientific source mutation.
    result = c.source(project, "f"*40)
    assert result["file_sha256"]["v3"] == "pinned"
    assert "src/hlt_classification/literature_context/transform.py" in required
    assert "src/hlt_classification/cms_proxy_ladder/inputs.py" in required
