"""Sealed real synthetic v1/v2 chain, queue protections and exports."""
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from test_literature_proxy_v2_campaign import parent
from hlt_classification.literature_proxy_v2 import campaign as c2, worker as w2
from hlt_classification.literature_proxy_v3 import campaign as c, inputs, worker as w
from hlt_classification.literature_proxy_v3.contracts import (
    load_json, reference, with_content_hash, write_immutable_json,
)


@pytest.fixture(scope="module")
def parent_v2(parent, tmp_path_factory):
    temp = tmp_path_factory.mktemp("noise-parent")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c2, "PROJECT", temp / "old-code")
        patch.setattr(c2, "source", lambda *a, **kw: dict(commit="b"*40, file_sha256={}))
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            patch.setenv(key, "1")
        patch.setenv("SLURM_CPUS_PER_TASK", "2")
        spec = c2.create(project=c2.PROJECT, commit="b"*40, parent_spec=parent, root=temp / "v2")
        w2.run(spec, workers=2)
    return Path(spec["root"]) / "study_spec.json"


@pytest.fixture
def pilot(parent_v2, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "PROJECT", tmp_path / "new-code")
    monkeypatch.setattr(c, "source", lambda *a, **kw: dict(commit="c"*40, file_sha256={}))
    return c.create(project=c.PROJECT, commit="c"*40, parent_spec=parent_v2, root=tmp_path / "v3")


def test_full_paired_pilot_reuses_control_not_calibration(pilot, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("No ROOT access or recalibration")
    monkeypatch.setattr(uproot, "open", forbidden)
    monkeypatch.setattr(w2, "count_task", forbidden)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    report = w.run(pilot, workers=2)
    parent = load_json(pilot["parent_spec"]["path"])
    saved = w2.results(parent)
    assert report["structure_exact"] and report["replay"]["exact"]
    assert report["counts"] == saved["counts"] and report["achieved_mean"] == saved["achieved_mean"]
    assert not report["calibration_refitted"] and report["calibration"] == saved["calibration"]
    assert not report["final_test_accessed"] and not report["production_qualified"]
    assert report["calibration"]["parents"] == dict(spec=parent["content_hash"])
    assert w.results(pilot) == w.run(pilot, workers=2) == report
    for key, row in saved["metrics"].items():
        if key.startswith(("OFFLINE/", "COUNT38_V2/")):
            assert report["metrics"][key] == row
    root = Path(pilot["root"])
    assert (root / "paired_response.pdf").read_bytes().startswith(b"%PDF")
    assert "NOISE_V3/single_parent/pt_ratio" in (root / "statistics.csv").read_text()
    seen = []
    for file in (root / "blocks").glob("*.npz"):
        with np.load(file, allow_pickle=False) as packed:
            seen.extend(packed["identity"].tolist())
            assert not any("label" in k or "ancestry" in k for k in packed.files)
            assert packed["NOISE_V3_offsets"][-1] == len(packed["NOISE_V3_p4"])
    assert set(seen) == {j for r in pilot["inputs"] for j in r["v1"]["selected"]["identities"]}
    with (root / "statistics.csv").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Output bytes"):
        w.results(pilot)


def test_no_rounded_calibration_changed_population_or_parent(pilot, tmp_path):
    for field in ("p_drop", "p_merge"):
        bad = copy.deepcopy(pilot)
        bad["calibration"][field] += 1e-12
        bad["calibration"] = with_content_hash(bad["calibration"])
        with pytest.raises(ValueError, match="parent"):
            c.validate_spec(with_content_hash(bad))
    bad = copy.deepcopy(pilot)
    bad["population"]["jets"] += 1
    with pytest.raises(ValueError, match="population"):
        c.validate_spec(with_content_hash(bad))
    bad = copy.deepcopy(pilot["parent_spec"])
    bad["sha256"] = "0"*64
    with pytest.raises(ValueError):
        inputs.authenticate(bad, pilot["parent_receipt"], pilot["calibration_ref"])
    bad = with_content_hash(dict(pilot, final_test_accessed=True))
    with pytest.raises(PermissionError):
        c.validate_spec(bad)


def test_corrupt_and_misaligned_saved_control(pilot, tmp_path):
    original = pilot["inputs"][0]
    record = copy.deepcopy(original)
    path = tmp_path / "bad.npz"
    path.write_bytes(Path(record["block"]["path"]).read_bytes()+b"changed")
    record["block"]["path"] = str(path)
    with pytest.raises(ValueError, match="checksum"):
        list(inputs.read_rows(record))
    with np.load(original["block"]["path"], allow_pickle=False) as data:
        packed = {k: data[k] for k in data.files}
    packed["COUNT38_V2_offsets"] = packed["COUNT38_V2_offsets"].astype(float)
    np.savez(path, **packed)
    record["block"] = reference(path)
    with pytest.raises(ValueError, match="offsets"):
        list(inputs.read_rows(record))
    record = copy.deepcopy(original)
    trace = load_json(record["lineage"]["path"])
    trace["rows"][0]["identity"] = "wrong"
    path = tmp_path / "bad.json"
    write_immutable_json(path, trace)
    record["lineage"] = reference(path)
    with pytest.raises(ValueError, match="identity"):
        list(inputs.read_rows(record))


def test_dry_exact_authorization_and_single_submission(pilot, monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kw["env"])
        return subprocess.CompletedProcess(argv, 0, "123456\n", "")
    monkeypatch.setattr(c.subprocess, "run", fake)
    plan = c.submit(pilot)["plan"]
    assert not calls and plan["jobs"] == 1 and not plan["automatic_followup"]
    assert plan["resources"]["cpus"] == 16 and plan["resources"]["gpus"] == 0
    with pytest.raises(PermissionError):
        c.submit(pilot, execute=True, reviewed_hash="wrong", authorization=c.AUTHORIZATION)
    result = c.submit(pilot, execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION)
    assert result["job"] == "123456" and len(calls) == 2
    assert "--test-only" in calls[0] and not any(v.startswith(("--gpus", "--gres")) for v in calls[1])
    assert c.submit(pilot, execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION) == result
    assert len(calls) == 2


def test_ambiguous_submit_is_not_retried(pilot, monkeypatch):
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "unclear", ""))
    args = dict(execute=True, reviewed_hash=c.plan(pilot)["content_hash"], authorization=c.AUTHORIZATION)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit(pilot, **args)
    with pytest.raises(FileExistsError):
        c.submit(pilot, **args)


def test_version_source_resource_guards(pilot, monkeypatch):
    for field, value in (("recipe", {}), ("resources", {})):
        with pytest.raises(ValueError, match="recipe/resources"):
            c.validate_spec(with_content_hash(dict(pilot, **{field: value})))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="allocated"):
        w.run(pilot, workers=3)
    monkeypatch.setattr(c, "source", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="Source"):
        c.validate_spec(pilot)


def test_fresh_root_and_no_parent_overwrite(pilot, parent_v2):
    with pytest.raises(ValueError, match="Fresh"):
        c.create(project=c.PROJECT, commit="c"*40, parent_spec=parent_v2, root=Path(parent_v2).parent / "inside")
    with pytest.raises(ValueError, match="Fresh"):
        c.create(project=c.PROJECT, commit="c"*40, parent_spec=parent_v2, root=pilot["root"])


def test_new_source_chain_and_queue_worker(tmp_path, monkeypatch):
    project = c.PROJECT
    shell = (project / "sbatch/run_jetclass2_literature_proxy_noise.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_literature_proxy_noise.py"' in shell
    assert "PYTHONNOUSERSITE=1" in shell and "LD_LIBRARY_PATH" in shell
    helper = (project / "scripts/queue_jetclass2_literature_proxy_noise.sh").read_text()
    assert c.AUTHORIZATION in helper and "--reviewed-hash" in helper
    monkeypatch.setattr(c.old, "source", lambda *a, **kw: dict(commit="c"*40, file_sha256={"v2": "pinned"}))
    requested = []
    monkeypatch.setattr(c.old.old, "git", lambda project, *args: requested.extend(args))
    paths = ["scripts/jetclass2_literature_proxy_noise.py", "scripts/queue_jetclass2_literature_proxy_noise.sh",
             "sbatch/run_jetclass2_literature_proxy_noise.sh", "docs/plans/JETCLASS2_LITERATURE_PROXY_NOISE_PILOT_PLAN.md",
             "docs/contracts/JETCLASS2_LITERATURE_PROXY_NOISE.md", "src/hlt_classification/literature_proxy_v3/kernel.py"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    src = c.source(tmp_path, "c"*40)
    assert src["file_sha256"]["v2"] == "pinned" and len(src["file_sha256"]) == 7
    assert all(p in requested for p in paths)
