"""Real synthetic native-parent integration; no Slurm jobs or real datasets."""
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from test_jetclass2_delphes import snapshot
from test_jetclass2_delphes_split_registry import registered
from hlt_classification.jetclass2_delphes.split_registry import select_profile
from hlt_classification.literature_proxy import campaign as old, worker as ow
from hlt_classification.literature_proxy_v2 import campaign as c, inputs, worker as w
from hlt_classification.literature_proxy_v2.contracts import load_json, reference, with_content_hash, write_immutable_json


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    temp = tmp_path_factory.mktemp("count38-parent")
    data, inventory, _, registry = registered.__wrapped__(snapshot.__wrapped__(temp))
    profile = select_profile(registry, inventory, "TRAIN_44")
    ip, pp = temp / "inventory.json", temp / "profile.json"
    write_immutable_json(ip, inventory)
    write_immutable_json(pp, profile)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(old, "source", lambda *a, **kw: dict(commit="a"*40, file_sha256={}))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            patch.setenv(name, "1")
        patch.setenv("SLURM_CPUS_PER_TASK", "2")
        spec = old.create(project=old.PROJECT, commit="a"*40, data_root=data, inventory=ip,
                          profile=pp, root=temp / "v1", count=12)
        ow.run(spec, workers=2)
    return Path(spec["root"]) / "study_spec.json"


@pytest.fixture
def pilot(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "source", lambda *a, **kw: dict(commit="b"*40, file_sha256={}))
    monkeypatch.setattr(c, "PROJECT", tmp_path / "clean-code")
    return c.create(project=c.PROJECT, commit="b"*40, parent_spec=parent, root=tmp_path / "v2")


def test_native_parent_and_v2_end_to_end(pilot, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("V2 must not read ROOT/labels/HLT")
    monkeypatch.setattr(uproot, "open", forbidden)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    # Spawned code only receives physical-block records, no ROOT reader capability.
    report = w.run(pilot, workers=2)
    assert report["replay"]["exact"] and report["jets"] == 12
    assert not report["final_test_accessed"] and not report["production_qualified"]
    # Fixture has three particles per jet; infeasible 38 target is not a failure.
    assert report["achieved_mean"] == 3.
    assert w.results(pilot) == w.run(pilot, workers=2) == report
    root = Path(pilot["root"])
    assert (root / "overlays.pdf").read_bytes().startswith(b"%PDF")
    assert "COUNT38_V2" in (root / "statistics.csv").read_text()
    observed = []
    for block in sorted((root / "blocks").glob("*.npz")):
        with np.load(block, allow_pickle=False) as data:
            observed.extend(data["identity"].tolist())
            assert not any("label" in k or "ancestry" in k for k in data.files)
            assert data["COUNT38_V2_offsets"][-1] == len(data["COUNT38_V2_p4"])
    assert set(observed) == {x for r in pilot["inputs"] for x in r["selected"]["identities"]}
    parent_spec = load_json(pilot["parent_spec"]["path"])
    old_report = ow.results(parent_spec)
    for name, row in old_report["metrics"].items():
        if name.startswith(("OFFLINE/", "NOMINAL/")) and not name.endswith("/count_delta"):
            assert report["metrics"][name] == row  # Paired control copied, never regenerated.
    with (root / "statistics.csv").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Output bytes"):
        w.results(pilot)


def test_parent_spec_and_block_corruption_fail_closed(pilot, tmp_path):
    bad = copy.deepcopy(pilot)
    bad["population"]["jets"] += 1
    with pytest.raises(ValueError, match="population"):
        c.validate_spec(with_content_hash(bad))
    record = copy.deepcopy(pilot["inputs"][0])
    block = tmp_path / "corrupt.npz"
    block.write_bytes(Path(record["block"]["path"]).read_bytes()+b"corrupt")
    record["block"]["path"] = str(block)
    with pytest.raises(ValueError, match="checksum"):
        list(inputs.read_rows(record))
    with np.load(pilot["inputs"][0]["block"]["path"], allow_pickle=False) as data:
        packed = {k: data[k] for k in data.files}
    packed["identity"] = np.array(["wrong"]*len(packed["identity"]))
    np.savez(block, **packed)
    record["block"] = reference(block)
    with pytest.raises(ValueError, match="identities"):
        list(inputs.read_rows(record))
    parent_bad = load_json(pilot["parent_spec"]["path"])
    parent_bad["recipe"]["drop_probability"] = .9
    parent_bad["recipe"] = with_content_hash(parent_bad["recipe"])
    path = tmp_path / "parent.json"
    write_immutable_json(path, with_content_hash(parent_bad))
    # A moved or edited spec cannot quietly authenticate as the same parent.
    with pytest.raises(ValueError):
        inputs.authenticate(reference(path), pilot["parent_receipt"])


def test_exact_submit_dry_default_duplicate_guard(pilot, monkeypatch):
    calls = []
    def fake(argv, **kw):
        calls.append(argv)
        assert not any(k.startswith(("SBATCH_", "SLURM_")) for k in kw["env"])
        return subprocess.CompletedProcess(argv, 0, "123456\n", "")
    monkeypatch.setattr(c.subprocess, "run", fake)
    assert c.submit(pilot)["dry_run"] and not calls
    p = c.plan(pilot)
    with pytest.raises(PermissionError):
        c.submit(pilot, execute=True, reviewed_hash="wrong", authorization=c.AUTHORIZATION)
    result = c.submit(pilot, execute=True, reviewed_hash=p["content_hash"], authorization=c.AUTHORIZATION)
    assert result["job"] == "123456" and len(calls) == 2
    assert "--test-only" in calls[0] and "--test-only" not in calls[1]
    assert not any(v.startswith(("--gres", "--gpus")) for v in calls[1])
    assert c.submit(pilot, execute=True, reviewed_hash=p["content_hash"], authorization=c.AUTHORIZATION) == result
    assert len(calls) == 2


def test_ambiguous_submit_lock_and_scope(pilot, monkeypatch):
    bad = with_content_hash(dict(pilot, final_test_accessed=True))
    with pytest.raises(PermissionError):
        c.validate_spec(bad)
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "unclear", ""))
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit(pilot, execute=True, reviewed_hash=c.plan(pilot)["content_hash"], authorization=c.AUTHORIZATION)
    assert (Path(pilot["root"]) / "submission.lock").exists()
    with pytest.raises(FileExistsError):
        c.submit(pilot, execute=True, reviewed_hash=c.plan(pilot)["content_hash"], authorization=c.AUTHORIZATION)


def test_source_recipe_resource_and_plan_guards(pilot, monkeypatch):
    bad = copy.deepcopy(pilot)
    bad["recipe"]["target_mean"] = 37.
    with pytest.raises(ValueError, match="recipe"):
        c.validate_spec(with_content_hash(bad))
    monkeypatch.setattr(c, "source", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="Source"):
        c.validate_spec(pilot)
    monkeypatch.setattr(c, "source", lambda *a, **kw: pilot["source"])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="allocated"):
        w.run(pilot, workers=3)
    path = Path(pilot["root"]) / "command_plan.json"
    p = load_json(path)
    p["argv"].append("--bad")
    import json
    path.write_text(json.dumps(with_content_hash(p)))
    with pytest.raises(ValueError, match="plan differs"):
        c.submit(pilot)


def test_queue_worker_environment():
    shell = (c.PROJECT / "sbatch/run_jetclass2_literature_proxy_count38.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_literature_proxy_count38.py"' in shell
    assert "PYTHONNOUSERSITE=1" in shell and "LD_LIBRARY_PATH" in shell
    helper = (c.PROJECT / "scripts/queue_jetclass2_literature_proxy_count38.sh").read_text()
    assert c.AUTHORIZATION in helper and "--reviewed-hash" in helper


def test_new_source_files_are_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(old, "source", lambda *a, **kw: dict(commit="a"*40, file_sha256={"v1": "original"}))
    requested = []
    monkeypatch.setattr(old, "git", lambda project, *args: requested.extend(args))
    package = tmp_path / "src/hlt_classification/literature_proxy_v2"
    package.mkdir(parents=True)
    (package / "kernel.py").write_text("# new kernel")
    for name in ("scripts/jetclass2_literature_proxy_count38.py", "scripts/queue_jetclass2_literature_proxy_count38.sh",
                 "sbatch/run_jetclass2_literature_proxy_count38.sh", "docs/plans/JETCLASS2_LITERATURE_PROXY_COUNT38_PILOT_PLAN.md",
                 "docs/contracts/JETCLASS2_LITERATURE_PROXY_COUNT38.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    source = c.source(tmp_path, "a"*40)
    assert source["file_sha256"]["v1"] == "original"
    assert "src/hlt_classification/literature_proxy_v2/kernel.py" in requested
    assert len(source["file_sha256"]) == 7


def test_offsets_and_trace_tampering(pilot, tmp_path):
    record = copy.deepcopy(pilot["inputs"][0])
    with np.load(record["block"]["path"], allow_pickle=False) as data:
        packed = {key: data[key] for key in data.files}
    packed["OFFLINE_offsets"] = packed["OFFLINE_offsets"].astype(float)+.5
    path = tmp_path / "offsets.npz"
    np.savez(path, **packed)
    record["block"] = reference(path)
    with pytest.raises(ValueError, match="offsets"):
        list(inputs.read_rows(record))
    record = copy.deepcopy(pilot["inputs"][0])
    trace = load_json(record["lineage"]["path"])
    trace["rows"][0]["identity"] = "different"
    path = tmp_path / "trace.json"
    write_immutable_json(path, trace)
    record["lineage"] = reference(path)
    with pytest.raises(ValueError, match="identity"):
        list(inputs.read_rows(record))
