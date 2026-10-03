"""Native synthetic ROOT integration and submission guards; no Slurm mutations."""
import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import uproot

from test_jetclass2_delphes import snapshot
from test_jetclass2_delphes_split_registry import registered
from hlt_classification.jetclass2_delphes.split_registry import select_profile
from hlt_classification.literature_proxy import campaign as c, population as p, worker as w
from hlt_classification.literature_proxy.contracts import artifact, load_json, write_immutable_json, with_content_hash


@pytest.fixture
def pilot(registered, tmp_path, monkeypatch):
    data, inventory, _, registry = registered
    profile = select_profile(registry, inventory, "TRAIN_44")
    ip, pp = tmp_path / "inventory.json", tmp_path / "profile.json"
    write_immutable_json(ip, inventory)
    write_immutable_json(pp, profile)
    src = dict(commit="a"*40, file_sha256={"stub": "b"*64})
    monkeypatch.setattr(c, "source", lambda *args, **kwargs: src)
    spec = c.create(project=c.PROJECT, commit="a"*40, data_root=data, inventory=ip,
                    profile=pp, root=tmp_path / "pilot", count=12)
    return spec, inventory, profile


def test_native_train_only_reader_and_chunks(pilot, monkeypatch):
    spec, inventory, profile = pilot
    selected = spec["population"]["files"]
    calls = []
    original = uproot.behaviors.TBranch.HasBranches.arrays

    def spy(self, expressions=None, *args, **kwargs):
        calls.append(tuple(expressions))
        assert set(expressions) == set(p.BRANCHES)
        assert not any(n.startswith("hlt_") or n == "jet_label" for n in expressions)
        return original(self, expressions, *args, **kwargs)

    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", spy)
    rows = [r for f in selected for r in p.read_file(spec["data_root"], inventory, profile, f, step=2)]
    other = [r for f in selected for r in p.read_file(spec["data_root"], inventory, profile, f, step=7)]
    assert len(rows) == 12 and [r[0] for r in rows] == [r[0] for r in other]
    assert calls
    assert all(np.array_equal(a[1].p4, b[1].p4) for a, b in zip(rows, other))
    validation_file = profile["memberships"]["validation"]["files"][0]["path"]
    with pytest.raises(PermissionError, match="training"):
        list(p.read_file(spec["data_root"], inventory, profile, dict(path=validation_file, entries=[0], identities=["x"])))
    bad = dict(selected[0], identities=["bad"] * len(selected[0]["entries"]))
    with pytest.raises(ValueError, match="identities"):
        list(p.read_file(spec["data_root"], inventory, profile, bad))


def test_population_and_science_tampering_fail_closed(pilot):
    spec, inventory, profile = pilot
    assert p.select(inventory, profile, 12) == spec["population"]
    for count in (0, 45, 20_001, True):
        with pytest.raises(ValueError):
            p.select(inventory, profile, count)
    bad = copy.deepcopy(spec)
    bad["recipe"]["drop_probability"] = .9
    bad["recipe"] = with_content_hash(bad["recipe"])
    with pytest.raises(ValueError, match="recipe"):
        c.validate_spec(with_content_hash(bad))
    bad = dict(spec, final_test_accessed=True)
    with pytest.raises(PermissionError):
        c.validate_spec(with_content_hash(bad))
    bad = copy.deepcopy(spec)
    bad["population"]["files"][0]["entries"][0] += 1
    bad["population"] = with_content_hash(bad["population"])
    with pytest.raises(ValueError, match="population"):
        c.validate_spec(with_content_hash(bad))


def test_source_corruption_before_and_after_read(pilot):
    spec, inventory, profile = pilot
    selected = spec["population"]["files"][0]
    path = Path(spec["data_root"]) / selected["path"]
    iterator = iter(p.read_file(spec["data_root"], inventory, profile, selected))
    next(iterator)
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed during"):
        list(iterator)
    with pytest.raises(ValueError, match="ROOT content differs"):
        list(p.read_file(spec["data_root"], inventory, profile, selected))


def test_dry_submit_exact_authorization_and_idempotence(pilot, monkeypatch):
    spec, _, _ = pilot
    calls = []

    def fake(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "12345\n", "")

    monkeypatch.setattr(c.subprocess, "run", fake)
    assert c.submit(spec)["dry_run"] and calls == []
    plan = c.plan(spec)
    with pytest.raises(PermissionError):
        c.submit(spec, execute=True, reviewed_hash="wrong", authorization=c.AUTHORIZATION)
    out = c.submit(spec, execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION)
    assert out["job"] == "12345" and len(calls) == 2
    assert "--test-only" in calls[0] and "--test-only" not in calls[1]
    assert not any(a.startswith("--gres") or a.startswith("--gpus") for a in calls[1])
    assert c.submit(spec, execute=True, reviewed_hash=plan["content_hash"], authorization=c.AUTHORIZATION) == out
    assert len(calls) == 2


def test_failed_or_ambiguous_submit_preserves_lock(pilot, monkeypatch):
    spec, _, _ = pilot
    monkeypatch.setattr(c.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "unclear", ""))
    with pytest.raises(RuntimeError, match="Ambiguous"):
        c.submit(spec, execute=True, reviewed_hash=c.plan(spec)["content_hash"], authorization=c.AUTHORIZATION)
    assert (Path(spec["root"]) / "submission.lock").exists()
    with pytest.raises(FileExistsError):
        c.submit(spec, execute=True, reviewed_hash=c.plan(spec)["content_hash"], authorization=c.AUTHORIZATION)


def test_full_native_miniature_publishes_blocks_plots_receipt(pilot, monkeypatch):
    spec, inventory, profile = pilot
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    report = w.run(spec, workers=2)
    assert report["jets"] == 12 and report["replay"]["exact"]
    assert not report["final_test_accessed"] and not report["production_qualified"]
    root = Path(spec["root"])
    assert w.results(spec) == report and w.run(spec, workers=2) == report
    assert (root / "overlays.pdf").read_bytes().startswith(b"%PDF")
    assert "q99_bin_approx" in (root / "statistics.csv").read_text()
    rows = 0
    for file in sorted((root / "blocks").glob("*.npz")):
        with np.load(file, allow_pickle=False) as data:
            rows += len(data["identity"])
            assert not any("ancestry" in name or "label" in name for name in data.files)
            for side in w.SIDES:
                offsets = data[f"{side}_offsets"]
                assert offsets[0] == 0 and len(offsets) == len(data["identity"]) + 1
                assert offsets[-1] == len(data[f"{side}_p4"])
    assert rows == 12
    selected = spec["population"]["files"][0]
    assert Path(w.file_task((spec, inventory, profile, selected))).is_file()
    with (root / "statistics.csv").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="Output bytes"):
        w.results(spec)


def test_clean_pushed_source_guard(tmp_path):
    # Actual git repo, not a mocked HEAD, covers tracked dirt and untracked shadow code.
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    files = ["scripts/jetclass2_literature_proxy.py", "sbatch/run_jetclass2_literature_proxy_pilot.sh",
             "docs/plans/JETCLASS2_LITERATURE_PROXY_PILOT_PLAN.md", "docs/contracts/JETCLASS2_LITERATURE_PROXY.md",
             "src/hlt_classification/literature_proxy/kernel.py", "src/hlt_classification/cms2jc2_response/bridge.py",
             "src/hlt_classification/data/cache_contracts.py", "src/hlt_classification/jetclass2_delphes/split_registry.py",
             "src/hlt_classification/jetclass2_delphes/inventory.py"]
    for name in files:
        pth = tmp_path / name
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text("test\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    commit = git("rev-parse", "HEAD")
    with pytest.raises(PermissionError, match="origin/main"):
        c.source(tmp_path, commit, pushed=True)
    git("update-ref", "refs/remotes/origin/main", commit)
    assert c.source(tmp_path, commit, pushed=True)["commit"] == commit
    (tmp_path / files[0]).write_text("changed\n")
    with pytest.raises(ValueError, match="clean"):
        c.source(tmp_path, commit)


def test_resources_and_worker_scope(pilot, monkeypatch):
    spec, _, _ = pilot
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(ValueError, match="allocated"):
        w.run(spec, workers=3)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OMP"):
        w.run(spec, workers=1)
    worker = (c.PROJECT / "sbatch/run_jetclass2_literature_proxy_pilot.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/jetclass2_literature_proxy.py"' in worker
    assert "PYTHONNOUSERSITE=1" in worker and "LD_LIBRARY_PATH" in worker


def test_per_file_reuse_authenticates_blocks(pilot):
    spec, inventory, profile = pilot
    selected = spec["population"]["files"][0]
    path = Path(w.file_task((spec, inventory, profile, selected)))
    report = load_json(path)
    block = next(name for name in report["outputs"] if name.endswith(".npz"))
    with (Path(spec["root"]) / block).open("ab") as handle:
        handle.write(b"not reusable")
    with pytest.raises(ValueError, match="corrupt"):
        w.file_task((spec, inventory, profile, selected))


def test_changed_saved_plan_rejected(pilot):
    spec, _, _ = pilot
    path = Path(spec["root"]) / "command_plan.json"
    plan = load_json(path)
    plan["argv"][plan["argv"].index("--cpus-per-task=16")] = "--cpus-per-task=72"
    import json
    path.write_text(json.dumps(with_content_hash(plan)))
    with pytest.raises(ValueError, match="plan differs"):
        c.submit(spec)
