"""Real synthetic ROOT, deterministic splits and fail-closed publication tests."""
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.data.cache_contracts import (
    sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.jetclass2_delphes.schema import PARTICLE_FIELDS, SIGNAL_IDS
from hlt_classification.luka_fullsim import foundation as f
from hlt_classification.luka_fullsim import inventory as inv
from hlt_classification.luka_fullsim.contracts import CLASS_NAMES, SCALARS, SOURCE, row_identity
from hlt_classification.luka_fullsim.splits import (
    build_splits, ordinary_rows, proportional_quotas, validate_splits,
)
from hlt_classification.provenance import capture_source_snapshot


def write_root(path, *, qcd=False, cycles=1, extra=False):
    labels = ([161] * 30 if qcd else list(SIGNAL_IDS) * 2) + [161, 0, 0, 0, 0]
    size = len(labels)
    branches = dict(jet_label="int32", hlt_matched="bool", jet_nparticles="int32",
                    hlt_jet_nparticles="int32", jet_pt="float32", hlt_jet_pt="float32")
    arrays = dict(jet_label=np.array(labels, np.int32), hlt_matched=np.ones(size, bool),
                  jet_nparticles=np.full(size, 2, np.int32),
                  hlt_jet_nparticles=np.full(size, 1, np.int32),
                  jet_pt=np.full(size, 201., np.float32),
                  hlt_jet_pt=np.full(size, 150., np.float32))  # Must NOT be cut.
    arrays["jet_pt"][-4:-2] = [200., 199.]
    arrays["hlt_matched"][-2] = False
    arrays["hlt_jet_nparticles"][-1] = 0
    for prefix in ("part_", "hlt_part_"):
        for field in PARTICLE_FIELDS:
            branches[prefix + field] = "var * float32"
            # Deliberately invalid particles: stage 1 must never decode these.
            arrays[prefix + field] = ak.Array([[float("nan")] for _ in labels])
    if extra:
        branches["extra"] = "float32"
        arrays["extra"] = np.zeros(size, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as handle:
        for _ in range(cycles):
            handle.mktree("tree", branches)
            handle["tree"].extend(arrays)


def manifest(container):
    (container / "source.txt").write_text(SOURCE + "\n")
    paths = sorted((container / "raw").rglob("*.root"))
    (container / "source.sha256").write_text("".join(
        f"{sha256_file(p)}  ./{p.relative_to(container / 'raw').as_posix()}\n" for p in paths))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    container = tmp_path_factory.mktemp("luka")
    for source in ("train_higgs2p", "train_qcd"):
        for i in range(6):
            write_root(container / "raw/jetclass2" / source / "fullsim_offline+hlt" / f"ntuple_{i}.root",
                       qcd=source == "train_qcd", cycles=2)
    manifest(container)
    inventory = inv.build_inventory(container, expected_files=12, step_size=7)
    splits = build_splits(inventory, train=66, validation=33)
    return container, inventory, splits


def test_scalar_selection_boundary_and_sources():
    batch = dict(jet_label=np.array([161, 187, 0, 1, 2, 3, 9]),
                 hlt_matched=np.array([True, True, True, True, False, True, True]),
                 jet_nparticles=np.array([1, 1, 1, 1, 1, 1, 1]),
                 hlt_jet_nparticles=np.array([1, 1, 1, 1, 1, 0, 1]),
                 jet_pt=np.array([201., 201., 200., 200.01, 201., 201., 199.]))
    keep, labels, flow = inv.eligible(batch, "train_higgs2p")
    assert np.flatnonzero(keep).tolist() == [3]
    assert labels.tolist() == [0, 0, 1, 2, 3, 4, 5]
    assert flow == dict(excluded_qcd_source=2, excluded_unmatched=1, excluded_empty=1,
                        excluded_offline_pt=2, selected=1)
    assert np.flatnonzero(inv.eligible(batch, "train_qcd")[0]).tolist() == [0, 1, 3]


@pytest.mark.parametrize("field,value", [("jet_pt", np.nan), ("jet_pt", np.inf),
                                        ("jet_label", 999), ("hlt_jet_nparticles", -1)])
def test_bad_required_scalars_fail_even_if_unmatched(field, value):
    batch = dict(jet_label=np.array([0]), hlt_matched=np.array([False]),
                 jet_nparticles=np.array([1]), hlt_jet_nparticles=np.array([1]), jet_pt=np.array([201.]))
    batch[field][0] = value
    with pytest.raises(ValueError):
        inv.eligible(batch, "train_higgs2p")


def test_natural_quotas_exact_not_balanced():
    counts = [276333, 12953, 12868, 12967, 13040, 13733, 6271, 6116, 4948, 4910, 9990]
    for total in (100000, 50000):
        quota = proportional_quotas(counts, total)
        assert sum(quota) == total
        assert all(abs(q - total * n / sum(counts)) < 1 for q, n in zip(quota, counts))
        assert quota[0] > .73 * total
    with pytest.raises(ValueError, match="zero class quota"):
        proportional_quotas([100000] + [1] * 10, 100)


def test_exact_replay_chunk_invariance_latest_cycle_and_no_particles(snapshot, monkeypatch):
    container, inventory, splits = snapshot
    original = uproot.behaviors.TBranch.HasBranches.iterate
    observed = []
    def guarded(self, expressions=None, *args, **kwargs):
        assert tuple(expressions) == SCALARS
        observed.append(tuple(expressions))
        return original(self, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "iterate", guarded)
    replay = inv.build_inventory(container, expected_files=12, step_size=1000)
    assert replay == inventory and observed
    assert all(r["tree_key"] == "tree;2" for r in inventory["files"])
    assert inventory["raw_entries"] == 6 * 25 + 6 * 35
    assert inventory["selected_class_counts"] == [186] + [12] * 10
    assert build_splits(replay, train=66, validation=33) == splits
    assert inventory["particle_arrays_decoded"] is False


def test_pair_keys_order_counts_and_sealing(snapshot):
    _, inventory, splits = snapshot
    grouped = {role: {g["sha256"] for g in splits["groups"] if g["role"] == role}
               for role in ("train", "validation", "final_test")}
    assert not grouped["train"] & grouped["validation"]
    assert not (grouped["train"] | grouped["validation"]) & grouped["final_test"]
    ids = set()
    for role, size in (("train", 66), ("validation", 33)):
        rows = list(ordinary_rows(inventory, splits, role=role))
        assert len(rows) == size
        assert [(r["file"], r["entry"]) for r in rows] == sorted((r["file"], r["entry"]) for r in rows)
        assert np.bincount([r["label"] for r in rows], minlength=11).tolist() == splits["quotas"][role]
        assert len({r["identity"] for r in rows}) == size
        assert not ids.intersection(r["identity"] for r in rows)
        ids.update(r["identity"] for r in rows)
    with pytest.raises(PermissionError, match="sealed"):
        list(ordinary_rows({}, {}, role="final_test"))  # Reject before any artifact access.
    all_count = sum(inventory["selected_class_counts"])
    unused = sum(sum(x) for x in splits["unused_ordinary_class_counts"].values())
    assert all_count == 66 + 33 + splits["selected_counts"]["final_test"] + unused


def test_identity_ignores_label_role_and_absolute_location(snapshot):
    record = snapshot[1]["files"][0]
    changed = dict(record, source="ignored", entries_by_class=[], absolute_path="elsewhere")
    assert row_identity(record, 4) == row_identity(changed, 4)
    assert row_identity(record, 4) != row_identity(record, 5)
    assert row_identity(record, 4) != row_identity(dict(record, sha256="f" * 64), 4)


@pytest.mark.parametrize("change", ["groups", "quotas", "memberships", "parents", "counts", "test"])
def test_rehashed_split_tampering_rejected(snapshot, change):
    _, inventory, splits = snapshot
    bad = deepcopy(splits)
    if change == "groups":
        bad["groups"][0]["role"] = "forged"
    elif change == "quotas":
        bad["quotas"]["train"][0] += 1
    elif change == "memberships":
        bad["memberships"]["train"][0]["entries_by_class"][0].append(999)
    elif change == "parents":
        bad["parents"]["inventory"] = "a" * 64
    elif change == "counts":
        bad["unused_ordinary_class_counts"]["train"][0] += 1
    else:
        bad["final_test_accessed"] = True
    with pytest.raises(ValueError, match="replay differs"):
        validate_splits(with_content_hash(bad), inventory)


def test_capacity_shortage_does_not_downsize_or_rebalance(snapshot):
    with pytest.raises(ValueError, match="capacity shortage"):
        build_splits(snapshot[1], train=250, validation=33)


@pytest.mark.parametrize("kind", ["checksum", "duplicate", "extra", "missing", "schema", "marker"])
def test_source_failures(snapshot, tmp_path, kind):
    source, _, _ = snapshot
    container = tmp_path / "copy"
    shutil.copytree(source, container)
    paths = sorted((container / "raw").rglob("*.root"))
    if kind == "checksum":
        with paths[0].open("ab") as stream:
            stream.write(b"corruption")
    elif kind == "duplicate":
        shutil.copyfile(paths[0], paths[1])
        manifest(container)
    elif kind == "extra":
        shutil.copyfile(paths[0], paths[0].with_name("extra.root"))
    elif kind == "missing":
        paths[0].unlink()
    elif kind == "schema":
        write_root(paths[0], extra=True)
        manifest(container)
    else:
        (container / "source.txt").write_text("another dataset")
    with pytest.raises(ValueError):
        inv.build_inventory(container, expected_files=12)


@pytest.mark.parametrize("name", ["../bad.root", "/bad.root", "C:/bad.root", "jetclass2\\bad.root"])
def test_manifest_paths_rejected(snapshot, tmp_path, name):
    container = tmp_path / "copy"
    shutil.copytree(snapshot[0], container)
    (container / "source.sha256").write_text("a" * 64 + "  " + name + "\n")
    with pytest.raises(ValueError):
        inv.transfer_manifest(container, 12)


def test_inventory_rehashed_entry_corruption(snapshot):
    bad = deepcopy(snapshot[1])
    entry = bad["files"][0]["entries_by_class"][1][0]
    bad["files"][0]["entries_by_class"][2][0] = entry
    with pytest.raises(ValueError, match="source entries"):
        inv.validate_inventory(with_content_hash(bad))


@pytest.fixture
def clean_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.txt").write_text("fixture implementation")
    def git(*args):
        return subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    git("init")
    git("add", "source.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    return project, capture_source_snapshot(project)


def test_real_publication_relocation_verify_and_immutable_outputs(snapshot, clean_project, tmp_path):
    container, inventory, splits = snapshot
    project, source = clean_project
    output = tmp_path / "foundation"
    output.mkdir()
    result = f._publish(output, container, project, source, inventory, splits)
    assert f.load_foundation(output) == (result, inventory, splits)
    f.print_summary(output)
    relocated = tmp_path / "relocated"
    shutil.copytree(container, relocated)
    assert f.verify(output, container=relocated) == result
    assert result["readiness"] == "scalar_metadata_only_not_training_admission"
    with pytest.raises(FileExistsError):
        write_immutable_json(output / "foundation.json", {"different": True})
    (output / "split_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        f.load_foundation(output)


def test_source_drift_prevents_completion(snapshot, clean_project, tmp_path):
    project, source = clean_project
    (project / "source.txt").write_text("changed")
    output = tmp_path / "incomplete"
    output.mkdir()
    with pytest.raises(ValueError, match="dirty"):
        f._publish(output, snapshot[0], project, source, snapshot[1], snapshot[2])
    assert not (output / "foundation.json").exists()


def test_late_source_corruption_prevents_completion(snapshot, clean_project, tmp_path):
    project, source = clean_project
    container = tmp_path / "changed_source"
    shutil.copytree(snapshot[0], container)
    path = container / "raw" / snapshot[1]["files"][0]["path"]
    with path.open("ab") as stream:
        stream.write(b"changed after scalar scan")
    output = tmp_path / "incomplete"
    output.mkdir()
    with pytest.raises(ValueError, match="checksum"):
        f._publish(output, container, project, source, snapshot[1], snapshot[2])
    assert not (output / "foundation.json").exists()


def test_interrupted_publication_preserves_partial_not_completion(snapshot, clean_project, tmp_path, monkeypatch):
    project, source = clean_project
    output = tmp_path / "incomplete"
    output.mkdir()
    original = f.write_immutable_json
    def interrupted(path, payload):
        if Path(path).name == "split_manifest.json":
            raise RuntimeError("injected interruption")
        return original(path, payload)
    monkeypatch.setattr(f, "write_immutable_json", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        f._publish(output, snapshot[0], project, source, snapshot[1], snapshot[2])
    assert (output / "inventory.json").exists()
    assert not (output / "foundation.json").exists()
    with pytest.raises(FileNotFoundError):
        f.load_foundation(output)


def test_production_build_rejects_existing_output_and_nested_raw(snapshot, clean_project, monkeypatch, tmp_path):
    project, source = clean_project
    monkeypatch.setattr(f, "_source", lambda *a: source)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        f.build(snapshot[0], existing, project=project, expected_commit=source["git_commit"])
    with pytest.raises(ValueError, match="separate"):
        f.build(snapshot[0], snapshot[0] / "outputs", project=project, expected_commit=source["git_commit"])


def test_source_pin_refuses_wrong_commit_and_outside_project(clean_project):
    project, _ = clean_project
    with pytest.raises(ValueError, match="full lowercase"):
        f._source(project, "short")
    with pytest.raises(ValueError, match="outside"):
        f._source(project, "a" * 40)


def test_no_scheduler_or_training_api():
    script = Path(__file__).resolve().parents[1] / "scripts/luka_fullsim_foundation.py"
    text = script.read_text()
    assert 'commands.add_parser("build")' in text
    assert '"--expected-commit"' in text
    assert "sbatch" not in text and "allow_test" not in text


def test_empty_root_file_has_zero_cutflow_not_a_schema_failure(snapshot, tmp_path):
    container = tmp_path / "copy"
    shutil.copytree(snapshot[0], container)
    path = sorted((container / "raw").rglob("*.root"))[0]
    # Preserve the same latest-cycle schema, but leave this file empty.
    with uproot.open(path) as handle:
        names = handle["tree"].keys()
    branches = {name: "var * float32" for name in names if name.startswith(("part_", "hlt_part_"))}
    branches.update(jet_label="int32", hlt_matched="bool", jet_nparticles="int32",
                    hlt_jet_nparticles="int32", jet_pt="float32", hlt_jet_pt="float32")
    with uproot.recreate(path) as handle:
        handle.mktree("tree", branches)
    manifest(container)
    inventory = inv.build_inventory(container, expected_files=12)
    record = inventory["files"][0]
    assert record["entries"] == 0 and set(record["cutflow"]) == set(inv.CUTFLOW)
    assert all(value == 0 for value in record["cutflow"].values())


def test_source_pin_checks_tracked_implementation_and_published_commit(monkeypatch):
    project = Path(f.__file__).resolve().parents[3]
    commit = "a" * 40
    monkeypatch.setattr(f, "capture_source_snapshot", lambda *a, **k: {"git_commit": commit})
    def git_success(argv, **kwargs):
        assert argv[:3] == ["git", "-C", str(project)]
        if "ls-files" in argv:
            return subprocess.CompletedProcess(argv, 0, "\n".join(f.REQUIRED_SOURCE))
        assert argv[3:] == ["merge-base", "--is-ancestor", commit, "origin/main"]
        return subprocess.CompletedProcess(argv, 0, b"")
    monkeypatch.setattr(f.subprocess, "run", git_success)
    assert f._source(project, commit)["git_commit"] == commit
    with pytest.raises(ValueError, match="commit differs"):
        f._source(project, "b" * 40)
    monkeypatch.setattr(f.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 0 if "ls-files" in argv else 1, "\n".join(f.REQUIRED_SOURCE)))
    with pytest.raises(ValueError, match="fetched origin/main"):
        f._source(project, commit)
    monkeypatch.setattr(f.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, ""))
    with pytest.raises(ValueError, match="Commit the new implementation"):
        f._source(project, commit)


def test_sporc_worker_is_metadata_only_absolute_and_isolated():
    project = Path(__file__).resolve().parents[1]
    worker = (project / "sbatch/run_luka_fullsim_foundation.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/luka_fullsim_foundation.py" build' in worker
    assert "PYTHONNOUSERSITE=1" in worker and "PYTHONDONTWRITEBYTECODE=1" in worker
    assert '${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' in worker
    assert "#SBATCH --cpus-per-task=1" in worker and "#SBATCH --mem=8G" in worker
    assert "#SBATCH --no-requeue" in worker
    assert "--gres" not in worker and "--gpus" not in worker and "scancel" not in worker


def test_cli_build_and_verify_with_real_production_data_flow(snapshot, clean_project, tmp_path, monkeypatch):
    # Only source placement and fixture size are substituted: same build/scanner/
    # splitter/publication/verification code. This is not RC/production evidence.
    project, source = clean_project
    monkeypatch.setattr(f, "_source", lambda *a: source)
    monkeypatch.setattr(f, "build_inventory", lambda path, **kw: inv.build_inventory(path, expected_files=12))
    monkeypatch.setattr(f, "build_splits", lambda inventory: build_splits(inventory, train=66, validation=33))
    root = tmp_path / "end_to_end"
    built = f.build(snapshot[0], root, project=project, expected_commit=source["git_commit"])
    assert f.verify(root) == built
    assert built["counts"]["train"] == 66 and built["counts"]["validation"] == 33
    with pytest.raises(FileExistsError):
        f.build(snapshot[0], root, project=project, expected_commit=source["git_commit"])
