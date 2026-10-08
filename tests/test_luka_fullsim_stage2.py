"""Synthetic ROOT stage-2 integration; not real SPORC acceptance evidence."""
from copy import deepcopy
from pathlib import Path
import subprocess

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.data.cache_contracts import (
    atomic_publish_bytes, deterministic_npz_bytes, sha256_file, with_content_hash, write_immutable_json,
)
from hlt_classification.luka_fullsim import inventory as inv, stage2 as s
from hlt_classification.luka_fullsim.contracts import SOURCE, artifact
from hlt_classification.luka_fullsim.splits import build_splits
from hlt_classification.luka_fullsim.particles import FIELDS, ParticleReader, physical, _ranges
from hlt_classification.luka_fullsim.particle_audit import Moments, scan
from hlt_classification.luka_fullsim.stage2_cache import prepare_cache
from hlt_classification.jetclass2_delphes.schema import SIGNAL_IDS
from hlt_classification.provenance import capture_source_snapshot


def raw_particles(n=2):
    a = np.zeros((n, len(FIELDS)))
    a[:, 0] = np.arange(n)+1.
    a[:, 3] = a[:, 0]+.1
    a[:, 4:6] = 1
    a[:, 10:14] = [.02, -.03, .004, .005]
    return a


def config(audit="a"*64, zero="error_only_unavailable", unit="mm"):
    return s.conventions(audit_sha256=audit, offline_unit=unit, hlt_unit=unit,
                         zero_error=zero, authority="operator_provisional",
                         evidence="Synthetic fixture, explicitly NOT producer confirmation")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    container = tmp_path_factory.mktemp("luka_particles")
    for source in ("train_higgs2p", "train_qcd"):
        for i in range(6):
            path = container / "raw/jetclass2" / source / "fullsim_offline+hlt" / f"{i}.root"
            path.parent.mkdir(parents=True, exist_ok=True)
            labels = [161]*30 if source == "train_qcd" else list(SIGNAL_IDS)*2
            n = len(labels)
            arrays = dict(jet_label=np.asarray(labels, np.int32), hlt_matched=np.ones(n, bool),
                jet_nparticles=np.full(n, 3, np.int32), hlt_jet_nparticles=np.full(n, 2, np.int32),
                jet_pt=np.full(n, 250, np.float32))
            schema = dict(jet_label="int32", hlt_matched="bool", jet_nparticles="int32",
                          hlt_jet_nparticles="int32", jet_pt="float32")
            for prefix, count in (("part_", 3), ("hlt_part_", 2)):
                raw = raw_particles(count)
                raw[0, 12] = 0  # valid d0 but explicitly unavailable error under fixture convention
                for col, field in enumerate(FIELDS):
                    arrays[prefix+field] = ak.Array([raw[:, col].tolist() for _ in labels])
                    schema[prefix+field] = "var * float32"
            with uproot.recreate(path) as handle:
                handle.mktree("tree", schema)
                handle["tree"].extend(arrays)
    (container / "source.txt").write_text(SOURCE+"\n")
    (container / "source.sha256").write_text("".join(
        f"{sha256_file(path)}  {path.relative_to(container/'raw').as_posix()}\n"
        for path in sorted((container / "raw").rglob("*.root"))))
    inventory = inv.build_inventory(container, expected_files=12)
    splits = build_splits(inventory, train=66, validation=33)
    return container, inventory, splits


@pytest.fixture(scope="module")
def prepared(dataset, tmp_path_factory):
    container, inventory, splits = dataset
    project = tmp_path_factory.mktemp("luka_source")
    (project / "tracked.txt").write_text("source fixture")
    for command in (["init"], ["add", "tracked.txt"], ["-c", "user.name=Fixture", "-c",
                     "user.email=fixture@example.invalid", "commit", "-m", "fixture"]):
        subprocess.run(["git", "-C", str(project), *command], check=True, capture_output=True)
    source = capture_source_snapshot(project, require_clean=True)
    foundation = artifact("FOUNDATION", counts=splits["selected_counts"], input_container=str(container))
    report = artifact("PARTICLE_AUDIT", parents=dict(foundation=foundation["content_hash"], source=source["content_hash"]),
                      source_snapshot=source, **scan(container, inventory, splits))
    cfg = config(report["content_hash"])
    root = tmp_path_factory.mktemp("luka_prepared")
    products = s.prepare_rows(container, inventory, splits, cfg, root)
    p = artifact("PREPARED", parents=dict(foundation=foundation["content_hash"], audit=report["content_hash"],
        conventions=cfg["content_hash"], source=source["content_hash"]), source_snapshot=source,
        foundation_root=str(root / "unused_fixture_locator"), audit=report, conventions=cfg,
        counts=dict(train=66, validation=33), **s.contracts(), **products,
        science_authorized=False, final_test_accessed=False)
    write_immutable_json(root / "prepared.json", p)
    return root, p, foundation, inventory, splits


def bind(prepared, monkeypatch):
    root, p, f, inv_, splits = prepared
    monkeypatch.setattr(s, "load_foundation", lambda _: (f, inv_, splits))
    return root, p


def test_selected_reader_order_no_test_or_unused_ranges(dataset, monkeypatch):
    container, inventory, splits = dataset
    calls = []
    original = uproot.behaviors.TBranch.HasBranches.arrays
    allowed = {}
    for m in splits["memberships"]["train"]:
        allowed[inventory["files"][m["file_index"]]["path"]] = set(sum(m["entries_by_class"], []))
    def guarded(tree, expressions, *args, **kwargs):
        path = Path(tree.file.file_path).relative_to(container / "raw").as_posix()
        assert path in allowed
        assert set(range(kwargs["entry_start"], kwargs["entry_stop"])) <= allowed[path]
        assert not any(b.startswith("part_") or b == "jet_nparticles" for b in expressions)
        calls.append(path)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", guarded)
    jets = list(ParticleReader(container, inventory, splits, role="train", chunk=3))
    assert len(jets) == 66 and calls and all(j.offline is None for j in jets)
    assert [(j.file_index, j.entry) for j in jets] == sorted((j.file_index, j.entry) for j in jets)
    assert np.bincount([j.label for j in jets], minlength=11).tolist() == splits["quotas"]["train"]
    with pytest.raises(PermissionError):
        ParticleReader("absent", {}, {}, role="final_test")
    with pytest.raises(PermissionError):
        prepare_cache("absent", role="final_test", coordinate="D000", max_ram_bytes=1)


def test_ranges_and_moments():
    assert list(_ranges([1, 2, 3, 7, 8], 2)) == [(1, 3), (3, 4), (7, 9)]
    m = Moments()
    m.add([0., -2., np.nan])
    m.add([2., np.inf])
    r = m.report()
    assert r["finite"] == 3 and r["nonfinite"] == 2 and r["zero"] == r["negative"] == 1
    assert r["mean"] == 0 and r["sd"] == pytest.approx(np.std([0, -2, 2]))
    assert sum(r["abs_bin_counts"]) == 3


def test_audit_zero_errors_not_inferred(dataset):
    r = scan(*dataset)
    assert r["counts"] == dict(train=66, validation=33)
    assert r["units"] == "stored_unconfirmed_no_conversion"
    assert r["statistics"]["train"]["hlt/charged/d0err"]["zero"] == 66
    assert r["statistics"]["train"]["offline/multiplicity"]["mean"] == 3
    assert not r["scientific_admission"] and not r["final_test_accessed"]


def test_jagged_failure_and_raw_diagnostics_do_not_fix_values(monkeypatch):
    from hlt_classification.luka_fullsim import particle_audit as audit_module
    from hlt_classification.luka_fullsim.particles import _matrix, RawJet
    arrays = {"hlt_part_"+f: ak.Array([[1., 2.]]) for f in FIELDS}
    arrays["hlt_part_dzerr"] = ak.Array([[1.]])
    with pytest.raises(ValueError, match="count mismatch"):
        _matrix(arrays, "hlt_part_", 0, 2)
    a = raw_particles()
    a[0, 12], a[1, 13] = -1, np.nan
    row = RawJet("1"*64, 0, 0, 0, a, a.copy())
    monkeypatch.setattr(audit_module, "ParticleReader", lambda *args, **kwargs: iter([row]))
    r = scan(None, None, None)
    for role in ("train", "validation"):
        assert r["issues"][role]["hlt"]["negative_error"] == 1
        assert r["issues"][role]["hlt"]["nonfinite_required"] == 1
        assert r["statistics"][role]["hlt/charged/dzerr"]["nonfinite"] == 1
    assert a[0, 12] == -1 and np.isnan(a[1, 13])


def test_reader_rejects_changed_source_before_particle_arrays(dataset, monkeypatch):
    from hlt_classification.luka_fullsim import particles as module
    monkeypatch.setattr(module, "sha256_file", lambda path: "f"*64)
    with pytest.raises(ValueError, match="checksum"):
        list(ParticleReader(*dataset, role="train"))


def test_physical_units_error_policy_and_neutral_applicability():
    raw = raw_particles()
    raw[0, 12] = 0
    mm = physical(raw, config(), "hlt")
    cm = physical(raw, config(unit="cm"), "offline")
    assert np.array_equal(cm.tracking, mm.tracking*10)
    assert mm.valid[0].tolist() == [True, True, False, True]
    missing = physical(raw, config(zero="value_and_error_unavailable"), "hlt")
    assert missing.valid[0].tolist() == [False, True, False, True]
    assert missing.tracking[0, 0] == 0 and mm.tracking[0, 1] < 0
    raw[0, 4:10] = [0, 0, 1, 0, 0, 0]
    neutral = physical(raw, config(), "hlt")
    assert not neutral.valid[0].any() and not neutral.tracking[0].any()
    raw[1, 5:10] = 0
    assert physical(raw, config(), "hlt").category[1] == 5


@pytest.mark.parametrize("col,value", [(12, -1), (10, np.nan), (4, 2), (6, 1), (0, 0), (3, -.1), (3, .2)])
def test_invalid_physics_fails_without_row_drops(col, value):
    raw = raw_particles()
    raw[0, col] = value
    with pytest.raises(ValueError):
        physical(raw, config(), "offline")


@pytest.mark.parametrize("key,value", [("zero_error", "guess"), ("evidence", ""),
    ("length_units", {"hlt": "mm", "offline": "unknown"}), ("authority", "automatically_inferred"),
    ("other_sentinels", "guess"), ("final_test_accessed", True)])
def test_conventions_fail_closed_even_rehashed(key, value):
    bad = config()
    bad[key] = value
    with pytest.raises(ValueError):
        s.validate_conventions(with_content_hash(bad))


def test_all_views_replay_and_hlt_only_independence(prepared, monkeypatch):
    root, p = bind(prepared, monkeypatch)
    original = uproot.behaviors.TBranch.HasBranches.arrays
    for coord in s.COORDINATES:
        cache = prepare_cache(root, role="train", coordinate=coord, max_ram_bytes=2**30)
        assert len(cache) == 66
        assert cache.batch(np.arange(3))["features"].shape[:2] == (3, 17)
    def hlt_only(tree, expressions, *args, **kwargs):
        assert not any(b.startswith("part_") for b in expressions)
        return original(tree, expressions, *args, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", hlt_only)
    cache = prepare_cache(root, role="train", coordinate="D000", max_ram_bytes=2**30)
    assert cache.nbytes < 2**30 and p["summaries"]["train"]["D000"]["particles"] == 132
    with pytest.raises(MemoryError):
        prepare_cache(root, role="train", coordinate="D000", max_ram_bytes=1)


def test_paired_batch_and_kernel_population(prepared, monkeypatch):
    import torch
    from torch import nn
    from hlt_classification.context_fusion.inputs import PairedCache
    from hlt_classification.jetclass2_delphes.runner import train_kernel
    root, _ = bind(prepared, monkeypatch)
    cache = lambda role, coordinate: prepare_cache(root, role=role, coordinate=coordinate, max_ram_bytes=2**30)
    train = PairedCache(cache("train", "U050"), cache("train", "U000"))
    val = PairedCache(cache("validation", "U050"), cache("validation", "U000"))
    assert train.batch(np.arange(3))["features"].shape[:3] == (3, 2, 17)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(17, 11)
        def forward(self, features, vectors, mask):
            return self.fc((features[:, 0]*mask[:, 0]).sum(-1)/mask[:, 0].sum(-1))
    q = np.full((66, 11), 1/11, np.float32)
    report, state = train_kernel(Tiny(), train, val, node=dict(node_id="fixture", coordinate="U050",
        teacher="fixture", sampler_seed=17), device="cpu", acceptance_passes=1,
        teacher_probabilities=q, teacher_identities=train.identities)
    assert report["acceptance_only"] and not report["scientific_fit"] and state
    with pytest.raises(ValueError, match="identity"):
        train_kernel(Tiny(), train, val, node=dict(node_id="fixture", coordinate="U050",
            teacher="fixture", sampler_seed=17), device="cpu", acceptance_passes=1,
            teacher_probabilities=q, teacher_identities=train.identities[::-1])


def test_artifact_corruption_and_semantic_tamper(prepared, monkeypatch, tmp_path):
    import shutil
    root, p = bind(prepared, monkeypatch)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    bad = deepcopy(p)
    bad["conventions"]["parents"]["audit"] = "b"*64
    bad["conventions"] = with_content_hash(bad["conventions"])
    bad["parents"]["conventions"] = bad["conventions"]["content_hash"]
    (copy / "prepared.json").write_text(__import__("json").dumps(with_content_hash(bad)))
    with pytest.raises(ValueError, match="lineage"):
        s.load_prepared(copy)
    (copy / "prepared.json").write_bytes((root / "prepared.json").read_bytes())
    ref = p["assignments"]["train"][0]
    (copy / ref["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        s.load_prepared(copy)


def test_rehashed_wrong_assignment_join_rejected(prepared, monkeypatch, tmp_path):
    import shutil
    import json
    root, p = bind(prepared, monkeypatch)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    bad = deepcopy(p)
    record = bad["assignments"]["train"][0]
    a = s.load_assignment(root, record)
    a["identities"][0, 0] ^= 1
    path = copy / record["path"]
    path.write_bytes(deterministic_npz_bytes(a))
    record.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    (copy / "prepared.json").write_text(json.dumps(with_content_hash(bad)))
    with pytest.raises(ValueError, match="identity/label"):
        prepare_cache(copy, role="train", coordinate="D000", max_ram_bytes=2**30)


def test_no_overlap_or_overwrite(tmp_path):
    protected = tmp_path / "source"
    protected.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        s.fresh(protected / "child", protected)
    out = s.fresh(tmp_path / "output", protected)
    with pytest.raises(FileExistsError):
        s.fresh(out, protected)


def test_audit_prepare_wrappers_publish_completion_last(dataset, prepared, monkeypatch, tmp_path):
    container, inventory, splits = dataset
    _, p, foundation, _, _ = prepared
    checks = []
    monkeypatch.setattr(s, "source", lambda *args: p["source_snapshot"])
    monkeypatch.setattr(s, "load_foundation", lambda *args: (foundation, inventory, splits))
    monkeypatch.setattr(s, "validate_source_snapshot", lambda *args, **kwargs: checks.append(kwargs["require_clean"]))
    parent = tmp_path / "foundation"
    project = tmp_path / "project"
    audit_root, output = tmp_path / "audit", tmp_path / "prepared"
    report = s.audit(parent, audit_root, project=project, expected_commit="a"*40)
    assert (audit_root / "audit.json").is_file()
    cfg = tmp_path / "conventions.json"
    write_immutable_json(cfg, config(report["content_hash"]))
    result = s.prepare(parent, audit_root / "audit.json", cfg, output,
                       project=project, expected_commit="a"*40)
    assert (output / "prepared.json").is_file() and checks == [True, True]
    assert s.load_prepared(output)[0] == result
    with pytest.raises(FileExistsError):
        s.audit(parent, audit_root, project=project, expected_commit="a"*40)
    # A source change at final validation must not publish an admission marker.
    def drift(*args, **kwargs):
        raise ValueError("source drift")
    monkeypatch.setattr(s, "validate_source_snapshot", drift)
    interrupted = tmp_path / "interrupted_audit"
    with pytest.raises(ValueError, match="source drift"):
        s.audit(parent, interrupted, project=project, expected_commit="a"*40)
    assert interrupted.is_dir() and not (interrupted / "audit.json").exists()


def test_worker_and_no_submission_surface():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "sbatch/run_luka_fullsim_stage2.sh").read_text()
    assert '"${PROJECT_DIR}/scripts/luka_fullsim_stage2.py"' in worker
    assert "PYTHONNOUSERSITE=1" in worker and 'LD_LIBRARY_PATH="${CONDA_PREFIX}/lib' in worker
    cli = (root / "scripts/luka_fullsim_stage2.py").read_text()
    assert "scancel" not in cli and "sbatch" not in cli
    gpu = (root / "src/hlt_classification/luka_fullsim/preflight.py").read_text()
    assert "acceptance_passes=1" in gpu and "production_submission_authorized=False" in gpu
    assert "train=100000, validation=50000" in gpu
