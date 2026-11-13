"""Hand-computed occupancy, sealed-role and real CLI/publication checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from hlt_classification.cms_proxy_ladder.contracts import file_ref
from hlt_classification.data.cache_contracts import (
    deterministic_npz_bytes, load_json, with_content_hash,
)
from hlt_classification.jetclass2_delphes.schema import CLASS_NAMES, SIGNAL_IDS
from hlt_classification.luka_fullstat.contracts import SOURCE, split_policy
from hlt_classification.operations import fullstat_pt_audit as audit
from hlt_classification.operations import fullstat_pt_metadata as meta

PROJECT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = with_content_hash(value)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path/"foundation"
    counts = dict(train=440, validation=1, final_test=1, unused=0)
    monkeypatch.setattr(audit, "COUNTS", counts)
    files = []
    quotas = {}
    for i, category in enumerate(("signal", "QCD_PT-200to300")):
        train_labels = np.repeat(np.arange(1, 11), 40) if i == 0 else np.zeros(40, dtype=int)
        labels = np.r_[train_labels, [1, 1]] if i == 0 else train_labels
        # Poison non-TRAIN pT to catch accidental summary/selection leakage.
        pt = np.tile([125., 225.], len(train_labels)//2)
        pt = np.r_[pt, [np.nan, -1.]] if i == 0 else pt
        roles = np.r_[np.zeros(len(train_labels), dtype=np.uint8), [1, 2]] if i == 0 else np.zeros(len(labels), dtype=np.uint8)
        raw = np.array([161 if c == 0 else SIGNAL_IDS[c-1] for c in labels], dtype=np.int32)
        path = root/"memberships"/f"{i:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(deterministic_npz_bytes(dict(labels=labels.astype(np.int8), jet_label=raw,
            roles=roles.astype(np.uint8), jet_pt=pt, particle_payload_trap=np.array([999]))))
        for c in np.unique(train_labels):
            quotas[f"signal:{c:02d}" if c else category] = 40
        files.append(dict(path=f"source{i}.root", category=category, sha256=str(i)*64,
            entries=len(labels), membership=file_ref(path, root=root),
            role_counts={name:int(np.count_nonzero(roles == j)) for j, name in enumerate(audit.ROLES)}))
    foundation = save(root/"foundation.json", dict(contract="LUKA_FULLSTAT_FOUNDATION/v1", schema_version=1,
        parents=dict(gate="a"*64), counts=counts, class_names=list(CLASS_NAMES), transfer=dict(source=SOURCE),
        policy=split_policy(), files=files, quotas=dict(train=quotas)))
    cache_root = tmp_path/"cache"
    cache = save(cache_root/"cache_manifest.json", dict(contract="LUKA_FULLSTAT_CACHE/v1", schema_version=1,
        parents=dict(gate="a"*64, foundation=foundation["content_hash"]), final_test_accessed=False,
        chunks=[dict(payload="DO_NOT_OPEN_PARTICLES")]))
    original_path = tmp_path/"original"/"campaign_spec.json"
    original = save(original_path, dict(contract="LUKA_FULLSTAT_RECOVERY_SPEC/v1", schema_version=1,
        root=str(original_path.parent), counts=counts, cache_root=str(cache_root), foundation_root=str(root),
        prepared_sha256=cache["content_hash"]))
    anchor = with_content_hash(dict(contract="LUKA_FULLSTAT_EMBEDDING_SWEEP_ANCHOR/v1", schema_version=1,
        parents=dict(parent=original["content_hash"]), spec=file_ref(original_path),
        evidence=dict(prepared=cache["content_hash"], foundation_root=str(root), cache_root=str(cache_root), foundation_gate="a"*64)))
    spec_path = tmp_path/"six"/"campaign_spec.json"
    save(spec_path, dict(contract="LUKA_FULLSTAT_SIX_TEACHER_SPEC/v1", schema_version=1,
                         root=str(spec_path.parent), counts=counts, anchor=anchor))
    return spec_path, root, foundation


def examples():
    return np.repeat(np.arange(11), 4), np.tile([100., 150., 200., 300.], 11)


def test_exact_histogram_edges_and_exposure():
    labels, pt = examples()
    data = audit.summarize(labels, pt, widths=[100], pt_range=[100, 300], draws=440, min_cell=3)
    c = data["candidates"][0]
    assert c["counts"] == [[2, 2]]*11  # 200 enters second bin; last upper edge 300 included
    assert c["sparse_nonempty_cells"] == 22
    assert c["feasible_on_this_range"] and c["outside_rows"] == 0
    risk = c["sampling_projection"]
    assert risk["max_expected_draws_per_jet_per_epoch"] == 10
    expected = 44*(1-(43/44)**440)
    assert risk["expected_unique_jets_per_epoch"] == pytest.approx(expected)
    assert risk["expected_repeat_draw_fraction_per_epoch"] == pytest.approx(1-expected/440)


def test_one_cell_per_class_exact_unique_probability():
    r = audit.summarize(np.arange(11), np.full(11, 1.), widths=[10], draws=11)["candidates"][0]
    assert r["sampling_projection"]["expected_unique_jets_per_epoch"] == pytest.approx(11*(1-(10/11)**11))


def test_empty_bins_are_not_silently_removed():
    labels, pt = examples()
    pt[labels == 0] = 125
    c = audit.summarize(labels, pt, widths=[100], pt_range=[100, 400])["candidates"][0]
    assert c["empty_cells"] == 2  # QCD has no jets in either upper bin
    assert c["sampling_projection"] is None
    assert c["feasible_on_this_range"] is False
    assert len(c["edges_GeV"]) == 4


def test_underflow_overflow_explicit_and_all_train_quantiles():
    labels = np.repeat(np.arange(11), 7)
    pt = np.tile([99, 100, 199, 200, 299, 300, 301], 11).astype(float)
    data = audit.summarize(labels, pt, widths=[100], pt_range=[100, 300])
    c = data["candidates"][0]
    assert c["counts"] == [[2, 3]]*11
    assert c["outside_rows"] == 22 and c["in_range_rows"] == 55
    assert c["underflow_by_class"] == c["overflow_by_class"] == [1]*11
    assert data["class_statistics"][0]["quantiles_GeV"]["max"] == 301


def test_supported_spans_preserve_internal_empty_bins():
    labels = np.repeat(np.arange(11), 4)
    pt = np.tile([125., 125., 325., 325.], 11)
    c = audit.summarize(labels, pt, widths=[100], min_cell=2)["candidates"][0]
    assert c["min_cell_bin_spans_GeV"] == [[100., 200.], [300., 400.]]
    assert c["populated_bin_spans_GeV"] == c["min_cell_bin_spans_GeV"]
    assert c["sampling_projection"] is None and c["empty_cells"] == 11


@pytest.mark.parametrize("kwargs", [dict(widths=[0]), dict(widths=[np.nan]), dict(widths=[50, 50]),
    dict(widths=[1e-6]), dict(widths=[100], pt_range=[1, 250]), dict(pt_range=[3, 1]),
    dict(draws=0), dict(draws=True), dict(min_cell=0), dict(epochs=0)])
def test_invalid_grid_and_budget(kwargs):
    with pytest.raises(ValueError):
        audit.summarize(*examples(), **kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1., 0.])
def test_nonfinite_or_nonpositive_train_pt_rejected(value):
    labels, pt = examples()
    pt[0] = value
    with pytest.raises(ValueError):
        audit.summarize(labels, pt)


def test_saved_lineage_train_only_and_no_science_payload_reads(campaign, monkeypatch):
    spec_path, root, foundation = campaign
    _, found_root, found, refs = audit.resolve_inputs(spec_path)
    assert found_root == root and found == foundation and len(refs) == 4
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    accesses = []
    def guarded(archive, key):
        assert key in {"roles", "labels", "jet_label", "jet_pt"}
        accesses.append(key)
        return original_getitem(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    labels, pt, memberships, digest = audit.read_train(root, foundation, progress=lambda _: None)
    assert len(pt) == 440 and np.isfinite(pt).all() and set(pt) == {125, 225}
    assert np.bincount(labels).tolist() == [40]*11
    assert len(memberships) == 2 and len(digest) == 64 and len(accesses) == 8


@pytest.mark.parametrize("field", ["prepared", "foundation_gate", "cache_root", "foundation_root"])
def test_alien_anchor_rejected(campaign, field):
    path, _, _ = campaign
    s = load_json(path)
    s["anchor"]["evidence"][field] = "alien"
    s["anchor"] = with_content_hash(s["anchor"])
    save(path, s)
    with pytest.raises(ValueError):
        audit.resolve_inputs(path)


def test_original_spec_supported(campaign):
    path, root, _ = campaign
    original = load_json(path)["anchor"]["spec"]["path"]
    _, actual, _, refs = audit.resolve_inputs(original)
    assert actual == root and len(refs) == 3


def test_corrupt_membership_and_quota_rejected(campaign):
    _, root, foundation = campaign
    bad = deepcopy(foundation)
    bad["quotas"]["train"]["signal:01"] += 1
    with pytest.raises(ValueError, match="quotas"):
        audit.read_train(root, bad, progress=lambda _: None)
    path = root/foundation["files"][0]["membership"]["path"]
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        audit.read_train(root, foundation, progress=lambda _: None)


def test_spec_hash_and_parent_file_binding_rejected(campaign):
    path, _, _ = campaign
    s = load_json(path)
    s["counts"]["train"] += 1
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="hash"):
        audit.resolve_inputs(path)


def test_path_escape_and_duplicate_source_rejected(campaign):
    _, root, f = campaign
    bad = deepcopy(f)
    bad["files"][0]["membership"]["path"] = "../outside.npz"
    with pytest.raises(ValueError, match="membership path"):
        audit.read_train(root, bad)
    bad = deepcopy(f)
    bad["files"][1]["sha256"] = bad["files"][0]["sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        audit.read_train(root, bad, progress=lambda _: None)


def test_publication_portable_show_and_immutable_inputs(campaign, tmp_path):
    spec, root, _ = campaign
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path/"audit"
    report = audit.run(spec, out, widths=[100], pt_range=[100, 300], draws=440)
    assert audit.load_report(out/"report.json") == report
    assert all(p.read_bytes() == b for p, b in before.items())
    assert {p.name for p in out.iterdir()} == {"report.json", "summary.txt", "cells.csv"}
    assert len((out/"cells.csv").read_text().splitlines()) == 23
    assert "No model inference" in (out/"summary.txt").read_text()
    result = subprocess.run([sys.executable, "-I", "-B", str(PROJECT/"scripts/audit_luka_fullstat_pt.py"),
                             "show", str(out/"report.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "TRAIN jets: 440" in result.stdout
    with pytest.raises(FileExistsError):
        audit.run(spec, out)
    with pytest.raises(PermissionError):
        audit.run(spec, spec.parent/"audit")
    (out/"cells.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="bytes differ"):
        audit.load_report(out/"report.json")


def test_failure_does_not_publish_completion(campaign, tmp_path):
    spec, _, _ = campaign
    output = tmp_path/"failed"
    with pytest.raises(ValueError):
        audit.run(spec, output, widths=[0])
    assert not output.exists()


def test_cli_numpy_only_no_torch_or_root_imports():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','weaver','uproot','awkward','scipy','sklearn'}:
            raise RuntimeError('forbidden heavy import: '+fullname)
sys.meta_path.insert(0, Guard())
from hlt_classification.operations import fullstat_pt_audit
print('numpy-only audit import passed')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PROJECT/"src")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "passed" in result.stdout


def test_real_cli_run_production_counts_and_portable_output(campaign, tmp_path):
    # Exercise the actual CLI with small immutable fixture counts injected only
    # by this test harness, never via a production CLI flag.
    spec, _, _ = campaign
    code = """
import runpy,sys,json
sys.path.insert(0, sys.argv[1])
from hlt_classification.operations import fullstat_pt_audit as a
a.COUNTS=json.loads(sys.argv[2])
sys.argv=sys.argv[3:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PROJECT/"src"),
        json.dumps(audit.COUNTS), str(PROJECT/"scripts/audit_luka_fullstat_pt.py"), "run",
        "--spec", str(spec), "--output", str(tmp_path/"cli"), "--widths", "100", "--pt-range", "100", "300"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Saved report:" in result.stdout
    assert audit.load_report(tmp_path/"cli/report.json")["analysis"]["train_rows"] == 440


def test_frozen_numpy_adapters_match_original_metadata_semantics(tmp_path):
    from hlt_classification.luka_fullstat import contracts as donor
    from hlt_classification.jetclass2_delphes.schema import map_labels
    from hlt_classification.data.cache_contracts import array_sha256
    assert meta.COUNTS == donor.COUNTS and meta.ROLES == donor.ROLES
    assert meta.SOURCE == donor.SOURCE and meta.split_policy() == donor.split_policy()
    assert meta.CLASS_NAMES == CLASS_NAMES
    raw = np.array([*SIGNAL_IDS, *range(161, 188)], dtype=np.int32)
    np.testing.assert_array_equal(meta.map_labels(raw), map_labels(raw))
    for array in (raw, np.array([1.5, 3.25], dtype=np.float64), raw.reshape(1, -1)):
        assert meta.array_sha256("x", array) == array_sha256("x", array)
    value = dict(contract="EXAMPLE/v1", schema_version=1, nested={"a": [1, 2.5]}, text="tau: \u03c4")
    assert meta.with_content_hash(value) == with_content_hash(value)
    path = tmp_path/"record.json"
    meta.write_immutable_json(path, meta.with_content_hash(value))
    assert meta.file_ref(path) == file_ref(path)
    assert meta.file_ref(path, root=tmp_path) == file_ref(path, root=tmp_path)
    assert meta.validate_file_ref(file_ref(path)) == path
    with pytest.raises(FileExistsError):
        meta.write_immutable_json(path, value)


def test_donor_generated_foundation_readback(tmp_path):
    # Genuine existing foundation producer on synthetic ROOT, rather than only
    # handcrafted metadata. The audit itself must never open these ROOT files.
    from test_luka_fullstat import write_root
    from hlt_classification.luka_fullstat import data
    from hlt_classification.data.cache_contracts import sha256_file
    container = tmp_path/"snapshot"
    write_root(container/"raw/jetclass2/train_higgs2p/merged/ntuple_merged_0.root", list(SIGNAL_IDS)*16+[161]*16)
    write_root(container/"raw/jetclass2sb/QCD_PT-200to300/merged/ntuple_merged_0.root", [161]*160)
    (container/"source.txt").write_text(SOURCE+"\n")
    (container/"source.sha256").write_text("".join(f"{sha256_file(p)}  {p.relative_to(container/'raw').as_posix()}\n"
        for p in sorted((container/"raw").rglob("*.root"))))
    counts = dict(train=242, validation=44, final_test=22, unused=12)
    root = tmp_path/"foundation"
    foundation = data.build(container, root, gate_hash="a"*64, counts=counts, expected_files=2)
    labels, pt, _, _ = audit.read_train(root, foundation, progress=lambda _: None)
    assert len(labels) == 242 and set(pt) == {199.}
    _, members = data.load(root)
    expected = np.concatenate([m["labels"][m["roles"] == 0] for m in members])
    np.testing.assert_array_equal(labels, expected)


def test_oscar_wrapper_interpreter_override_and_syntax(campaign, tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else Path(shutil.which("bash") or "/absent")
    if not bash.is_file():
        pytest.skip("Bash unavailable")
    wrapper = PROJECT/"scripts/audit_luka_fullstat_pt.sh"
    result = subprocess.run([str(bash), "-n", wrapper.as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    spec, _, _ = campaign
    output = tmp_path/"shell"
    audit.run(spec, output, widths=[100])
    result = subprocess.run([str(bash), wrapper.as_posix(), "show", (output/"report.json").as_posix()],
        capture_output=True, text=True, env={**os.environ, "AUDIT_PYTHON": Path(sys.executable).as_posix()})
    assert result.returncode == 0, result.stderr
    assert "TRAIN jets: 440" in result.stdout
