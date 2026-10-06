"""Relocation of actual synthetic ROOT/banks, never real final-test data."""
import copy
from contextlib import closing
from itertools import islice
from pathlib import Path
import shutil
import runpy

import numpy as np
import pytest
import uproot

from test_literature_context_production import (
    parent, parent_v2, final_pilot, study, initial, admit, serial_bulk,
)
from hlt_classification.data.cache_contracts import load_json, sha256_file, with_content_hash
from hlt_classification.literature_context_production import campaign, engine, output, population
from hlt_classification import literature_context_consumer as consumer


@pytest.fixture
def relocated(study, tmp_path, monkeypatch):
    serial_bulk(monkeypatch)
    attempt = load_json(initial(study))
    admit(study, attempt)
    bundle = campaign.bundle(study)
    for shard in study["shards"]:
        engine.generate_shard(study, attempt, shard, bundle["calibration"])
    manifest = output.manifest(study)
    for role in ("train", "validation"):
        output.manifest(study, role)
    root = tmp_path / "oscar-copy"
    proxy = root / consumer.PROXY_DIRECTORY
    snapshot = root / consumer.SNAPSHOT_DIRECTORY
    offline = snapshot / "jetclass2"
    metadata = root / "oscar_provenance_v1"
    shutil.copytree(study["root"], proxy)
    shutil.copytree(study["data_root"], offline)
    for record in (study["profile"], study["donor_profile"]):
        path = snapshot / Path(record["path"]).relative_to(Path(study["data_root"]).parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], path)
    for record in (study["inventory"], bundle["pilot"], bundle["pilot_receipt"], bundle["pilot_report"]):
        original = Path(record["path"])
        path = metadata / original.parent.name / original.name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
    return root, manifest["content_hash"], study


def opened(fixture):
    root, digest, _ = fixture
    return consumer.RelocatedDataset.from_oscar_copy(root, offline_root=root/consumer.SNAPSHOT_DIRECTORY/"jetclass2", expected_manifest_sha256=digest)


def test_actual_relocation_no_original_reads_and_no_writes(relocated, monkeypatch):
    root, digest, original = relocated
    before = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    old_open = Path.open
    forbidden = [Path(original["root"]).resolve(), Path(original["data_root"]).resolve()]
    forbidden += [Path(original["inventory"]["path"]).parent.resolve()]
    forbidden += [Path(campaign.bundle(original)["pilot"]["path"]).parent.resolve()]
    def guarded(path, mode="r", *args, **kwargs):
        assert not any(path.resolve().is_relative_to(p) for p in forbidden), str(path)
        assert not any(c in mode for c in "wax+"), mode
        return old_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    data = opened(relocated)
    report = data.describe()
    assert report["parents"]["manifest"] == digest and report["metadata_authenticated"]
    assert not report["physical_banks_verified"] and not report["final_test_accessed"]
    rows = list(data.iter_pairs("train", labels=True))
    assert len(rows) == 44 and len({r.identity for r in rows}) == 44
    assert all(0 <= row.label < 11 for row in rows)
    for row in rows:
        np.testing.assert_allclose(row.offline.tracking[:, 0], .3)  # Native mm, never /10.
        np.testing.assert_allclose(row.offline.tracking[:, 1], -.2)
    for row, (identity, p) in zip(rows, data.iter_proxy("train")):
        assert row.identity == identity
        for name in ("p4", "charge", "category", "tracking", "valid"):
            np.testing.assert_array_equal(getattr(row.proxy, name), getattr(p, name))
    assert before == {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_no_native_hlt_branches_no_labels_by_default(relocated, monkeypatch):
    data = opened(relocated)
    old = uproot.behaviors.TBranch.HasBranches.arrays
    requests = []
    def tracked(tree, expressions, **kwargs):
        requests.append(expressions)
        assert not any(name.startswith("hlt_part_") for name in expressions)
        assert "jet_label" not in expressions and "hlt_matched" not in expressions
        assert kwargs["entry_stop"]-kwargs["entry_start"] <= 512
        return old(tree, expressions, **kwargs)
    monkeypatch.setattr(uproot.behaviors.TBranch.HasBranches, "arrays", tracked)
    rows = list(data.iter_pairs("validation"))
    assert len(rows) == 11 and all(r.label is None for r in rows) and requests


@pytest.mark.parametrize("method", ["iter_proxy", "iter_pairs"])
def test_test_role_rejected_before_any_open(relocated, monkeypatch, method):
    data = opened(relocated)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Forbidden role opened data"))
    with pytest.raises(PermissionError, match="sealed"):
        list(getattr(data, method)("final_test"))
    test_shard = next(s["shard_id"] for s in data.study["shards"] if s["role"] == "final_test")
    with pytest.raises(PermissionError, match="subset"):
        list(getattr(data, method)("train", shard_ids=[test_shard]))


def test_train_reader_ignores_corrupt_test_banks_and_receipts(relocated):
    data = opened(relocated)
    test = next(s for s in data.study["shards"] if s["role"] == "final_test")
    path = data.proxy_root / "shards" / (test["shard_id"]+".json")
    receipt = load_json(path)
    (data.proxy_root / receipt["blocks"][0]["relative"]).write_bytes(b"sealed test corruption")
    path.write_bytes(b"sealed test receipt corruption")
    fresh = opened(relocated)
    assert len(list(fresh.iter_proxy("train"))) == 44


@pytest.mark.parametrize("target", ["inventory", "profile", "pilot", "pilot_receipt", "pilot_report", "bundle"])
def test_corrupt_provenance_rejected(relocated, target):
    data = opened(relocated)
    path = Path(data.local_references[target])
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError, match="checksum|size"):
        opened(relocated)


def test_wrong_manifest_and_unknown_recipe_rejected(relocated):
    root, _, _ = relocated
    with pytest.raises(ValueError, match="manifest"):
        consumer.RelocatedDataset.from_oscar_copy(root,
            offline_root=root/consumer.SNAPSHOT_DIRECTORY/"jetclass2",
            expected_manifest_sha256="0"*64)
    data = opened(relocated)
    path = data.proxy_root / "dataset_manifest.json"
    value = copy.deepcopy(data.manifest)
    value["recipe"] = "CMS_FITTED"
    import json
    path.write_text(json.dumps(with_content_hash(value)))
    with pytest.raises(ValueError, match="manifest"):
        opened(relocated)


@pytest.mark.parametrize("target", ["receipt", "block", "offline", "attempt", "preflight", "release"])
def test_corrupt_ordinary_payload_rejected(relocated, target):
    data = opened(relocated)
    shard = next(s for s in data.study["shards"] if s["role"] == "train")
    receipt_path = data.proxy_root / "shards" / (shard["shard_id"]+".json")
    receipt = load_json(receipt_path)
    paths = dict(receipt=receipt_path, block=data.proxy_root / receipt["blocks"][0]["relative"],
        offline=data.offline_root / shard["path"],
        attempt=data.proxy_root / f"attempts/{receipt['attempt']}/attempt_spec.json",
        preflight=data.proxy_root / f"attempts/{receipt['attempt']}/preflight.json",
        release=data.proxy_root / "releases/train.json")
    paths[target].write_bytes(b"corrupt")
    with pytest.raises((ValueError, KeyError)):
        list(data.iter_pairs("train"))


def test_proxy_path_requires_no_root_access_and_sparse_shard_subset(relocated, monkeypatch):
    data = opened(relocated)
    monkeypatch.setattr(uproot, "open", lambda *a, **k: pytest.fail("Proxy-only route opened ROOT"))
    shard = next(s for s in data.study["shards"] if s["role"] == "train")
    rows = list(data.iter_proxy("train", shard_ids=[shard["shard_id"]]))
    assert len(rows) == shard["jets"]
    record, entries = population.entries_for(data.study["population"], shard)
    assert [i for i, _ in rows] == list(population.ids(data.inventory["content_hash"], record, entries))


def test_bad_proxy_identity_detected_even_with_updated_block_reference(relocated, monkeypatch):
    data = opened(relocated)
    original = output.arrays
    def swapped(path):
        values = original(path)
        values["jet_identity"][[0, 1]] = values["jet_identity"][[1, 0]]
        return values
    monkeypatch.setattr(output, "arrays", swapped)
    with pytest.raises(ValueError, match="identity join"):
        list(data.iter_proxy("train"))


def test_bounded_pair_iterator_closes_root(relocated, monkeypatch):
    data = opened(relocated)
    original = consumer.authenticated_open
    closed = []
    from contextlib import contextmanager
    @contextmanager
    def tracked(path, digest):
        try:
            with original(path, digest) as handle:
                yield handle
        finally:
            closed.append(path)
    monkeypatch.setattr(consumer, "authenticated_open", tracked)
    with closing(data.iter_pairs("train")) as rows:
        assert len(list(islice(rows, 2))) == 2
    assert len(closed) == 1
