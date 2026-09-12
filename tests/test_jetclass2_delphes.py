from pathlib import Path
import shutil
from fractions import Fraction

import awkward as ak
import numpy as np
import pytest
import uproot

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes.contracts import relative_file
from hlt_classification.jetclass2_delphes.inputs import build_inputs, wrap_phi
from hlt_classification.jetclass2_delphes.inventory import build_inventory, validate_inventory, verify_snapshot
from hlt_classification.jetclass2_delphes.reader import DatasetReader, Particles
from hlt_classification.jetclass2_delphes.schema import PARTICLE_FIELDS, SIGNAL_IDS, map_labels
from hlt_classification.jetclass2_delphes.selection import selected_mask, selection_policy
from hlt_classification.jetclass2_delphes.splits import build_splits, validate_splits
from hlt_classification.jetclass2_delphes.views import build_view, match_particles, pairing_matrices
from hlt_classification.jetclass2_delphes.reader import Jet


def particle_values(n=2):
    p = np.zeros((n, 14), np.float32)
    p[:, 0] = np.arange(n) + 1
    p[:, 3] = np.arange(n) + 2
    p[:, 4] = 1
    p[:, 5] = 1
    p[:, 10] = .3
    p[:, 12] = -.2
    return p


def root_file(path, *, multiple_cycles=False, bad_length=False):
    labels = list(SIGNAL_IDS) + [161, 187]
    n = len(labels)
    branches = {"jet_label": "int32", "hlt_matched": "bool",
                "jet_nparticles": "int32", "hlt_jet_nparticles": "int32"}
    arrays = dict(jet_label=np.array(labels, np.int32), hlt_matched=np.ones(n, bool),
                  jet_nparticles=np.full(n, 3, np.int32), hlt_jet_nparticles=np.full(n, 2, np.int32))
    arrays["hlt_matched"][-1] = False
    for prefix, count in (("part_", 3), ("hlt_part_", 2)):
        raw = particle_values(count)
        for column, name in enumerate(PARTICLE_FIELDS):
            branches[prefix + name] = "var * float32"
            values = [raw[:, column].tolist() for _ in range(n)]
            if prefix == "hlt_part_":
                values[-1] = [float("nan")]  # unmatched dummy must not be validated
            if bad_length and prefix == "hlt_part_" and column == 0:
                values[0] = [1]
            arrays[prefix + name] = ak.Array(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        for _ in range(2 if multiple_cycles else 1):
            f.mktree("tree", branches)
            f["tree"].extend(arrays)


@pytest.fixture
def snapshot(tmp_path):
    data = tmp_path / "data"
    for source in ("train_higgs2p", "train_qcd"):
        for i in range(6):
            root_file(data / source / f"ntuple_{i}.root", multiple_cycles=True)
    inventory = build_inventory(data, step_size=5)
    return data, inventory, build_splits(inventory)


def test_label_and_source_policies():
    raw = np.array([0, 14, 161, 187])
    assert map_labels(raw).tolist() == [1, 10, 0, 0]
    flags = np.array([True, False, True, True])
    mask, _ = selected_mask(raw, flags, "train_higgs2p", selection_policy())
    assert mask.tolist() == [True, False, False, False]
    mask, _ = selected_mask(raw, flags, "train_higgs2p", selection_policy(include_signal_file_qcd=True))
    assert mask.tolist() == [True, False, True, True]
    with pytest.raises(ValueError, match="Unregistered"):
        map_labels(np.array([4]))
    with pytest.raises(ValueError, match="boolean"):
        selected_mask(raw, flags.astype(int), "train_qcd", selection_policy())


def test_inventory_latest_cycle_splits_and_reader(snapshot):
    data, inventory, splits = snapshot
    assert inventory["entries"] == 12 * 12  # not both ROOT cycles
    assert inventory["files"][0]["tree_key"] == "tree;2"
    assert inventory["selected_class_counts"][0] == 6
    assert build_splits(inventory) == splits
    all_ids = set()
    for role in ("train", "validation"):
        jets = list(DatasetReader(data, inventory, splits, role=role, include_offline=True, step_size=3))
        assert len(jets) == splits["role_counts"][role]
        ids = {j.identity for j in jets}
        assert len(ids) == len(jets) and not ids & all_ids
        all_ids |= ids
        assert all(len(j.hlt) == 2 and len(j.offline) == 3 for j in jets)
        hlt_only = list(DatasetReader(data, inventory, splits, role=role, step_size=7))
        assert [j.identity for j in jets] == [j.identity for j in hlt_only]
        assert all(j.offline is None for j in hlt_only)
    with pytest.raises(PermissionError):
        DatasetReader(data, inventory, splits, role="final_test")


def test_relocation_and_hash_corruption(snapshot, tmp_path):
    data, inventory, splits = snapshot
    other = tmp_path / "relocated"
    shutil.copytree(data, other)
    verify_snapshot(other, inventory)
    original = list(DatasetReader(data, inventory, splits, role="train"))
    copied = list(DatasetReader(other, inventory, splits, role="train"))
    assert [j.identity for j in original] == [j.identity for j in copied]
    path = other / inventory["files"][0]["path"]
    with path.open("ab") as f:
        f.write(b"corrupted")
    with pytest.raises(ValueError, match="content differs"):
        verify_snapshot(other, inventory)


def test_split_tampering_and_missing_classes(snapshot):
    _, inventory, splits = snapshot
    bad = {**splits, "groups": [{**g} for g in splits["groups"]]}
    bad["groups"][0]["group_id"] = bad["groups"][1]["group_id"]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(with_content_hash(bad), inventory)
    bad = {**inventory, "selected_class_counts": [0] * 11}
    with pytest.raises(ValueError, match="totals"):
        validate_inventory(with_content_hash(bad))


def test_duplicate_content_rejected(snapshot):
    data, _, _ = snapshot
    shutil.copyfile(data / "train_qcd/ntuple_0.root", data / "train_qcd/copy.root")
    with pytest.raises(ValueError, match="Duplicate ROOT content"):
        build_inventory(data)


@pytest.mark.parametrize("name", ["../escape.root", "C:/escape.root", "/escape.root", "a\\b.root"])
def test_paths_fail_closed(name, tmp_path):
    with pytest.raises(ValueError):
        relative_file(tmp_path, name)


def test_reduced_features_zero_errors_overflow_and_validation():
    raw = particle_values()
    particles = Particles(raw)
    assert not particles.uncertainty_valid.any()
    inputs = build_inputs(particles, capacity=16)
    assert inputs.features.shape == (17, 16)
    assert set(inputs.model_inputs()) == {"features", "vectors", "mask"}
    assert inputs.features[11, 0] == pytest.approx(np.tanh(.3))
    assert inputs.features[12, 0] == 0
    assert inputs.features[13, 0] == pytest.approx(np.tanh(-.2))
    assert not inputs.features[:, 2:].any()
    assert not inputs.vectors[:, 2:].any()
    assert wrap_phi(np.pi) == pytest.approx(-np.pi)
    with pytest.raises(ValueError, match="overflow"):
        build_inputs(Particles(particle_values(17)), capacity=16)
    raw[0, 11] = -1
    with pytest.raises(ValueError, match="Negative"):
        Particles(raw)
    raw[0, 11] = 0
    raw[0, 6] = 1
    with pytest.raises(ValueError, match="one-hot"):
        Particles(raw)
    raw[0, 6] = 0
    raw[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        Particles(raw)


@pytest.mark.parametrize("nh,no", [(2, 3), (3, 2), (3, 3), (1, 4), (4, 1)])
def test_views_exact_endpoints_rectangular_and_reference(nh, no):
    from hlt_classification.scouting.hcwdl_fullcard_bottleneck_matcher import reference_pairing_from_matrices
    h, o = particle_values(nh), particle_values(no)
    o[:, 1] = .03
    o[:, 3] += .2
    jet = Jet("a" * 64, 1, Particles(h), Particles(o))
    mapping = match_particles(jet.hlt, jet.offline)
    np.testing.assert_array_equal(mapping, reference_pairing_from_matrices(**pairing_matrices(jet.hlt, jet.offline)))
    assert sum(mapping >= 0) == min(nh, no)
    assert len(set(mapping[mapping >= 0])) == min(nh, no)
    assert build_view(jet, u=Fraction(0), f=Fraction(0)) is jet.offline
    assert build_view(jet, u=Fraction(1), f=Fraction(1)) is jet.hlt
    # Poison offline access entirely: D000 must not evaluate it.
    poisoned = Jet(jet.identity, jet.label, jet.hlt, object())
    assert build_view(poisoned, u=Fraction(1), f=Fraction(1)) is jet.hlt
    counts = []
    for numerator in range(11):
        view = build_view(jet, u=Fraction(numerator, 10), f=Fraction(0), mapping=mapping)
        counts.append(len(view))
    assert counts == sorted(counts, reverse=no >= nh)
    u100 = build_view(jet, u=Fraction(1), f=Fraction(0), mapping=mapping)
    assert len(u100) == nh
    for i, j in enumerate(mapping):
        np.testing.assert_array_equal(u100.values[i], o[j] if j >= 0 else h[i])
    for numerator in range(11):
        view = build_view(jet, u=Fraction(1), f=Fraction(numerator, 10), mapping=mapping)
        assert len(view) == nh
        assert np.isfinite(build_inputs(view, capacity=16).features).all()


def test_atomic_type_and_zero_uncertainty_transitions():
    h, o = particle_values(1), particle_values(1)
    o[0, 4:10] = [0, 0, 1, 0, 0, 0]  # neutral hadron
    o[0, 10:14] = [0, 0, 0, 0]
    h[0, 11] = .2
    jet = Jet("b" * 64, 0, Particles(h), Particles(o))
    for i in range(11):
        raw = build_view(jet, u=Fraction(1), f=Fraction(i, 10)).values[0]
        np.testing.assert_array_equal(raw[10:14], h[0, 10:14] if raw[4] else o[0, 10:14])


def test_bad_jagged_length_and_forbidden_file_subset(tmp_path):
    data = tmp_path / "bad"
    # Three broad files provide each class to each role; one has a corrupt row.
    for i in range(3):
        root_file(data / "train_qcd" / f"{i}.root", bad_length=True)
    inv = build_inventory(data)
    splits = build_splits(inv)
    with pytest.raises(ValueError, match="Jagged"):
        list(DatasetReader(data, inv, splits, role="train", include_offline=True))
    final_file = next(g["path"] for g in splits["groups"] if g["role"] == "final_test")
    with pytest.raises(PermissionError):
        DatasetReader(data, inv, splits, role="train", file_paths=(final_file,))


def test_foundation_compact_publication_cache_join_and_graph(snapshot, tmp_path):
    from hlt_classification.jetclass2_delphes.foundation import (
        build_foundation_spec, build_assignment_shard, load_assignments, build_foundation_lock,
    )
    from hlt_classification.jetclass2_delphes.cache import prepare_cache
    from hlt_classification.jetclass2_delphes.campaign import build_campaign_plan
    data, inv, splits = snapshot
    spec = build_foundation_spec(inv, splits)
    root = tmp_path / "foundation"
    for task in spec["assignment_tasks"]:
        result = build_assignment_shard(spec, data_root=data, output_root=root, file_index=task["file_index"])
        assert result["array_bytes"] < 20_000
    lock = build_foundation_lock(spec, root)
    assert not lock["particle_views_persisted"]
    assert not lock["production_runtime_accepted"]
    plan = build_campaign_plan(spec)
    assert plan["fresh_fit_count"] == 31
    assert plan["probability_publication_count"] == 26
    assert plan["imported_models"] == [] and not plan["executable"]
    for branch in plan["branches"].values():
        assert next(n for n in plan["nodes"] if n["node_id"] == branch[-1])["deployable"]
    cache = prepare_cache(spec, data_root=data, foundation_root=root, role="train",
                          coordinate_name="U100", max_ram_bytes=10_000_000)
    assert len(cache) == splits["role_counts"]["train"]
    assert cache.batch(np.arange(len(cache)))["mask"].sum() == 2 * len(cache)
    other = prepare_cache(spec, data_root=data, foundation_root=root, role="train",
                          coordinate_name="U100", workers=2, max_ram_bytes=10_000_000)
    for key in ("features", "vectors", "mask", "identities", "labels"):
        np.testing.assert_array_equal(cache.batch(np.arange(len(cache)))[key], other.batch(np.arange(len(other)))[key])
    first = spec["assignment_tasks"][0]["file_index"]
    array_path = root / "assignments" / f"{first:04d}.npz"
    with array_path.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_assignments(spec, root=root, file_index=first)
