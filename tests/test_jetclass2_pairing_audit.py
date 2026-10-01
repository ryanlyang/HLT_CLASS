"""Jet-level pairing diagnostics distinguish same-class wrong partners."""
import copy

import numpy as np
import pytest

from test_jetclass2_delphes import snapshot
from hlt_classification.jetclass2_delphes import pairing_audit as audit
from hlt_classification.jetclass2_delphes.reader import DatasetReader, Particles
from hlt_classification.jetclass2_delphes.split_registry import build_registry, select_profile


def toy_rows(n=150):
    rng = np.random.default_rng(37)
    rows = []
    for i in range(n):
        pt = float(rng.uniform(100, 900))
        hlt = dict(pt=pt, mass=float(rng.uniform(20, 150)), count=int(rng.integers(10, 90)),
                   charged_count=int(rng.integers(3, 10)), charged_pt_fraction=float(rng.uniform(.1, .9)),
                   eta=float(rng.uniform(-2, 2)), phi=float(rng.uniform(-np.pi, np.pi)), negative_mass2=False)
        offline = dict(hlt, pt=pt * 1.05, eta=hlt["eta"] + .002, phi=hlt["phi"] + .003)
        rows.append(dict(identity=f"{i:064x}", source="signal", file=f"signal/{i % 3}.root",
                         label=i % 2 + 1, hlt=hlt, offline=offline, producer_dr=None, branch_axis_dr=None))
    return rows


def test_global_directions_wrap_and_native_summary():
    raw = np.zeros((2, 14), np.float32)
    raw[:, 0] = -10
    raw[:, 1] = [-.01, .01]
    raw[:, 3] = 12
    raw[:, 4:6] = 1
    result = audit.describe(Particles(raw))
    assert result["pt"] == 20
    assert result["mass"] == pytest.approx(np.sqrt(24**2 - 20**2))
    assert result["charged_pt_fraction"] == 1
    assert result["count"] == result["charged_count"] == 2
    assert audit.delta_r(dict(eta=0, phi=np.pi - .01), dict(eta=0, phi=-np.pi + .01)) == pytest.approx(.02)


@pytest.mark.parametrize("control", ["class_source", "class_source_pt", "class_file"])
def test_shuffles_preserve_groups_and_exclude_self_and_singletons(control):
    rows = toy_rows()
    rows.append(dict(rows[0], label=10))
    partners = audit.shuffled_partners(rows, seed=12, control=control)
    assert np.array_equal(partners, audit.shuffled_partners(rows, seed=12, control=control))
    valid = np.flatnonzero(partners >= 0)
    assert partners[-1] == -1
    assert np.all(partners[valid] != valid)
    assert set(partners[valid]) == set(valid)
    for i in valid:
        left, right = rows[i], rows[partners[i]]
        assert left["label"] == right["label"] and left["source"] == right["source"]
        if control == "class_file":
            assert left["file"] == right["file"]
        if control == "class_source_pt":
            assert np.floor(np.log(left["hlt"]["pt"]) / .2) == np.floor(np.log(right["hlt"]["pt"]) / .2)


def test_wrong_same_class_pairing_is_detectable_without_label_changes():
    rows = toy_rows()
    result = audit.summarize(rows, repeats=5)
    comparison = result["controls"]["class_source"]["by_class"]["ALL"]
    assert comparison["actual"]["axis_dr"]["q99"] < .01
    assert comparison["shuffled_median_dr"]["q50"] > 1
    assert comparison["actual"]["spearman"]["pt"] == pytest.approx(1)
    assert abs(comparison["shuffled_spearman"]["pt"]["q50"]) < .25
    broken = copy.deepcopy(rows)
    p = audit.shuffled_partners(rows, seed=77, control="class_source")
    for i in range(len(rows)):
        broken[i]["offline"] = rows[p[i]]["offline"]
    failed = audit.summarize(broken, repeats=5)
    assert failed["actual"]["axis_dr"]["q50"] > 1
    assert abs(failed["actual"]["spearman"]["pt"]) < .25
    assert [r["label"] for r in broken] == [r["label"] for r in rows]


def test_empty_classes_constant_metrics_and_optional_producer_diagnostics():
    rows = toy_rows(3)
    rows[0]["producer_dr"] = .1
    rows[0]["branch_axis_dr"] = .101
    result = audit.summarize(rows, repeats=2)
    assert result["producer_dr"]["count"] == 1
    assert result["producer_vs_branch_axis_absolute_difference"]["q50"] == pytest.approx(.001)
    assert audit.spearman([1, 1, 1], [2, 3, 4]) is None
    assert result["controls"]["class_source"]["by_class"]["X_mm"]["actual"]["rows"] == 0
    assert result["controls"]["class_source"]["by_class"]["X_mm"]["actual"]["axis_dr"]["q50"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        audit.distribution([float("nan")])


def test_raw_sample_uses_exact_registered_rows_without_opening_test(snapshot, monkeypatch):
    data, inventory, reservoirs = snapshot
    registry = build_registry(data, inventory, reservoirs, training_sizes=(33,), validation_size=11, test_size=11)
    profile = select_profile(registry, inventory, "TRAIN_33")
    allowed = {g["path"] for g in profile["groups"] if g["role"] == "train"}
    verify = audit.verify_file
    opened = []
    def checked(root, record, inventory):
        assert record["path"] in allowed
        opened.append(record["path"])
        return verify(root, record, inventory)
    monkeypatch.setattr(audit, "verify_file", checked)
    rows, files = audit.sample_pairs(inventory, profile, data_root=data, files_per_source=2, rows_per_file=4)
    native = {j.identity: j for j in DatasetReader(data, inventory, profile, role="train", include_offline=True)}
    assert opened and rows and {r["identity"] for r in rows} <= set(native)
    assert len(rows) <= 16
    for row in rows:
        assert row["hlt"] == audit.describe(native[row["identity"]].hlt)
        assert row["offline"] == audit.describe(native[row["identity"]].offline)
        assert row["producer_dr"] is None
    second, second_files = audit.sample_pairs(inventory, profile, data_root=data, files_per_source=2, rows_per_file=4)
    assert second == rows and second_files == files
    with pytest.raises(PermissionError, match="final_test"):
        audit.sample_pairs(inventory, profile, data_root=data, role="final_test")
    bad = copy.deepcopy(profile)
    bad["memberships"]["train"]["rows"] += 1
    with pytest.raises(ValueError):
        audit.sample_pairs(inventory, bad, data_root=data)
