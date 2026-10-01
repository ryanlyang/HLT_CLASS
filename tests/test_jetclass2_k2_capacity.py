from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from hlt_classification.data.cache_contracts import with_content_hash
from hlt_classification.jetclass2_delphes.capacity_audit import (
    audit_scope, count_histogram, distribution, grouped_summary, summarize,
)
from hlt_classification.jetclass2_delphes.partial_snapshot import build_partial_snapshot_plan
from hlt_classification.jetclass2_delphes.splits import build_splits
from test_jetclass2_delphes_partial_snapshot import synthetic_inventory
from test_jetclass2_delphes import snapshot


def test_exact_capacity_predicate_and_particle_denominators():
    h = np.array([2, 2, 3, 1])
    o = np.array([4, 5, 3, 5])
    hist = count_histogram(h, o, np.array([0, 0, 1, 1]))
    result = summarize(hist)
    assert result["jets"] == 4
    assert result["overflow_jets"] == 2
    assert result["overflow_fraction"] == .5
    assert result["equality_jets"] == 1
    assert result["offline_particles_to_crop"] == 4
    assert result["fraction_all_offline_particles_cropped"] == 4 / 17
    assert result["fillers"] == 3
    assert result["overflow_excess"]["mean"] == 2
    assert result["overflow_crop_fraction"]["mean"] == .4
    assert grouped_summary(hist)["by_class"]["QCD"]["overflow_jets"] == 1


def test_histogram_aggregation_and_nearest_rank_are_chunk_invariant():
    h = np.array([1, 2, 4, 4, 2, 1])
    o = np.array([1, 5, 8, 9, 5, 2])
    c = np.zeros(len(h), np.int64)
    whole = count_histogram(h, o, c)
    pieces = count_histogram(h[:3], o[:3], c[:3]) + count_histogram(h[3:], o[3:], c[3:])
    assert whole == pieces
    assert summarize(whole) == summarize(pieces)
    d = distribution([10, 20], [3, 1])
    assert d["quantiles"]["0.5"] == 10
    assert d["quantiles"]["0.9"] == 20
    assert d["mean"] == 12.5
    assert summarize(Counter()) == {"jets": 0}


@pytest.mark.parametrize("hlt,offline,labels", [([0], [1], [0]), ([1], [-1], [0]),
                                              ([1.5], [1], [0]), ([1], [1], [11]),
                                              ([1, 2], [1], [0])])
def test_invalid_count_arrays_fail_closed(hlt, offline, labels):
    with pytest.raises(ValueError):
        count_histogram(hlt, offline, labels)


def test_scope_seals_union_of_both_test_roles_and_replays_parent_hashes():
    inventory = synthetic_inventory()
    local = build_splits(inventory)
    plan = build_partial_snapshot_plan(inventory, training_sizes=(2_000,),
                                       validation_size=1_000, test_size=1_000)
    scope = audit_scope(inventory, local, plan)
    for path in scope["included_paths"]:
        assert scope["local_roles"][path] != "final_test"
        assert scope["sporc_roles"].get(path) != "final_test"
    for path in scope["excluded_test_union_paths"]:
        assert "final_test" in (scope["local_roles"][path], scope["sporc_roles"].get(path))
    assert set(scope["included_paths"]).isdisjoint(scope["excluded_test_union_paths"])
    assert len(scope["included_paths"]) + len(scope["excluded_test_union_paths"]) == inventory["file_count"]
    corrupt = deepcopy(plan)
    corrupt["projected_inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit_scope(inventory, local, with_content_hash(corrupt))


def test_real_root_audit_opens_only_safe_files_and_is_immutable(snapshot, tmp_path, monkeypatch):
    from hlt_classification.data.cache_contracts import write_immutable_json, load_json
    from hlt_classification.jetclass2_delphes import capacity_audit as audit
    data, inventory, local = snapshot
    plan = build_partial_snapshot_plan(inventory, training_sizes=(30,),
                                       validation_size=11, test_size=11, headroom_fraction=0)
    scope = audit_scope(inventory, local, plan)
    inv_path, split_path, plan_path = [tmp_path / n for n in ("inventory.json", "splits.json", "plan.json")]
    for path, value in ((inv_path, inventory), (split_path, local), (plan_path, plan)):
        write_immutable_json(path, value)
    real_open = audit.uproot.open
    opened = []
    def guarded_open(path, **kwargs):
        relative = Path(path).relative_to(data).as_posix()
        assert relative in scope["included_paths"]
        assert relative not in scope["excluded_test_union_paths"]
        opened.append(relative)
        return real_open(path, **kwargs)
    monkeypatch.setattr(audit.uproot, "open", guarded_open)
    args = dict(data_root=data, inventory_path=inv_path, splits_path=split_path,
                partial_plan_path=plan_path, output_root=tmp_path / "audit", step_size=2)
    report = audit.run_audit(**args)
    assert opened == scope["included_paths"]
    assert report["groups"]["all"]["overall"]["overflow_jets"] == 0
    assert report["groups"]["all"]["overall"]["jets"] == sum(
        sum(r["selected_class_counts"]) for r in inventory["files"] if r["path"] in opened)
    assert report["particle_arrays_read"] is False
    assert report["final_test_accessed"] is False
    assert load_json(args["output_root"] / "report.json") == report
    with pytest.raises(ValueError, match="fresh output"):
        audit.run_audit(**args)
    from hlt_classification.jetclass2_delphes.capacity_pt_audit import run_pt_audit
    pt_report = run_pt_audit(count_root=args["output_root"], inventory_path=inv_path,
                             output_root=tmp_path / "pt_audit", step_size=3)
    assert pt_report["selected_jets_rechecked"] == report["groups"]["all"]["overall"]["jets"]
    assert pt_report["groups"]["all"]["minimum_dropped_scalar_pt"] == 0
    assert pt_report["final_test_accessed"] is False
    assert opened == 2 * scope["included_paths"]


def test_minimum_pt_loss_is_a_count_constrained_lower_bound():
    import itertools
    from hlt_classification.jetclass2_delphes.capacity_pt_audit import minimum_pt_crop
    pt = [1., 2., 4., 8., 16., 32.]
    result = minimum_pt_crop(pt, 2)
    assert result["dropped_particles"] == 2
    assert result["minimum_dropped_scalar_pt"] == 3
    assert result["minimum_dropped_pt_fraction"] == 3 / 63
    assert all(sum(pair) >= 3 for pair in itertools.combinations(pt, 2))
    assert minimum_pt_crop(pt, 3)["minimum_dropped_scalar_pt"] == 0
    for bad in ([0.], [float("nan")], [-1.]):
        with pytest.raises(ValueError):
            minimum_pt_crop(bad, 1)
