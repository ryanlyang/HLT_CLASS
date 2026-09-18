from __future__ import annotations

from types import SimpleNamespace
import pytest

from hlt_classification.cms2jc2_response.contracts import artifact, with_content_hash
from hlt_classification.cms2jc2_response.graph import science_graph, validate_graph
from hlt_classification.cms2jc2_response.parallel import ShardedRecords
from hlt_classification.cms2jc2_response.response import collect
from hlt_classification.cms2jc2_response.association import policy
from hlt_classification.cms2jc2_response import storage
from test_cms2jc2_response_science import pairs


def test_science_graph_covers_33_fits_and_combined_cpu_budget():
    graph = science_graph()
    validate_graph(graph)
    assert len(graph["tasks"]) == 71
    fits = [r for r in graph["tasks"] if r["kind"] == "fit"]
    evals = [r for r in graph["tasks"] if r["kind"] == "evaluate"]
    assert len(fits) == len(evals) == 33
    # Each kind occupies a fixed number of serial dependency lanes, including
    # the sensitivity wave. Cross-kind overlap is bounded by 2*16+4*8=64.
    for rows, width in ((fits, 2), (evals, 4)):
        for i in range(width, len(rows)):
            assert rows[i-width]["task_id"] in rows[i]["depends_on"]
    assert 2*fits[0]["resources"]["cpus"]+4*evals[0]["resources"]["cpus"] == 64
    graph["tasks"][1]["resources"] = {**graph["tasks"][1]["resources"], "cpus": 72}
    with pytest.raises(ValueError):
        validate_graph(with_content_hash(graph))


def test_sharded_records_preserve_each_sampling_weight():
    a, _ = collect(pairs("a", 10), policy(), cap=72)
    b, _ = collect(pairs("b", 20), policy(), cap=72)
    combined = ShardedRecords([a, b])
    x, y, w, ids = combined.arrays("singleton")
    assert w.sum() == 60 and len(x) == len(y) == len(w) == len(ids)
    assert set(ids) <= {p.identity for name, count in (("a", 10), ("b", 20)) for p in pairs(name, count)}


def test_storage_counts_failed_outputs_and_rejects_budget_growth(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))
    value = artifact("TEST", value=1)
    path = storage.publish_json(tmp_path, "reports/one.json", value, "TEST", remaining_bytes=10000)
    assert path.is_file()
    assert storage.publish_json(tmp_path, "reports/one.json", value, "TEST", remaining_bytes=10000) == path
    monkeypatch.setattr(storage, "TOTAL_CAP", path.stat().st_size+5)
    with pytest.raises(OSError, match="12 GiB"):
        storage.publish_json(tmp_path, "failed_attempt/report.json", value, "TEST", remaining_bytes=10000)
    with pytest.raises(ValueError):
        storage.publish_json(tmp_path, "../escape.json", value, "TEST", remaining_bytes=10000)
    assert not (tmp_path/".publication_lock").exists()


def test_storage_report_budget_and_free_headroom(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100*storage.GIB))
    monkeypatch.setattr(storage, "REPORT_CAP", 10)
    with pytest.raises(OSError, match="2 GiB"):
        storage.publish_json(tmp_path, "reports/one.json", artifact("TEST"), "TEST", remaining_bytes=10000)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="headroom"):
        storage.publish_json(tmp_path, "fits/one.json", artifact("TEST"), "TEST", remaining_bytes=10000)


def test_fit_engine_rejects_partial_population_and_does_not_write_records(tmp_path, monkeypatch):
    from hlt_classification.cms2jc2_response import worker
    from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
    monkeypatch.setattr(worker, "validate_inventory", lambda value: None)
    monkeypatch.setattr(worker, "validate_roles", lambda *args: None)
    monkeypatch.setattr(worker, "validate_memberships", lambda *args: None)
    monkeypatch.setattr(worker, "validate_compatibility", lambda *args, **kwargs: None)

    def collect_fake(**kwargs):
        return collect(pairs(kwargs["fit_role"], 12), kwargs["rules"], cap=720)

    monkeypatch.setattr(worker, "parallel_collect", collect_fake)
    roles = {"content_hash": "b"*64}
    membership = {"content_hash": "c"*64, "budgets": {"FULL": {
        "fit_location": [{"selected_entries": 12}], "fit_residual": [{"selected_entries": 12}]}}}
    args = dict(cms_root=tmp_path, inventory={}, roles=roles, membership=membership,
                review=provisional_compatibility(inventory_hash="a"*64), source_hash="d"*64,
                candidate_id="A_L", budget="FULL", cpus=1, cap=720)
    args["inventory"] = {"content_hash": "a"*64}
    model, report = worker.fit_task(**args)
    assert report["full_registered_fit"] and model["budget"] == "FULL"
    assert not report["durable_calibration_records"] and not list(tmp_path.iterdir())
    membership["budgets"]["FULL"]["fit_location"][0]["selected_entries"] = 13
    with pytest.raises(ValueError, match="declared population"):
        worker.fit_task(**args)
    with pytest.raises(ValueError, match="declared population"):
        worker.fit_task(**args, probe_jets=20_000)
