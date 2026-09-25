from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hlt_classification.cms2jc2_response import b_tracking as campaign, b_tracking_worker as worker
from hlt_classification.cms2jc2_response import dev_campaign as dev, dev_data as data, dev_submission as submission
from hlt_classification.cms2jc2_response import dev_worker as old_worker
from hlt_classification.cms2jc2_response.b_tracking_generation import (
    TrackingGenerator, VARIANTS, MASKS, residual_draw, check_tracking_replay)
from hlt_classification.cms2jc2_response.b_tracking_metrics import TrackingCounters
from hlt_classification.cms2jc2_response.contracts import artifact, load_json, with_content_hash, sha256_file
from hlt_classification.cms2jc2_response.c_diagnostic_metrics import combine
from hlt_classification.cms2jc2_response.response import Generator, collect, fit_response
from hlt_classification.cms2jc2_response.assumptions import provisional_compatibility
from test_cms2jc2_response_science import particles, pairs
from test_cms2jc2_frozen_c import backend_fixture, frozen_root, free_disk


@pytest.fixture(scope="module")
def fitted_b():
    from threadpoolctl import threadpool_limits
    rules = dev.POLICIES["BASE"]
    loc, lr = collect(pairs("location"), rules, cap=7200)
    res, rr = collect(pairs("residual"), rules, cap=7200)
    with threadpool_limits(limits=1):
        return fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="B_L",
            review=provisional_compatibility(inventory_hash="a"*64), rules=rules, budget="TEST", source_hash="b"*64)


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_b_fit_exact_replay_and_unchanged_fields(fitted_b, variant):
    original, diagnostic = Generator(fitted_b), TrackingGenerator(fitted_b, variant)
    for p in (particles(), particles().take([1, 0]), particles().take([])):
        for replica in range(3):
            expected, old_info = original(p, jet="held", replica=replica, trace=True)
            actual, info = diagnostic(p, jet="held", replica=replica)
            check_tracking_replay(expected, old_info, actual, info, variant=variant)
            repeated, _ = diagnostic(p, jet="held", replica=replica)
            np.testing.assert_array_equal(actual.tracking, repeated.tracking)
            counter = TrackingCounters(); counter.add(info, variant)
            assert sum(counter.value["selected_residual_cells"].values()) == len(info["events"])
    with pytest.raises(ValueError):
        TrackingGenerator(with_content_hash(dict(fitted_b, candidate_id="C_L")), variant)


@pytest.mark.parametrize("dim", [8, 16])
def test_tracking_masks_apply_to_both_split_siblings(dim):
    backend = backend_fixture(dim)
    args = dict(jet="held", replica=1, object_key="particle", module="emission2", factors={})
    original, old_info, _ = residual_draw(backend, [0, 0, 0, 0], variant="FULL", **args)
    for v in VARIANTS:
        result, info, detail = residual_draw(backend, [0, 0, 0, 0], variant=v, **args)
        mask = np.array([j % 8 in MASKS[v] for j in range(dim)])
        np.testing.assert_array_equal(result[mask], 0)
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(detail["raw_residual"], original)
        assert info == old_info and detail["selected_residual_cell"] == "0"
    result, _, detail = residual_draw(backend, [5, 0, 0, 0], variant="TRACK_CENTRAL", **args)
    assert result is None and detail["selected_residual_cell"] == "missing_category"


@pytest.mark.parametrize("missing", ["module", "backend"])
def test_fallbacks_keep_original_behavior_and_expose_reason(fitted_b, missing):
    response = deepcopy(fitted_b)
    if missing == "module":
        del response["modules"]["emission1_value"]
    else:
        response["modules"]["emission1_value"]["backends"] = []
    response = with_content_hash(response)
    original, old_info = Generator(response)(particles(), jet="held", trace=True)
    for v in VARIANTS:
        actual, info = TrackingGenerator(response, v)(particles(), jet="held")
        check_tracking_replay(original, old_info, actual, info, variant=v)
        stats = TrackingCounters(); stats.add(info, v)
        suffix = "identity_fallback" if missing == "module" else "missing_state_backend"
        assert all(k.endswith(suffix) for k in stats.value["selected_residual_cells"])


@pytest.mark.parametrize("mechanism", ["merged", "split", "additional", "loss"])
def test_traversal_parity_across_topologies(fitted_b, mechanism):
    from hlt_classification.cms2jc2_response.bridge import Particles, p4_from_coordinates
    response = deepcopy(fitted_b)
    p = particles()
    p = Particles(p4_from_coordinates([2., 10.], [0., .01], [0., .01], [.1, .2]),
                  p.charge, p.category, p.tracking, p.valid, p.keys)
    modules = response["modules"]
    modules["merge"] = dict(states=[[int(mechanism == "merged")]], model=None)
    modules["singleton"] = dict(states=[[2 if mechanism == "split" else 0 if mechanism == "loss" else 1]], model=None)
    modules["additional_count"] = dict(states=[[int(mechanism == "additional")]], model=None)
    if mechanism == "additional":
        modules["additional_state"] = deepcopy(modules["emission1_state"])
        modules["additional_value"] = deepcopy(modules["emission1_value"])
    if mechanism == "split":
        modules["emission2_state"] = dict(states=[[0, 1, 15, 1, 1, 0, 0, 1]], model=None)
    response = with_content_hash(response)
    expected, old_info = Generator(response)(p, jet="held", trace=True)
    for v in VARIANTS:
        actual, info = TrackingGenerator(response, v)(p, jet="held")
        check_tracking_replay(expected, old_info, actual, info, variant=v)


def test_counters_filter_invalid_placeholders_and_split_clipping(fitted_b):
    _, info = TrackingGenerator(fitted_b, "FULL")(particles(), jet="held")
    event = deepcopy(info["events"][0])
    event.update(state=[0, 1, 15, 1, 1, 0, 0, 1], central=np.full(16, 2.),
        raw_increment=np.full(16, 1.), applied_increment=np.full(16, 1.),
        response=(np.full(16, 3.), np.ones(16)), response_limits=[[-1.]*16, [1.]*16],
        scale=(np.full(16, 2.), np.ones(16)), residual_available=True)
    event.pop("output", None)
    info["events"] = [event]
    stats = TrackingCounters(); stats.add(info, "DZ_CENTRAL")
    total = combine(stats.value, stats.value)
    for key, row in total["tracking_coordinates"].items():
        if "/pid0/" in key:
            assert row["applicable"] == row["high_clipped"] == row["central_outside"] == 2
            assert row["scale_evaluated"] == row["scale_clipped"] == 2
            assert row["absolute_correction"]["sum"] == 4
            assert row["residual_disabled"] == (2 if key.endswith("/dz") else 0)
        else:
            assert row == dict(applicable=0, inapplicable=2)
    import json
    text = json.dumps(total, allow_nan=False)
    assert "held" not in text and "input_keys" not in text
    info["events"][0]["central"][5] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        TrackingCounters().add(info, "FULL")


@pytest.fixture
def b_root(frozen_root, tmp_path, monkeypatch):
    _, ctx, put, _, donor = frozen_root
    for module in (campaign, worker):
        monkeypatch.setattr(module, "COUNTS", data.COUNTS)
    loc, lr = old_worker.collected(ctx, "location", dev.POLICIES["BASE"], 1)
    res, rr = old_worker.collected(ctx, "residual", dev.POLICIES["BASE"], 1)
    study = dev.validate_stage(donor)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        fitted = fit_response(loc, res, location_report=lr, residual_report=rr, candidate_id="B_L",
            review=ctx["review"], rules=dev.POLICIES["BASE"], budget="SYNTHETIC_DEV", source_hash=study["source"]["content_hash"])
    put(donor, "candidate_B_L", dict(response=fitted, fit=artifact("DEV_FIT",
        parents={"response": fitted["content_hash"], "samples": ctx["samples"]["content_hash"]}, candidate="B_L")))
    for i in range(4):
        task = next(t for t in donor["tasks"] if t["task_id"] == f"evaluate_B_{i}")
        put(donor, task["task_id"], dict(result=old_worker.evaluate_task(ctx, donor, task)))
    put(donor, "report_B", dict(result=old_worker.report_task(ctx, donor, dict(params=dict(candidate="B_L")))))
    before = {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    spec = campaign.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir=tmp_path/"b_project",
                           source_commit="f"*40, root=tmp_path/"b_audit")
    return spec, ctx, put, before, donor


def test_end_to_end_b_frozen_reuse_processes_reports_and_submission(b_root, monkeypatch):
    spec, ctx, put, before, donor = b_root
    plan = submission.submit(spec)
    assert len(plan["commands"]) == 6 and plan["gpus"] == 0 and plan["cpu_upper_bound"] == 144
    assert [t["cpus"] for t in spec["tasks"]] == [2, 36, 36, 36, 36, 1]
    assert all("--partition=tier3" in r["argv"] and not any("--gres" in a for a in r["argv"]) for r in plan["commands"])
    bad = deepcopy(spec); bad["variants"] = ["FULL"]
    with pytest.raises(ValueError, match="registration"):
        dev.validate_stage(with_content_hash(bad))
    changed = deepcopy(dev.validate_stage(spec)); changed["source"]["files"]["scientific.py"] = "d"*64
    with pytest.raises(ValueError, match="scientific source"):
        campaign.reuse(donor, changed)
    with pytest.raises(PermissionError, match="disjoint"):
        campaign.create(parent_spec=dev.stage_dir(donor)/"stage_spec.json", project_dir="unused",
                        source_commit="f"*40, root=donor["root"])
    calls = []
    def scheduler(argv):
        calls.append(argv)
        out = "PartitionName=tier3 State=UP" if argv[0] == "scontrol" else str(4000+len(calls))+"\n"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(submission, "scheduler", scheduler)
    with pytest.raises(PermissionError):
        submission.submit(spec, execute=True, reviewed_plan_hash=plan["content_hash"])
    ledger = submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["btrack"], reviewed_plan_hash=plan["content_hash"])
    live = [cmd for cmd in calls if cmd[0] == "sbatch" and "--test-only" not in cmd]
    assert len(live) == 6
    assert all("--dependency=afterok:"+ledger["jobs"]["bt_acceptance"] in cmd for cmd in live[1:5])
    assert "--dependency=afterok:"+":".join(ledger["jobs"][f"bt_eval_{i}"] for i in range(4)) in live[-1]
    assert submission.submit(spec, execute=True, authorization_phrase=dev.PHRASES["btrack"], reviewed_plan_hash=plan["content_hash"]) == ledger
    accepted = worker.acceptance(ctx, spec)
    assert accepted["serial_process_parity"] and accepted["unchanged_kinematics_and_state"]
    put(spec, "bt_acceptance", dict(result=accepted))
    for task in spec["tasks"][1:5]:
        task = dict(task, cpus=2 if task["params"]["shard"] == 0 else 1)
        put(spec, task["task_id"], dict(result=worker.evaluate(ctx, spec, task)))
    from hlt_classification.cms2jc2_response import dev_diagnostics as diag
    def one_page(*args, cohort, **kw):
        if cohort != "all": return
        pages = diag.plot_pages(*args, cohort=cohort, **kw)
        try:
            page = next(pages, None)
            if page is not None: yield page
        finally: pages.close()
    monkeypatch.setattr(worker, "plot_pages", one_page)
    result = worker.report(ctx, spec)
    assert result["jets"] == 12 and result["figures"] and not result["final_test_accessed"]
    assert set(result["comparisons"]) == set(VARIANTS)
    assert all(result["instrumentation"][v]["jet_replicas"] == 36 for v in VARIANTS)
    put(spec, "bt_report", dict(result=result))
    from hlt_classification.cms2jc2_response.b_tracking_results import read, render
    snapshot = {p: sha256_file(p) for p in Path(spec["root"]).rglob("*") if p.is_file()}
    def forbidden(*a, **kw): raise AssertionError("Read-only results attempted generation/data/scheduler/publication")
    monkeypatch.setattr(worker, "sample_stream", forbidden)
    monkeypatch.setattr(Generator, "__call__", forbidden)
    monkeypatch.setattr(submission, "scheduler", forbidden)
    monkeypatch.setattr(dev, "write", forbidden)
    assert "DZERR_CENTRAL" in render(read(spec))
    assert before == {p: sha256_file(p) for p in Path(donor["root"]).rglob("*") if p.is_file()}
    assert snapshot == {p: sha256_file(p) for p in Path(spec["root"]).rglob("*") if p.is_file()}
    path = Path(donor["root"])/dev.verified_outputs(donor, "candidate_B_L")["outputs"]["response"]["relative"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        read(spec)
